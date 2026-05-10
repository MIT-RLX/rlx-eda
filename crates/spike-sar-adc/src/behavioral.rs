//! Behavioral non-ideal N-bit SAR ADC for system-level
//! characterization (INL/DNL, ENOB/SFDR/SNDR, Monte Carlo mismatch,
//! comparator metastability, etc.).
//!
//! This is a *system model*, not a transistor netlist — it implements
//! the SAR algorithm directly with parameterizable noise, mismatch,
//! and offset sources at the same hooks an analog designer cares
//! about. The transistor-level SAR netlist (`SarAdc<N>` in this same
//! crate) is the artifact we *verify against silicon*; the behavioral
//! model is what we *characterize the architecture with*.

use std::cell::Cell;

/// All quantities are SI (volts, seconds). Defaults are sensible for
/// a 1.8 V, 8-bit Sky130-class SAR with hand-built R-2R DAC: 0.5 %
/// resistor σ, 1 mV comparator offset σ, 2 mV input-referred noise σ,
/// no S/H droop.
#[derive(Debug, Clone)]
pub struct BehavioralSar {
    pub n_bits: usize,
    pub vref:   f64,
    /// Per-resistor relative σ (e.g. `5e-3` for 0.5 % resistor matching
    /// from a hand-trimmed R-2R network).
    pub r_mismatch_sigma: f64,
    /// Comparator input-referred offset (1-σ), volts.
    pub comp_offset_sigma: f64,
    /// Comparator input-referred thermal noise (1-σ per decision), volts.
    pub comp_noise_sigma: f64,
    /// S/H droop coefficient: vhold(t) = vsample · exp(-t / τ_droop). τ in seconds.
    /// Set to `None` for an ideal track-and-hold.
    pub sh_droop_tau: Option<f64>,
    /// Time the held sample sits before the comparator decisions complete.
    pub conversion_time: f64,
    /// Comparator decision time (regenerative latch RC). Used by the
    /// metastability model: the longer you wait, the smaller the
    /// undecided window.
    pub comp_decision_time: f64,
    /// τ of the comparator's regenerative latch — determines the
    /// metastability window: P(undecided) ∝ exp(−t_decision / τ).
    pub comp_latch_tau: f64,
}

impl Default for BehavioralSar {
    fn default() -> Self {
        Self {
            n_bits: 8,
            vref:   1.8,
            r_mismatch_sigma:   5e-3,
            comp_offset_sigma:  1e-3,
            comp_noise_sigma:   2e-3,
            sh_droop_tau:       None,
            conversion_time:    1e-6,
            comp_decision_time: 1e-9,
            comp_latch_tau:     50e-12,    // 50 ps regenerative latch τ (Sky130-class)
        }
    }
}

/// One realization of the random per-instance mismatches: per-DAC-bit
/// resistor errors + a single comparator offset. Sampled once at
/// "manufacture time" (== call site of `Realization::sample`), reused
/// across many `convert(vin)` calls.
#[derive(Debug, Clone)]
pub struct Realization {
    pub bit_weight_err: Vec<f64>,  // multiplicative error per bit weight; len = n_bits
    pub comp_offset:    f64,
}

impl Realization {
    /// Sample one realization of the per-instance random parameters.
    pub fn sample(spec: &BehavioralSar, rng: &mut dyn RngF64) -> Self {
        // R-2R DAC: each bit's effective weight has σ = sqrt(2)·σ_R
        // (R + 2R series in the ladder rung). Approximated as a single
        // multiplicative Gaussian per bit weight.
        let bit_weight_err = (0..spec.n_bits)
            .map(|_| 1.0 + std::f64::consts::SQRT_2 * spec.r_mismatch_sigma * rng.gauss())
            .collect();
        let comp_offset = spec.comp_offset_sigma * rng.gauss();
        Self { bit_weight_err, comp_offset }
    }

    pub fn ideal() -> Self {
        Self { bit_weight_err: vec![1.0; 64], comp_offset: 0.0 }
    }
}

impl BehavioralSar {
    /// Quantize one input sample with this realization. Adds per-decision
    /// comparator thermal noise from `rng`. Optional `t_hold_before_decision`
    /// applies S/H droop before the decision starts.
    pub fn convert(
        &self,
        vin: f64,
        real: &Realization,
        rng: &mut dyn RngF64,
    ) -> u32 {
        // S/H: optional exponential droop over the conversion window.
        let v_held = match self.sh_droop_tau {
            Some(tau) if tau > 0.0 => vin * (-self.conversion_time / tau).exp(),
            _ => vin,
        };

        let lsb_ideal = self.vref / (1u32 << self.n_bits) as f64;
        let mut code = 0u32;
        let mut v_dac = 0.0_f64;
        for i in (0..self.n_bits).rev() {
            // Trial DAC voltage with per-bit mismatch.
            let bit_v = (1u32 << i) as f64 * lsb_ideal
                * real.bit_weight_err.get(i).copied().unwrap_or(1.0);
            let trial = v_dac + bit_v;
            // Comparator decision = (v_held − trial) > Vos + n.
            let n = self.comp_noise_sigma * rng.gauss();
            if (v_held - trial) > (real.comp_offset + n) {
                code |= 1 << i;
                v_dac = trial;
            }
        }
        code
    }

    /// One LSB in volts (ideal).
    pub fn lsb(&self) -> f64 { self.vref / (1u32 << self.n_bits) as f64 }

    /// Number of output codes (= 2^N).
    pub fn levels(&self) -> u32 { 1u32 << self.n_bits }

    /// Probability that the regenerative comparator latch has NOT
    /// resolved within `t_decision`, as a function of the differential
    /// input at the latch's input. Standard model: latch output grows
    /// as exp(t/τ); needs to clear ~Vdd/2 to register as a logic level.
    /// Returns 1.0 when v_diff = 0 (perfect tie).
    pub fn metastability_prob(&self, v_diff: f64) -> f64 {
        let need = self.vref * 0.5;
        let initial = v_diff.abs().max(1e-12);
        let gain_required = need / initial;
        let t_required = self.comp_latch_tau * gain_required.ln().max(0.0);
        let surplus = self.comp_decision_time - t_required;
        if surplus <= 0.0 { 1.0 } else { (-surplus / self.comp_latch_tau).exp() }
    }
}

/// f64-producing RNG abstraction so call sites can pick determinism vs.
/// system entropy. Minimal — just `gauss()` and `uniform()`.
pub trait RngF64 {
    fn uniform(&mut self) -> f64;
    fn gauss(&mut self) -> f64 {
        // Box-Muller, returns one of the two normals each call by
        // caching the second.
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Deterministic 64-bit LCG → uniform[0, 1). Splitmix-style; not
/// cryptographic, fine for Monte Carlo mismatch sweeps.
#[derive(Debug, Clone)]
pub struct Lcg64 { pub state: u64 }
impl Lcg64 {
    pub fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) } }
}
impl RngF64 for Lcg64 {
    fn uniform(&mut self) -> f64 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // [0, 1) via 53-bit mantissa.
        ((z >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }
}

/// INL/DNL of one realization, one LSB units. Drives a slow input
/// ramp through `n_samples_per_code · 2^N` samples and computes the
/// actual transition voltages.
pub fn measure_inl_dnl(
    spec: &BehavioralSar,
    real: &Realization,
    samples_per_code: usize,
) -> (Vec<f64>, Vec<f64>) {
    let levels = spec.levels() as usize;
    let n = levels * samples_per_code;
    // Walk vin from 0 to vref; histogram each output code.
    let mut hist = vec![0u32; levels];
    let mut rng_noise = Lcg64::new(42);
    for k in 0..n {
        let vin = (k as f64 + 0.5) * spec.vref / n as f64;
        let code = spec.convert(vin, real, &mut rng_noise) as usize;
        if code < levels { hist[code] += 1; }
    }
    // Each code's actual width = hist count / total ramp samples; the
    // ideal width = vref/levels. DNL_i = (width_i / ideal) − 1, in LSB.
    let ideal_width = (n / levels) as f64;
    let dnl: Vec<f64> = hist.iter().map(|&h| h as f64 / ideal_width - 1.0).collect();
    // INL = cumulative DNL.
    let mut inl = Vec::with_capacity(levels);
    let mut acc = 0.0;
    for &d in &dnl { acc += d; inl.push(acc); }
    // Center INL by removing best-fit straight line (endpoint correction).
    if inl.len() > 1 {
        let last = *inl.last().unwrap();
        let m = last / (inl.len() as f64 - 1.0);
        for (i, v) in inl.iter_mut().enumerate() { *v -= m * i as f64; }
    }
    (inl, dnl)
}

/// Convenience: thread-safe single-shot Realization::sample with a fresh LCG.
pub fn realization_seed(spec: &BehavioralSar, seed: u64) -> Realization {
    let mut rng = Lcg64::new(seed);
    Realization::sample(spec, &mut rng)
}

// One-shot RNG for convenience in places that don't want to thread
// state. NOT a global — each instance is fresh.
pub fn fresh_lcg(seed: u64) -> Lcg64 { Lcg64::new(seed) }

/// Used as a tiny shim where an `&mut dyn RngF64` is needed but the
/// caller already has an LCG instance.
pub fn as_dyn(r: &mut Lcg64) -> &mut dyn RngF64 { r }

// `Cell<u64>` import is unused in the public surface but keeps the
// option open for thread-cell-based RNGs in follow-ups.
#[allow(dead_code)] fn _keep_cell_import() -> Cell<u64> { Cell::new(0) }
