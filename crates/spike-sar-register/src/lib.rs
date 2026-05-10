//! `SarRegister<N>` — N-bit successive-approximation register.
//!
//! The "brain" of a SAR ADC's digital side. One `DffSR` per bit plus
//! one `Inverter` per bit (to derive `set_b`); the rest is wiring. The
//! algorithmic logic lives in **how the inputs are clocked**, not in
//! per-stage logic — which keeps the emitted netlist small and the
//! per-bit timing correctness easy to reason about.
//!
//! ## Algorithm
//!
//! For each bit i (0 ≤ i < N), the SAR algorithm is:
//!
//! 1. Trial: drive the DAC with `bit[i] = 1` and the previously-decided
//!    higher bits.
//! 2. Wait for DAC + comparator to settle.
//! 3. Latch: if comparator says vin > v_dac then keep `bit[i] = 1`,
//!    else clear it back to 0.
//!
//! In a phased system that maps directly onto a set-priority `DffSR`:
//!
//! ```text
//!   set_b[i]  = !phase[i]      ⇒ q[i] forced 1 during eval
//!   d[i]      = comparator_out
//!   clk[i]    = capture[i]     ⇒ latches comparator after eval settles
//!   reset_b   = global         (cleared at start of conversion)
//! ```
//!
//! ## Why a separate capture clock per bit
//!
//! Naively wiring `bit[i].clk = phase[i-1]` (capture on the next
//! phase's rising edge) creates a race: at the instant of the rising
//! edge, `bit[i-1].set` *also* activates, propagating through the DAC
//! and (essentially-instantaneous) comparator. The data path
//! (`set → DAC → cmp`) is ~1 NAND-gate delay; the clock path
//! (`phase → clk_inv → master.en → master locks`) is ~2 gate delays.
//! So the comparator's *new* output reaches `bit[i].master`'s d input
//! before master goes opaque — master captures the wrong value.
//!
//! Decoupling the capture clock from the next phase's rising edge
//! sidesteps this entirely. The caller generates `capture[i]` as a
//! clean pulse after `phase[i]` falls but before `phase[i-1]` rises.
//! With even ~50 ns of separation, the comparator settles fully on the
//! current trial before master locks.
//!
//! ## Phase / bit indexing convention
//!
//! Bits are LSB-first to match `spike-ripple-counter`'s `q[0]` = LSB.
//! The conversion fires phases **from MSB down to LSB**, so:
//!
//!   - `phase[N-1]` fires first (MSB trial)
//!   - `phase[0]` fires last (LSB trial)
//!   - `capture[i]` fires after `phase[i]` ends and before any other
//!     phase begins
//!
//! Realistic SAR clock generators produce both arrays from a shared
//! ring counter; the test harness uses PWL.
//!
//! ## Net order
//!
//! `[phase_0, phase_1, ..., phase_{N-1},
//!   cap_0,   cap_1,   ..., cap_{N-1},
//!   cmp, reset_b,
//!   bit_0,   bit_1,   ..., bit_{N-1},
//!   vdd, gnd]`
//!
//! Total: `3N + 4` terminals.

use eda_spice_emit::{EmitError, Netlist, SpiceEmit};
use spike_cmos_gates::{DffSR, Inverter};

pub mod mna;

#[derive(Debug, Clone, Copy)]
pub struct SarRegister<const N: usize = 8> {
    pub stage: DffSR,
    pub inv:   Inverter,
}

impl<const N: usize> Default for SarRegister<N> {
    fn default() -> Self {
        Self { stage: DffSR::default(), inv: Inverter::default() }
    }
}

impl<const N: usize> SarRegister<N> {
    pub const N_TERMINALS: usize = 3 * N + 4;
}

impl<const N: usize> SpiceEmit for SarRegister<N> {
    fn n_terminals(&self) -> usize { Self::N_TERMINALS }

    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != Self::N_TERMINALS {
            return Err(EmitError::ArityMismatch {
                block: format!("SarRegister<{N}>"),
                expected: Self::N_TERMINALS,
                got: nets.len(),
            });
        }

        let phases   = &nets[0..N];
        let captures = &nets[N..2 * N];
        let cmp      = nets[2 * N];
        let reset_b  = nets[2 * N + 1];
        let bits     = &nets[2 * N + 2..3 * N + 2];
        let vdd      = nets[3 * N + 2];
        let gnd      = nets[3 * N + 3];

        for i in 0..N {
            // set_b[i] = !phase[i].
            let set_b = format!("{id}_setb_{i}");
            self.inv.emit_spice(
                n,
                &[phases[i], &set_b, vdd, gnd],
                &format!("{id}_inv_{i}"),
            )?;

            // The DffSR's q is held internally; we then pass it through a
            // 2-inverter buffer chain before exposing as `bit[i]`. The
            // buffer drops the source impedance the DAC's resistive
            // ladder sees by a factor of ~2-3 — without it, the held bit
            // values droop dramatically when neighbouring bits transition
            // (R-2R + finite gate output impedance forms a voltage
            // divider that pulls held values below threshold). Two
            // cascaded inverters preserve polarity and isolate the
            // master-slave's internal SR latch from the DAC's load.
            let q_int = format!("{id}_qint_{i}");
            let qb_net = format!("{id}_qb_{i}");
            // DffSR: [d, clk, set_b, reset_b, q, qb, vdd, gnd]
            self.stage.emit_spice(
                n,
                &[cmp, captures[i], &set_b, reset_b, &q_int, &qb_net, vdd, gnd],
                &format!("{id}_dff_{i}"),
            )?;

            // Buffer chain: q_int → buf_mid → bits[i].
            let buf_mid = format!("{id}_bufmid_{i}");
            self.inv.emit_spice(
                n,
                &[&q_int, &buf_mid, vdd, gnd],
                &format!("{id}_buf_a_{i}"),
            )?;
            self.inv.emit_spice(
                n,
                &[&buf_mid, bits[i], vdd, gnd],
                &format!("{id}_buf_b_{i}"),
            )?;
        }
        Ok(())
    }
}

/// Pure-Rust SAR algorithm reference. Given `vin` in `[0, vref)`,
/// returns the N-bit code that the SAR algorithm would converge to,
/// assuming an ideal comparator and DAC.
///
/// Used by integration tests as the analytic witness against the
/// SPICE-simulated output.
pub fn ideal_sar_code(vin: f64, vref: f64, n_bits: usize) -> u32 {
    let mut code = 0u32;
    let mut v_dac = 0.0_f64;
    let lsb = vref / (1u32 << n_bits) as f64;
    // Use `>=` (boundary biases toward the upper code). With strict `>`
    // a vin landing exactly on a quantization boundary would round
    // down by 1 LSB; `>=` matches `floor(vin / lsb)` at the boundary.
    for i in (0..n_bits).rev() {
        let trial = v_dac + (1u32 << i) as f64 * lsb;
        if vin >= trial {
            code |= 1 << i;
            v_dac = trial;
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_matches() {
        let r4: SarRegister<4> = SarRegister::default();
        assert_eq!(r4.n_terminals(), 16);  // 4 phase + 4 cap + cmp + rb + 4 bit + vdd + gnd
        let r8: SarRegister<8> = SarRegister::default();
        assert_eq!(r8.n_terminals(), 28);
    }

    #[test]
    fn arity_mismatch_caught() {
        let mut net = Netlist::new("t");
        let r: SarRegister<4> = SarRegister::default();
        let err = r.emit_spice(&mut net, &["p0"], "u1").unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 16, got: 1, .. }));
    }

    #[test]
    fn emit_produces_per_bit_set_b_and_dff() {
        let mut net = Netlist::new("t");
        let r: SarRegister<4> = SarRegister::default();
        let nets = vec![
            "p0", "p1", "p2", "p3",
            "c0", "c1", "c2", "c3",
            "cmp", "rb",
            "b0", "b1", "b2", "b3",
            "vdd", "0",
        ];
        r.emit_spice(&mut net, &nets, "u1").unwrap();
        let body = net.body.join("\n");
        for i in 0..4 {
            assert!(body.contains(&format!("u1_setb_{i}")), "missing set_b net for bit {i}");
            assert!(body.contains(&format!("u1_dff_{i}")),  "missing DFF prefix for bit {i}");
        }
    }

    #[test]
    fn each_bit_uses_its_own_capture_clock() {
        let mut net = Netlist::new("t");
        let r: SarRegister<4> = SarRegister::default();
        let nets = vec![
            "p0", "p1", "p2", "p3",
            "c0", "c1", "c2", "c3",
            "cmp", "rb",
            "b0", "b1", "b2", "b3",
            "vdd", "0",
        ];
        r.emit_spice(&mut net, &nets, "u1").unwrap();
        let body = net.body.join("\n");
        // The DFF's internal clk-inverter line shows the input clk net.
        for i in 0..4 {
            let civ_line = body
                .lines()
                .find(|l| l.contains(&format!("u1_dff_{i}_civ")))
                .unwrap_or_else(|| panic!("missing bit-{i} clk-inverter line"));
            assert!(civ_line.contains(&format!(" c{i} ")),
                "bit-{i} clk should be `c{i}`, line was: {civ_line}");
        }
    }

    #[test]
    fn ideal_sar_code_matches_floor_quantization() {
        let vref = 1.8;
        let n = 8;
        for &vin in &[0.0, 0.1, 0.5, 0.9, 1.0, 1.4, 1.799999] {
            let code = ideal_sar_code(vin, vref, n);
            let expected = (vin / vref * 256.0).floor() as u32;
            assert_eq!(code, expected,
                "vin={vin}: SAR returned {code}, expected {expected}");
        }
    }

    #[test]
    fn ideal_sar_code_at_full_scale_is_all_ones() {
        let vref = 1.8;
        let n = 8;
        let lsb = vref / 256.0;
        let code = ideal_sar_code(vref - lsb / 2.0, vref, n);
        assert_eq!(code, 255);
    }
}
