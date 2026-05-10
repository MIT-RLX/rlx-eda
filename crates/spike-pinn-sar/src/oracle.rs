//! Deterministic oracle: ideal 8-bit `BehavioralSar`.
//!
//! All mismatch / noise σ are zero, `Realization::ideal()` provides
//! unit per-bit weights and zero comparator offset. The output is
//! the exact 256-step quantisation `code = floor(x · 256)` for
//! `x ∈ [0, 1)`, with the boundary case `x = 1.0` clamped to 255.

use spike_sar_adc::behavioral::{BehavioralSar, Lcg64, Realization};

use crate::config::*;

fn ideal_spec() -> BehavioralSar {
    BehavioralSar {
        n_bits: N_BITS,
        vref:   VREF as f64,
        r_mismatch_sigma:   0.0,
        comp_offset_sigma:  0.0,
        comp_noise_sigma:   0.0,
        sh_droop_tau:       None,
        conversion_time:    1e-6,
        comp_decision_time: 1e-9,
        comp_latch_tau:     50e-12,
    }
}

/// Predict `code/256 ∈ [0, 1]` for a single normalised input `x = vin/vref`.
pub fn truth_norm(x_norm: f32) -> f32 {
    let spec = ideal_spec();
    let real = Realization::ideal();
    // RNG output is multiplied by σ=0, so its sequence is irrelevant
    // to the result; using a fixed seed makes tests reproducible.
    let mut rng = Lcg64::new(0xC0DE_C0DE);
    let code = spec.convert(x_norm as f64, &real, &mut rng);
    code as f32 / LEVELS as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_values_quantise_correctly() {
        // x = 0   → code 0    → 0/256 = 0.0
        // x = 0.5 → code 128  → 128/256 = 0.5
        // x = 0.999 → code 255 → 255/256 ≈ 0.996
        assert!((truth_norm(0.0) - 0.0).abs() < 1e-6);
        assert!((truth_norm(0.5) - 0.5).abs() < 1e-3);
        assert!((truth_norm(0.999) - (255.0 / 256.0)).abs() < 1e-3);
    }

    #[test]
    fn output_is_monotone_non_decreasing() {
        let xs: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let mut prev = 0.0_f32;
        for x in xs {
            let y = truth_norm(x);
            assert!(y + 1e-9 >= prev, "non-monotone at x={x}: prev={prev}, y={y}");
            prev = y;
        }
    }
}
