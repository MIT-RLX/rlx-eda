//! Non-ideal R-2R evaluator: agreement with the closed form at nominal,
//! plus a perturbation-sensitivity sanity check.

use spike_dac_r2r::ideal_vout;
use spike_dado_r2r::{r_in_idx, r_value, solve_r2r, Design, DEVIATIONS, L, N_BITS, N_CODES};

#[test]
fn matches_ideal_at_nominal_for_every_code() {
    let nominal = [2u8; L]; // index 2 → 0% deviation
    for code in 0..N_CODES as u32 {
        let v = solve_r2r(&nominal, code, 1.0, 0.0);
        let target = ideal_vout(code, N_BITS as u32, 1.0, 0.0);
        assert!((v - target).abs() < 1e-12, "code {code}: {v} vs {target}");
    }
}

#[test]
fn deviation_indexing_recovers_nominal_2r_and_r() {
    // Index 2 is the centre of DEVIATIONS — a 0% perturbation. The
    // resolved ohms must match the textbook 2R / R values.
    let nominal: Design = [2u8; L];
    // r_term and r_in[*] are 2R = 20 kΩ.
    for idx in 0..=8 {
        assert!((r_value(&nominal, idx) - 20_000.0).abs() < 1e-9);
    }
    // Spine resistors are R = 10 kΩ.
    for idx in 9..=15 {
        assert!((r_value(&nominal, idx) - 10_000.0).abs() < 1e-9);
    }
    // The deviation alphabet is symmetric around zero.
    assert_eq!(DEVIATIONS[2], 0.0);
}

#[test]
fn uniform_scaling_leaves_vout_unchanged() {
    // Scaling every resistor by the same factor preserves all voltage
    // ratios in the network. Vout = ideal_vout(code, ...) regardless,
    // even when the ratios are far from the nominal R/2R relationship.
    // This is a strong topology invariant the solver must respect.
    let all_plus_5: Design = [4u8; L]; // every resistor +5%
    for code in [0u32, 1, 42, 128, 255] {
        let v = solve_r2r(&all_plus_5, code, 1.0, 0.0);
        let target = ideal_vout(code, N_BITS as u32, 1.0, 0.0);
        assert!((v - target).abs() < 1e-12,
                "code {code}: uniform scaling broke ideal vout ({v} vs {target})");
    }
}

#[test]
fn perturbing_a_single_resistor_breaks_ideal_vout_on_some_code() {
    // A perturbation of just one resistor away from nominal must move
    // vout off the ideal staircase for some code. (We don't assert
    // direction — that depends on the resistor's role in the ladder.)
    let mut perturbed: Design = [2u8; L];
    perturbed[r_in_idx(7)] = 4; // +5% on MSB feeder
    let mut max_shift = 0.0_f64;
    for code in 0..N_CODES as u32 {
        let v = solve_r2r(&perturbed, code, 1.0, 0.0);
        let target = ideal_vout(code, N_BITS as u32, 1.0, 0.0);
        max_shift = max_shift.max((v - target).abs());
    }
    assert!(max_shift > 1e-3, "expected a measurable shift, got {max_shift}");
}
