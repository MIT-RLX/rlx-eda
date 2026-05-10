//! Tier 1: rlx Newton-unrolled-in-graph matches a pure-Rust Newton
//! implementation of the same recurrence.

use eda_validate::assert_close;
use spike_diode::*;

#[test]
fn rlx_forward_matches_rust_reference() {
    let cases = &[
        (1.0_f32, 1_000.0_f32, 1e-15_f32),     // typical silicon diode
        (3.3,     2_200.0,     1e-12),          // smaller-Vd schottky-ish
        (5.0,     10_000.0,    1e-15),
        (0.8,     1_000.0,     1e-15),          // sub-Vd, lightly forward
    ];
    for &(v, r, is_) in cases {
        let rlx = run_forward(v, r, is_, VT, 20);
        let ref_ = ref_newton(v, r, is_, VT, 20);
        assert_close(rlx, ref_, 1e-5, 1e-9,
            &format!("Newton DC @ V={v}, R={r}, Is={is_:.0e}"));
    }
}

#[test]
fn newton_converges_to_a_stable_value() {
    // Doubling the iteration count past a "well-converged" level should
    // change Vmid by less than f32 ulp-scale relative error. Catches
    // divergent Newton settings. The diode's exp non-linearity needs ~30
    // iters to reach this regime without damping.
    let v = 1.0_f32;
    let r = 1_000.0_f32;
    let is_ = 1e-15_f32;

    let v30 = run_forward(v, r, is_, VT, 30);
    let v60 = run_forward(v, r, is_, VT, 60);

    assert_close(v60, v30, 1e-5, 1e-9, "Vmid not converged at 30 iters");
}

#[test]
fn typical_silicon_diode_has_sensible_vd() {
    // For Is=1e-15, R=1kΩ, V=1V: Vmid should be ~0.6–0.7 V (the
    // textbook silicon diode forward drop).
    let vmid = run_forward(1.0, 1_000.0, 1e-15, VT, 20);
    assert!(vmid > 0.6 && vmid < 0.7,
        "Vmid = {vmid} outside expected silicon-diode range [0.6, 0.7] V");
}
