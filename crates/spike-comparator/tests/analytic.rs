//! Tier 1: rlx forward `vout` vs the closed-form `vout_smooth`. Both
//! evaluate the same expression — this catches graph-construction
//! bugs (wrong op order, wrong wiring) without needing a simulator.

use spike_comparator::{run_vout, vout_ideal, vout_smooth};

const VOH: f64 = 1.8;
const VOL: f64 = 0.0;
const K: f64 = 1000.0;

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] rlx={a:+.6e} closed={b:+.6e} |Δ|={:.3e} env={env:.3e}",
            (a - b).abs());
    }
}

/// Sample at a wide spread of (v+ − v−) values: deep negatives, in the
/// transition window, and deep positives.
const PROBES_DV: &[f64] = &[
    -50e-3, -10e-3, -5e-3, -2e-3, -1e-3, -0.5e-3, -0.1e-3,
    0.0,
    0.1e-3, 0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3, 50e-3,
];

#[test]
fn rlx_matches_closed_form_at_common_mode_zero() {
    // v_minus = 0, sweep v_plus across the transition.
    for &dv in PROBES_DV {
        let rlx = run_vout(dv, 0.0, K, VOH, VOL);
        let closed = vout_smooth(dv, 0.0, K, VOH, VOL);
        assert_close(rlx, closed, 1e-12, 1e-12,
            &format!("dv={:.4e}", dv));
    }
}

#[test]
fn rlx_matches_closed_form_at_common_mode_half_vdd() {
    // Common-mode shifted to vdd/2: should give identical results
    // since vout depends only on v+ − v−.
    let v_minus = VOH / 2.0;
    for &dv in PROBES_DV {
        let v_plus = v_minus + dv;
        let rlx = run_vout(v_plus, v_minus, K, VOH, VOL);
        let closed = vout_smooth(v_plus, v_minus, K, VOH, VOL);
        assert_close(rlx, closed, 1e-12, 1e-12,
            &format!("dv={dv:.4e}, v_minus={v_minus}"));
    }
}

#[test]
fn rlx_at_zero_difference_is_midrail() {
    let rlx = run_vout(0.5, 0.5, K, VOH, VOL);
    assert_close(rlx, 0.5 * (VOH + VOL), 1e-12, 1e-12, "midrail");
}

#[test]
fn rlx_far_from_threshold_approaches_rails() {
    // At Δv = 50 mV with k=1000, tanh(50) ≈ 1 to f64 precision.
    let v_high = run_vout(0.05, 0.0, K, VOH, VOL);
    let v_low  = run_vout(-0.05, 0.0, K, VOH, VOL);
    assert!((v_high - VOH).abs() < 1e-12, "v_high = {v_high}, expected {VOH}");
    assert!(v_low.abs() < 1e-12, "v_low = {v_low}, expected 0");
}

#[test]
fn smooth_approaches_ideal_in_high_gain_limit() {
    // Increase k from 100 to 1e6 — vout(±5 mV) should approach the
    // ideal step monotonically. Check that high-k matches ideal to
    // within atol = 1e-9 V.
    for &k in &[100.0_f64, 1e3, 1e4, 1e6] {
        let v_high = run_vout(5e-3, 0.0, k, VOH, VOL);
        let v_low  = run_vout(-5e-3, 0.0, k, VOH, VOL);
        let ideal_high = vout_ideal(5e-3, 0.0, VOH, VOL);
        let ideal_low  = vout_ideal(-5e-3, 0.0, VOH, VOL);
        let envelope = if k >= 1e4 { 1e-9 } else { 1.0 }; // loose for k=100
        if k >= 1e4 {
            assert!((v_high - ideal_high).abs() < envelope,
                "k={k}: v_high={v_high}, ideal={ideal_high}");
            assert!((v_low - ideal_low).abs() < envelope,
                "k={k}: v_low={v_low}, ideal={ideal_low}");
        }
    }
}
