//! Tier 2: AD `∂vout/∂{k, voh, vol}` against analytic and centered FD.
//!
//! All three derivatives have closed forms (see `analytic_dvout_d*`),
//! so we triangulate AD ↔ analytic ↔ FD-on-rlx-forward. AD vs FD on
//! `v_plus` exercises the input-side gradient that an SAR-loop
//! optimizer would also want.

use spike_comparator::*;

const VOH: f64 = 1.8;
const VOL: f64 = 0.0;
const K: f64 = 1000.0;

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] AD={a:+.6e} ref={b:+.6e} |Δ|={:.3e} env={env:.3e}",
            (a - b).abs());
    }
}

/// Two operating points: deep saturation (`tanh(±5) ≈ ±1`, gradient
/// largest in `voh`/`vol` and smallest in `k`) and the active region
/// (`tanh(0.5)`, gradient largest in `k`).
const POINTS: &[(f64, f64, &str)] = &[
    (-5e-3, 0.0, "deep negative (vol-rail)"),
    (-0.5e-3, 0.0, "active negative"),
    (0.0, 0.0, "threshold"),
    (0.5e-3, 0.0, "active positive"),
    (5e-3, 0.0, "deep positive (voh-rail)"),
];

#[test]
fn ad_grads_match_analytic_at_each_operating_point() {
    for &(v_plus, v_minus, label) in POINTS {
        let (_v, ad_dk, ad_dvoh, ad_dvol) =
            run_vout_grad(v_plus, v_minus, K, VOH, VOL);
        let an_dk = analytic_dvout_dk(v_plus, v_minus, K, VOH, VOL);
        let an_dvoh = analytic_dvout_dvoh(v_plus, v_minus, K, VOH, VOL);
        let an_dvol = analytic_dvout_dvol(v_plus, v_minus, K, VOH, VOL);

        assert_close(ad_dk,   an_dk,   1e-9, 1e-12,
            &format!("∂vout/∂k AD vs analytic ({label})"));
        assert_close(ad_dvoh, an_dvoh, 1e-9, 1e-12,
            &format!("∂vout/∂voh AD vs analytic ({label})"));
        assert_close(ad_dvol, an_dvol, 1e-9, 1e-12,
            &format!("∂vout/∂vol AD vs analytic ({label})"));
    }
}

#[test]
fn ad_grads_match_finite_difference_at_active_point() {
    // FD step sizes chosen to balance truncation vs round-off:
    // - k step 0.1 (0.01% of nominal 1000)
    // - voh/vol step 1e-5 V (0.001% of 1.8 V)
    let v_plus = 0.5e-3;  // active region — non-trivial gradients on all three params
    let v_minus = 0.0;
    let (_v, ad_dk, ad_dvoh, ad_dvol) =
        run_vout_grad(v_plus, v_minus, K, VOH, VOL);

    let eps_k = 0.1;
    let fd_dk = (run_vout(v_plus, v_minus, K + eps_k, VOH, VOL)
               - run_vout(v_plus, v_minus, K - eps_k, VOH, VOL)) / (2.0 * eps_k);

    let eps_v = 1e-5;
    let fd_dvoh = (run_vout(v_plus, v_minus, K, VOH + eps_v, VOL)
                 - run_vout(v_plus, v_minus, K, VOH - eps_v, VOL)) / (2.0 * eps_v);
    let fd_dvol = (run_vout(v_plus, v_minus, K, VOH, VOL + eps_v)
                 - run_vout(v_plus, v_minus, K, VOH, VOL - eps_v)) / (2.0 * eps_v);

    assert_close(ad_dk,   fd_dk,   1e-4, 1e-10, "∂vout/∂k AD vs FD");
    assert_close(ad_dvoh, fd_dvoh, 1e-4, 1e-10, "∂vout/∂voh AD vs FD");
    assert_close(ad_dvol, fd_dvol, 1e-4, 1e-10, "∂vout/∂vol AD vs FD");
}

#[test]
fn ad_input_slope_matches_finite_difference() {
    // ∂vout/∂(v+) is the comparator's effective small-signal gain at
    // the operating point. AD on inputs isn't done by run_vout_grad
    // (it only exposes param gradients), so we verify FD ↔ analytic.
    // This is a sanity check on the *closed-form* derivative against
    // the rlx forward — same role as the analytic↔rlx test in tier 1
    // but at the slope level.
    let v_plus = 0.5e-3;
    let v_minus = 0.0;
    let an = analytic_dvout_dvplus(v_plus, v_minus, K, VOH, VOL);

    let eps = 1e-6;
    let fd = (run_vout(v_plus + eps, v_minus, K, VOH, VOL)
            - run_vout(v_plus - eps, v_minus, K, VOH, VOL)) / (2.0 * eps);

    assert_close(an, fd, 1e-4, 1e-10, "∂vout/∂v+ analytic vs FD");
}
