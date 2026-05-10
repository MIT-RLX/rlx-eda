//! Tier 2: rlx single-step AD against centered finite differences of the
//! same rlx forward (independent witness — doesn't reuse the analytic
//! formula).
//!
//! ## Note on parameter conditioning for FD
//!
//! Centered FD requires the function to be smooth on the perturbation
//! window: truncation error scales as `eps² · f'''(x)`. For our BE step,
//! `∂²f/∂C² ∝ 1/(h · s²)` where `s = 1/R + C/h`; at very small C with
//! commensurately small h, that's huge — FD breaks down at any usable
//! eps. We deliberately use **well-conditioned** parameters (C in mF/µF
//! range, h ~ RC/10) so FD is honest, and let the tier-1 analytic test
//! cover the small-C regime where AD's superiority is the whole point.

use spike_rc_transient::*;

fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    if !close(a, b, rtol, atol) {
        panic!(
            "[{label}] not close:\n  a    = {a:+.15e}\n  b    = {b:+.15e}\n  |a-b|= {diff:.3e}",
            diff = (a - b).abs()
        );
    }
}

/// Relative perturbation, no fixed-magnitude floor — at small parameter
/// scales a 1e-9 floor would dwarf the parameter itself.
fn fd_eps_rel(x: f64, rel: f64) -> f64 { rel * x.abs() }

#[test]
fn ad_matches_fd_single_step() {
    // Well-conditioned: RC = 1 s, h = 0.1 s. Gradients are O(1).
    let v = 1.0_f64;
    let vout_prev = 0.3_f64;
    let r = 1_000.0_f64;
    let c = 1e-3_f64;
    let h = r * c / 10.0;

    let (_, dr_ad, dc_ad) = run_step_and_grad(v, vout_prev, r, c, h);

    let rel = 1e-5;
    let hr = fd_eps_rel(r, rel);
    let hc = fd_eps_rel(c, rel);
    let dr_fd = (run_step_once(v, vout_prev, r + hr, c, h)
              -  run_step_once(v, vout_prev, r - hr, c, h)) / (2.0 * hr);
    let dc_fd = (run_step_once(v, vout_prev, r, c + hc, h)
              -  run_step_once(v, vout_prev, r, c - hc, h)) / (2.0 * hc);

    assert_close(dr_ad, dr_fd, 1e-7, 1e-15, "∂vout_n/∂R: AD vs FD");
    assert_close(dc_ad, dc_fd, 1e-7, 1e-15, "∂vout_n/∂C: AD vs FD");
}

#[test]
fn ad_matches_fd_swept_h_over_rc() {
    // Sweep timestep:RC ratio to stress conditioning of `g1+gc`. Keep RC
    // = 1 s so the gradient magnitudes stay O(1) and FD is well-posed.
    let v = 1.0_f64;
    let vout_prev = 0.0_f64;
    let r = 1_000.0_f64;
    let c = 1e-3_f64;
    let rc = r * c;

    for &h_ratio in &[0.01_f64, 0.1, 1.0, 10.0, 100.0] {
        let h = rc * h_ratio;

        let (_, dr_ad, dc_ad) = run_step_and_grad(v, vout_prev, r, c, h);

        let rel = 1e-5;
        let hr = fd_eps_rel(r, rel);
        let hc = fd_eps_rel(c, rel);
        let dr_fd = (run_step_once(v, vout_prev, r + hr, c, h)
                  -  run_step_once(v, vout_prev, r - hr, c, h)) / (2.0 * hr);
        let dc_fd = (run_step_once(v, vout_prev, r, c + hc, h)
                  -  run_step_once(v, vout_prev, r, c - hc, h)) / (2.0 * hc);

        // FD truncation grows when `gc/g1` is far from 1 (asymmetric
        // conditioning). Keep rtol ≤ 1e-5 across the whole sweep.
        assert_close(dr_ad, dr_fd, 1e-5, 1e-15, &format!("∂R @ h={h_ratio}·RC"));
        assert_close(dc_ad, dc_fd, 1e-5, 1e-15, &format!("∂C @ h={h_ratio}·RC"));
    }
}
