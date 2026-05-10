//! Tier 2: AD `∂|H|²/∂R, ∂|H|²/∂C` from `run_ac_grad` against centered
//! FD on the same forward, plus against the analytic Bode-magnitude
//! gradient. Three witnesses on the AC gradient surface.

use spike_ac::{
    analytic_dmagsq_dc, analytic_dmagsq_dr, run_ac_grad, run_ac_point,
};

const R: f64 = 1_000.0;
const C: f64 = 1e-9;

fn mag_sq(omega: f64, r: f64, c: f64) -> f64 {
    let (re, im) = run_ac_point(omega, r, c);
    re * re + im * im
}

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] not close: a={a:+.6e} b={b:+.6e} |a-b|={:.3e} env={env:.3e}",
            (a - b).abs());
    }
}

#[test]
fn ad_grad_matches_analytic_at_one_decade_spread() {
    // Test at three frequencies: well below, at, and well above corner.
    let fc = 1.0 / (2.0 * std::f64::consts::PI * R * C);
    for f in [fc * 0.1, fc, fc * 10.0] {
        let omega = 2.0 * std::f64::consts::PI * f;
        let (_re, _im, ad_dr, ad_dc) = run_ac_grad(omega, R, C);
        let ana_dr = analytic_dmagsq_dr(omega, R, C);
        let ana_dc = analytic_dmagsq_dc(omega, R, C);
        assert_close(ad_dr, ana_dr, 1e-7, 1e-15,
            &format!("∂|H|²/∂R AD vs analytic at f={f:.2e}"));
        assert_close(ad_dc, ana_dc, 1e-7, 1e-15,
            &format!("∂|H|²/∂C AD vs analytic at f={f:.2e}"));
    }
}

#[test]
fn ad_grad_matches_finite_difference_at_corner() {
    // Centered FD on the rlx forward — independent of the analytic
    // formula, so any common bug between AD and analytic shows up here.
    let fc = 1.0 / (2.0 * std::f64::consts::PI * R * C);
    let omega = 2.0 * std::f64::consts::PI * fc;

    let (_re, _im, ad_dr, ad_dc) = run_ac_grad(omega, R, C);

    let eps_r = R * 1e-4;
    let eps_c = C * 1e-4;
    let fd_dr = (mag_sq(omega, R + eps_r, C) - mag_sq(omega, R - eps_r, C)) / (2.0 * eps_r);
    let fd_dc = (mag_sq(omega, R, C + eps_c) - mag_sq(omega, R, C - eps_c)) / (2.0 * eps_c);

    assert_close(ad_dr, fd_dr, 1e-4, 1e-12, "∂|H|²/∂R AD vs FD");
    assert_close(ad_dc, fd_dc, 1e-4, 1e-12, "∂|H|²/∂C AD vs FD");
}
