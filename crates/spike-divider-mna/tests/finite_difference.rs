//! Tier 2: rlx AD against centered finite differences of the **rlx forward
//! itself** (not the analytic closed form). This independently validates the
//! whole stack: graph build → DenseSolve → narrow → autodiff. If AD and FD
//! agree, the implicit-function VJP is producing the right linear-algebra
//! identities.

use spike_divider_mna::*;

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

/// f64 centered FD wants `eps ≈ ε_f64^(1/3) · |x| ≈ 6e-6 · |x|`. Use
/// relative perturbation; floor at 1e-9 so tiny values still get a real step.
fn fd_eps(x: f64) -> f64 { (1e-5 * x.abs()).max(1e-9) }

#[test]
fn ad_matches_fd_at_nominal() {
    let v = 1.0_f64;
    let r1 = 1_000.0_f64;
    let r2 = 1_000.0_f64;

    let (_, d_r1_ad, d_r2_ad) = run_forward_and_grad_mna(v, r1, r2);

    let h1 = fd_eps(r1);
    let h2 = fd_eps(r2);
    let d_r1_fd = (run_forward_mna(v, r1 + h1, r2) - run_forward_mna(v, r1 - h1, r2)) / (2.0 * h1);
    let d_r2_fd = (run_forward_mna(v, r1, r2 + h2) - run_forward_mna(v, r1, r2 - h2)) / (2.0 * h2);

    assert_close(d_r1_ad, d_r1_fd, 1e-7, 1e-15, "dVout/dR1: AD vs FD");
    assert_close(d_r2_ad, d_r2_fd, 1e-7, 1e-15, "dVout/dR2: AD vs FD");
}

#[test]
fn ad_matches_fd_swept_resistance_ratios() {
    let v = 1.0_f64;

    for &(r1, r2) in &[
        (1_000.0_f64, 1_000.0_f64),
        (1_000.0,     10_000.0),
        (10_000.0,    1_000.0),
        (47_000.0,    330.0),
        (220.0,       47_000.0),
    ] {
        let (_, d_r1_ad, d_r2_ad) = run_forward_and_grad_mna(v, r1, r2);
        let h1 = fd_eps(r1);
        let h2 = fd_eps(r2);
        let d_r1_fd = (run_forward_mna(v, r1 + h1, r2) - run_forward_mna(v, r1 - h1, r2)) / (2.0 * h1);
        let d_r2_fd = (run_forward_mna(v, r1, r2 + h2) - run_forward_mna(v, r1, r2 - h2)) / (2.0 * h2);

        // f64 truncation error of centered FD at eps_rel=1e-5 is ~ eps² ~ 1e-10
        // relative; loosen slightly for the asymmetric cases.
        assert_close(d_r1_ad, d_r1_fd, 1e-7, 1e-15, &format!("dR1 @ ({r1},{r2})"));
        assert_close(d_r2_ad, d_r2_fd, 1e-7, 1e-15, &format!("dR2 @ ({r1},{r2})"));
    }
}
