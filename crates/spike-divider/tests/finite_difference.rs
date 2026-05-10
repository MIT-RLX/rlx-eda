//! Tier 2: rlx AD against centered finite differences.
//!
//! This is the second witness: even if the analytic formula is mistyped, the
//! AD-vs-FD agreement still independently confirms the autodiff is working.
//! Triangulation = AD ≈ analytic ≈ FD; if any two agree and one diverges, the
//! disagreement tells you which side is wrong.
//!
//! ## On the eps choice
//!
//! In f32, an *absolute* eps of 1e-3 catastrophically cancels when the
//! parameter is ~1000 (r1+1e-3 rounds back to r1). Centered FD wants
//! `eps ≈ ε^(1/3) · |x|` for f32 — i.e. relative perturbation. We use
//! `eps_rel = 1e-3` which keeps the perturbation well above f32 noise for
//! any reasonable resistance.

use eda_validate::assert_close;
use spike_divider::*;

const EPS_REL: f32 = 1e-3;

/// Centered FD with relative perturbation. Defends against f32 cancellation
/// without coupling tolerance to magnitude.
fn fd_eps(x: f32) -> f32 {
    (EPS_REL * x.abs()).max(1e-6)
}

#[test]
fn ad_matches_finite_difference_at_nominal() {
    let v = 1.0_f32;
    let r1 = 1_000.0_f32;
    let r2 = 1_000.0_f32;

    let (_, d_r1_ad, d_r2_ad) = run_forward_and_grad(v, r1, r2);

    let h1 = fd_eps(r1);
    let h2 = fd_eps(r2);
    let d_r1_fd = (run_forward(v, r1 + h1, r2) - run_forward(v, r1 - h1, r2)) / (2.0 * h1);
    let d_r2_fd = (run_forward(v, r1, r2 + h2) - run_forward(v, r1, r2 - h2)) / (2.0 * h2);

    // Centered FD truncation is O(h²·f''') ~ O((rel·x)² · 1/x³). For our
    // divider that's ~ 6e-7 · 1/r at the nominal point — comfortably under
    // 1e-3 relative.
    assert_close(d_r1_ad, d_r1_fd, 1e-3, 1e-9, "dVout/dR1: AD vs FD");
    assert_close(d_r2_ad, d_r2_fd, 1e-3, 1e-9, "dVout/dR2: AD vs FD");
}

#[test]
fn ad_matches_fd_swept_resistance_ratios() {
    let v = 1.0_f32;

    for &(r1, r2) in &[
        (1_000.0_f32, 1_000.0_f32),
        (1_000.0,     10_000.0),
        (10_000.0,    1_000.0),
        (47_000.0,    330.0),
        (220.0,       47_000.0),
    ] {
        let (_, d_r1_ad, d_r2_ad) = run_forward_and_grad(v, r1, r2);
        let h1 = fd_eps(r1);
        let h2 = fd_eps(r2);
        let d_r1_fd = (run_forward(v, r1 + h1, r2) - run_forward(v, r1 - h1, r2)) / (2.0 * h1);
        let d_r2_fd = (run_forward(v, r1, r2 + h2) - run_forward(v, r1, r2 - h2)) / (2.0 * h2);

        // Asymmetric ratios push gradients down to ~2e-5, where f32 centered
        // FD's roundoff floor (~ε_f32 · |f| / numerator) dominates. AD agrees
        // with analytic to <1 ulp (tier-1 test); we expect ~1% relative
        // error from FD here. That's f32's structural limit, not a bug.
        assert_close(d_r1_ad, d_r1_fd, 1e-2, 1e-9, &format!("dR1 @ ({r1},{r2})"));
        assert_close(d_r2_ad, d_r2_fd, 1e-2, 1e-9, &format!("dR2 @ ({r1},{r2})"));
    }
}
