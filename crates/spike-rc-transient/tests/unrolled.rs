//! End-to-end AD on the unrolled multi-step BE transient.
//!
//! Validates that `grad_with_loss` produces correct `∂vout_N/∂R, ∂vout_N/∂C`
//! when we chain N BE steps in a single rlx graph (R, C shared, A reused).
//! Three witnesses:
//! - **Forward**: rlx unrolled graph vs `analytic_transient_with_ic` (closed-form
//!   geometric recurrence).
//! - **Gradient analytic**: AD vs `analytic_dtransient_d{r,c}`.
//! - **Gradient FD**: AD vs centered FD on the unrolled forward — independent
//!   of the analytic derivative formula.

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

#[test]
fn unrolled_forward_matches_analytic_transient() {
    // Cases mix n_steps and h/RC ratios so the geometric closed form is
    // exercised at different α values.
    let cases = &[
        // (V, vout_0, R,    C,      N,    h)
        (1.0_f64, 0.0_f64, 1_000.0_f64, 1e-9_f64,    100, 1e-8_f64),
        (1.0,     0.5,     1_000.0,     1e-9,         50, 2e-8),
        (3.3,     0.0,     2_200.0,     2.2e-9,      200, 1e-8),
        (-1.0,    0.0,     10_000.0,    100e-12,     150, 5e-9),
    ];
    for &(v, vout_0, r, c, n, h) in cases {
        let rlx = run_unrolled_forward(v, vout_0, h, r, c, n);
        let an  = analytic_transient_with_ic(v, vout_0, n, h, r, c);
        // 5 ops/step × 100 steps × ~ulps = a few hundred ulps of f64
        // accumulation; rtol 1e-11 is honest.
        assert_close(rlx, an, 1e-11, 1e-15,
            &format!("unrolled fwd @ V={v}, vout_0={vout_0}, R={r}, C={c}, N={n}, h={h:.2e}"));
    }
}

#[test]
fn unrolled_grad_matches_analytic() {
    // Use vout_0 ≠ V so both ∂R and ∂C gradients are nonzero (the
    // recurrence's α-derivative scales with `vout_0 − V`).
    let v       = 1.0_f64;
    let vout_0  = 0.0_f64;       // discharged cap
    let r       = 1_000.0_f64;
    let c       = 1e-9_f64;
    let h       = 1e-8_f64;
    let n       = 100;

    let (v_n_rlx, d_r_rlx, d_c_rlx) = run_unrolled_and_grad(v, vout_0, h, r, c, n);
    let v_n_an = analytic_transient_with_ic(v, vout_0, n, h, r, c);
    let d_r_an = analytic_dtransient_dr(v, vout_0, n, h, r, c);
    let d_c_an = analytic_dtransient_dc(v, vout_0, n, h, r, c);

    // Forward: same expectation as the forward-only test.
    assert_close(v_n_rlx, v_n_an, 1e-11, 1e-15, "vout_N");

    // Gradient: the implicit-function VJP runs N times in reverse.
    // Each step's reverse adds one DenseSolve (via Aᵀ); 100 of them.
    // Expected accumulation ~ N·κ(A)·ε_f64 ≈ 100·O(1)·1e-16 ≈ 1e-14.
    assert_close(d_r_rlx, d_r_an, 1e-9, 1e-18, "∂vout_N/∂R");
    assert_close(d_c_rlx, d_c_an, 1e-9, 1e-18, "∂vout_N/∂C");
}

#[test]
fn unrolled_grad_matches_finite_difference() {
    // Independent witness: don't trust the analytic gradient formula —
    // FD against the unrolled forward itself.
    let v       = 1.0_f64;
    let vout_0  = 0.0_f64;
    let r       = 1_000.0_f64;
    let c       = 1e-3_f64;       // µF — well-conditioned for FD (per the
    let h       = 1e-4_f64;       //       reasoning in finite_difference.rs)
    let n       = 50;

    let (_, d_r_ad, d_c_ad) = run_unrolled_and_grad(v, vout_0, h, r, c, n);

    let rel = 1e-5;
    let hr = rel * r;
    let hc = rel * c;
    let d_r_fd = (run_unrolled_forward(v, vout_0, h, r + hr, c, n)
              -  run_unrolled_forward(v, vout_0, h, r - hr, c, n)) / (2.0 * hr);
    let d_c_fd = (run_unrolled_forward(v, vout_0, h, r, c + hc, n)
              -  run_unrolled_forward(v, vout_0, h, r, c - hc, n)) / (2.0 * hc);

    assert_close(d_r_ad, d_r_fd, 1e-6, 1e-15, "∂vout_N/∂R: AD vs FD (unrolled)");
    assert_close(d_c_ad, d_c_fd, 1e-6, 1e-15, "∂vout_N/∂C: AD vs FD (unrolled)");
}

#[test]
fn unrolled_forward_matches_outer_loop_transient() {
    // Sanity: the unrolled-graph forward must agree with the outer-loop
    // forward (which calls a single-step graph N times). Same math, two
    // different graph topologies.
    let cases = &[
        (1.0_f64, 1_000.0_f64, 1e-9_f64,  100, 1e-8_f64),
        (3.3,     2_200.0,     2.2e-9,     50, 2e-8),
    ];
    for &(v_dc, r, c, n, h) in cases {
        let unrolled = run_unrolled_forward(v_dc, 0.0, h, r, c, n);
        let looped   = run_transient(n, h, r, c, 0.0, |_| v_dc);
        assert_close(unrolled, looped, 1e-12, 1e-15,
            &format!("unrolled vs loop @ V={v_dc},R={r},C={c},N={n}"));
    }
}
