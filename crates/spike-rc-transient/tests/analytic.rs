//! Tier 1: rlx forward + AD vs analytic ground truth.
//!
//! Two halves:
//! - **Single-step**: `vout_n` and its gradients vs the closed-form 3×3
//!   MNA solve. f64 throughout, so tolerances are tight.
//! - **Multi-step**: outer-loop rlx transient vs the closed-form
//!   geometric recurrence `V·(1 − α^N)` with `α = RC/(h+RC)`.

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

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 32) as f64) / (u32::MAX as f64) // [0, 1)
    }
}

#[test]
fn single_step_forward_and_grad_match_analytic() {
    let mut rng = lcg(0xBE5705CFu64);
    for _ in 0..10 {
        let v          = 0.1 + 9.9 * rng();
        let vout_prev  = -0.5 + 1.5 * rng();
        let r          = 100.0 + 99_900.0 * rng();
        let c          = 1e-12 + 1e-9 * rng();
        let h          = (r * c) * (0.01 + 0.99 * rng()); // h on the order of RC

        let (vn_rlx, dr_rlx, dc_rlx) = run_step_and_grad(v, vout_prev, r, c, h);
        let vn_an = analytic_step(v, vout_prev, r, c, h);
        let dr_an = analytic_dstep_dr(v, vout_prev, r, c, h);
        let dc_an = analytic_dstep_dc(v, vout_prev, r, c, h);

        assert_close(vn_rlx, vn_an, 1e-12, 1e-15, "vout_n");
        assert_close(dr_rlx, dr_an, 1e-9,  1e-18, "∂vout_n/∂R");
        assert_close(dc_rlx, dc_an, 1e-9,  1e-18, "∂vout_n/∂C");
    }
}

#[test]
fn multistep_rlx_loop_matches_pure_rust_be() {
    // The two should agree to nearly f64 ulp — they're solving the same
    // 3×3 MNA system at each step, with one path through rlx and one in
    // straight Rust.
    let cases = &[
        (1.0_f64, 1_000.0_f64, 1e-9_f64,    100, 1e-9_f64),
        (5.0,     2_200.0,     2.2e-9,       50, 5e-9),
        (3.3,     10_000.0,    100e-12,     200, 1e-10),
    ];
    for &(v_dc, r, c, n, h) in cases {
        let rlx = run_transient(n, h, r, c, 0.0, |_| v_dc);
        let pr  = ref_transient(n, h, r, c, 0.0, |_| v_dc);
        assert_close(rlx, pr, 1e-12, 1e-15,
            &format!("multistep rlx vs pure-Rust @ V={v_dc},R={r},C={c},N={n},h={h:.2e}"));
    }
}

#[test]
fn multistep_pure_rust_matches_geometric_closed_form() {
    let cases = &[
        (1.0_f64, 1_000.0_f64, 1e-9_f64,   100, 1e-9_f64),
        (2.5,     5_000.0,     0.5e-9,      50, 1e-10),
        (3.3,     10_000.0,    100e-12,    200, 1e-11),
    ];
    for &(v_dc, r, c, n, h) in cases {
        let pr = ref_transient(n, h, r, c, 0.0, |_| v_dc);
        let an = analytic_transient_dc(v_dc, n, h, r, c);
        // N iterations accumulate ~N ulps of f64 roundoff; rtol 1e-11
        // covers the worst case in the table without hiding real bugs.
        assert_close(pr, an, 1e-11, 1e-15,
            &format!("BE recurrence vs V·(1−α^N) @ V={v_dc},R={r},C={c},N={n},h={h:.2e}"));
    }
}

#[test]
fn be_converges_to_continuum_as_h_shrinks() {
    // Halving h should roughly halve the BE-vs-continuum error
    // (Backward Euler is O(h) accurate). Coarse linear-rate check.
    let v_dc = 1.0_f64;
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let t_stop = r * c; // 1·RC

    let err = |h: f64| {
        let n = (t_stop / h).round() as usize;
        let h = t_stop / n as f64; // rounded
        let be = analytic_transient_dc(v_dc, n, h, r, c);
        let an = continuum_transient_dc(v_dc, t_stop, r, c);
        (be - an).abs()
    };

    let e_coarse = err(t_stop / 50.0);
    let e_fine   = err(t_stop / 500.0);
    // BE is O(h); halving h × 10 should drop error by ~10×. Allow 5×–20×
    // for the local-truncation-vs-asymptotic-rate slop.
    let ratio = e_coarse / e_fine;
    assert!(ratio > 5.0 && ratio < 20.0,
        "expected ~10× error reduction with 10× smaller h, got ratio = {ratio:.2} \
         (e_coarse = {e_coarse:.3e}, e_fine = {e_fine:.3e})");
}
