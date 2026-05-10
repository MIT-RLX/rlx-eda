//! Tier 1: rlx MNA forward + AD vs analytic ground truth.
//!
//! With f64 throughout (DenseSolve is F64-native on CPU), tolerances tighten
//! by orders of magnitude vs the f32 closed-form spike.

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

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 32) as f64) / (u32::MAX as f64) // [0, 1)
    }
}

fn random_points(seed: u64) -> Vec<(f64, f64, f64)> {
    let mut rng = lcg(seed);
    (0..10)
        .map(|_| {
            let v  = 0.1 + 9.9 * rng();
            let r1 = 100.0 + 99_900.0 * rng();
            let r2 = 100.0 + 99_900.0 * rng();
            (v, r1, r2)
        })
        .collect()
}

#[test]
fn forward_matches_analytic() {
    for (v, r1, r2) in random_points(0xA110_CA7Eu64) {
        let rlx = run_forward_mna(v, r1, r2);
        let an = analytic_vout(v, r1, r2);
        // 3x3 dense solve in f64: a few ulps from analytic.
        assert_close(rlx, an, 1e-12, 1e-15, &format!("Vout @ V={v}, R1={r1}, R2={r2}"));
    }
}

#[test]
fn grad_matches_analytic() {
    for (v, r1, r2) in random_points(0xC1C1_EA7Fu64) {
        let (vout, d_r1, d_r2) = run_forward_and_grad_mna(v, r1, r2);
        let vout_an = analytic_vout(v, r1, r2);
        let d_r1_an = analytic_dvout_dr1(v, r1, r2);
        let d_r2_an = analytic_dvout_dr2(v, r1, r2);

        assert_close(vout, vout_an, 1e-12, 1e-15, "Vout");

        // Gradient through the implicit-function VJP: d_b = solve(Aᵀ, e1)
        // and d_A = -d_b ⊗ xᵀ. f64 dense solve precision ~ κ(A) · ε_f64;
        // our 3x3 system is well-conditioned, so 1e-10 rtol is honest.
        assert_close(d_r1, d_r1_an, 1e-10, 1e-18, "dVout/dR1");
        assert_close(d_r2, d_r2_an, 1e-10, 1e-18, "dVout/dR2");
    }
}
