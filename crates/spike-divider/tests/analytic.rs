//! Tier 1: rlx forward + AD vs analytic ground truth at 10 random points.
//!
//! This is the load-bearing test of the whole architecture: if rlx-opt's
//! reverse-mode VJP doesn't agree with the analytic gradient on a 5-node
//! graph, nothing further in the stack can be trusted.

use eda_validate::assert_close;
use spike_divider::*;

/// Deterministic LCG so the test is reproducible without an `rand` dependency.
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 32) as f32) / (u32::MAX as f32)   // [0, 1)
    }
}

fn random_points(seed: u64) -> Vec<(f32, f32, f32)> {
    let mut rng = lcg(seed);
    (0..10)
        .map(|_| {
            let v  = 0.1 + 9.9 * rng();        // 0.1 V .. 10 V
            let r1 = 100.0 + 99_900.0 * rng(); // 100 Ω .. 100 kΩ
            let r2 = 100.0 + 99_900.0 * rng();
            (v, r1, r2)
        })
        .collect()
}

#[test]
fn forward_matches_analytic() {
    for (v, r1, r2) in random_points(0xA110_CA7Eu64) {
        let rlx = run_forward(v, r1, r2);
        let an  = analytic_vout(v, r1, r2);
        // f32 single-op chain: relative error ~ a few ulps.
        assert_close(rlx, an, 1e-5, 1e-7, &format!("Vout @ V={v}, R1={r1}, R2={r2}"));
    }
}

#[test]
fn grad_matches_analytic() {
    for (v, r1, r2) in random_points(0xC1C1EA7Fu64) {
        let (vout, d_r1, d_r2) = run_forward_and_grad(v, r1, r2);
        let vout_an = analytic_vout(v, r1, r2);
        let d_r1_an = analytic_dvout_dr1(v, r1, r2);
        let d_r2_an = analytic_dvout_dr2(v, r1, r2);

        assert_close(vout, vout_an, 1e-5, 1e-7, "Vout");
        // Gradients magnitude ~1e-5..1e-2, so atol must be small enough not
        // to mask sign flips: pick rtol=1e-4, atol=1e-9.
        assert_close(d_r1, d_r1_an, 1e-4, 1e-9, "dVout/dR1");
        assert_close(d_r2, d_r2_an, 1e-4, 1e-9, "dVout/dR2");
    }
}
