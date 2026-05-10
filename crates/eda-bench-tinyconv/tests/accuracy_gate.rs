//! Accuracy-gate-specific tests for `inner::run`.
//!
//! Verifies the `λ · max(0, k_acc · σ − ε)` term:
//!   - With gate disabled (`noise_model = None`), Adam parks at min
//!     bounds (already covered by `inner_loop.rs`).
//!   - With gate enabled, Adam can be pushed away from low-Vdd /
//!     small-W regions where σ blows up — the gate counteracts the
//!     pure-power-min pull toward bounds.
//!   - The final loss with the gate enabled differs from the gate-
//!     disabled version (proves the gate is actually contributing
//!     to the loss surface).

use eda_bench_tinyconv::optimization::{
    inner::{run, InnerConfig},
    LossWeights,
};
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, NoiseModel, TileParams};

fn digital_tile(id: &str, params: TileParams) -> Mac8x8Tile {
    Mac8x8Tile::with_topology(id, params, MacTopology::Digital)
}

#[test]
fn inner_loss_differs_with_and_without_gate() {
    // Same starting params + same step count; only `noise_model`
    // differs. The loss curves must diverge — proves the gate is
    // wired into the residual.
    let init = TileParams { vdd: 1.0, ..TileParams::default() };
    let with_gate = InnerConfig {
        max_steps: 3, // short — just need the loss to differ
        noise_model: Some(NoiseModel::default()),
        ..InnerConfig::default()
    };
    let without_gate = InnerConfig {
        noise_model: None,
        ..with_gate.clone_for_test()
    };

    let trace_with = run(&digital_tile("u_with", init), &with_gate).expect("with-gate runs");
    let trace_without =
        run(&digital_tile("u_without", init), &without_gate).expect("without-gate runs");

    let l_with = trace_with[0].p_total;
    let l_without = trace_without[0].p_total;
    assert!(
        (l_with - l_without).abs() > 1e-6,
        "gate should change the loss: with={} without={}",
        l_with,
        l_without,
    );
    // Gate adds a non-negative term, so loss-with >= loss-without.
    assert!(l_with >= l_without);
}

#[test]
fn gate_engages_below_nominal_vdd_but_clamps_above() {
    // The gate term is `λ · max(0, k_acc · σ − ε)`, with σ ramping
    // up only when Vdd drops below the noise model's `vdd_nominal`.
    // Compare gate-on vs gate-off at the same operating point in
    // both regimes:
    //   - Vdd = 0.7 (well below nominal): gate adds a large term.
    //   - Vdd = 2.5 (above nominal): supply contribution clamps to
    //     zero, gate adds (at most) the Pelgrom + thermal floor.
    let droop_params = TileParams { vdd: 0.7, w_l_n: 2.0, w_l_p: 2.0, ..TileParams::default() };
    let high_params = TileParams { vdd: 2.5, w_l_n: 2.0, w_l_p: 2.0, ..TileParams::default() };

    let with = InnerConfig {
        max_steps: 0,
        noise_model: Some(NoiseModel::default()),
        ..InnerConfig::default()
    };
    let without = InnerConfig {
        noise_model: None,
        ..with.clone_for_test()
    };

    let droop_with = run(&digital_tile("d_w", droop_params), &with).unwrap()[0].p_total;
    let droop_without = run(&digital_tile("d_o", droop_params), &without).unwrap()[0].p_total;
    let high_with = run(&digital_tile("h_w", high_params), &with).unwrap()[0].p_total;
    let high_without = run(&digital_tile("h_o", high_params), &without).unwrap()[0].p_total;

    let droop_gate = droop_with - droop_without;
    let high_gate = high_with - high_without;

    assert!(
        droop_gate > 100.0,
        "gate at Vdd=0.7 should add a large positive term: got {droop_gate}"
    );
    assert!(
        droop_gate > high_gate * 5.0,
        "gate at Vdd=0.7 should dominate gate at Vdd=2.5: \
         droop_gate={droop_gate} high_gate={high_gate}"
    );
}

#[test]
fn high_lambda_pushes_adam_away_from_low_vdd() {
    // With a heavy λ, Adam should refuse to drag Vdd to its lower
    // bound (0.6) — the gate term pushes back. Without the gate,
    // Adam would clamp to 0.6 (per the existing `clamps_to_min_bounds`
    // test). So this test asserts: with gate, final Vdd > min Vdd
    // bound (Adam stops short of the bound because the gate term
    // outweighs the energy savings).
    let init = TileParams { vdd: 1.5, w_l_n: 1.0, w_l_p: 1.0, ..TileParams::default() };
    let cfg = InnerConfig {
        max_steps: 100,
        learning_rate: 0.05,
        weights: LossWeights {
            // Crank λ so the gate dominates as Vdd drops.
            lambda_acc: 10_000.0,
            ..LossWeights::default()
        },
        noise_model: Some(NoiseModel::default()),
        ..InnerConfig::default()
    };
    let trace = run(&digital_tile("u_gate_push", init), &cfg).expect("Adam runs");
    let final_vdd = trace.last().unwrap().vdd;
    let min_vdd = cfg.min_params.vdd as f32;
    assert!(
        final_vdd > min_vdd + 0.05,
        "high-λ gate should keep Vdd above min bound: got {} vs min {}",
        final_vdd,
        min_vdd,
    );
}

// Helper: clone of InnerConfig used by `inner_loss_differs_with_and_without_gate`
// — `InnerConfig` doesn't derive `Clone` because `NoiseModel` is large; we
// only need a shallow copy for the gate-on/gate-off comparison.
trait CloneForTest {
    fn clone_for_test(&self) -> Self;
}
impl CloneForTest for InnerConfig {
    fn clone_for_test(&self) -> Self {
        InnerConfig {
            max_steps: self.max_steps,
            learning_rate: self.learning_rate,
            weights: self.weights,
            min_params: self.min_params,
            max_params: self.max_params,
            noise_model: self.noise_model,
        }
    }
}
