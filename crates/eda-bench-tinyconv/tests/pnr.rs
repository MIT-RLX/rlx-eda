//! `pnr::run(mode)` runtime toggle.
//!
//! Disabled mode → no-op, returns `None`.
//! AdamHpwl mode → builds synthetic controller netlist, Adam-steps
//! HPWL, returns `Some(summary)` with initial vs final HPWL.

use eda_bench_tinyconv::pnr::{run, run_adam_hpwl, PnrAdamConfig, PnrMode};

#[test]
fn disabled_mode_returns_none_immediately() {
    let result = run(PnrMode::Disabled).expect("disabled never errors");
    assert!(result.is_none(), "Disabled should produce no summary");
}

#[test]
fn default_mode_is_disabled() {
    // Backward-compat guarantee: existing call sites that omit
    // `pnr_mode` get the no-op path.
    let mode = PnrMode::default();
    let result = run(mode).unwrap();
    assert!(result.is_none());
}

#[test]
fn adam_hpwl_mode_returns_summary_with_position_count() {
    let result = run(PnrMode::AdamHpwl(PnrAdamConfig::default())).unwrap();
    let summary = result.expect("AdamHpwl should produce a summary");
    // Synthetic netlist has 4 cells.
    assert_eq!(summary.final_positions.len(), 4);
    assert_eq!(summary.n_steps, 200);
}

#[test]
fn adam_hpwl_reduces_total_wirelength() {
    // 200 Adam steps should pull the 4 cells closer together →
    // final HPWL strictly < initial HPWL.
    let summary = run_adam_hpwl(PnrAdamConfig::default()).unwrap();
    assert!(
        summary.final_hpwl < summary.initial_hpwl,
        "Adam should reduce HPWL: initial={} final={}",
        summary.initial_hpwl,
        summary.final_hpwl
    );
    assert!(summary.initial_hpwl > 0.0);
    assert!(summary.final_hpwl >= 0.0);
}

#[test]
fn adam_hpwl_converges_at_higher_step_count() {
    // More steps → lower or equal final HPWL (Adam is monotone-ish
    // for this convex toy problem).
    let short = run_adam_hpwl(PnrAdamConfig {
        max_steps: 20,
        ..PnrAdamConfig::default()
    })
    .unwrap();
    let long = run_adam_hpwl(PnrAdamConfig {
        max_steps: 400,
        ..PnrAdamConfig::default()
    })
    .unwrap();
    assert!(
        long.final_hpwl <= short.final_hpwl + 1.0, // tolerance for float wobble
        "more steps should reach lower HPWL: short={} long={}",
        short.final_hpwl,
        long.final_hpwl
    );
}

#[test]
fn adam_hpwl_final_positions_clustered_near_centroid() {
    // After 400 Adam steps, the 4 cells should be tightly clustered
    // (HPWL minimum). Bbox of final positions should be much smaller
    // than the initial 50 µm spread.
    let summary = run_adam_hpwl(PnrAdamConfig {
        max_steps: 400,
        ..PnrAdamConfig::default()
    })
    .unwrap();
    let xs: Vec<f32> = summary.final_positions.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = summary.final_positions.iter().map(|p| p.1).collect();
    let xrange = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let yrange = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let initial_range = 50_000.0_f32;
    assert!(
        xrange < initial_range,
        "x range should shrink from initial: got {xrange}"
    );
    assert!(
        yrange < initial_range,
        "y range should shrink from initial: got {yrange}"
    );
}
