//! End-to-end inner Adam loop test.
//!
//! Builds a `Mac8x8Tile` (Digital topology), runs the inner loop
//! against its `add_to_dc` residual, and verifies that:
//!   - Adam follows the gradient (total power monotone-ish ↓),
//!   - the trace has the right length,
//!   - bound clamps engage as expected,
//!   - non-digital topologies are rejected (no body wired yet).
//!
//! v1 loss = total power only. The full PLAN.md loss
//!   `α·energy + β·delay + γ·area + λ·max(0, acc_drop_pp − ε)`
//! reduces to `α·energy` until the noise model + delay/area model
//! land. So Adam will trivially drive params to min bounds — that's
//! the right behaviour under the partial loss.

use eda_bench_tinyconv::optimization::{
    inner::{run, InnerConfig, InnerStep},
    OptError,
};
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};

fn digital_tile(id: &str) -> Mac8x8Tile {
    Mac8x8Tile::with_topology(id, TileParams::default(), MacTopology::Digital)
}

fn cfg(max_steps: usize) -> InnerConfig {
    InnerConfig {
        max_steps,
        ..InnerConfig::default()
    }
}

#[test]
fn inner_run_returns_trace_for_digital_tile() {
    let trace = run(&digital_tile("u_smoke"), &cfg(5)).expect("Adam runs");
    // max_steps = 5 → 6 records (step 0..=5 inclusive).
    assert_eq!(trace.len(), 6);
    let s0 = trace[0];
    assert_eq!(s0.step, 0);
    // Step 0 reflects the initial TileParams::default.
    let init = TileParams::default();
    assert!((s0.w_l_n as f64 - init.w_l_n).abs() < 1e-6);
    assert!((s0.vdd as f64 - init.vdd).abs() < 1e-6);
}

#[test]
fn inner_run_drives_total_power_down() {
    // 50 Adam steps should reliably drop p_total — partial loss
    // (energy only) has its minimum at the param bounds.
    let trace = run(&digital_tile("u_descend"), &cfg(50)).expect("Adam runs");
    let p0 = trace[0].p_total;
    let pn = trace.last().unwrap().p_total;
    assert!(
        pn < p0,
        "total power should decrease under Adam: p0={p0} pn={pn}"
    );
}

#[test]
fn inner_run_respects_clamp_bounds() {
    // With the multi-term loss (energy + delay + area, optionally
    // + accuracy gate), the optimum is interior — delay grows as
    // (W/L) shrinks, balancing the energy savings, so Adam doesn't
    // necessarily park at min bounds. But it must still respect
    // them: every recorded step has params inside [mins, maxes].
    let cfg = InnerConfig {
        max_steps: 200,
        learning_rate: 0.1,
        noise_model: None,
        ..InnerConfig::default()
    };
    let trace = run(&digital_tile("u_clamp"), &cfg).expect("Adam runs");
    let mins = InnerConfig::default().min_params;
    let maxes = InnerConfig::default().max_params;
    for s in &trace {
        assert!(
            s.w_l_n >= mins.w_l_n as f32 - 1e-6 && s.w_l_n <= maxes.w_l_n as f32 + 1e-6,
            "step {}: w_l_n {} outside bounds [{}, {}]",
            s.step, s.w_l_n, mins.w_l_n, maxes.w_l_n,
        );
        assert!(
            s.vdd >= mins.vdd as f32 - 1e-6 && s.vdd <= maxes.vdd as f32 + 1e-6,
            "step {}: vdd {} outside bounds [{}, {}]",
            s.step, s.vdd, mins.vdd, maxes.vdd,
        );
    }
}

#[test]
fn inner_run_rejects_non_digital_topology() {
    let cr = Mac8x8Tile::with_topology(
        "u_cr",
        TileParams::default(),
        MacTopology::ChargeRedistribution,
    );
    match run(&cr, &cfg(5)) {
        Err(OptError::InnerDiverged { steps: 0 }) => {}
        other => panic!("expected InnerDiverged, got {other:?}"),
    }
}

#[test]
fn inner_step_records_carry_finite_gradients() {
    let trace: Vec<InnerStep> = run(&digital_tile("u_grad"), &cfg(20)).expect("Adam runs");
    for s in &trace {
        assert!(s.p_total.is_finite(), "step {} p_total not finite", s.step);
        assert!(s.grad_w_l_n.is_finite(), "step {} dL/dw_l_n not finite", s.step);
        assert!(s.grad_w_l_p.is_finite(), "step {} dL/dw_l_p not finite", s.step);
        assert!(s.grad_vdd.is_finite(), "step {} dL/dvdd not finite", s.step);
    }
}
