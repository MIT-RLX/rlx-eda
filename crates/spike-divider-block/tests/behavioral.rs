//! Behavioral tier — same `RcDivider` instance drives layout AND
//! simulation, with parameter slots keyed by `Block::name()`.
//!
//! This is the architectural payoff of the HIR scaffold: one Rust value
//! produces both the GDS-bound `Cell` (via `Layout<RcDemo>`) and the
//! rlx-graph param slots (via `DcBehavioral`). Layout uses `length`;
//! simulation uses the same Block's `name()` to key the rlx `Param`.
//! Caller maps physical → electrical via a sheet-resistance constant.

use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_divider_block::*;

/// Map "RES body length" to an electrical resistance value, mirroring a
/// real PDK's sheet-rho × length / width relationship. Picked here so
/// numbers are easy to read: with `length=10000 DBU = 10 µm`, R ≈ 1 kΩ.
fn r_from_length(length_dbu: i64) -> f32 {
    // sheet rho = 100 Ω/sq, body width = 1 µm = 1000 DBU.
    let length_um = length_dbu as f32 / 1000.0;
    100.0 * length_um / 1.0
}

#[test]
fn one_block_drives_both_layout_and_simulation() {
    // Single block instance. Both flows touch the same Rust value.
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );

    // ── Layout flow ──────────────────────────────────────────────────
    let lib = RcDemo::new_library("dual_flow");
    let pdk = RcDemo::register(&lib);
    let top = eda_hir::Layout::layout(&div, &lib, &pdk);
    let layout_cell = lib.get(top);
    assert_eq!(layout_cell.ports().len(), 3, "layout flow produced 3 top ports");

    // ── Simulation flow (same `div` instance) ────────────────────────
    let (fwd, r1_id, r2_id) = div.build_dc_graph();
    let bwd = grad_with_loss(&fwd, &[r1_id, r2_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);

    // Param names come from `Block::name()` — same identity that drives
    // the layout's CellName. Pulling the param names from the divider
    // (not hard-coding) is the contract.
    let [n1, n2] = div.dc_param_names();
    let r1_val = r_from_length(div.r1.length);
    let r2_val = r_from_length(div.r2.length);
    compiled.set_param(&n1, &[r1_val]);
    compiled.set_param(&n2, &[r2_val]);

    let v_in = 1.0_f32;
    let outs = compiled.run(&[("V", &[v_in]), ("d_output", &[1.0_f32])]);
    let vout  = outs[0][0];
    let d_r1  = outs[1][0];
    let d_r2  = outs[2][0];

    // Forward: matches V·R2/(R1+R2) at f32 tolerance.
    let expected = v_in * r2_val / (r1_val + r2_val);
    assert!((vout - expected).abs() < 1e-5,
        "vout = {vout}, expected {expected}");

    // Gradients: ∂Vout/∂R1 = -V·R2/(R1+R2)², ∂Vout/∂R2 = +V·R1/(R1+R2)².
    let denom = (r1_val + r2_val).powi(2);
    let exp_dr1 = -v_in * r2_val / denom;
    let exp_dr2 =  v_in * r1_val / denom;
    assert!((d_r1 - exp_dr1).abs() < (exp_dr1.abs() * 1e-3).max(1e-9),
        "∂R1: got {d_r1}, expected {exp_dr1}");
    assert!((d_r2 - exp_dr2).abs() < (exp_dr2.abs() * 1e-3).max(1e-9),
        "∂R2: got {d_r2}, expected {exp_dr2}");
}

#[test]
fn distinct_resistor_names_produce_distinct_param_slots() {
    // Two resistors with different `length`s give different `name()`s,
    // which drive different rlx `Param` slots. If two slots collided
    // on name, set_param would silently set the wrong one.
    let div = RcDivider::new(
        Resistor { length: 1_000,  id: "R1".into() },
        Resistor { length: 99_999, id: "R2".into() },
    );
    let [n1, n2] = div.dc_param_names();
    assert_ne!(n1, n2, "param names must be distinct: {n1} vs {n2}");

    // And the names match the block names — same identity drives layout
    // (CellName) and sim (Param name).
    use eda_hir::Block;
    assert_eq!(n1, div.r1.name());
    assert_eq!(n2, div.r2.name());
}
