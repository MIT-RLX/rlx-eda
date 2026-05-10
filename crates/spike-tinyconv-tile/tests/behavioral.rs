//! `DcBehavioral::add_to_dc` tests for the Digital MAC residual.
//! Pure-graph-construction tests — verify the structure (param
//! count, naming, output shape) without invoking an evaluator.
//!
//! Numerical analytic / FD / ngspice cross-validation lives in
//! `tests/{analytic,fd,ngspice}.rs` and lights up once an evaluator
//! is wired into this crate.

use eda_hir::{Block, DcBehavioral};
use rlx_ir::{shape::Dim, DType, Graph, NodeId, Op};
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};

fn digital_tile(id: &str) -> Mac8x8Tile {
    Mac8x8Tile::with_topology(id, TileParams::default(), MacTopology::Digital)
}

/// Tile name is the prefix `add_to_dc` uses for every param it adds.
fn tile_name(t: &Mac8x8Tile) -> String {
    <Mac8x8Tile as Block>::name(t)
}

fn param_names(g: &Graph) -> Vec<String> {
    g.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn find_param(g: &Graph, full_name: &str) -> NodeId {
    g.nodes()
        .iter()
        .enumerate()
        .find_map(|(i, n)| match &n.op {
            Op::Param { name } if name == full_name => Some(NodeId(i as u32)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("param {full_name:?} not in graph"))
}

#[test]
fn add_to_dc_creates_three_params_per_tile() {
    let mut g = Graph::new("dc-test");
    let tile = digital_tile("u0");
    let _id = tile.add_to_dc(&mut g);
    let prefix = tile_name(&tile);
    let count = g
        .nodes()
        .iter()
        .filter(|n| matches!(&n.op, Op::Param { name } if name.starts_with(&prefix)))
        .count();
    assert_eq!(
        count,
        3,
        "expected exactly 3 params (w_l_n, w_l_p, vdd) — got names: {:?}",
        param_names(&g),
    );
}

#[test]
fn add_to_dc_returns_scalar_node() {
    let mut g = Graph::new("scalar-test");
    let id = digital_tile("u0").add_to_dc(&mut g);
    let shape = g.shape(id);
    assert_eq!(shape.dims(), &[Dim::Static(1)]);
    assert_eq!(shape.dtype(), DType::F32);
}

#[test]
fn param_names_are_tile_instance_keyed() {
    let mut g = Graph::new("naming-test");
    let _ = digital_tile("uA").add_to_dc(&mut g);
    let _ = digital_tile("uB").add_to_dc(&mut g);

    let names = param_names(&g);
    let unique: std::collections::HashSet<_> = names.iter().collect();
    assert_eq!(unique.len(), 6, "param names should all be unique: {names:?}");
    assert!(names.iter().any(|n| n.contains("uA") && n.ends_with("__w_l_n")));
    assert!(names.iter().any(|n| n.contains("uB") && n.ends_with("__vdd")));
}

#[test]
fn output_node_is_add_of_two_inputs() {
    // Principal output = Add(Pdyn, Pleak). Sanity: 2 inputs, top
    // op is Binary::Add.
    let mut g = Graph::new("structure-test");
    let id = digital_tile("u0").add_to_dc(&mut g);
    let node = g.node(id);
    assert_eq!(
        node.inputs.len(),
        2,
        "principal output should be Add(Pdyn, Pleak)"
    );
    assert!(
        matches!(&node.op, Op::Binary(rlx_ir::op::BinaryOp::Add)),
        "expected top op to be Binary(Add); got {:?}",
        node.op,
    );
}

#[test]
fn add_loss_to_dc_uses_supplied_area_baseline() {
    use rlx_runtime::{Device, Session};
    use spike_tinyconv_tile::LossWeights;

    // Two graphs: one with the placeholder baseline (None), one
    // with a supplied baseline (Some(1000.0)). Evaluate both at
    // identical params; the loss difference should equal
    // gamma_area · (1000 − placeholder).
    //
    // placeholder = N_CELLS_DIGITAL · A0_PER_CELL = 202 · 0.05 = 10.1
    // override   = 1000.0
    // gamma_area = 0.25 (default)
    // ⇒ Δloss = 0.25 · (1000.0 − 10.1) = 247.475
    let tile = digital_tile("u_baseline");

    fn final_loss(tile: &Mac8x8Tile, baseline: Option<f32>) -> f32 {
        let mut g = Graph::new("baseline-test");
        let weights = LossWeights {
            area_baseline_um2: baseline,
            ..LossWeights::default()
        };
        let id = tile.add_loss_to_dc(&mut g, weights, None);
        g.set_outputs(vec![id]);
        let mut s = Session::new(Device::Cpu).compile(g);
        let prefix = tile_name(tile);
        s.set_param(&format!("{prefix}__w_l_n"), &[1.0]);
        s.set_param(&format!("{prefix}__w_l_p"), &[1.0]);
        s.set_param(&format!("{prefix}__vdd"), &[1.5]);
        s.run(&[])[0][0]
    }

    let placeholder_loss = final_loss(&tile, None);
    let override_loss = final_loss(&tile, Some(1_000.0));
    let delta = override_loss - placeholder_loss;
    let expected = 0.25 * (1_000.0 - 10.1);
    assert!(
        (delta - expected).abs() < 1e-3,
        "baseline override should shift loss by γ·Δbaseline: \
         got Δ={delta}, expected {expected}"
    );
}

#[test]
fn vdd_param_is_reused_across_pdyn_and_pleak() {
    // Vdd appears in both Pdyn (as Vdd²) and Pleak (linearly), so it
    // must be reused. Verifies the closed-form factoring didn't
    // silently duplicate the param.
    let mut g = Graph::new("vdd-reuse-test");
    let tile = digital_tile("u0");
    let _ = tile.add_to_dc(&mut g);

    let vdd_id = find_param(&g, &format!("{}__vdd", tile_name(&tile)));
    // `use_count` counts distinct downstream consumer nodes (not
    // input slots). Even with `add_to_dc` returning only P_total,
    // `build_digital_terms` constructs ALL three terms (delay +
    // area as well, both feeding off the same Param triple) so
    // they're available to `add_loss_to_dc`. Vdd is therefore
    // consumed by:
    //   1. `Mul(vdd, vdd)` for Vdd² (Pdyn)
    //   2. `Mul(p_leak_partial, vdd)` in Pleak
    //   3. `Mul(avg_wl, vdd)` in delay denominator
    assert_eq!(
        g.use_count(vdd_id),
        3,
        "vdd should be reused across Vdd² (Pdyn), Pleak, and delay; \
         got use_count = {}",
        g.use_count(vdd_id),
    );
}
