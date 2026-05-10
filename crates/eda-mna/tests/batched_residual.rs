// SPDX-License-Identifier: GPL-3.0-only
// RLX-EDA — circuit-level building blocks on top of rlx.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! End-to-end parity for `build_batched_residual_graph`.
//!
//! Uses the same two-resistor divider topology that `newton_solve.rs`
//! validates the scalar Newton path against. Builds the scalar
//! residual graph + the vmap-lifted batched residual graph, evaluates
//! both at N independent (V_in, V_mid) draws, and asserts the batched
//! per-draw outputs match the scalar evaluations within f32 epsilon.
//!
//! What this proves:
//!   - `rlx_opt::vmap::vmap` correctly lifts an MNA-shape graph (only
//!     uses Op::Input, Param, Constant, Binary).
//!   - `eda_mna::build_batched_residual_graph` finds the right input
//!     names (`v_<id>` per net) and threads them through.
//!   - The lifted graph is structurally valid for downstream lowering
//!     (compiles + runs through `rlx_runtime::Session::Cpu`).
//!
//! Solving the batched system through `Op::BatchedDenseSolve` is
//! phase 3 work — needs either a batched-Newton driver or an
//! explicit Jacobian + RHS extraction off the residual graph.

use eda_hir::Block;
use eda_mna::{build_batched_residual_graph, build_residual_graph, Circuit, NetId};
use rlx_ir::{DType, Graph};
use rlx_runtime::{Device, Session};
use spike_divider_block::Resistor;

// Resistor::currents binds R as a graph Param keyed by Block::name(&r).
// For a `Resistor { length: 10_000, id: "R1" }` the convention used by
// the newton_solve tests is to provide R = 1000 Ω directly.
const R1_VAL: f32 = 1_000.0;
const R2_VAL: f32 = 3_000.0;

fn build_divider_circuit() -> (Circuit, Resistor, Resistor, NetId, NetId) {
    let mut c = Circuit::new();
    let v_in_net  = c.alloc_boundary_net();
    let v_mid_net = c.alloc_unknown_net();
    let r1 = Resistor { length: 10_000, id: "R1".into() };
    let r2 = Resistor { length: 30_000, id: "R2".into() };
    c.add_device(r1.clone(), &[v_in_net, v_mid_net]);
    c.add_device(r2.clone(), &[v_mid_net, NetId::GND]);
    (c, r1, r2, v_in_net, v_mid_net)
}

/// Bind R params + run a scalar graph at one (V_in, V_mid) point and
/// return the single residual scalar.
fn run_scalar_residual(
    graph: &Graph, r1_name: &str, r2_name: &str,
    v_in: f32, v_mid: f32, v_in_net: NetId, v_mid_net: NetId,
) -> f32 {
    let mut compiled = Session::new(Device::Cpu).compile(graph.clone());
    compiled.set_param_typed(r1_name, &R1_VAL.to_le_bytes(), DType::F32);
    compiled.set_param_typed(r2_name, &R2_VAL.to_le_bytes(), DType::F32);
    let outs = compiled.run_typed(&[
        (&format!("v_{}", v_in_net.0),  &v_in.to_le_bytes(),  DType::F32),
        (&format!("v_{}", v_mid_net.0), &v_mid.to_le_bytes(), DType::F32),
    ]);
    let bytes = &outs[0].0;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[..4]);
    f32::from_le_bytes(buf)
}

/// Bind R params + run the batched graph at N stacked (V_in, V_mid)
/// points and return N residual scalars.
fn run_batched_residual(
    graph: &Graph, r1_name: &str, r2_name: &str,
    v_in_draws: &[f32], v_mid_draws: &[f32],
    v_in_net: NetId, v_mid_net: NetId,
) -> Vec<f32> {
    assert_eq!(v_in_draws.len(), v_mid_draws.len());
    let mut compiled = Session::new(Device::Cpu).compile(graph.clone());
    compiled.set_param_typed(r1_name, &R1_VAL.to_le_bytes(), DType::F32);
    compiled.set_param_typed(r2_name, &R2_VAL.to_le_bytes(), DType::F32);

    // Each batched input is shape [N, 1] f32 — N · 1 · 4 bytes.
    let pack = |xs: &[f32]| -> Vec<u8> {
        let mut b = Vec::with_capacity(xs.len() * 4);
        for x in xs { b.extend_from_slice(&x.to_le_bytes()); }
        b
    };
    let v_in_bytes  = pack(v_in_draws);
    let v_mid_bytes = pack(v_mid_draws);
    let outs = compiled.run_typed(&[
        (&format!("v_{}", v_in_net.0),  &v_in_bytes,  DType::F32),
        (&format!("v_{}", v_mid_net.0), &v_mid_bytes, DType::F32),
    ]);
    // Output is [N, 1] f32 — N · 4 bytes; unpack as N scalars.
    let bytes = &outs[0].0;
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[i*4..i*4 + 4]);
        out.push(f32::from_le_bytes(buf));
    }
    out
}

#[test]
fn batched_residual_matches_per_draw_scalar() {
    let (c, r1, r2, v_in_net, v_mid_net) = build_divider_circuit();
    let r1_name = Block::name(&r1);
    let r2_name = Block::name(&r2);

    let scalar  = build_residual_graph(&c);
    let batched = build_batched_residual_graph(&c, 4);

    // Four draws — three near the analytic solution (where residual ≈ 0)
    // and one deliberately off so we exercise non-zero residuals too.
    let v_in_draws:  Vec<f32> = vec![1.0, 2.0, 3.0, 1.0];
    let v_mid_draws: Vec<f32> = vec![
        1.0 * R2_VAL / (R1_VAL + R2_VAL),   // analytic, residual ≈ 0
        2.0 * R2_VAL / (R1_VAL + R2_VAL),   // analytic, residual ≈ 0
        3.0 * R2_VAL / (R1_VAL + R2_VAL),   // analytic, residual ≈ 0
        0.5,                                 // off-analytic, residual ≠ 0
    ];

    let scalar_outs: Vec<f32> = (0..v_in_draws.len()).map(|i| {
        run_scalar_residual(
            &scalar.graph, &r1_name, &r2_name,
            v_in_draws[i], v_mid_draws[i],
            v_in_net, v_mid_net,
        )
    }).collect();

    let batched_outs = run_batched_residual(
        &batched.graph, &r1_name, &r2_name,
        &v_in_draws, &v_mid_draws,
        v_in_net, v_mid_net,
    );

    assert_eq!(batched_outs.len(), scalar_outs.len());
    for (i, (b, s)) in batched_outs.iter().zip(scalar_outs.iter()).enumerate() {
        let drift = (b - s).abs();
        assert!(
            drift < 1e-6,
            "draw {i}: batched={b}, scalar={s}, drift={drift:.3e}",
        );
    }

    // Sanity checks on the values themselves (not just inter-path parity).
    for i in 0..3 {
        assert!(
            scalar_outs[i].abs() < 1e-5,
            "analytic draw {i} should produce ≈0 residual, got {}", scalar_outs[i],
        );
    }
    assert!(
        scalar_outs[3].abs() > 1e-5,
        "off-analytic draw 3 should produce non-trivial residual, got {}", scalar_outs[3],
    );
}

#[test]
fn batched_residual_graph_carries_metadata_through() {
    let (c, _, _, _v_in_net, v_mid_net) = build_divider_circuit();
    let batched = build_batched_residual_graph(&c, 8);

    // Net topology metadata is unchanged by vmap — same NetId ordering
    // means downstream code that indexes by NetId keeps working.
    assert_eq!(batched.unknown_nets.len(), 1);
    assert_eq!(batched.unknown_nets[0], v_mid_net);
    assert_eq!(batched.all_nets.len(), 2);
    assert_eq!(batched.branches.len(), 0);

    // Every per-draw input now has a leading [8, ...] dim. Look it up
    // by name and check.
    let g = &batched.graph;
    for n in g.nodes() {
        if let rlx_ir::Op::Input { name } = &n.op {
            if name.starts_with("v_") {
                assert_eq!(
                    n.shape.dims().len(), 2,
                    "batched input {name} expected rank 2, got {:?}", n.shape,
                );
                assert_eq!(
                    n.shape.dim(0), rlx_ir::shape::Dim::Static(8),
                    "batched input {name} leading dim should be 8",
                );
            }
        }
    }
}
