//! Voltage divider via Modified Nodal Analysis stamps and `DenseSolve`.
//!
//! Where `spike-divider` evaluates the closed-form `V·R2/(R1+R2)` directly,
//! this spike builds the **MNA system** that any real SPICE-style simulator
//! produces, solves it via rlx's `DenseSolve` op, and takes gradients through
//! the linear solve via the implicit-function VJP.
//!
//! ## The system
//!
//! Three nodes (vin, vout, gnd-as-ref) plus one branch current for the
//! voltage source — an N+M = 2+1 system:
//!
//! ```text
//!   indices: 0 = vin, 1 = vout, 2 = i_V1
//!
//!   [  g1     -g1     1 ]   [ vin  ]   [ 0 ]
//!   [ -g1   g1+g2     0 ] · [ vout ] = [ 0 ]
//!   [   1      0      0 ]   [ i_V1 ]   [ V ]
//!
//!   where  g1 = 1/R1,  g2 = 1/R2.
//! ```
//!
//! Solving gives `vout = V * R2 / (R1 + R2)` — the same answer, via the
//! same formulation ngspice uses internally.
//!
//! ## Why this matters architecturally
//!
//! The CPU backend's `DenseSolve` is F64-native and has a closed-form VJP
//! (`d_b = solve(Aᵀ, upstream)`, `d_A = -d_b ⊗ xᵀ`). That means rlx can
//! differentiate **through the linear solve** with no custom op needed. The
//! whole circuit residual — stamps + solve + extraction — is just an rlx
//! graph, and `grad_with_loss` returns `∂output/∂R1`, `∂output/∂R2` for free.
//!
//! ## Upstream rlx-cpu F64 gaps discovered while writing this spike
//!
//! `rlx-cpu/src/thunk.rs` has F64 paths for `Constant`, `Reshape` (via
//! `CopyF64`), `Binary` (via `BinaryFullF64`), `Reduce-Sum`
//! (`ReduceSumF64`), `Transpose` (`TransposeF64`), `DenseSolve`
//! (`DenseSolveF64`). It does **not** yet have F64 paths for:
//!
//! - `Concat` — silently copies f32-sized chunks through f64 data → corruption
//!   → dgesv reports singular matrix.
//! - `Narrow` — same f32-only `sl`/`sl_mut` slicing.
//! - `ScatterAdd` — same.
//!
//! For now we build the matrix purely from `Constant` stamp patterns
//! combined via broadcast `Mul` and `Add`, and extract `x[1]` via a masked
//! dot-product (`x · e1` then `Reduce-Sum`). All those ops have working F64
//! paths. Adding F64 to `Concat`/`Narrow`/`ScatterAdd` upstream is a small,
//! mechanical change — worth doing once we need it (real circuits will).

pub mod mc;

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

fn scalar() -> Shape { Shape::new(&[1], DType::F64) }
fn vec3()   -> Shape { Shape::new(&[3], DType::F64) }
fn mat3()   -> Shape { Shape::new(&[3, 3], DType::F64) }

fn const_scalar(g: &mut Graph, x: f64) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar())
}

fn const_vec3_f64(g: &mut Graph, x: &[f64; 3]) -> NodeId {
    let mut bytes = Vec::with_capacity(24);
    for v in x { bytes.extend_from_slice(&v.to_le_bytes()); }
    g.add_node(Op::Constant { data: bytes }, vec![], vec3())
}

fn const_mat3_f64(g: &mut Graph, x: &[f64; 9]) -> NodeId {
    let mut bytes = Vec::with_capacity(72);
    for v in x { bytes.extend_from_slice(&v.to_le_bytes()); }
    g.add_node(Op::Constant { data: bytes }, vec![], mat3())
}

/// Build the forward graph and return `(graph, R1, R2)`.
pub fn build_forward_mna() -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new("voltage_divider_mna");

    let v  = g.input("V",  scalar());
    let r1 = g.param("R1", scalar());
    let r2 = g.param("R2", scalar());

    // Conductances: g1 = 1/R1, g2 = 1/R2.
    let one = const_scalar(&mut g, 1.0);
    let g1 = g.binary(BinaryOp::Div, one, r1, scalar());
    let g2 = g.binary(BinaryOp::Div, one, r2, scalar());

    // Stamp patterns. Each is a [3,3] sparse pattern that, when scaled by
    // its corresponding conductance (or 1 for the voltage source) and
    // summed, reconstructs the MNA matrix.
    //
    //   pattern_vsrc · 1   +  pattern_r1 · g1  +  pattern_r2 · g2  =  A
    let pattern_vsrc = const_mat3_f64(&mut g, &[
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
    ]);
    let pattern_r1 = const_mat3_f64(&mut g, &[
        1.0, -1.0, 0.0,
        -1.0, 1.0, 0.0,
        0.0,  0.0, 0.0,
    ]);
    let pattern_r2 = const_mat3_f64(&mut g, &[
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]);

    // [3,3] · [1] broadcasts to [3,3] in BinaryFullF64.
    let r1_term = g.binary(BinaryOp::Mul, pattern_r1, g1, mat3());
    let r2_term = g.binary(BinaryOp::Mul, pattern_r2, g2, mat3());
    let a_partial = g.binary(BinaryOp::Add, pattern_vsrc, r1_term, mat3());
    let a_mat = g.binary(BinaryOp::Add, a_partial, r2_term, mat3());

    // b = pattern_b · V, where pattern_b = [0, 0, 1].
    let pattern_b = const_vec3_f64(&mut g, &[0.0, 0.0, 1.0]);
    let b_vec = g.binary(BinaryOp::Mul, pattern_b, v, vec3());

    let x = g.dense_solve(a_mat, b_vec, vec3());

    // Extract vout = x[1] via masked dot product (avoids Op::Narrow which
    // is f32-only on the CPU backend today).
    let e1 = const_vec3_f64(&mut g, &[0.0, 1.0, 0.0]);
    let masked = g.binary(BinaryOp::Mul, x, e1, vec3());
    let vout = g.reduce(masked, ReduceOp::Sum, vec![0], /*keep_dim=*/true, scalar());

    g.set_outputs(vec![vout]);
    (g, r1, r2)
}

/// Forward only. Returns Vout in f64.
pub fn run_forward_mna(v: f64, r1: f64, r2: f64) -> f64 {
    let (graph, _r1, _r2) = build_forward_mna();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("R1", &r1.to_le_bytes(), DType::F64);
    compiled.set_param_typed("R2", &r2.to_le_bytes(), DType::F64);
    let outs = compiled.run_typed(&[("V", &v.to_le_bytes(), DType::F64)]);
    decode_f64_scalar(&outs[0].0)
}

/// Forward + reverse-mode AD. `(Vout, ∂Vout/∂R1, ∂Vout/∂R2)`.
pub fn run_forward_and_grad_mna(v: f64, r1: f64, r2: f64) -> (f64, f64, f64) {
    let (fwd, r1_id, r2_id) = build_forward_mna();
    let bwd = grad_with_loss(&fwd, &[r1_id, r2_id]);

    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("R1", &r1.to_le_bytes(), DType::F64);
    compiled.set_param_typed("R2", &r2.to_le_bytes(), DType::F64);

    let one = 1.0_f64.to_le_bytes();
    let v_b = v.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("V",        &v_b, DType::F64),
        ("d_output", &one, DType::F64),
    ]);

    (
        decode_f64_scalar(&outs[0].0),
        decode_f64_scalar(&outs[1].0),
        decode_f64_scalar(&outs[2].0),
    )
}

fn decode_f64_scalar(bytes: &[u8]) -> f64 {
    assert!(bytes.len() >= 8, "expected ≥8 bytes for f64 scalar, got {}", bytes.len());
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    f64::from_le_bytes(buf)
}

// ---- Closed-form references ----

pub fn analytic_vout(v: f64, r1: f64, r2: f64) -> f64 { v * r2 / (r1 + r2) }
pub fn analytic_dvout_dr1(v: f64, r1: f64, r2: f64) -> f64 { -v * r2 / (r1 + r2).powi(2) }
pub fn analytic_dvout_dr2(v: f64, r1: f64, r2: f64) -> f64 {  v * r1 / (r1 + r2).powi(2) }

pub fn spice_deck(v: f64, r1: f64, r2: f64) -> String {
    format!(
        "* voltage divider (mna spike)\n\
         V1 vin 0 {v}\n\
         R1 vin vout {r1}\n\
         R2 vout 0 {r2}\n",
    )
}
