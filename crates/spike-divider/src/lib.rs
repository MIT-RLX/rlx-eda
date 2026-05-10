//! Voltage divider as an rlx graph — the architectural smoke test.
//!
//! Forward: `Vout = V * R2 / (R1 + R2)`.
//! Analytic gradients: `∂Vout/∂R1 = -V*R2/(R1+R2)^2`, `∂Vout/∂R2 = V*R1/(R1+R2)^2`.
//!
//! This module exposes the graph builders + thin wrappers so integration tests
//! can drive forward+gradient runs without re-doing the rlx plumbing.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

/// Scalar f32 shape used everywhere in this spike. The reference test in
/// rlx (`cpu_grad_finite_difference.rs`) uses `[1]` for scalars — we follow
/// that convention rather than `Shape::scalar`, so the autodiff codepaths
/// we exercise match the ones rlx itself tests.
pub fn scalar_shape() -> Shape {
    Shape::new(&[1], DType::F32)
}

/// Build the forward graph: V * R2 / (R1 + R2).
///
/// Returns the graph plus the param node ids so callers can hand them to
/// `grad_with_loss`. `V` is an `Op::Input` (run-time injected); `R1` and `R2`
/// are `Op::Param`s (set once via `set_param`).
pub fn build_forward() -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new("voltage_divider");
    let v  = g.input("V",  scalar_shape());
    let r1 = g.param("R1", scalar_shape());
    let r2 = g.param("R2", scalar_shape());

    let num = g.binary(BinaryOp::Mul, v,  r2, scalar_shape());
    let den = g.binary(BinaryOp::Add, r1, r2, scalar_shape());
    let vout = g.binary(BinaryOp::Div, num, den, scalar_shape());

    g.set_outputs(vec![vout]);
    (g, r1, r2)
}

/// Run the forward graph with concrete `V`, `R1`, `R2` and return `Vout`.
pub fn run_forward(v: f32, r1_val: f32, r2_val: f32) -> f32 {
    let (g, _r1, _r2) = build_forward();
    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param("R1", &[r1_val]);
    compiled.set_param("R2", &[r2_val]);
    let outs = compiled.run(&[("V", &[v])]);
    outs[0][0]
}

/// Run forward + reverse-mode AD. Returns `(Vout, ∂Vout/∂R1, ∂Vout/∂R2)`.
///
/// `grad_with_loss` produces a graph with outputs `[loss, ∂/∂R1, ∂/∂R2]`
/// and an extra `Op::Input` named `"d_output"` seeded with the upstream
/// gradient (here `[1.0]` — we differentiate the output directly).
pub fn run_forward_and_grad(v: f32, r1_val: f32, r2_val: f32) -> (f32, f32, f32) {
    let (fwd, r1, r2) = build_forward();
    let bwd = grad_with_loss(&fwd, &[r1, r2]);

    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param("R1", &[r1_val]);
    compiled.set_param("R2", &[r2_val]);

    let outs = compiled.run(&[
        ("V",        &[v][..]),
        ("d_output", &[1.0_f32][..]),
    ]);

    (outs[0][0], outs[1][0], outs[2][0])
}

/// Closed-form forward.
pub fn analytic_vout(v: f32, r1: f32, r2: f32) -> f32 {
    v * r2 / (r1 + r2)
}

/// Closed-form `∂Vout/∂R1`.
pub fn analytic_dvout_dr1(v: f32, r1: f32, r2: f32) -> f32 {
    -v * r2 / (r1 + r2).powi(2)
}

/// Closed-form `∂Vout/∂R2`.
pub fn analytic_dvout_dr2(v: f32, r1: f32, r2: f32) -> f32 {
    v * r1 / (r1 + r2).powi(2)
}

/// SPICE deck for the same divider — used by the optional ngspice cross-check.
/// Caller appends `.control` block via the `Invoker`.
pub fn spice_deck(v: f32, r1: f32, r2: f32) -> String {
    format!(
        "* voltage divider\n\
         V1 vin 0 {v}\n\
         R1 vin vout {r1}\n\
         R2 vout 0 {r2}\n",
    )
}
