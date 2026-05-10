//! Diode-resistor DC circuit — first nonlinear spike.
//!
//! Topology:
//!
//! ```text
//!     V ──[R]── Vmid ──[D]── gnd
//! ```
//!
//! KCL at `Vmid` (currents into Vmid sum to zero):
//!
//! ```text
//!     (V − Vmid) / R   =   Is · (exp(Vmid / Vt) − 1)
//! ```
//!
//! Define `f(Vmid) = (V − Vmid)/R − Is·(exp(Vmid/Vt) − 1) = 0`. The root
//! is the DC operating point.
//!
//! ## How we solve it
//!
//! Newton's method, **unrolled inside an rlx graph**:
//!
//!   `Vmid_{k+1} = Vmid_k − f(Vmid_k) / f'(Vmid_k)`
//!
//! With `n_newton` unrolled iterations, the entire DC solve is one
//! compiled rlx graph. This means `grad_with_loss` walks gradients back
//! through every Newton step automatically — no implicit-function-theorem
//! custom op needed for the MVP. (A real production stack would use IFT
//! to skip the unroll cost; for ≤20 iterations the unrolled-graph cost
//! is small and the gradient story is fully type-checked.)
//!
//! Initial guess: `Vmid_0 = V / 2`. Conservative — the function is
//! monotonic in `Vmid` so Newton converges from any positive guess; the
//! exponential's sensitivity dominates only above ~Vt·ln(V/(R·Is)).
//!
//! ## What we validate
//!
//! - rlx forward matches a pure-Rust Newton implementation (same
//!   discretization, just a different evaluator).
//! - AD `∂Vmid/∂R` matches the implicit-function-theorem analytic
//!   `∂Vmid/∂R = −(∂f/∂R) / (∂f/∂Vmid)` evaluated at the converged Vmid.
//! - AD vs centered FD on the rlx forward (independent witness — no
//!   shared formula with the analytic).

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

pub mod op_ift;
pub use op_ift::{build_graph_ift, run_forward_ift, run_forward_and_grad_ift, run_jvp_ift};

pub mod transient;
pub use transient::{
    build_transient_graph, build_transient_graph_checkpointed,
    run_transient_forward, run_transient_and_grad,
    run_transient_and_grad_checkpointed,
    ref_dc_op, ref_transient,
};

pub mod optimize;
pub use optimize::{
    AdamState, build_optimization_graph, optimize_diode_rc,
    run_loss_and_grad, synthesize_target,
};
#[cfg(feature = "ngspice")]
pub use transient::spice_deck;

/// Thermal voltage at 300 K (kT/q) in volts.
pub const VT: f32 = 0.025_852;

fn scalar() -> Shape { Shape::new(&[1], DType::F32) }

fn const_scalar(g: &mut Graph, x: f32) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar())
}

/// Build the unrolled-Newton DC graph. Returns `(graph, R_id, Is_id)`.
///
/// Inputs (set per-call): `V`, `Vt`.
/// Params (set once, AD-targets): `R`, `Is`.
/// Output: `Vmid` (the DC operating point after `n_newton` Newton steps).
pub fn build_graph(n_newton: usize) -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new("rd_diode_dc");
    let v       = g.input("V",  scalar());
    let vt      = g.input("Vt", scalar());
    let r       = g.param("R",  scalar());
    let is_node = g.param("Is", scalar());

    // Initial guess: Vmid = min(V/2, 0.7). The cap prevents the first
    // exp(Vmid/Vt) call from overflowing f32 when V is large — a typical
    // silicon diode never gets above ~0.8 V at sane currents anyway.
    let half     = const_scalar(&mut g, 0.5);
    let v_half   = g.binary(BinaryOp::Mul, v, half, scalar());
    // 0.6 V puts us just below a typical silicon-diode operating point.
    // Higher caps overshoot through the exponential's steep region;
    // lower caps make Newton crawl. 0.6 is a good compromise that gives
    // ~1 ulp convergence in ~30 iters across the test cases.
    let cap      = const_scalar(&mut g, 0.6);
    let mut vmid = g.binary(BinaryOp::Min, v_half, cap, scalar());

    let one     = const_scalar(&mut g, 1.0);
    let neg_one = const_scalar(&mut g, -1.0);

    for _ in 0..n_newton {
        // f(Vmid) = (V − Vmid)/R − Is·(exp(Vmid/Vt) − 1)
        let v_minus_vmid = g.binary(BinaryOp::Sub, v, vmid, scalar());
        let i_r          = g.binary(BinaryOp::Div, v_minus_vmid, r, scalar());

        let vmid_over_vt = g.binary(BinaryOp::Div, vmid, vt, scalar());
        let exp_v        = g.activation(Activation::Exp, vmid_over_vt, scalar());
        let exp_minus_1  = g.binary(BinaryOp::Sub, exp_v, one, scalar());
        let i_d          = g.binary(BinaryOp::Mul, is_node, exp_minus_1, scalar());

        let f_val        = g.binary(BinaryOp::Sub, i_r, i_d, scalar());

        // f'(Vmid) = −1/R − (Is/Vt)·exp(Vmid/Vt)
        let inv_r        = g.binary(BinaryOp::Div, one, r, scalar());
        let neg_inv_r    = g.binary(BinaryOp::Mul, neg_one, inv_r, scalar());
        let is_over_vt   = g.binary(BinaryOp::Div, is_node, vt, scalar());
        let dexp         = g.binary(BinaryOp::Mul, is_over_vt, exp_v, scalar());
        let neg_dexp     = g.binary(BinaryOp::Mul, neg_one, dexp, scalar());
        let fp           = g.binary(BinaryOp::Add, neg_inv_r, neg_dexp, scalar());

        // Vmid ← Vmid − f / f'
        let dvmid        = g.binary(BinaryOp::Div, f_val, fp, scalar());
        vmid             = g.binary(BinaryOp::Sub, vmid, dvmid, scalar());
    }

    g.set_outputs(vec![vmid]);
    (g, r, is_node)
}

pub fn run_forward(v: f32, r: f32, is_: f32, vt: f32, n_newton: usize) -> f32 {
    let (graph, _r, _is) = build_graph(n_newton);
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param("R",  &[r]);
    compiled.set_param("Is", &[is_]);
    let outs = compiled.run(&[("V", &[v][..]), ("Vt", &[vt][..])]);
    outs[0][0]
}

/// Forward + reverse-mode AD: returns `(Vmid, ∂Vmid/∂R, ∂Vmid/∂Is)`.
/// Gradients flow through every unrolled Newton step automatically.
pub fn run_forward_and_grad(
    v: f32, r: f32, is_: f32, vt: f32, n_newton: usize,
) -> (f32, f32, f32) {
    let (fwd, r_id, is_id) = build_graph(n_newton);
    let bwd = grad_with_loss(&fwd, &[r_id, is_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param("R",  &[r]);
    compiled.set_param("Is", &[is_]);
    let outs = compiled.run(&[
        ("V",        &[v][..]),
        ("Vt",       &[vt][..]),
        ("d_output", &[1.0_f32][..]),
    ]);
    (outs[0][0], outs[1][0], outs[2][0])
}

// ── Pure-Rust reference implementations ─────────────────────────────────

/// Pure-Rust Newton on the same `f(Vmid)`. Identical math, no rlx in
/// the loop. Used as the "did rlx forward correctly?" oracle.
pub fn ref_newton(v: f32, r: f32, is_: f32, vt: f32, n_newton: usize) -> f32 {
    let mut vmid = (v / 2.0).min(0.6);    // matches the rlx graph's cap
    for _ in 0..n_newton {
        let exp_v = (vmid / vt).exp();
        let f  = (v - vmid) / r - is_ * (exp_v - 1.0);
        let fp = -1.0 / r - (is_ / vt) * exp_v;
        vmid -= f / fp;
    }
    vmid
}

/// Implicit-function-theorem analytic gradient `∂Vmid/∂R` evaluated at
/// the converged operating point.
///
/// At `f(Vmid, R) = 0`, differentiating implicitly:
///   `∂f/∂R + ∂f/∂Vmid · ∂Vmid/∂R = 0`
///   `∂Vmid/∂R = − (∂f/∂R) / (∂f/∂Vmid)`
///
/// `∂f/∂R = − (V − Vmid) / R²`
/// `∂f/∂Vmid = − 1/R − (Is/Vt) · exp(Vmid/Vt)`
pub fn analytic_dvmid_dr(v: f32, r: f32, is_: f32, vt: f32, vmid: f32) -> f32 {
    let exp_v = (vmid / vt).exp();
    let df_dr   = -(v - vmid) / (r * r);
    let df_dvm  = -1.0 / r - (is_ / vt) * exp_v;
    -df_dr / df_dvm
}

/// IFT analytic gradient `∂Vmid/∂Is`.
/// `∂f/∂Is = -(exp(Vmid/Vt) - 1)`
pub fn analytic_dvmid_dis(v: f32, r: f32, is_: f32, vt: f32, vmid: f32) -> f32 {
    let exp_v   = (vmid / vt).exp();
    let df_dis  = -(exp_v - 1.0);
    let df_dvm  = -1.0 / r - (is_ / vt) * exp_v;
    -df_dis / df_dvm
}
