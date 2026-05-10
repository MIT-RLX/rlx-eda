//! IFT (implicit-function-theorem) variant of the diode-DC graph.
//!
//! Same forward as [`crate::build_graph`] — `n_newton` unrolled Newton
//! steps to converge `Vmid`. The difference is **how the gradient is
//! computed**:
//!
//! - [`crate::build_graph`] differentiates **through** the Newton loop.
//!   Reverse-mode AD walks every step's primitives, accumulating
//!   gradients along the way. Cost: O(n_newton) primitives per backward.
//! - [`build_graph_ift`] wraps the unrolled Newton in `Op::CustomFn`
//!   and overrides the AD rule with the closed-form IFT gradient. The
//!   backward is O(1) primitives — one `df_dVmid` evaluation, then a
//!   scalar divide per parameter. The forward is identical.
//!
//! The IFT result is exact at the converged operating point. With ~30
//! Newton iterations both paths agree to f32 ulps.
//!
//! ## IFT derivation
//!
//! At converged `Vmid*` we have `f(Vmid*, V, Vt, R, Is) = 0` where
//!
//!   `f(Vmid, V, Vt, R, Is) = (V − Vmid)/R − Is·(exp(Vmid/Vt) − 1)`.
//!
//! Differentiating implicitly:
//!
//!   `∂f/∂p + ∂f/∂Vmid · ∂Vmid/∂p = 0`     (for p ∈ {V, Vt, R, Is})
//!   `∂Vmid/∂p = − (∂f/∂p) / (∂f/∂Vmid)`
//!
//! The reverse-mode VJP for output `y = Vmid*` and upstream `d_output`:
//!
//!   `dp = (∂Vmid/∂p) · d_output = − (∂f/∂p) · d_output / (∂f/∂Vmid)`.
//!
//! Component derivatives (let `e = exp(Vmid*/Vt)`):
//!
//!   `∂f/∂V    =  1/R`
//!   `∂f/∂Vt   =  Is · Vmid* / Vt² · e`
//!   `∂f/∂R    = −(V − Vmid*) / R²`
//!   `∂f/∂Is   = −(e − 1)`
//!   `∂f/∂Vmid = −1/R − (Is/Vt)·e`

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_opt::autodiff_fwd::jvp;
use rlx_runtime::{Device, Session};

pub(crate) fn scalar() -> Shape { Shape::new(&[1], DType::F32) }

pub(crate) fn const_scalar(g: &mut Graph, x: f32) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar())
}

/// Forward body: same Newton recurrence as the outer-graph version, but
/// with **all four** primals as `Op::Input` nodes (not `Op::Param`) so
/// the body can be embedded as a `CustomFn` sub-graph. Inputs are
/// declared in the order `(V, Vt, R, Is)` — that order is what the
/// outer `custom_fn` wires to the body's primal slots by NodeId.
pub(crate) fn build_fwd_body(n_newton: usize) -> Graph {
    let mut b = Graph::new("diode_fwd_body");
    let v   = b.input("V",  scalar());
    let vt  = b.input("Vt", scalar());
    let r   = b.input("R",  scalar());
    let is_ = b.input("Is", scalar());

    let half     = const_scalar(&mut b, 0.5);
    let v_half   = b.binary(BinaryOp::Mul, v, half, scalar());
    let cap      = const_scalar(&mut b, 0.6);
    let mut vmid = b.binary(BinaryOp::Min, v_half, cap, scalar());

    let one     = const_scalar(&mut b, 1.0);
    let neg_one = const_scalar(&mut b, -1.0);

    for _ in 0..n_newton {
        let v_minus_vmid = b.binary(BinaryOp::Sub, v, vmid, scalar());
        let i_r          = b.binary(BinaryOp::Div, v_minus_vmid, r, scalar());
        let vmid_over_vt = b.binary(BinaryOp::Div, vmid, vt, scalar());
        let exp_v        = b.activation(Activation::Exp, vmid_over_vt, scalar());
        let exp_minus_1  = b.binary(BinaryOp::Sub, exp_v, one, scalar());
        let i_d          = b.binary(BinaryOp::Mul, is_, exp_minus_1, scalar());
        let f_val        = b.binary(BinaryOp::Sub, i_r, i_d, scalar());

        let inv_r        = b.binary(BinaryOp::Div, one, r, scalar());
        let neg_inv_r    = b.binary(BinaryOp::Mul, neg_one, inv_r, scalar());
        let is_over_vt   = b.binary(BinaryOp::Div, is_, vt, scalar());
        let dexp         = b.binary(BinaryOp::Mul, is_over_vt, exp_v, scalar());
        let neg_dexp     = b.binary(BinaryOp::Mul, neg_one, dexp, scalar());
        let fp           = b.binary(BinaryOp::Add, neg_inv_r, neg_dexp, scalar());

        let dvmid        = b.binary(BinaryOp::Div, f_val, fp, scalar());
        vmid             = b.binary(BinaryOp::Sub, vmid, dvmid, scalar());
    }

    b.set_outputs(vec![vmid]);
    b
}

/// VJP body: 4 primals + `primal_output` (= Vmid*) + `d_output` →
/// `(dV, dVt, dR, dIs)` via the closed-form IFT identities.
///
/// Body Inputs (declaration order):
///   `V`, `Vt`, `R`, `Is`, `primal_output`, `d_output`.
///
/// Outputs (set_outputs order — matches the outer custom_fn's primal
/// inputs by position):
///   `dV`, `dVt`, `dR`, `dIs`.
pub(crate) fn build_vjp_body() -> Graph {
    let mut b = Graph::new("diode_vjp_body");
    let v    = b.input("V",             scalar());
    let vt   = b.input("Vt",            scalar());
    let r    = b.input("R",             scalar());
    let is_  = b.input("Is",            scalar());
    let vmid = b.input("primal_output", scalar());
    let dout = b.input("d_output",      scalar());

    // Common subexpressions.
    let one      = const_scalar(&mut b, 1.0);
    let neg_one  = const_scalar(&mut b, -1.0);
    let vmid_vt  = b.binary(BinaryOp::Div, vmid, vt, scalar());
    let e        = b.activation(Activation::Exp, vmid_vt, scalar());
    let inv_r    = b.binary(BinaryOp::Div, one, r, scalar());
    let is_vt    = b.binary(BinaryOp::Div, is_, vt, scalar());

    // df_dVmid = -1/R - (Is/Vt) * e
    let dexp     = b.binary(BinaryOp::Mul, is_vt, e, scalar());
    let neg_inv_r= b.binary(BinaryOp::Mul, neg_one, inv_r, scalar());
    let neg_dexp = b.binary(BinaryOp::Mul, neg_one, dexp, scalar());
    let df_dvmid = b.binary(BinaryOp::Add, neg_inv_r, neg_dexp, scalar());

    // lambda = -d_output / df_dVmid    (so that  dp = ∂f/∂p · lambda)
    let neg_dout       = b.binary(BinaryOp::Mul, neg_one, dout, scalar());
    let lambda         = b.binary(BinaryOp::Div, neg_dout, df_dvmid, scalar());

    // ∂f/∂V = 1/R
    let df_dv          = inv_r;
    let dv             = b.binary(BinaryOp::Mul, df_dv, lambda, scalar());

    // ∂f/∂Vt = (Is * Vmid / Vt²) * e
    let is_vmid        = b.binary(BinaryOp::Mul, is_, vmid, scalar());
    let vt_sq          = b.binary(BinaryOp::Mul, vt, vt, scalar());
    let coef_vt        = b.binary(BinaryOp::Div, is_vmid, vt_sq, scalar());
    let df_dvt         = b.binary(BinaryOp::Mul, coef_vt, e, scalar());
    let d_vt           = b.binary(BinaryOp::Mul, df_dvt, lambda, scalar());

    // ∂f/∂R = -(V - Vmid) / R²
    let v_minus_vmid   = b.binary(BinaryOp::Sub, v, vmid, scalar());
    let r_sq           = b.binary(BinaryOp::Mul, r, r, scalar());
    let pos_dfdr       = b.binary(BinaryOp::Div, v_minus_vmid, r_sq, scalar());
    let df_dr          = b.binary(BinaryOp::Mul, neg_one, pos_dfdr, scalar());
    let d_r            = b.binary(BinaryOp::Mul, df_dr, lambda, scalar());

    // ∂f/∂Is = -(e - 1)
    let e_minus_1      = b.binary(BinaryOp::Sub, e, one, scalar());
    let df_dis         = b.binary(BinaryOp::Mul, neg_one, e_minus_1, scalar());
    let d_is           = b.binary(BinaryOp::Mul, df_dis, lambda, scalar());

    b.set_outputs(vec![dv, d_vt, d_r, d_is]);
    b
}

/// JVP body: forward-mode IFT, evaluated at the converged `Vmid*`.
/// 4 primals + `primal_output` + 4 tangents → 1 output tangent.
///
///   `dVmid = − (∂f/∂V·tV + ∂f/∂Vt·tVt + ∂f/∂R·tR + ∂f/∂Is·tIs) / ∂f/∂Vmid`
///
/// All Jacobian terms evaluated at `Vmid*` (the cached forward output
/// passed in via the `"primal_output"` Input — same convention as
/// `vjp_body`).
///
/// Body Inputs (declaration order): `V`, `Vt`, `R`, `Is`,
/// `primal_output`, `tangent_0`, `tangent_1`, `tangent_2`, `tangent_3`.
pub(crate) fn build_jvp_body() -> Graph {
    let mut b = Graph::new("diode_jvp_body");
    let v    = b.input("V",             scalar());
    let vt   = b.input("Vt",            scalar());
    let r    = b.input("R",             scalar());
    let is_  = b.input("Is",            scalar());
    let vmid = b.input("primal_output", scalar());
    let t_v  = b.input("tangent_0",     scalar());
    let t_vt = b.input("tangent_1",     scalar());
    let t_r  = b.input("tangent_2",     scalar());
    let t_is = b.input("tangent_3",     scalar());

    let one      = const_scalar(&mut b, 1.0);
    let neg_one  = const_scalar(&mut b, -1.0);
    let vmid_vt  = b.binary(BinaryOp::Div, vmid, vt, scalar());
    let e        = b.activation(Activation::Exp, vmid_vt, scalar());
    let inv_r    = b.binary(BinaryOp::Div, one, r, scalar());
    let is_vt    = b.binary(BinaryOp::Div, is_, vt, scalar());
    let dexp     = b.binary(BinaryOp::Mul, is_vt, e, scalar());
    let neg_inv_r= b.binary(BinaryOp::Mul, neg_one, inv_r, scalar());
    let neg_dexp = b.binary(BinaryOp::Mul, neg_one, dexp, scalar());
    let df_dvmid = b.binary(BinaryOp::Add, neg_inv_r, neg_dexp, scalar());

    // ∂f/∂V · tV
    let df_dv    = inv_r;
    let term_v   = b.binary(BinaryOp::Mul, df_dv, t_v, scalar());

    // ∂f/∂Vt · tVt
    let is_vmid  = b.binary(BinaryOp::Mul, is_, vmid, scalar());
    let vt_sq    = b.binary(BinaryOp::Mul, vt, vt, scalar());
    let coef_vt  = b.binary(BinaryOp::Div, is_vmid, vt_sq, scalar());
    let df_dvt   = b.binary(BinaryOp::Mul, coef_vt, e, scalar());
    let term_vt  = b.binary(BinaryOp::Mul, df_dvt, t_vt, scalar());

    // ∂f/∂R · tR
    let v_minus_vmid = b.binary(BinaryOp::Sub, v, vmid, scalar());
    let r_sq         = b.binary(BinaryOp::Mul, r, r, scalar());
    let pos_dfdr     = b.binary(BinaryOp::Div, v_minus_vmid, r_sq, scalar());
    let df_dr        = b.binary(BinaryOp::Mul, neg_one, pos_dfdr, scalar());
    let term_r       = b.binary(BinaryOp::Mul, df_dr, t_r, scalar());

    // ∂f/∂Is · tIs
    let e_minus_1    = b.binary(BinaryOp::Sub, e, one, scalar());
    let df_dis       = b.binary(BinaryOp::Mul, neg_one, e_minus_1, scalar());
    let term_is      = b.binary(BinaryOp::Mul, df_dis, t_is, scalar());

    let sum_a        = b.binary(BinaryOp::Add, term_v, term_vt, scalar());
    let sum_b        = b.binary(BinaryOp::Add, sum_a, term_r, scalar());
    let sum_all      = b.binary(BinaryOp::Add, sum_b, term_is, scalar());

    let neg_sum      = b.binary(BinaryOp::Mul, neg_one, sum_all, scalar());
    let t_vmid       = b.binary(BinaryOp::Div, neg_sum, df_dvmid, scalar());

    b.set_outputs(vec![t_vmid]);
    b
}

/// Build the IFT-shape diode-DC graph. Returns `(graph, R_id, Is_id)`.
///
/// Inputs / params identical to [`crate::build_graph`]: `V`, `Vt` are
/// runtime Inputs; `R`, `Is` are Params. The body is wrapped in
/// `Op::CustomFn` so reverse- and forward-mode AD use the closed-form
/// IFT bodies instead of differentiating through the Newton loop.
pub fn build_graph_ift(n_newton: usize) -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new("rd_diode_dc_ift");
    let v       = g.input("V",  scalar());
    let vt      = g.input("Vt", scalar());
    let r       = g.param("R",  scalar());
    let is_node = g.param("Is", scalar());

    let fwd_body = build_fwd_body(n_newton);
    let vjp_body = build_vjp_body();
    let jvp_body = build_jvp_body();

    let vmid = g.custom_fn(
        vec![v, vt, r, is_node],
        fwd_body, Some(vjp_body), Some(jvp_body),
    );
    g.set_outputs(vec![vmid]);
    (g, r, is_node)
}

pub fn run_forward_ift(v: f32, r: f32, is_: f32, vt: f32, n_newton: usize) -> f32 {
    let (graph, _r, _is) = build_graph_ift(n_newton);
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param("R",  &[r]);
    compiled.set_param("Is", &[is_]);
    let outs = compiled.run(&[("V", &[v][..]), ("Vt", &[vt][..])]);
    outs[0][0]
}

/// Reverse-mode AD via IFT custom_vjp. Returns `(Vmid, ∂Vmid/∂R, ∂Vmid/∂Is)`.
pub fn run_forward_and_grad_ift(
    v: f32, r: f32, is_: f32, vt: f32, n_newton: usize,
) -> (f32, f32, f32) {
    let (fwd, r_id, is_id) = build_graph_ift(n_newton);
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

/// Forward-mode AD via IFT custom_jvp. Returns `(Vmid, t_Vmid)` where
/// `t_Vmid` is the directional derivative of `Vmid*` along the seeded
/// tangent `(tV, tVt, tR, tIs)`.
pub fn run_jvp_ift(
    v: f32, r: f32, is_: f32, vt: f32,
    t_v: f32, t_vt: f32, t_r: f32, t_is: f32,
    n_newton: usize,
) -> (f32, f32) {
    let (fwd, r_id, is_id) = build_graph_ift(n_newton);
    let v_id  = fwd.nodes().iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "V"))
        .map(|n| n.id).expect("V input");
    let vt_id = fwd.nodes().iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "Vt"))
        .map(|n| n.id).expect("Vt input");

    let fwd_g = jvp(&fwd, &[v_id, vt_id, r_id, is_id]);
    let mut compiled = Session::new(Device::Cpu).compile(fwd_g);
    compiled.set_param("R",  &[r]);
    compiled.set_param("Is", &[is_]);
    // Seeded tangents follow JAX's "tangent_<name>" naming convention.
    let outs = compiled.run(&[
        ("V",          &[v][..]),
        ("Vt",         &[vt][..]),
        ("tangent_V",  &[t_v][..]),
        ("tangent_Vt", &[t_vt][..]),
        ("tangent_R",  &[t_r][..]),
        ("tangent_Is", &[t_is][..]),
    ]);
    (outs[0][0], outs[1][0])
}
