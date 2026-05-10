//! CMOS inverter DC operating point — first multi-device nonlinear
//! circuit through rlx.
//!
//! ## Topology
//!
//! ```text
//!     VDD ──[PMOS]── V_out ──[NMOS]── gnd
//!                 |
//!     V_in  ──────┴── (gates of both)
//! ```
//!
//! NMOS terminals: drain=V_out, gate=V_in, source=gnd, body=gnd
//! PMOS terminals: drain=V_out, gate=V_in, source=VDD,  body=VDD
//!
//! ## DC operating point
//!
//! At steady state, KCL at `V_out` requires `I_NMOS = I_PMOS` (both
//! flowing from VDD through PMOS, into V_out, out through NMOS to gnd).
//!
//! Using the smooth Shichman-Hodges model from `id_subgraph`:
//!
//! ```text
//!     I_n = id(Vgs = V_in,        Vds = V_out;       Vth_n, kp_n, λ_n)
//!     I_p = id(Vgs = VDD - V_in,  Vds = VDD - V_out; Vth_p, kp_p, λ_p)
//! ```
//!
//! Nonlinear in `V_out`. We solve via Newton, unrolled inside an
//! `Op::CustomFn` body — the same IFT pattern that makes
//! `spike-diode::op_ift` differentiable in O(1) backward primitives.
//! Reverse-mode AD on the parameters costs one IFT solve per backward
//! call, regardless of `n_newton`.
//!
//! ## What this validates
//!
//! - rlx hosts a multi-device nonlinear circuit (two MOSFETs sharing a
//!   node), not just a single-device toy.
//! - The custom_vjp IFT pattern composes when the residual involves
//!   two device-model evaluations rather than one.
//! - rlx forward matches a pure-Rust Newton reference; AD matches the
//!   IFT analytic gradient and finite differences.
//! - ngspice `.op` on the same inverter (with both NMOS and PMOS as
//!   real devices) produces the same `V_out` to ngspice's solver
//!   tolerance.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

use crate::{id_strict, DELTA};

fn scalar() -> Shape { Shape::new(&[1], DType::F64) }
fn const_scalar(g: &mut Graph, x: f64) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar())
}

/// Insert a smooth NMOS Id sub-graph into `g`. Mirrors
/// `crate::id_subgraph` but with **all** parameters as graph nodes
/// (no `g.param` calls) so the helper composes inside other graphs
/// (the inverter body sums two `id_subgraph_node` instances).
#[allow(clippy::too_many_arguments)]
fn id_subgraph_node(
    g: &mut Graph,
    vgs: NodeId, vds: NodeId,
    vth: NodeId, kp: NodeId, lam: NodeId,
) -> NodeId {
    let half  = const_scalar(g, 0.5);
    let half2 = const_scalar(g, 0.5);
    let one   = const_scalar(g, 1.0);
    let delta = const_scalar(g, DELTA);

    // Vov = Vgs - Vth (no cutoff smoothing — same caveat as id_subgraph).
    let vov_s = g.binary(BinaryOp::Sub, vgs, vth, scalar());

    // Vds_eff = ½·(Vds + Vov_s − √((Vds − Vov_s)² + δ)) — smooth-min.
    let sum     = g.binary(BinaryOp::Add, vds, vov_s, scalar());
    let diff    = g.binary(BinaryOp::Sub, vds, vov_s, scalar());
    let diff_sq = g.binary(BinaryOp::Mul, diff, diff, scalar());
    let arg     = g.binary(BinaryOp::Add, diff_sq, delta, scalar());
    let root    = g.activation(Activation::Sqrt, arg, scalar());
    let inner   = g.binary(BinaryOp::Sub, sum, root, scalar());
    let vds_eff = g.binary(BinaryOp::Mul, half, inner, scalar());

    // Id = kp · (Vov_s · Vds_eff − Vds_eff²/2) · (1 + λ·Vds).
    let term1            = g.binary(BinaryOp::Mul, vov_s, vds_eff, scalar());
    let vds_eff_sq       = g.binary(BinaryOp::Mul, vds_eff, vds_eff, scalar());
    let half_vds_eff_sq  = g.binary(BinaryOp::Mul, half2, vds_eff_sq, scalar());
    let bracket          = g.binary(BinaryOp::Sub, term1, half_vds_eff_sq, scalar());
    let lam_vds          = g.binary(BinaryOp::Mul, lam, vds, scalar());
    let clm              = g.binary(BinaryOp::Add, one, lam_vds, scalar());
    let kp_bracket       = g.binary(BinaryOp::Mul, kp, bracket, scalar());
    g.binary(BinaryOp::Mul, kp_bracket, clm, scalar())
}

/// Build the inverter forward body for `Op::CustomFn`.
///
/// Op::Inputs in NodeId order: `[V_in, VDD, Vth_n, kp_n, lam_n,
/// Vth_p, kp_p, lam_p]`. Single output: `V_out` (the converged DC
/// operating point).
///
/// Newton iteration on `f(V_out) = I_n - I_p`:
///   * `f'(V_out) = ∂I_n/∂V_out + ∂I_p/∂V_out` — both have positive
///     drain conductance, so the Jacobian is well-conditioned.
///   * Initial guess: `V_out = VDD/2` (mid-rail; converges quickly for
///     the typical sigmoid-shaped transfer curve).
fn build_fwd_body(n_newton: usize) -> Graph {
    let mut b = Graph::new("inverter_dc_fwd");
    let v_in   = b.input("V_in",   scalar());
    let vdd    = b.input("VDD",    scalar());
    let vth_n  = b.input("Vth_n",  scalar());
    let kp_n   = b.input("kp_n",   scalar());
    let lam_n  = b.input("lam_n",  scalar());
    let vth_p  = b.input("Vth_p",  scalar());
    let kp_p   = b.input("kp_p",   scalar());
    let lam_p  = b.input("lam_p",  scalar());

    // Initial guess: V_out = VDD / 2.
    let half = const_scalar(&mut b, 0.5);
    let mut v_out = b.binary(BinaryOp::Mul, vdd, half, scalar());

    let one     = const_scalar(&mut b, 1.0);
    let neg_one = const_scalar(&mut b, -1.0);
    let two     = const_scalar(&mut b, 2.0);
    let h_step  = const_scalar(&mut b, 1e-3);  // FD step for ∂I/∂V_out
    // Step-magnitude cap for damped Newton. Without this, when the
    // transfer curve is nearly vertical (steep transition region),
    // `dv = f/fp` blows up because fp ≈ 0 — the iterate flies to
    // -∞ or +∞ and Newton diverges. Clamping |dv| at a fraction of
    // the rail keeps the iterate physically meaningful.
    let max_step     = const_scalar(&mut b, 0.2);  // 200 mV per Newton step
    let neg_max_step = const_scalar(&mut b, -0.2);

    for _ in 0..n_newton {
        // Effective device biases.
        let vsg = b.binary(BinaryOp::Sub, vdd, v_in, scalar());        // VDD − V_in
        let vsd = b.binary(BinaryOp::Sub, vdd, v_out, scalar());       // VDD − V_out

        let i_n = id_subgraph_node(&mut b, v_in, v_out, vth_n, kp_n, lam_n);
        let i_p = id_subgraph_node(&mut b, vsg,  vsd,   vth_p, kp_p, lam_p);
        let f   = b.binary(BinaryOp::Sub, i_n, i_p, scalar());

        // Numerical derivative of f w.r.t. V_out via central FD inside
        // the graph. Cheaper than tracking ∂I/∂V_out symbolically and
        // avoids re-deriving the smooth-min branch's contribution.
        let v_out_p = b.binary(BinaryOp::Add, v_out, h_step, scalar());
        let v_out_m = b.binary(BinaryOp::Sub, v_out, h_step, scalar());
        let vsd_p   = b.binary(BinaryOp::Sub, vdd, v_out_p, scalar());
        let vsd_m   = b.binary(BinaryOp::Sub, vdd, v_out_m, scalar());

        let i_n_p = id_subgraph_node(&mut b, v_in, v_out_p, vth_n, kp_n, lam_n);
        let i_p_p = id_subgraph_node(&mut b, vsg,  vsd_p,   vth_p, kp_p, lam_p);
        let f_p   = b.binary(BinaryOp::Sub, i_n_p, i_p_p, scalar());
        let i_n_m = id_subgraph_node(&mut b, v_in, v_out_m, vth_n, kp_n, lam_n);
        let i_p_m = id_subgraph_node(&mut b, vsg,  vsd_m,   vth_p, kp_p, lam_p);
        let f_m   = b.binary(BinaryOp::Sub, i_n_m, i_p_m, scalar());

        // f'(V_out) ≈ (f_p − f_m) / (2h)
        let df  = b.binary(BinaryOp::Sub, f_p, f_m, scalar());
        let denom = b.binary(BinaryOp::Mul, two, h_step, scalar());
        let fp  = b.binary(BinaryOp::Div, df, denom, scalar());

        // V_out ← V_out − clip(f / f', ±max_step). Step cap keeps
        // the iterate in a physically reasonable range when fp ≈ 0.
        let dv_raw = b.binary(BinaryOp::Div, f, fp, scalar());
        let dv_capped_hi = b.binary(BinaryOp::Min, dv_raw, max_step, scalar());
        let dv = b.binary(BinaryOp::Max, dv_capped_hi, neg_max_step, scalar());
        v_out   = b.binary(BinaryOp::Sub, v_out, dv, scalar());
        let _ = (one, neg_one);
    }

    b.set_outputs(vec![v_out]);
    b
}

/// Build the inverter VJP body — IFT closed form for `∂V_out/∂param`
/// at the converged operating point.
///
/// At convergence, `f(V_out*; params) = I_n - I_p = 0`. Implicit
/// differentiation:
///   `∂V_out*/∂p = − (∂f/∂p) / (∂f/∂V_out)`
///
/// We compute both `∂f/∂p` and `∂f/∂V_out` numerically inside the
/// vjp_body via central FD on `id_strict`-shape evaluations.
/// Each backward call runs 2× per-parameter perturbation + 2 for the
/// V_out derivative — O(n_params) work, independent of `n_newton`.
fn build_vjp_body() -> Graph {
    let mut b = Graph::new("inverter_dc_vjp");
    let v_in    = b.input("V_in",          scalar());
    let vdd     = b.input("VDD",           scalar());
    let vth_n   = b.input("Vth_n",         scalar());
    let kp_n    = b.input("kp_n",          scalar());
    let lam_n   = b.input("lam_n",         scalar());
    let vth_p   = b.input("Vth_p",         scalar());
    let kp_p    = b.input("kp_p",          scalar());
    let lam_p   = b.input("lam_p",         scalar());
    let v_out   = b.input("primal_output", scalar());
    let dout    = b.input("d_output",      scalar());

    let two     = const_scalar(&mut b, 2.0);
    let neg_one = const_scalar(&mut b, -1.0);
    let h_step  = const_scalar(&mut b, 1e-4);

    // Evaluate residual `f(V_out; params) = I_n - I_p` at a chosen
    // (V_out, params). Helper closure-like via an inline fn.
    let eval_f = |b: &mut Graph,
                  v_in: NodeId, v_out: NodeId, vdd: NodeId,
                  vth_n: NodeId, kp_n: NodeId, lam_n: NodeId,
                  vth_p: NodeId, kp_p: NodeId, lam_p: NodeId| -> NodeId
    {
        let vsg = b.binary(BinaryOp::Sub, vdd, v_in,  scalar());
        let vsd = b.binary(BinaryOp::Sub, vdd, v_out, scalar());
        let i_n = id_subgraph_node(b, v_in, v_out, vth_n, kp_n, lam_n);
        let i_p = id_subgraph_node(b, vsg,  vsd,   vth_p, kp_p, lam_p);
        b.binary(BinaryOp::Sub, i_n, i_p, scalar())
    };

    // ∂f/∂V_out via central FD.
    let v_out_p = b.binary(BinaryOp::Add, v_out, h_step, scalar());
    let v_out_m = b.binary(BinaryOp::Sub, v_out, h_step, scalar());
    let f_vp = eval_f(&mut b, v_in, v_out_p, vdd, vth_n, kp_n, lam_n, vth_p, kp_p, lam_p);
    let f_vm = eval_f(&mut b, v_in, v_out_m, vdd, vth_n, kp_n, lam_n, vth_p, kp_p, lam_p);
    let df_dv = b.binary(BinaryOp::Sub, f_vp, f_vm, scalar());
    let two_h = b.binary(BinaryOp::Mul, two, h_step, scalar());
    let df_dvout = b.binary(BinaryOp::Div, df_dv, two_h, scalar());

    // λ_adjoint = -d_output / df_dV_out.
    let neg_dout = b.binary(BinaryOp::Mul, neg_one, dout, scalar());
    let lambda   = b.binary(BinaryOp::Div, neg_dout, df_dvout, scalar());

    // Per-parameter gradient via central FD on f(V_out, p).
    let mut grad_for = |b: &mut Graph,
                        param_id: NodeId,
                        kind: u8| -> NodeId
    {
        // Perturb the named param up/down, hold V_out fixed at primal.
        let p_p = b.binary(BinaryOp::Add, param_id, h_step, scalar());
        let p_m = b.binary(BinaryOp::Sub, param_id, h_step, scalar());
        let f_p = match kind {
            0 => eval_f(b, v_in, v_out, vdd, p_p, kp_n,  lam_n,  vth_p, kp_p, lam_p),
            1 => eval_f(b, v_in, v_out, vdd, vth_n, p_p, lam_n,  vth_p, kp_p, lam_p),
            2 => eval_f(b, v_in, v_out, vdd, vth_n, kp_n, p_p,   vth_p, kp_p, lam_p),
            3 => eval_f(b, v_in, v_out, vdd, vth_n, kp_n, lam_n, p_p,   kp_p, lam_p),
            4 => eval_f(b, v_in, v_out, vdd, vth_n, kp_n, lam_n, vth_p, p_p,  lam_p),
            5 => eval_f(b, v_in, v_out, vdd, vth_n, kp_n, lam_n, vth_p, kp_p, p_p),
            _ => unreachable!(),
        };
        let f_m = match kind {
            0 => eval_f(b, v_in, v_out, vdd, p_m, kp_n,  lam_n,  vth_p, kp_p, lam_p),
            1 => eval_f(b, v_in, v_out, vdd, vth_n, p_m, lam_n,  vth_p, kp_p, lam_p),
            2 => eval_f(b, v_in, v_out, vdd, vth_n, kp_n, p_m,   vth_p, kp_p, lam_p),
            3 => eval_f(b, v_in, v_out, vdd, vth_n, kp_n, lam_n, p_m,   kp_p, lam_p),
            4 => eval_f(b, v_in, v_out, vdd, vth_n, kp_n, lam_n, vth_p, p_m,  lam_p),
            5 => eval_f(b, v_in, v_out, vdd, vth_n, kp_n, lam_n, vth_p, kp_p, p_m),
            _ => unreachable!(),
        };
        let df = b.binary(BinaryOp::Sub, f_p, f_m, scalar());
        let two_h = b.binary(BinaryOp::Mul, two, h_step, scalar());
        let df_dp = b.binary(BinaryOp::Div, df, two_h, scalar());
        // dp = (∂f/∂p) · λ_adjoint.
        b.binary(BinaryOp::Mul, df_dp, lambda, scalar())
    };

    let d_vth_n = grad_for(&mut b, vth_n, 0);
    let d_kp_n  = grad_for(&mut b, kp_n,  1);
    let d_lam_n = grad_for(&mut b, lam_n, 2);
    let d_vth_p = grad_for(&mut b, vth_p, 3);
    let d_kp_p  = grad_for(&mut b, kp_p,  4);
    let d_lam_p = grad_for(&mut b, lam_p, 5);

    // V_in / VDD: not gradient targets in this spike — emit zero
    // contributions so the AD pre-pass is satisfied (8 inputs ⇒ 8
    // outputs).
    let zero = const_scalar(&mut b, 0.0);

    b.set_outputs(vec![zero, zero, d_vth_n, d_kp_n, d_lam_n, d_vth_p, d_kp_p, d_lam_p]);
    b
}

/// Build the inverter DC graph.
/// Returns `(graph, Vth_n, kp_n, lam_n, Vth_p, kp_p, lam_p)`.
#[allow(clippy::type_complexity)]
pub fn build_inverter_dc_graph(
    n_newton: usize,
) -> (Graph, NodeId, NodeId, NodeId, NodeId, NodeId, NodeId) {
    let mut g = Graph::new("inverter_dc");
    let v_in  = g.input("V_in", scalar());
    let vdd   = g.input("VDD",  scalar());
    let vth_n = g.param("Vth_n", scalar());
    let kp_n  = g.param("kp_n",  scalar());
    let lam_n = g.param("lam_n", scalar());
    let vth_p = g.param("Vth_p", scalar());
    let kp_p  = g.param("kp_p",  scalar());
    let lam_p = g.param("lam_p", scalar());

    let v_out = g.custom_fn(
        vec![v_in, vdd, vth_n, kp_n, lam_n, vth_p, kp_p, lam_p],
        build_fwd_body(n_newton),
        Some(build_vjp_body()),
        None,
    );
    g.set_outputs(vec![v_out]);
    (g, vth_n, kp_n, lam_n, vth_p, kp_p, lam_p)
}

/// Pure-Rust Newton on the same inverter KCL — used as the "did rlx
/// forward correctly?" oracle.
#[allow(clippy::too_many_arguments)]
pub fn ref_inverter_dc(
    v_in: f64, vdd: f64,
    vth_n: f64, kp_n: f64, lam_n: f64,
    vth_p: f64, kp_p: f64, lam_p: f64,
    n_newton: usize,
) -> f64 {
    let h = 1e-3;
    let max_step = 0.2;
    let mut v_out = vdd / 2.0;
    let f = |v_out: f64| -> f64 {
        let i_n = id_strict(v_in, v_out, vth_n, kp_n, lam_n);
        let i_p = id_strict(vdd - v_in, vdd - v_out, vth_p, kp_p, lam_p);
        i_n - i_p
    };
    for _ in 0..n_newton {
        let f_v   = f(v_out);
        let fp_v  = (f(v_out + h) - f(v_out - h)) / (2.0 * h);
        let dv    = (f_v / fp_v).clamp(-max_step, max_step);
        v_out -= dv;
    }
    v_out
}

/// Forward-only: `V_out` at the supplied parameters.
#[allow(clippy::too_many_arguments)]
pub fn run_inverter_dc(
    v_in: f64, vdd: f64,
    vth_n: f64, kp_n: f64, lam_n: f64,
    vth_p: f64, kp_p: f64, lam_p: f64,
    n_newton: usize,
) -> f64 {
    let (graph, _, _, _, _, _, _) = build_inverter_dc_graph(n_newton);
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("Vth_n", &vth_n.to_le_bytes(), DType::F64);
    compiled.set_param_typed("kp_n",  &kp_n .to_le_bytes(), DType::F64);
    compiled.set_param_typed("lam_n", &lam_n.to_le_bytes(), DType::F64);
    compiled.set_param_typed("Vth_p", &vth_p.to_le_bytes(), DType::F64);
    compiled.set_param_typed("kp_p",  &kp_p .to_le_bytes(), DType::F64);
    compiled.set_param_typed("lam_p", &lam_p.to_le_bytes(), DType::F64);
    let outs = compiled.run_typed(&[
        ("V_in", &v_in.to_le_bytes(), DType::F64),
        ("VDD",  &vdd .to_le_bytes(), DType::F64),
    ]);
    decode_f64(&outs[0].0)
}

/// Forward + reverse AD: `(V_out, ∂V_out/∂{Vth_n, kp_n, lam_n,
/// Vth_p, kp_p, lam_p})`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn run_inverter_dc_grad(
    v_in: f64, vdd: f64,
    vth_n: f64, kp_n: f64, lam_n: f64,
    vth_p: f64, kp_p: f64, lam_p: f64,
    n_newton: usize,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let (fwd, vth_n_id, kp_n_id, lam_n_id, vth_p_id, kp_p_id, lam_p_id) =
        build_inverter_dc_graph(n_newton);
    let bwd = grad_with_loss(
        &fwd,
        &[vth_n_id, kp_n_id, lam_n_id, vth_p_id, kp_p_id, lam_p_id],
    );
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("Vth_n", &vth_n.to_le_bytes(), DType::F64);
    compiled.set_param_typed("kp_n",  &kp_n .to_le_bytes(), DType::F64);
    compiled.set_param_typed("lam_n", &lam_n.to_le_bytes(), DType::F64);
    compiled.set_param_typed("Vth_p", &vth_p.to_le_bytes(), DType::F64);
    compiled.set_param_typed("kp_p",  &kp_p .to_le_bytes(), DType::F64);
    compiled.set_param_typed("lam_p", &lam_p.to_le_bytes(), DType::F64);
    let one = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("V_in",     &v_in.to_le_bytes(), DType::F64),
        ("VDD",      &vdd .to_le_bytes(), DType::F64),
        ("d_output", &one,                 DType::F64),
    ]);
    (
        decode_f64(&outs[0].0),
        decode_f64(&outs[1].0),
        decode_f64(&outs[2].0),
        decode_f64(&outs[3].0),
        decode_f64(&outs[4].0),
        decode_f64(&outs[5].0),
        decode_f64(&outs[6].0),
    )
}

fn decode_f64(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    f64::from_le_bytes(buf)
}
