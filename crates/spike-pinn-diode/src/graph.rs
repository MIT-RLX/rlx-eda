//! Training-loss and inference graphs.
//!
//! Training graph (per §7) takes
//! - `x`, `x_plus`, `x_minus`, `x_ic`  : `[B, 5]` — central-FD shifts
//! - `coef_a`, `coef_b`, `coef_isn`     : `[B, 1]` — per-sample residual factors
//! - `y_truth`                          : `[B, 1]` — `Vmid/V_REF` from oracle
//! and produces a scalar loss
//! `L = λ_phys·Σres² + λ_ic·Σv_ic² + λ_data·Σ(v − y_truth)²`.
//! `λ_*` are baked as `Op::Constant` per ablation row, so each
//! ablation gets its own compiled graph (cheap; build is microseconds).
//!
//! Inference graph: input `x [N, 5]`, output `v [N, 1]`. Same MLP
//! parameter names so trained weights transfer by `set_param`.

use eda_nn::{Mlp, ParamSpec};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::config::*;
use crate::encoding::K1_DIODE_EXP;

fn s2(a: usize, b: usize) -> Shape { Shape::new(&[a, b], DType::F32) }
fn s1(a: usize)             -> Shape { Shape::new(&[a],    DType::F32) }
fn const_f32(g: &mut Graph, x: f32) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], s1(1))
}

const ACT: Activation = Activation::Tanh;

pub struct TrainingGraph {
    pub graph: Graph,
    pub param_ids: Vec<NodeId>,
    pub specs: Vec<ParamSpec>,
}

/// Build the loss graph for a given ablation row + batch size.
pub fn build_training_graph(abl: Ablation, batch: usize) -> TrainingGraph {
    let b = batch;
    let mut g = Graph::new("pinn_diode_train");

    let x          = g.input("x",          s2(b, 5));
    let x_plus     = g.input("x_plus",     s2(b, 5));
    let x_minus    = g.input("x_minus",    s2(b, 5));
    let x_ic       = g.input("x_ic",       s2(b, 5));
    let alpha      = g.input("alpha",      s2(b, 1));
    let coef_diode = g.input("coef_diode", s2(b, 1));
    let y_truth    = g.input("y_truth",    s2(b, 1));

    let inv_2eps    = const_f32(&mut g, 1.0 / (2.0 * EPS_T_NORM));
    let one_const   = const_f32(&mut g, 1.0);
    let k1_const    = const_f32(&mut g, K1_DIODE_EXP);
    // Lambdas are runtime inputs (not constants) so the §16b warmup
    // schedule can vary `λ_phys` per step without rebuilding the
    // graph. `abl` selects the *target* lambda values, but the
    // training loop multiplies by the schedule's per-step gating
    // factor before pushing them as inputs.
    let _ = abl;
    let lam_phys = g.input("lam_phys", s1(1));
    let lam_data = g.input("lam_data", s1(1));
    let lam_ic   = g.input("lam_ic",   s1(1));

    // Shared MLP: forward applied to all four input slices (weight
    // sharing — same param node ids). The MLP's last linear is
    // followed by a sigmoid to bound `v_pred ∈ (0, 1)`, which is
    // physically required (Vmid is always between 0 and the diode
    // forward drop ≈ 0.7 V) and crucial for training stability:
    // without it the pre-sigmoid output at Glorot init can be ±3,
    // pushing `K1·v ≈ ±116` and `exp(K1·v)` to +inf, which
    // immediately NaNs the loss. The sigmoid keeps `K1·v ≤ 38.7`
    // and `exp(K1·v) ≤ 6.5e16` — still large but finite, and
    // gradient clipping in the optimizer handles the remainder.
    let mlp = Mlp::new(&mut g, "m", ARCH_DIMS, ACT);
    let raw_pred  = mlp.forward(&mut g, x,       b);
    let v_pred    = g.activation(Activation::Sigmoid, raw_pred,  s2(b, 1));
    let raw_plus  = mlp.forward(&mut g, x_plus,  b);
    let v_plus    = g.activation(Activation::Sigmoid, raw_plus,  s2(b, 1));
    let raw_minus = mlp.forward(&mut g, x_minus, b);
    let v_minus   = g.activation(Activation::Sigmoid, raw_minus, s2(b, 1));
    let raw_ic    = mlp.forward(&mut g, x_ic,    b);
    let v_ic      = g.activation(Activation::Sigmoid, raw_ic,    s2(b, 1));

    // Central FD: dv/dt_n = (v_plus − v_minus) / (2ε)
    let dv      = g.binary(BinaryOp::Sub, v_plus, v_minus, s2(b, 1));
    let dv_dt_n = g.binary(BinaryOp::Mul, dv, inv_2eps, s2(b, 1));

    // KCL residual normalised by drive current `V_dc/R`:
    //   r_n = 1 − α·v − coef_diode·(exp(K1·v) − 1) − α·dv/dt_n
    let alpha_v       = g.binary(BinaryOp::Mul, alpha, v_pred, s2(b, 1));
    let one_minus_av  = g.binary(BinaryOp::Sub, one_const, alpha_v, s2(b, 1));
    let k1_v          = g.binary(BinaryOp::Mul, k1_const, v_pred, s2(b, 1));
    let exp_term      = g.activation(Activation::Exp, k1_v, s2(b, 1));
    let exp_minus_1   = g.binary(BinaryOp::Sub, exp_term, one_const, s2(b, 1));
    let i_diode       = g.binary(BinaryOp::Mul, coef_diode, exp_minus_1, s2(b, 1));
    let i_c           = g.binary(BinaryOp::Mul, alpha, dv_dt_n, s2(b, 1));
    let r1            = g.binary(BinaryOp::Sub, one_minus_av, i_diode, s2(b, 1));
    let res           = g.binary(BinaryOp::Sub, r1, i_c, s2(b, 1));

    let res_sq    = g.binary(BinaryOp::Mul, res, res, s2(b, 1));
    let v_ic_sq   = g.binary(BinaryOp::Mul, v_ic, v_ic, s2(b, 1));
    let v_err     = g.binary(BinaryOp::Sub, v_pred, y_truth, s2(b, 1));
    let v_err_sq  = g.binary(BinaryOp::Mul, v_err, v_err, s2(b, 1));

    let l_phys_raw = g.reduce(res_sq,   ReduceOp::Sum, vec![0, 1], false, s1(1));
    let l_ic_raw   = g.reduce(v_ic_sq,  ReduceOp::Sum, vec![0, 1], false, s1(1));
    let l_data_raw = g.reduce(v_err_sq, ReduceOp::Sum, vec![0, 1], false, s1(1));

    let l_phys = g.binary(BinaryOp::Mul, l_phys_raw, lam_phys, s1(1));
    let l_ic   = g.binary(BinaryOp::Mul, l_ic_raw,   lam_ic,   s1(1));
    let l_data = g.binary(BinaryOp::Mul, l_data_raw, lam_data, s1(1));
    let pi     = g.binary(BinaryOp::Add, l_phys, l_ic, s1(1));
    let loss   = g.binary(BinaryOp::Add, pi,     l_data, s1(1));

    g.set_outputs(vec![loss]);

    let specs = mlp.param_specs();
    let pids  = mlp.param_ids();
    TrainingGraph { graph: g, param_ids: pids, specs }
}

/// Inference graph: same MLP shape and same final sigmoid as
/// training, batch=N. Param names are stable so trained weights
/// transfer by `set_param`.
pub fn build_inference_graph(batch: usize) -> (Graph, Vec<ParamSpec>) {
    let mut g = Graph::new("pinn_diode_infer");
    let x = g.input("x_inf", s2(batch, 5));
    let mlp = Mlp::new(&mut g, "m", ARCH_DIMS, ACT);
    let raw = mlp.forward(&mut g, x, batch);
    let v = g.activation(Activation::Sigmoid, raw, s2(batch, 1));
    g.set_outputs(vec![v]);
    (g, mlp.param_specs())
}

