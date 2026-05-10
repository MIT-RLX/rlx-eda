//! Training and inference graphs for the SAR PINN.
//!
//! Pure data-MSE regression — no physics residual, no IC, no FD
//! shifts. `v_pred = sigmoid(MLP(x))`, loss is sum of squared errors
//! over batch.

use eda_nn::{Mlp, ParamSpec};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::config::*;

fn s2(a: usize, b: usize) -> Shape { Shape::new(&[a, b], DType::F32) }
fn s1(a: usize)             -> Shape { Shape::new(&[a],    DType::F32) }

const ACT: Activation = Activation::Tanh;

pub struct TrainingGraph {
    pub graph: Graph,
    pub param_ids: Vec<NodeId>,
    pub specs: Vec<ParamSpec>,
}

pub fn build_training_graph(batch: usize) -> TrainingGraph {
    let b = batch;
    let mut g = Graph::new("pinn_sar_train");

    let x        = g.input("x",       s2(b, 1));
    let y_truth  = g.input("y_truth", s2(b, 1));

    let mlp = Mlp::new(&mut g, "m", ARCH_DIMS, ACT);
    let raw = mlp.forward(&mut g, x, b);
    let v_pred = g.activation(Activation::Sigmoid, raw, s2(b, 1));

    let v_err    = g.binary(BinaryOp::Sub, v_pred, y_truth, s2(b, 1));
    let v_err_sq = g.binary(BinaryOp::Mul, v_err, v_err,    s2(b, 1));
    let loss     = g.reduce(v_err_sq, ReduceOp::Sum, vec![0, 1], false, s1(1));

    g.set_outputs(vec![loss]);

    let specs = mlp.param_specs();
    let pids  = mlp.param_ids();
    TrainingGraph { graph: g, param_ids: pids, specs }
}

pub fn build_inference_graph(batch: usize) -> (Graph, Vec<ParamSpec>) {
    let mut g = Graph::new("pinn_sar_infer");
    let x = g.input("x_inf", s2(batch, 1));
    let mlp = Mlp::new(&mut g, "m", ARCH_DIMS, ACT);
    let raw = mlp.forward(&mut g, x, batch);
    let v = g.activation(Activation::Sigmoid, raw, s2(batch, 1));
    g.set_outputs(vec![v]);
    (g, mlp.param_specs())
}
