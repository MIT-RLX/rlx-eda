//! Run a trained PINN on a batch of inputs.

use rlx_runtime::{Device, Session};

use crate::graph::build_inference_graph;
use crate::train::Trained;

pub fn predict(trained: &Trained, xs: &[f32], device: Device) -> Vec<f32> {
    let n = xs.len();
    let (g, _specs) = build_inference_graph(n);
    let mut compiled = Session::new(device).compile(g);

    let mut off = 0;
    for sp in &trained.specs {
        compiled.set_param(&sp.name, &trained.weights[off..off + sp.n]);
        off += sp.n;
    }

    let outs = compiled.run(&[("x_inf", xs)]);
    outs[0].clone()
}
