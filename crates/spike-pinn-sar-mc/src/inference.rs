use rlx_runtime::{Device, Session};

use crate::config::INPUT_DIM;
use crate::graph::build_inference_graph;
use crate::sample::McSample;
use crate::train::Trained;

pub fn predict(trained: &Trained, samples: &[McSample], device: Device) -> Vec<f32> {
    let n = samples.len();
    let (g, _specs) = build_inference_graph(n);
    let mut compiled = Session::new(device).compile(g);

    let mut off = 0;
    for sp in &trained.specs {
        compiled.set_param(&sp.name, &trained.weights[off..off + sp.n]);
        off += sp.n;
    }

    let mut x_flat = Vec::with_capacity(n * INPUT_DIM);
    for s in samples {
        x_flat.extend_from_slice(&s.encode());
    }
    let outs = compiled.run(&[("x_inf", &x_flat[..])]);
    outs[0].clone()
}
