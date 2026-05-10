//! Run a `Trained` PINN over a batch of samples on a chosen device.
//!
//! Inference is one MLP forward pass with `batch = N`. Trained
//! weights transfer from training by name (param ids are graph-local
//! but the names are stable: `m.l0.W`, `m.l1.b`, …).

use rlx_runtime::{Device, Session};

use crate::config::V_REF;
use crate::encoding::Sample;
use crate::graph::build_inference_graph;
use crate::train::Trained;

/// Returns `Vmid` predictions in physical units (volts).
pub fn predict(trained: &Trained, samples: &[Sample], device: Device) -> Vec<f32> {
    let n = samples.len();
    let (g, _specs) = build_inference_graph(n);
    let mut compiled = Session::new(device).compile(g);

    let mut off = 0;
    for sp in &trained.specs {
        compiled.set_param(&sp.name, &trained.weights[off..off + sp.n]);
        off += sp.n;
    }

    let mut x_flat = Vec::with_capacity(n * 5);
    for s in samples {
        x_flat.extend_from_slice(&s.encode());
    }

    let outs = compiled.run(&[("x_inf", &x_flat[..])]);
    outs[0].iter().map(|&v_n| v_n * V_REF).collect()
}
