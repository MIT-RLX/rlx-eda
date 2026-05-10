//! Adam loop for the SAR PINN. Pure data-MSE.

use eda_nn::{init_glorot, Adam, ParamSpec, Rng};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

use crate::config::*;
use crate::graph::{build_training_graph, TrainingGraph};
use crate::oracle::truth_norm;

pub struct Trained {
    pub weights: Vec<f32>,
    pub specs: Vec<ParamSpec>,
    pub losses: Vec<f32>,
}

pub fn train(seed: u32, train_x: &[f32], device: Device) -> Trained {
    let TrainingGraph { graph: fwd, param_ids, specs } = build_training_graph(BATCH);
    let bwd = grad_with_loss(&fwd, &param_ids);
    let mut compiled = Session::new(device).compile(bwd);

    let total: usize = specs.iter().map(|s| s.n).sum();
    let mut rng = Rng::new(seed.wrapping_add(1));
    let mut weights = init_glorot(&specs, &mut rng);
    let mut adam = Adam::new(LR, total);

    let truth: Vec<f32> = train_x.iter().map(|x| truth_norm(*x)).collect();
    let n_train = train_x.len();
    let mut losses = Vec::with_capacity(N_STEPS);

    for _ in 0..N_STEPS {
        // Push current weights.
        let mut off = 0;
        for sp in &specs {
            compiled.set_param(&sp.name, &weights[off..off + sp.n]);
            off += sp.n;
        }

        let mut x_batch = Vec::with_capacity(BATCH);
        let mut y_batch = Vec::with_capacity(BATCH);
        for _ in 0..BATCH {
            let idx = (rng.next() as usize) % n_train;
            x_batch.push(train_x[idx]);
            y_batch.push(truth[idx]);
        }

        let outs = compiled.run(&[
            ("x",        &x_batch[..]),
            ("y_truth",  &y_batch[..]),
            ("d_output", &[1.0_f32][..]),
        ]);

        let loss = outs[0][0] / BATCH as f32;
        losses.push(loss);

        let mut grads = Vec::with_capacity(total);
        for i in 0..specs.len() {
            grads.extend_from_slice(&outs[1 + i]);
        }
        // Per-element gradient clipping; matches diode protocol's
        // hygiene clause from §16b.
        for g in grads.iter_mut() {
            if g.is_nan() || g.is_infinite() { *g = 0.0; }
            *g = g.clamp(-1.0, 1.0);
        }
        adam.step(&mut weights, &grads);
    }

    Trained { weights, specs, losses }
}
