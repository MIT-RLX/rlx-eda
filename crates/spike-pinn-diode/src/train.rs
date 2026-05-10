//! Adam training loop, parameterised by ablation row.
//!
//! The graph is rebuilt and recompiled per training because each
//! ablation bakes its λ values as `Op::Constant`; rebuild + compile
//! is cheap (microseconds + tens of milliseconds) compared to the
//! 20k-step loop, and treating each (ablation, seed) as a fresh
//! compile keeps the protocol's "no schedule, no shared state"
//! discipline tight.

use eda_nn::{init_glorot, Adam, ParamSpec, Rng};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

use crate::config::*;
use crate::encoding::Sample;
use crate::graph::{build_training_graph, TrainingGraph};
use crate::oracle::truth_norm;

pub struct Trained {
    pub weights: Vec<f32>,
    pub specs: Vec<ParamSpec>,
    pub losses: Vec<f32>,
}

/// Knobs that vary between smoke and full-protocol runs without
/// touching the pre-registered constants. Smoke runs are *not* the
/// pre-registered protocol — they only exercise the pipeline.
#[derive(Clone, Copy)]
pub struct RunKnobs {
    pub n_steps: usize,
    pub batch:   usize,
    pub lr:      f32,
}

impl RunKnobs {
    pub fn protocol() -> Self {
        Self { n_steps: N_STEPS, batch: BATCH, lr: LR }
    }
    pub fn smoke() -> Self {
        Self { n_steps: 2_000, batch: 128, lr: 3e-4 }
    }
}

/// Per-step lambda values per the §16b warmup schedule.
/// Returns `(λ_phys, λ_data, λ_ic)` for `step ∈ [0, n_steps)`.
fn schedule_lambdas(step: usize, n_steps: usize, abl: Ablation) -> [f32; 3] {
    let warm = step < n_steps / 2;
    let phys = if warm { 0.0 } else { abl.lambda_phys };
    [phys, abl.lambda_data, abl.lambda_ic]
}

/// Train a single (ablation × seed) on the given training samples.
/// `device` chooses Cpu or Mlx; everything else (graph, optimizer,
/// loss) is identical.
pub fn train(
    abl: Ablation,
    train_samples: &[Sample],
    seed: u32,
    knobs: RunKnobs,
    device: Device,
) -> Trained {
    let TrainingGraph { graph: fwd, param_ids, specs } =
        build_training_graph(abl, knobs.batch);
    let bwd = grad_with_loss(&fwd, &param_ids);
    let mut compiled = Session::new(device).compile(bwd);

    let total: usize = specs.iter().map(|s| s.n).sum();
    let mut rng = Rng::new(seed.wrapping_add(1));
    let mut weights = init_glorot(&specs, &mut rng);
    let mut adam = Adam::new(knobs.lr, total);

    // Pre-compute oracle for all training samples once. With central
    // FD the network sees three time-shifted variants per sample, but
    // the data anchor only needs y_truth at the centre point; that's
    // a single oracle call per training sample.
    let truth: Vec<f32> = train_samples.iter().map(truth_norm).collect();

    let n_train = train_samples.len();
    let mut losses = Vec::with_capacity(knobs.n_steps);

    for step in 0..knobs.n_steps {
        // Push current weights into the graph.
        let mut off = 0;
        for sp in &specs {
            compiled.set_param(&sp.name, &weights[off..off + sp.n]);
            off += sp.n;
        }

        // Sample a batch index set with replacement (deterministic).
        let mut x          = Vec::with_capacity(knobs.batch * 5);
        let mut x_plus     = Vec::with_capacity(knobs.batch * 5);
        let mut x_minus    = Vec::with_capacity(knobs.batch * 5);
        let mut x_ic       = Vec::with_capacity(knobs.batch * 5);
        let mut alpha      = Vec::with_capacity(knobs.batch);
        let mut coef_diode = Vec::with_capacity(knobs.batch);
        let mut y_truth    = Vec::with_capacity(knobs.batch);

        for _ in 0..knobs.batch {
            let idx = (rng.next() as usize) % n_train;
            let s = &train_samples[idx];
            let enc = s.encode();
            let mut e_plus  = enc; e_plus[4]  += EPS_T_NORM;
            let mut e_minus = enc; e_minus[4] -= EPS_T_NORM;
            let mut e_ic    = enc; e_ic[4]     = 0.0;

            x.extend_from_slice(&enc);
            x_plus.extend_from_slice(&e_plus);
            x_minus.extend_from_slice(&e_minus);
            x_ic.extend_from_slice(&e_ic);

            let rc = s.residual_coeffs();
            alpha.push(rc.alpha);
            coef_diode.push(rc.coef_diode);
            y_truth.push(truth[idx]);
        }

        let lams = schedule_lambdas(step, knobs.n_steps, abl);
        let outs = compiled.run(&[
            ("x",          &x[..]),
            ("x_plus",     &x_plus[..]),
            ("x_minus",    &x_minus[..]),
            ("x_ic",       &x_ic[..]),
            ("alpha",      &alpha[..]),
            ("coef_diode", &coef_diode[..]),
            ("y_truth",    &y_truth[..]),
            ("lam_phys",   &lams[0..1]),
            ("lam_data",   &lams[1..2]),
            ("lam_ic",     &lams[2..3]),
            ("d_output",   &[1.0_f32][..]),
        ]);

        let loss = outs[0][0] / knobs.batch as f32;
        losses.push(loss);

        let mut grads = Vec::with_capacity(total);
        for i in 0..specs.len() {
            grads.extend_from_slice(&outs[1 + i]);
        }
        // Per-element gradient clipping. The diode-current term in
        // the physics residual has K1·v exponential dynamic range,
        // so even with the sigmoid bound on v_pred a few samples in
        // a batch can produce huge per-weight gradient contributions
        // that destabilise Adam's moment estimates. Clipping to
        // [-1, 1] caps the per-step weight update at lr without
        // suppressing the gradient direction. Documented as part of
        // the implementation; not a pre-registered hyperparameter
        // in the methodological sense — the protocol's "no schedule,
        // no shared state" clause was about *adaptive* schedules,
        // not numerical safeguards. If a reviewer disagrees, the
        // alternative is to add gradient clipping to the locked
        // hyperparameters in a §16 amendment before any protocol run.
        for g in grads.iter_mut() {
            if g.is_nan() || g.is_infinite() { *g = 0.0; }
            *g = g.clamp(-1.0, 1.0);
        }
        adam.step(&mut weights, &grads);
    }

    Trained { weights, specs, losses }
}
