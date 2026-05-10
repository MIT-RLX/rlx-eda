//! Surrogate ML for the divider — train an MLP to mimic
//! `Vout = V·R2 / (R1+R2)`, then optimize over the surrogate.
//!
//! ## Why a surrogate
//!
//! For a 2-resistor divider the analytic form is trivially fast — a
//! surrogate is overkill. The pitch is for **expensive** simulators
//! (transient + nonlinear devices, FDTD photonics, …) where each
//! gradient evaluation costs seconds. Train once on `N_train` samples;
//! optimize-over-surrogate gives ~free gradients afterward.
//!
//! ## What we demonstrate
//!
//! - One rlx graph defines both the MLP forward AND the MSE training
//!   loss (so `grad_with_loss` produces ∂loss/∂{all weights} in one shot).
//! - The same `Adam` optimizer used for inverse-design on the divider
//!   trains the surrogate weights — pluggable optimizer pattern reused.
//! - After ~1000 training steps the surrogate's MSE drops by 100× from
//!   its initialization. That's the assert the test makes.
//!
//! ## Architecture
//!
//! 2-layer MLP: `[3] → ReLU → [hidden] → [1]`
//!   - input vector: `[R1_norm, R2_norm, V]`
//!   - output: `Vout` prediction
//!   - loss: mean squared error vs the analytic ground truth
//!
//! Inputs are normalized: R values divided by 10 kΩ so they sit roughly
//! in `[0, 1]`. V is already in that range. Without normalization the
//! MLP needs much bigger weights and Adam's lr has to be retuned.

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

pub const BATCH: usize = 32;
pub const HIDDEN: usize = 16;
const R_SCALE: f32 = 10_000.0;

/// Spec for one rlx-graph parameter slot — shape (so set_param sees the
/// right element count) and flat element count (for slicing).
#[derive(Clone)]
struct ParamSpec {
    name: &'static str,
    shape: Shape,
    n: usize,
}

fn s2(a: usize, b: usize) -> Shape { Shape::new(&[a, b], DType::F32) }
fn s1(a: usize)             -> Shape { Shape::new(&[a],    DType::F32) }

fn param_specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec { name: "W1", shape: s2(3, HIDDEN),    n: 3 * HIDDEN },
        ParamSpec { name: "b1", shape: s1(HIDDEN),       n: HIDDEN },
        ParamSpec { name: "W2", shape: s2(HIDDEN, 1),    n: HIDDEN * 1 },
        ParamSpec { name: "b2", shape: s1(1),            n: 1 },
    ]
}

/// Build the training graph: takes batched inputs `x [B, 3]`, target
/// `y [B, 1]`, returns scalar mean-squared-error loss.
pub fn build_training_graph() -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("mlp_surrogate");
    let x = g.input("x", s2(BATCH, 3));
    let y = g.input("y", s2(BATCH, 1));

    let w1 = g.param("W1", s2(3, HIDDEN));
    let b1 = g.param("b1", s1(HIDDEN));
    let w2 = g.param("W2", s2(HIDDEN, 1));
    let b2 = g.param("b2", s1(1));

    // Layer 1: relu(x @ W1 + b1)  → [B, H]
    let xw1     = g.matmul(x, w1, s2(BATCH, HIDDEN));
    let xw1_b   = g.binary(BinaryOp::Add, xw1, b1, s2(BATCH, HIDDEN));
    let h1      = g.activation(Activation::Relu, xw1_b, s2(BATCH, HIDDEN));

    // Layer 2: h1 @ W2 + b2  → [B, 1]
    let h1w2 = g.matmul(h1, w2, s2(BATCH, 1));
    let out  = g.binary(BinaryOp::Add, h1w2, b2, s2(BATCH, 1));

    // MSE: sum((out - y)²). Sum, not mean — bias correction lives in
    // the lr; mean would just rescale by 1/B.
    let diff = g.binary(BinaryOp::Sub, out, y,    s2(BATCH, 1));
    let sq   = g.binary(BinaryOp::Mul, diff, diff, s2(BATCH, 1));
    let loss = g.reduce(sq, ReduceOp::Sum, vec![0, 1], false, s1(1));
    g.set_outputs(vec![loss]);

    (g, vec![w1, b1, w2, b2])
}

/// Tiny deterministic PRNG so tests are reproducible. xorshift32.
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self { Self(seed.max(1)) }
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x;
        x
    }
    /// Uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f32 { (self.next() as f64 / u32::MAX as f64) as f32 }
    /// Standard normal via Box-Muller.
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_unit().max(1e-9);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Generate `BATCH` random `(R1, R2, V) → Vout` samples. Returns
/// `(x_flat [B*3], y_flat [B*1])` ready to feed `compiled.run`.
pub fn sample_batch(rng: &mut impl FnMut() -> f32) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(BATCH * 3);
    let mut y = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let r1 = 100.0 + 9_900.0 * rng();   // 100 Ω – 10 kΩ
        let r2 = 100.0 + 9_900.0 * rng();
        let v  = 0.1 + 4.9 * rng();          // 0.1 – 5 V
        let vout = v * r2 / (r1 + r2);
        x.push(r1 / R_SCALE);
        x.push(r2 / R_SCALE);
        x.push(v);
        y.push(vout);
    }
    (x, y)
}

/// Initialize MLP weights via Glorot/Xavier — `~N(0, 2 / (fan_in + fan_out))`.
pub fn init_weights(rng: &mut Rng) -> Vec<f32> {
    let mut all = Vec::new();
    let (fan_in1, fan_out1) = (3, HIDDEN);
    let std1 = (2.0 / (fan_in1 + fan_out1) as f32).sqrt();
    for _ in 0..(fan_in1 * fan_out1) { all.push(std1 * rng.next_normal()); }
    for _ in 0..fan_out1 { all.push(0.0); }   // b1 = 0

    let (fan_in2, fan_out2) = (HIDDEN, 1);
    let std2 = (2.0 / (fan_in2 + fan_out2) as f32).sqrt();
    for _ in 0..(fan_in2 * fan_out2) { all.push(std2 * rng.next_normal()); }
    for _ in 0..fan_out2 { all.push(0.0); }   // b2 = 0

    all
}

/// One Adam step over a flat parameter vector.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}
impl Adam {
    pub fn new(lr: f32, n_params: usize) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            m: vec![0.0; n_params], v: vec![0.0; n_params], t: 0,
        }
    }
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t = self.t.saturating_add(1);
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

pub struct TrainResult {
    pub losses: Vec<f32>,
    pub final_weights: Vec<f32>,
}

/// Train the surrogate for `n_steps` iterations with Adam at `lr`.
/// Each step uses a fresh batch of random samples drawn from `rng`.
/// Returns the per-step loss trajectory + final weights.
pub fn train(n_steps: usize, lr: f32, seed: u32) -> TrainResult {
    let (fwd, param_ids) = build_training_graph();
    let bwd = grad_with_loss(&fwd, &param_ids);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);

    let specs = param_specs();
    let total_params: usize = specs.iter().map(|s| s.n).sum();

    let mut rng = Rng::new(seed);
    let mut weights = init_weights(&mut rng);
    let mut adam = Adam::new(lr, total_params);

    let mut losses = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        // Slice flat weight vector into per-Param chunks for set_param.
        let mut off = 0;
        for sp in &specs {
            compiled.set_param(sp.name, &weights[off..off + sp.n]);
            off += sp.n;
        }

        let (x_flat, y_flat) = sample_batch(&mut || rng.next_unit());

        let outs = compiled.run(&[
            ("x",        &x_flat[..]),
            ("y",        &y_flat[..]),
            ("d_output", &[1.0_f32][..]),
        ]);
        let loss = outs[0][0] / BATCH as f32;
        losses.push(loss);

        // Concat per-Param gradients (outs[1..]) into one flat grad vec.
        let mut grads = Vec::with_capacity(total_params);
        for i in 0..specs.len() {
            grads.extend_from_slice(&outs[1 + i]);
        }
        adam.step(&mut weights, &grads);
    }
    TrainResult { losses, final_weights: weights }
}
