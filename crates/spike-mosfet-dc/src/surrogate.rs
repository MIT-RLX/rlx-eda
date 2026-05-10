//! Thermal MOSFET surrogate: an MLP that learns
//! `(Vgs, Vds, T_celsius) → Id` from samples drawn against
//! [`crate::run_id_at_temp`]. Same shape as `spike-surrogate` for the
//! resistor divider, generalized to a 3-d input + temperature corner.
//!
//! ## Why
//!
//! - **Speed.** Once trained, the surrogate evaluates `Id` in a single
//!   batched matmul — no per-sample graph compile, no Newton inside.
//!   Useful when the outer optimization loop wants 10⁴–10⁶ `Id`
//!   evaluations across (Vgs, Vds, T) and the analytic graph's overhead
//!   per call (compile + AD + run) becomes the bottleneck.
//! - **Replaces SPICE in the loop.** The training labels here come from
//!   our analytic model, but the architecture is identical for SPICE-
//!   labeled training (just swap the `label_fn`). When sky130 BSIM4 is
//!   the truth and we lack closed-form, this is the path: batch ngspice
//!   to generate (Vgs, Vds, T, Id) tuples, then train.
//! - **Differentiable through the temperature corner.** Unlike
//!   `run_id_at_temp` where T enters via a host-side parameter remap
//!   (no AD edge for T), the surrogate has T as a graph input — so
//!   downstream code can differentiate `Id` w.r.t. T directly. Useful
//!   for thermal sensitivity analysis or for picking a worst-corner T
//!   via gradient ascent.
//!
//! ## Architecture
//!
//! 2-layer MLP: `[3] → ReLU → [HIDDEN] → [1]`.
//!
//! Inputs are normalized:
//!   - `Vgs_norm = Vgs / V_NORM`  (V_NORM = 1.8 V)
//!   - `Vds_norm = Vds / V_NORM`
//!   - `T_norm   = (T_celsius − T_NOM_C) / T_SPAN`  (T_SPAN = 100 °C)
//!
//! Output is `Id × ID_SCALE` where `ID_SCALE = 1e4` so a typical
//! saturation current of 100 µA lands at ~1.0 — matches the dynamic
//! range MLPs train cleanly on.

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

use crate::{run_id_at_temp, T_NOM_C};

pub const BATCH:  usize = 64;
pub const HIDDEN: usize = 32;

pub const V_NORM:    f32 = 1.8;
pub const T_SPAN:    f32 = 100.0;
pub const ID_SCALE:  f32 = 1e4;

// Default device parameters used to label training samples. Chosen to
// match the existing `spike-mosfet-dc` test harness.
const VTH0_DEFAULT: f64 = 0.5;
const KP0_DEFAULT:  f64 = 100e-6;
const LAM_DEFAULT:  f64 = 0.02;

#[derive(Clone)]
struct ParamSpec { name: &'static str, shape: Shape, n: usize }

fn s2(a: usize, b: usize) -> Shape { Shape::new(&[a, b], DType::F32) }
fn s1(a: usize)             -> Shape { Shape::new(&[a],    DType::F32) }

fn param_specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec { name: "W1", shape: s2(3, HIDDEN), n: 3 * HIDDEN },
        ParamSpec { name: "b1", shape: s1(HIDDEN),    n: HIDDEN },
        ParamSpec { name: "W2", shape: s2(HIDDEN, 1), n: HIDDEN },
        ParamSpec { name: "b2", shape: s1(1),         n: 1 },
    ]
}

/// Build the training graph: batched MLP forward + MSE loss against
/// labels `y [B,1]`.
pub fn build_training_graph() -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("mosfet_thermal_surrogate");
    let x = g.input("x", s2(BATCH, 3));
    let y = g.input("y", s2(BATCH, 1));

    let w1 = g.param("W1", s2(3, HIDDEN));
    let b1 = g.param("b1", s1(HIDDEN));
    let w2 = g.param("W2", s2(HIDDEN, 1));
    let b2 = g.param("b2", s1(1));

    let xw1   = g.matmul(x, w1, s2(BATCH, HIDDEN));
    let xw1_b = g.binary(BinaryOp::Add, xw1, b1, s2(BATCH, HIDDEN));
    let h1    = g.activation(Activation::Relu, xw1_b, s2(BATCH, HIDDEN));

    let h1w2 = g.matmul(h1, w2, s2(BATCH, 1));
    let out  = g.binary(BinaryOp::Add, h1w2, b2, s2(BATCH, 1));

    let diff = g.binary(BinaryOp::Sub, out, y,    s2(BATCH, 1));
    let sq   = g.binary(BinaryOp::Mul, diff, diff, s2(BATCH, 1));
    let loss = g.reduce(sq, ReduceOp::Sum, vec![0, 1], false, s1(1));
    g.set_outputs(vec![loss]);

    (g, vec![w1, b1, w2, b2])
}

/// Inference graph: same MLP forward, no loss head, batch-1.
pub fn build_inference_graph() -> Graph {
    let mut g = Graph::new("mosfet_thermal_surrogate_inference");
    let x  = g.input("x", s2(1, 3));
    let w1 = g.param("W1", s2(3, HIDDEN));
    let b1 = g.param("b1", s1(HIDDEN));
    let w2 = g.param("W2", s2(HIDDEN, 1));
    let b2 = g.param("b2", s1(1));

    let xw1   = g.matmul(x, w1, s2(1, HIDDEN));
    let xw1_b = g.binary(BinaryOp::Add, xw1, b1, s2(1, HIDDEN));
    let h1    = g.activation(Activation::Relu, xw1_b, s2(1, HIDDEN));
    let h1w2  = g.matmul(h1, w2, s2(1, 1));
    let out   = g.binary(BinaryOp::Add, h1w2, b2, s2(1, 1));
    g.set_outputs(vec![out]);
    g
}

/// Tiny xorshift PRNG for reproducible tests.
pub struct Rng(u32);
impl Rng {
    pub fn new(seed: u32) -> Self { Self(seed.max(1)) }
    fn raw(&mut self) -> u32 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x; x
    }
    pub fn unit(&mut self) -> f32 { (self.raw() as f64 / u32::MAX as f64) as f32 }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-9);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Encode raw `(Vgs, Vds, T_celsius, Id)` into normalized `(x, y)`.
pub fn encode(vgs: f32, vds: f32, t_celsius: f32, id: f32) -> ([f32; 3], f32) {
    let x = [
        vgs / V_NORM,
        vds / V_NORM,
        (t_celsius - T_NOM_C as f32) / T_SPAN,
    ];
    let y = id * ID_SCALE;
    (x, y)
}

/// Inverse of `encode`'s `y` channel: surrogate output → physical Id.
#[inline] pub fn decode_id(y_norm: f32) -> f32 { y_norm / ID_SCALE }

/// Draw one training batch by sampling `(Vgs, Vds, T)` uniformly from
/// the operating envelope and labeling via `run_id_at_temp` with
/// nominal device params (`Vth0=0.5`, `kp0=100 µA`, `λ=0.02`).
pub fn sample_batch(rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(BATCH * 3);
    let mut y = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let vgs = rng.unit() * V_NORM;
        let vds = rng.unit() * V_NORM;
        let t   = -40.0 + 165.0 * rng.unit();   // [-40, 125] °C
        let id  = run_id_at_temp(
            vgs as f64, vds as f64,
            VTH0_DEFAULT, KP0_DEFAULT, LAM_DEFAULT,
            t as f64,
        ) as f32;
        let (xi, yi) = encode(vgs, vds, t, id);
        x.extend_from_slice(&xi);
        y.push(yi);
    }
    (x, y)
}

/// Glorot/Xavier weight init, biases zero. Returns a flat vector
/// matching `param_specs()` order.
pub fn init_weights(rng: &mut Rng) -> Vec<f32> {
    let mut all = Vec::new();
    let std1 = (2.0 / (3 + HIDDEN) as f32).sqrt();
    for _ in 0..(3 * HIDDEN) { all.push(std1 * rng.normal()); }
    for _ in 0..HIDDEN { all.push(0.0); }
    let std2 = (2.0 / (HIDDEN + 1) as f32).sqrt();
    for _ in 0..HIDDEN { all.push(std2 * rng.normal()); }
    all.push(0.0);
    all
}

/// Adam optimizer over a flat parameter vector. Identical to the one
/// in `spike-surrogate` — kept local to avoid pulling that crate into
/// our dep graph.
pub struct Adam {
    pub lr: f32, beta1: f32, beta2: f32, eps: f32,
    m: Vec<f32>, v: Vec<f32>, t: u32,
}
impl Adam {
    pub fn new(lr: f32, n: usize) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
               m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }
    pub fn step(&mut self, p: &mut [f32], g: &[f32]) {
        self.t = self.t.saturating_add(1);
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..p.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g[i] * g[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            p[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

pub struct TrainResult {
    pub losses: Vec<f32>,
    pub final_weights: Vec<f32>,
}

/// Train the surrogate for `n_steps` Adam iterations at learning rate
/// `lr`. Returns per-step mean loss + final flat weights.
pub fn train(n_steps: usize, lr: f32, seed: u32) -> TrainResult {
    let (fwd, param_ids) = build_training_graph();
    let bwd = grad_with_loss(&fwd, &param_ids);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);

    let specs = param_specs();
    let n_params: usize = specs.iter().map(|s| s.n).sum();

    let mut rng = Rng::new(seed);
    let mut weights = init_weights(&mut rng);
    let mut adam = Adam::new(lr, n_params);

    let mut losses = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let mut off = 0;
        for sp in &specs {
            compiled.set_param(sp.name, &weights[off..off + sp.n]);
            off += sp.n;
        }
        let (x, y) = sample_batch(&mut rng);
        let outs = compiled.run(&[
            ("x", &x[..]),
            ("y", &y[..]),
            ("d_output", &[1.0_f32][..]),
        ]);
        losses.push(outs[0][0] / BATCH as f32);

        let mut grads = Vec::with_capacity(n_params);
        for i in 0..specs.len() { grads.extend_from_slice(&outs[1 + i]); }
        adam.step(&mut weights, &grads);
    }
    TrainResult { losses, final_weights: weights }
}

/// Evaluate the trained surrogate at a single `(Vgs, Vds, T)` point.
/// `weights` is the flat vector returned by `train`.
pub fn predict(weights: &[f32], vgs: f32, vds: f32, t_celsius: f32) -> f32 {
    let g = build_inference_graph();
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let specs = param_specs();
    let mut off = 0;
    for sp in &specs {
        compiled.set_param(sp.name, &weights[off..off + sp.n]);
        off += sp.n;
    }
    let ([x0, x1, x2], _) = encode(vgs, vds, t_celsius, 0.0);
    let outs = compiled.run(&[("x", &[x0, x1, x2][..])]);
    decode_id(outs[0][0])
}
