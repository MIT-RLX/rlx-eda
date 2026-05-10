//! Minimal NN layer + training primitives over rlx graphs.
//!
//! rlx exposes Graph IR + AD + runtime, but no `nn::Linear`/`Sequential`.
//! Every spike that wants an MLP today (e.g. `spike-surrogate`) re-rolls
//! the same `g.param` + `g.matmul` + Glorot-init + Adam scaffolding.
//! This crate factors that out so PINN-style trainers in `eda-pinn` and
//! future surrogate spikes share one well-understood path.
//!
//! Scope is deliberately narrow:
//! - `Linear`, `Mlp` — composable subgraph builders that hand back a
//!   `NodeId` so they can be wired into a larger loss graph.
//! - `ParamSpec` — name/shape/element-count triple that lets the trainer
//!   slice a flat weight vector into per-parameter chunks.
//! - `Adam` — same implementation used in spike-surrogate, lifted verbatim.
//! - `Rng` — xorshift32 for deterministic Glorot init and batch sampling.
//!
//! No dataset/dataloader, no high-level `Module` trait, no checkpointing.
//! Add only when a second consumer needs it.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Shape};

/// Spec for one rlx-graph parameter slot. The trainer keeps a flat
/// `Vec<f32>` of weights; `ParamSpec` tells it how to slice that vector
/// when calling `compiled.set_param(name, slice)` and how to read
/// per-parameter gradients back from `compiled.run` outputs.
#[derive(Clone, Debug)]
pub struct ParamSpec {
    pub name: String,
    pub shape: Shape,
    pub n: usize,
    /// Glorot fan-in/out used to seed this slot. Biases use `(0, 0)` and
    /// are zero-initialised; weight matrices use the layer's `(in, out)`.
    pub fan: (usize, usize),
}

fn s2(a: usize, b: usize) -> Shape { Shape::new(&[a, b], DType::F32) }
fn s1(a: usize)             -> Shape { Shape::new(&[a],    DType::F32) }

/// Affine layer: `y = x @ W + b`. Holds the rlx node ids of its
/// parameters so the trainer can address them by name.
#[derive(Clone, Debug)]
pub struct Linear {
    pub w_id: NodeId,
    pub b_id: NodeId,
    pub w_name: String,
    pub b_name: String,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    /// Allocate `W [in,out]` and `b [out]` parameters in `g`. `prefix`
    /// becomes the scope for parameter naming, e.g. `"l0"` → `"l0.W"`,
    /// `"l0.b"`. Names must be unique within the graph.
    pub fn new(g: &mut Graph, prefix: &str, in_dim: usize, out_dim: usize) -> Self {
        let w_name = format!("{prefix}.W");
        let b_name = format!("{prefix}.b");
        let w_id = g.param(&w_name, s2(in_dim, out_dim));
        let b_id = g.param(&b_name, s1(out_dim));
        Self { w_id, b_id, w_name, b_name, in_dim, out_dim }
    }

    /// Forward: takes `x [B, in_dim]`, returns `[B, out_dim]`.
    pub fn forward(&self, g: &mut Graph, x: NodeId, batch: usize) -> NodeId {
        let xw = g.matmul(x, self.w_id, s2(batch, self.out_dim));
        g.binary(BinaryOp::Add, xw, self.b_id, s2(batch, self.out_dim))
    }

    pub fn param_specs(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec {
                name: self.w_name.clone(),
                shape: s2(self.in_dim, self.out_dim),
                n: self.in_dim * self.out_dim,
                fan: (self.in_dim, self.out_dim),
            },
            ParamSpec {
                name: self.b_name.clone(),
                shape: s1(self.out_dim),
                n: self.out_dim,
                fan: (0, 0),
            },
        ]
    }
}

/// Plain feedforward MLP: a stack of `Linear` layers separated by one
/// activation. The final layer is *un*-activated (callers append their
/// own loss/output transform).
#[derive(Clone, Debug)]
pub struct Mlp {
    pub layers: Vec<Linear>,
    pub act: Activation,
}

impl Mlp {
    /// Construct an MLP with shape `dims = [in, h0, h1, ..., out]`.
    /// Hidden activations all use `act`; the output is linear.
    pub fn new(g: &mut Graph, prefix: &str, dims: &[usize], act: Activation) -> Self {
        assert!(dims.len() >= 2, "Mlp needs at least input + output dim");
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, ab)| Linear::new(g, &format!("{prefix}.l{i}"), ab[0], ab[1]))
            .collect();
        Self { layers, act }
    }

    /// Forward over a batch of size `batch`.
    pub fn forward(&self, g: &mut Graph, x: NodeId, batch: usize) -> NodeId {
        let last = self.layers.len() - 1;
        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(g, h, batch);
            if i != last {
                h = g.activation(self.act, h, s2(batch, layer.out_dim));
            }
        }
        h
    }

    pub fn param_specs(&self) -> Vec<ParamSpec> {
        self.layers.iter().flat_map(|l| l.param_specs()).collect()
    }

    pub fn param_ids(&self) -> Vec<NodeId> {
        self.layers.iter().flat_map(|l| [l.w_id, l.b_id]).collect()
    }
}

/// Glorot/Xavier init for a flat parameter vector laid out per
/// `param_specs`. Biases (`fan == (0, 0)`) are zeroed.
pub fn init_glorot(specs: &[ParamSpec], rng: &mut Rng) -> Vec<f32> {
    let total: usize = specs.iter().map(|s| s.n).sum();
    let mut out = Vec::with_capacity(total);
    for sp in specs {
        let (fi, fo) = sp.fan;
        if fi == 0 && fo == 0 {
            out.extend(std::iter::repeat(0.0).take(sp.n));
        } else {
            let std = (2.0 / (fi + fo) as f32).sqrt();
            for _ in 0..sp.n {
                out.push(std * rng.next_normal());
            }
        }
    }
    out
}

/// Adam optimiser over a flat parameter vector. Identical to the
/// implementation that's been validated in spike-surrogate; lifted here
/// so PINN trainers in `eda-pinn` can reuse it without re-importing
/// from a spike crate.
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
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), self.m.len());
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

/// Deterministic xorshift32 PRNG. Matches the seed scheme used in
/// existing spike crates so cross-comparisons are reproducible.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self { Self(seed.max(1)) }

    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`.
    pub fn next_unit(&mut self) -> f32 {
        (self.next() as f64 / u32::MAX as f64) as f32
    }

    /// Standard normal via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_unit().max(1e-9);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_param_count_is_consistent() {
        let mut g = Graph::new("t");
        let mlp = Mlp::new(&mut g, "m", &[3, 8, 1], Activation::Relu);
        let specs = mlp.param_specs();
        let total: usize = specs.iter().map(|s| s.n).sum();
        assert_eq!(total, 3 * 8 + 8 + 8 * 1 + 1);
        assert_eq!(specs.len(), 4);
        assert_eq!(specs[0].name, "m.l0.W");
        assert_eq!(specs[3].name, "m.l1.b");
    }

    #[test]
    fn glorot_init_zeros_biases() {
        let mut g = Graph::new("t");
        let mlp = Mlp::new(&mut g, "m", &[2, 4, 1], Activation::Tanh);
        let specs = mlp.param_specs();
        let mut rng = Rng::new(42);
        let w = init_glorot(&specs, &mut rng);
        let w1n = 2 * 4;
        let b1n = 4;
        for x in &w[w1n..w1n + b1n] {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn adam_decreases_quadratic_loss() {
        let mut x = vec![0.0_f32];
        let mut opt = Adam::new(0.1, 1);
        for _ in 0..200 {
            let g = vec![2.0 * (x[0] - 3.0)];
            opt.step(&mut x, &g);
        }
        assert!((x[0] - 3.0).abs() < 1e-2, "x = {}", x[0]);
    }
}
