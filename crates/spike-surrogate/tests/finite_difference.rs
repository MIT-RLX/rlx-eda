//! AD-vs-FD parity for the surrogate's training-loss gradient.
//!
//! The existing `training.rs` test confirms loss drops 10× over 3000
//! Adam steps — but loss could be falling under noisy or systematically
//! biased gradients and still hit that bar. This test fixes a batch
//! and a deterministic weight vector, asks `grad_with_loss` for
//! `dloss/dW_i` at a sample of indices, and compares each to a
//! centered finite-difference estimate computed from the same compiled
//! graph re-run with `W_i ± ε`.
//!
//! Catches: per-parameter sign flips, missing VJP rules, broadcast
//! bugs in `Op::Reduce` of the MSE sum, and ReLU's f32 backward.

use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_surrogate::*;

/// Per-Param flat-weight layout. Mirrors `param_specs()` in lib.rs;
/// kept in lockstep here because the lib doesn't export that detail.
const W1_LEN: usize = 3 * HIDDEN;
const B1_LEN: usize = HIDDEN;
const W2_LEN: usize = HIDDEN;
const B2_LEN: usize = 1;
const TOTAL_LEN: usize = W1_LEN + B1_LEN + W2_LEN + B2_LEN;

const PARAM_NAMES: [&str; 4] = ["W1", "b1", "W2", "b2"];
const PARAM_LENS:  [usize; 4] = [W1_LEN, B1_LEN, W2_LEN, B2_LEN];

/// Deterministic synthetic weights — small uniform values, no
/// dependence on a particular RNG. Keeps the test self-contained.
fn fixed_weights() -> Vec<f32> {
    (0..TOTAL_LEN)
        .map(|i| 0.2 * ((i as f32 * 0.137).sin()))
        .collect()
}

/// Deterministic synthetic batch — fully self-contained (no Rng
/// dependency), same shape as `sample_batch`.
fn fixed_batch() -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(BATCH * 3);
    let mut y = Vec::with_capacity(BATCH);
    for k in 0..BATCH {
        let r1 = 100.0 + 9_900.0 * (k as f32 / BATCH as f32);
        let r2 = 100.0 + 9_900.0 * ((BATCH - k) as f32 / BATCH as f32);
        let v  = 0.1 + 4.9 * ((k as f32 * 0.31).fract());
        let vout = v * r2 / (r1 + r2);
        x.push(r1 / 10_000.0);
        x.push(r2 / 10_000.0);
        x.push(v);
        y.push(vout);
    }
    (x, y)
}

/// Run the compiled bwd graph with `weights` set across the four
/// Params and return `(loss, all_grads_flat)`.
fn run_loss_and_grad(
    compiled: &mut rlx_runtime::CompiledGraph,
    weights: &[f32], x: &[f32], y: &[f32],
) -> (f32, Vec<f32>) {
    let mut off = 0;
    for (name, &n) in PARAM_NAMES.iter().zip(PARAM_LENS.iter()) {
        compiled.set_param(name, &weights[off..off + n]);
        off += n;
    }
    let outs = compiled.run(&[
        ("x", x), ("y", y), ("d_output", &[1.0_f32][..]),
    ]);
    let loss = outs[0][0];
    let mut grads = Vec::with_capacity(TOTAL_LEN);
    for i in 0..PARAM_NAMES.len() {
        grads.extend_from_slice(&outs[1 + i]);
    }
    (loss, grads)
}

/// Pick a spread of weight indices across all four Params — at least
/// one element from each so a missing-VJP-on-one-param bug doesn't
/// hide. Sampling instead of every-element keeps the test fast (each
/// FD costs two compiled forward+backward runs).
fn sample_indices() -> Vec<usize> {
    let mut idx = Vec::new();
    // 4 elements from W1
    for k in [0, 11, 23, 47] { idx.push(k); }
    // 2 from b1
    let off = W1_LEN;
    for k in [0, 7] { idx.push(off + k); }
    // 2 from W2
    let off = W1_LEN + B1_LEN;
    for k in [0, 9] { idx.push(off + k); }
    // 1 from b2
    let off = W1_LEN + B1_LEN + W2_LEN;
    idx.push(off);
    idx
}

#[test]
fn loss_grad_matches_finite_differences() {
    let (fwd, param_ids) = build_training_graph();
    let bwd = grad_with_loss(&fwd, &param_ids);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);

    let weights = fixed_weights();
    let (x, y) = fixed_batch();

    let (_loss0, grads_ad) = run_loss_and_grad(&mut compiled, &weights, &x, &y);
    assert_eq!(grads_ad.len(), TOTAL_LEN, "expected {TOTAL_LEN} flat grad elements");

    // FD step: 2e-3 on weights of order 0.2. Loss is O(8) (32 batch
    // elements × ~0.5² MSE), so f32 ulp on the loss difference is
    // ~1e-6. Numerator (loss_p − loss_m) is ~2εf' ~ 1e-3, giving an
    // ulp/signal ratio ~1e-3 — the floor of central FD on f32 here.
    // Truncation O(ε²·f'') stays below that for ε=2e-3 with the
    // weights we set up. Net: ~1% relative envelope is honest.
    let eps: f32 = 2e-3;

    for &i in sample_indices().iter() {
        let mut w_p = weights.clone(); w_p[i] += eps;
        let mut w_m = weights.clone(); w_m[i] -= eps;
        let (loss_p, _) = run_loss_and_grad(&mut compiled, &w_p, &x, &y);
        let (loss_m, _) = run_loss_and_grad(&mut compiled, &w_m, &x, &y);
        let fd = (loss_p - loss_m) / (2.0 * eps);

        let ad = grads_ad[i];
        // 2% relative + 1e-2 absolute floor. Sign flips and missing
        // VJP rules produce |Δ|/|fd| of order 1+, so this catches
        // them; f32 cancellation noise sits ~1% relative on this graph.
        let env = 2e-2 * fd.abs() + 1e-2;
        let diff = (ad - fd).abs();
        assert!(diff <= env,
            "[grad i={i}] AD={ad:+.6e} FD={fd:+.6e} |Δ|={diff:.3e} env={env:.3e}");
    }
}
