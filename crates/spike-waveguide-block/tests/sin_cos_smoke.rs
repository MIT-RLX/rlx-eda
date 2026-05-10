//! Smoke test for the newly-added `Activation::Sin` / `Activation::Cos`
//! ops in rlx-ir. Builds a tiny graph `loss = sin(x)·cos(x)` (i.e.
//! `0.5·sin(2x)`), checks the forward value at a few points, and
//! cross-checks the AD gradient against analytic + finite differences.
//!
//! Lives in this crate because rlx-eda was the consumer that prompted
//! the IR addition. Once a more general "rlx-eda math smoke" test
//! suite exists, this can move there.

use rlx_ir::{
    op::{Activation, BinaryOp},
    DType, Graph, NodeId, Shape as TensorShape,
};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

fn build_sin_cos_loss() -> (Graph, NodeId) {
    let mut g = Graph::new("sin_cos_smoke");
    let s = TensorShape::new(&[1], DType::F32);
    let x = g.param("x", s.clone());
    let sx = g.activation(Activation::Sin, x, s.clone());
    let cx = g.activation(Activation::Cos, x, s.clone());
    let loss = g.binary(BinaryOp::Mul, sx, cx, s);
    g.set_outputs(vec![loss]);
    (g, x)
}

#[test]
fn sin_cos_forward_matches_double_angle_identity() {
    // loss(x) = sin(x)·cos(x) = 0.5·sin(2x)
    let (fwd, x_id) = build_sin_cos_loss();
    let bwd = grad_with_loss(&fwd, &[x_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    for &x in &[0.0_f32, 0.3, 0.7, 1.1, 2.5, -0.4] {
        sess.set_param("x", &[x]);
        let outs = sess.run(&[("d_output", &[1.0_f32])]);
        let got = outs[0][0];
        let expected = 0.5 * (2.0 * x).sin();
        assert!(
            (got - expected).abs() < 1e-5,
            "sin(x)·cos(x) at x={x}: got {got}, expected {expected}",
        );
    }
}

#[test]
fn sin_cos_grad_matches_double_angle_derivative() {
    // d/dx (sin(x)·cos(x)) = cos²(x) − sin²(x) = cos(2x)
    let (fwd, x_id) = build_sin_cos_loss();
    let bwd = grad_with_loss(&fwd, &[x_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    for &x in &[0.0_f32, 0.2, 0.85, 1.4, -0.6] {
        sess.set_param("x", &[x]);
        let outs = sess.run(&[("d_output", &[1.0_f32])]);
        let ad = outs[1][0];
        let analytic = (2.0 * x).cos();
        let rel = (ad - analytic).abs() / analytic.abs().max(1e-6);
        assert!(
            rel < 1e-3,
            "d/dx at x={x}: AD = {ad}, analytic = {analytic}, rel = {rel:.2e}",
        );

        // FD cross-check.
        let h = 1e-3_f32;
        sess.set_param("x", &[x + h]);
        let lp = sess.run(&[("d_output", &[1.0_f32])])[0][0];
        sess.set_param("x", &[x - h]);
        let lm = sess.run(&[("d_output", &[1.0_f32])])[0][0];
        let fd = (lp - lm) / (2.0 * h);
        let rel_fd = (ad - fd).abs() / fd.abs().max(1e-6);
        assert!(
            rel_fd < 1e-2,
            "AD vs FD at x={x}: AD = {ad}, FD = {fd}, rel = {rel_fd:.2e}",
        );
    }
}
