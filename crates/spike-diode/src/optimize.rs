//! Gradient-descent fit of diode-RC parameters to a target waveform.
//!
//! The first end-to-end check that the rlx + custom_vjp + scan + AD
//! pipeline is **useful** for circuit design, not just numerically
//! correct: take a target Vmid(t) trajectory, run Adam on `(R, Is, C)`
//! to minimise `mean((Vmid_rlx(t) - target(t))²)`, and recover the
//! parameters that generated the target.
//!
//! ## Parameterisation
//!
//! `R, Is, C` span many decades (kΩ, fA, nF). Linear-space optimisation
//! is brittle — Adam's per-parameter scale assumption breaks across
//! 10⁹ ranges. We optimise `log R`, `log Is`, `log C` instead and
//! materialise the physics values via `exp(log_*)` inside the graph.
//! Adam in log-space takes ~5% relative steps regardless of the
//! parameter's absolute magnitude.
//!
//! ## What this validates
//!
//! 1. The gradients produced by `grad_with_loss` actually drive useful
//!    optimisation — sign + magnitude are both correct, and f32
//!    precision is sufficient for the Adam update direction.
//! 2. The full stack (custom_vjp DC IC + bcast scan + Newton-in-body
//!    + AD-through-trajectory + outer MSE) composes correctly under
//!    repeated forward+backward passes.
//! 3. Adam converges from a perturbed initialisation to the true
//!    parameters within a few hundred iterations.

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

use crate::op_ift::{build_fwd_body, build_jvp_body, build_vjp_body, const_scalar, scalar};
use crate::transient::ref_transient;

fn vec_n(n: usize) -> Shape { Shape::new(&[n, 1], DType::F32) }

fn pair_shape() -> Shape { Shape::new(&[2], DType::F32) }

/// Per-step body that accumulates squared-error loss into the carry.
///
/// Carry shape `[2]`: `carry[0] = vmid`, `carry[1] = loss_acc`. Body
/// extracts both via `Narrow`, runs Newton to advance vmid, computes
/// `(vmid_new - target_n)²`, adds it to loss_acc, then `Concat`s the
/// new pair as the next carry.
///
/// Op::Inputs in NodeId order: `[carry, R, Is, C, Vt, h, V_n, target_n]`.
/// Output: shape `[2]` next carry.
fn build_step_body(n_newton: usize) -> Graph {
    let mut b = Graph::new("opt_step_body");
    let carry_in = b.input("carry",   pair_shape());
    let r        = b.input("R",       scalar());
    let is_      = b.input("Is",      scalar());
    let c        = b.input("C",       scalar());
    let vt       = b.input("Vt",      scalar());
    let h        = b.input("h",       scalar());
    let v_n      = b.input("V_n",     scalar());
    let target_n = b.input("target_n", scalar());

    // Extract vmid and loss_acc from carry.
    let vmid_prev = b.add_node(
        Op::Narrow { axis: 0, start: 0, len: 1 },
        vec![carry_in], scalar());
    let loss_acc_prev = b.add_node(
        Op::Narrow { axis: 0, start: 1, len: 1 },
        vec![carry_in], scalar());

    let one     = const_scalar(&mut b, 1.0);
    let neg_one = const_scalar(&mut b, -1.0);

    let mut vmid = vmid_prev;
    for _ in 0..n_newton {
        let v_minus_vmid = b.binary(BinaryOp::Sub, v_n, vmid, scalar());
        let i_r          = b.binary(BinaryOp::Div, v_minus_vmid, r, scalar());
        let vmid_vt      = b.binary(BinaryOp::Div, vmid, vt, scalar());
        let exp_v        = b.activation(Activation::Exp, vmid_vt, scalar());
        let exp_m1       = b.binary(BinaryOp::Sub, exp_v, one, scalar());
        let i_d          = b.binary(BinaryOp::Mul, is_, exp_m1, scalar());
        let dvmid_step   = b.binary(BinaryOp::Sub, vmid, vmid_prev, scalar());
        let c_dvmid      = b.binary(BinaryOp::Mul, c, dvmid_step, scalar());
        let i_c          = b.binary(BinaryOp::Div, c_dvmid, h, scalar());
        let f_a          = b.binary(BinaryOp::Sub, i_r, i_d, scalar());
        let f_val        = b.binary(BinaryOp::Sub, f_a, i_c, scalar());

        let inv_r        = b.binary(BinaryOp::Div, one, r, scalar());
        let neg_inv_r    = b.binary(BinaryOp::Mul, neg_one, inv_r, scalar());
        let is_vt        = b.binary(BinaryOp::Div, is_, vt, scalar());
        let dexp         = b.binary(BinaryOp::Mul, is_vt, exp_v, scalar());
        let neg_dexp     = b.binary(BinaryOp::Mul, neg_one, dexp, scalar());
        let c_over_h     = b.binary(BinaryOp::Div, c, h, scalar());
        let neg_c_over_h = b.binary(BinaryOp::Mul, neg_one, c_over_h, scalar());
        let fp_a         = b.binary(BinaryOp::Add, neg_inv_r, neg_dexp, scalar());
        let fp           = b.binary(BinaryOp::Add, fp_a, neg_c_over_h, scalar());

        let dvm_step     = b.binary(BinaryOp::Div, f_val, fp, scalar());
        vmid             = b.binary(BinaryOp::Sub, vmid, dvm_step, scalar());
    }

    // Update loss accumulator: loss_acc_new = loss_acc_prev + (vmid - target_n)².
    let resid    = b.binary(BinaryOp::Sub, vmid, target_n, scalar());
    let resid_sq = b.binary(BinaryOp::Mul, resid, resid, scalar());
    let new_loss = b.binary(BinaryOp::Add, loss_acc_prev, resid_sq, scalar());

    // Pack [new_vmid, new_loss] as the next carry.
    let next_carry = b.add_node(
        Op::Concat { axis: 0 },
        vec![vmid, new_loss], pair_shape());
    b.set_outputs(vec![next_carry]);
    b
}

/// Build the optimisation graph: scalar MSE loss between simulated
/// Vmid(t) trajectory and a target trajectory. Inputs supply the
/// time-varying drive + target + temperature/timestep; params are the
/// log-space `(R, Is, C)` Adam optimises.
///
/// Returns `(graph, log_R, log_Is, log_C)`.
pub fn build_optimization_graph(
    n_steps: usize, n_newton_dc: usize, n_newton_step: usize,
) -> (Graph, NodeId, NodeId, NodeId) {
    assert!(n_steps > 0);
    let mut g = Graph::new("rd_optimize");

    let v_dc       = g.input("V_dc",       scalar());
    let v_per_step = g.input("V_per_step", vec_n(n_steps));
    let target     = g.input("target",     vec_n(n_steps));
    let vt         = g.input("Vt",         scalar());
    let h          = g.input("h",          scalar());

    // log-space params; physics values via exp().
    let log_r  = g.param("log_R",  scalar());
    let log_is = g.param("log_Is", scalar());
    let log_c  = g.param("log_C",  scalar());
    let r   = g.activation(Activation::Exp, log_r,  scalar());
    let is_ = g.activation(Activation::Exp, log_is, scalar());
    let c   = g.activation(Activation::Exp, log_c,  scalar());

    // DC initial condition via custom_vjp (IFT gradient).
    let vmid_0 = g.custom_fn(
        vec![v_dc, vt, r, is_],
        build_fwd_body(n_newton_dc),
        Some(build_vjp_body()),
        Some(build_jvp_body()),
    );

    // Pack init carry = [vmid_0, 0.0] (loss accumulator starts at 0).
    let zero = const_scalar(&mut g, 0.0);
    let init_carry = g.add_node(
        Op::Concat { axis: 0 },
        vec![vmid_0, zero],
        pair_shape());

    // Forward simulation: scan_with_bcasts_and_xs returns the final
    // carry [vmid_N, total_loss]. No save_trajectory needed — the
    // scan body accumulates the squared-error loss directly into
    // carry[1] each iteration.
    let body = build_step_body(n_newton_step);
    let final_carry = g.scan_with_bcasts_and_xs(
        init_carry,
        &[r, is_, c, vt, h],         // bcasts (5 scalars)
        &[v_per_step, target],       // xs ([n_steps, 1] each)
        body,
        n_steps as u32,
    );

    // Extract loss_acc from final_carry, divide by n_steps for MSE.
    let total_loss = g.add_node(
        Op::Narrow { axis: 0, start: 1, len: 1 },
        vec![final_carry], scalar());
    let inv_n   = const_scalar(&mut g, 1.0 / (n_steps as f32));
    let loss    = g.binary(BinaryOp::Mul, total_loss, inv_n, scalar());

    g.set_outputs(vec![loss]);
    (g, log_r, log_is, log_c)
}

/// Run forward + reverse-mode AD once. Returns `(loss, d_logR, d_logIs, d_logC)`.
pub fn run_loss_and_grad(
    v_dc: f32, v_per_step: &[f32], target_per_step: &[f32], vt: f32, h: f32,
    log_r: f32, log_is: f32, log_c: f32,
    n_newton_dc: usize, n_newton_step: usize,
) -> (f32, f32, f32, f32) {
    assert_eq!(v_per_step.len(), target_per_step.len());
    let n = v_per_step.len();
    let (fwd, log_r_id, log_is_id, log_c_id) =
        build_optimization_graph(n, n_newton_dc, n_newton_step);
    let bwd = grad_with_loss(&fwd, &[log_r_id, log_is_id, log_c_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param("log_R",  &[log_r]);
    compiled.set_param("log_Is", &[log_is]);
    compiled.set_param("log_C",  &[log_c]);
    let outs = compiled.run(&[
        ("V_dc",       &[v_dc][..]),
        ("V_per_step", v_per_step),
        ("target",     target_per_step),
        ("Vt",         &[vt][..]),
        ("h",          &[h][..]),
        ("d_output",   &[1.0_f32][..]),
    ]);
    (outs[0][0], outs[1][0], outs[2][0], outs[3][0])
}

/// Adam state for a single scalar parameter. `m` is the first-moment
/// estimate (mean gradient), `v` the second-moment (mean squared
/// gradient). Both decay toward zero per Adam's bias-correction.
#[derive(Default, Clone, Copy, Debug)]
pub struct AdamState {
    pub value: f32,
    pub m:     f32,
    pub v:     f32,
}

impl AdamState {
    pub fn new(initial: f32) -> Self { Self { value: initial, m: 0.0, v: 0.0 } }

    /// One Adam update step. `t` is 1-based iteration index for the
    /// bias correction.
    pub fn step(&mut self, grad: f32, t: u32, lr: f32) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPS:   f32 = 1e-8;
        self.m = BETA1 * self.m + (1.0 - BETA1) * grad;
        self.v = BETA2 * self.v + (1.0 - BETA2) * grad * grad;
        let bias1 = 1.0 - BETA1.powi(t as i32);
        let bias2 = 1.0 - BETA2.powi(t as i32);
        let m_hat = self.m / bias1;
        let v_hat = self.v / bias2;
        self.value -= lr * m_hat / (v_hat.sqrt() + EPS);
    }
}

/// Run Adam in log-space on `(log_R, log_Is, log_C)` for `n_iters`
/// iterations. Returns `(R*, Is*, C*, loss_history)`.
///
/// `target` is the desired Vmid(t) waveform, sampled at the same
/// timestep `h` for `n_steps` steps starting from `V_dc` (which is
/// also held fixed across optimisation).
#[allow(clippy::too_many_arguments)]
pub fn optimize_diode_rc(
    v_dc: f32, v_per_step: &[f32], target_per_step: &[f32],
    vt: f32, h: f32,
    init_r: f32, init_is: f32, init_c: f32,
    n_iters: usize, lr: f32,
    n_newton_dc: usize, n_newton_step: usize,
) -> (f32, f32, f32, Vec<f32>) {
    let mut log_r  = AdamState::new(init_r .ln());
    let mut log_is = AdamState::new(init_is.ln());
    let mut log_c  = AdamState::new(init_c .ln());

    let mut history = Vec::with_capacity(n_iters);
    for t in 1..=n_iters {
        let (loss, d_log_r, d_log_is, d_log_c) = run_loss_and_grad(
            v_dc, v_per_step, target_per_step, vt, h,
            log_r.value, log_is.value, log_c.value,
            n_newton_dc, n_newton_step,
        );
        history.push(loss);
        log_r .step(d_log_r,  t as u32, lr);
        log_is.step(d_log_is, t as u32, lr);
        log_c .step(d_log_c,  t as u32, lr);
    }

    (log_r.value.exp(), log_is.value.exp(), log_c.value.exp(), history)
}

/// Generate a target waveform by running the pure-Rust transient at
/// the supplied `(R, Is, C)`. Mirrors what an external "ground truth"
/// data source would provide in a real fit — but lets us test
/// optimisation against a known answer.
pub fn synthesize_target(
    v_dc: f32, n_steps: usize, h: f32,
    r: f32, is_: f32, c: f32,
    vt: f32, n_newton_dc: usize, n_newton_step: usize,
) -> Vec<f32> {
    let v_per_step: Vec<f32> = vec![v_dc; n_steps];
    // Reuse the existing pure-Rust reference, but capture the FULL
    // trajectory by re-running it for incrementing prefixes.
    // (`ref_transient` only returns the final Vmid; for a target we
    // need every step's value.)
    let mut traj = Vec::with_capacity(n_steps);
    for k in 1..=n_steps {
        let v = ref_transient(
            v_dc, &v_per_step[..k], vt, h,
            r, is_, c, n_newton_dc, n_newton_step,
        );
        traj.push(v);
    }
    traj
}
