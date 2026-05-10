//! Diode-RC transient — first nonlinear time-stepping spike with end-to-end
//! AD through both the DC initial condition and the time loop.
//!
//! Topology adds a capacitor in parallel with the diode (D ∥ C — gnd):
//!
//! ```text
//!     V(t) ──[R]── Vmid ──┬──[D]── gnd
//!                         └──[C]── gnd
//! ```
//!
//! KCL at `Vmid`:
//!
//! ```text
//!     (V − Vmid)/R   =   Is·(exp(Vmid/Vt) − 1)   +   C · dVmid/dt
//! ```
//!
//! Backward Euler discretisation gives a per-step nonlinear residual
//!
//! ```text
//!     f(Vmid_n; Vmid_{n−1}, V_n, params) =
//!         (V_n − Vmid_n)/R − Is·(exp(Vmid_n/Vt) − 1) − C·(Vmid_n − Vmid_{n−1})/h
//! ```
//!
//! solved per timestep by Newton's method (unrolled inside the scan body).
//! The DC initial condition `Vmid_0` is the steady-state operating point at
//! `V(0)` (i.e. `dVmid/dt = 0`, which collapses back to the diode-DC
//! equation). We get it from `op_ift::build_graph_ift` — that path uses
//! a `custom_vjp` IFT body, so the gradient through `Vmid_0` is O(1)
//! primitives instead of O(n_newton_dc).
//!
//! ## How constants reach the body
//!
//! Today `Op::Scan` body sees only its own `Op::Input` nodes — there is
//! no "broadcast input" channel for values that are constant across all
//! timesteps. We work around that by materialising
//! `(R, Is, C, Vt, h)` into `[n_steps, 1]` per-step xs via a broadcast
//! `Mul` against an `[n_steps, 1]` ones constant. Wasteful (an extra
//! `5 × n_steps × 4` bytes), but it keeps the body purely a function
//! of `(carry, xs_per_step)`. Adding a real broadcast channel to
//! `Op::Scan` is a clean follow-on for rlx.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

use crate::op_ift::{build_fwd_body, build_jvp_body, build_vjp_body, const_scalar, scalar};

fn vec_n(n: usize) -> Shape { Shape::new(&[n, 1], DType::F32) }

/// The per-step Newton body.
///
/// Op::Inputs in NodeId-declaration order match the outer scan's
/// `[init, bcast_0..bcast_{B-1}, xs_0..xs_{X-1}]` layout:
///   carry  : `vmid_prev`
///   bcasts : `R`, `Is`, `C`, `Vt`, `h`  (constant across iterations)
///   xs     : `V_n`                       (per-step source-voltage waveform)
/// Output: `vmid_n`, the converged Vmid at this step.
fn build_step_body(n_newton: usize) -> Graph {
    let mut b = Graph::new("rd_transient_step_body");
    let vmid_prev = b.input("vmid_prev", scalar());
    let r         = b.input("R",         scalar());
    let is_       = b.input("Is",        scalar());
    let c         = b.input("C",         scalar());
    let vt        = b.input("Vt",        scalar());
    let h         = b.input("h",         scalar());
    let v_n       = b.input("V_n",       scalar());

    let one     = const_scalar(&mut b, 1.0);
    let neg_one = const_scalar(&mut b, -1.0);

    // Initial guess: vmid_n ≈ vmid_prev. The continuous solution is
    // smooth on the time scale of `h`, so the prior carry is a much
    // better seed than the V/2 cap used for the cold DC solve.
    let mut vmid = vmid_prev;

    for _ in 0..n_newton {
        // f(Vmid) = (V - Vmid)/R - Is·(exp(Vmid/Vt) - 1) - C·(Vmid - vmid_prev)/h
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

        // f'(Vmid) = -1/R - (Is/Vt)·exp(Vmid/Vt) - C/h
        let inv_r        = b.binary(BinaryOp::Div, one, r, scalar());
        let neg_inv_r    = b.binary(BinaryOp::Mul, neg_one, inv_r, scalar());
        let is_vt        = b.binary(BinaryOp::Div, is_, vt, scalar());
        let dexp         = b.binary(BinaryOp::Mul, is_vt, exp_v, scalar());
        let neg_dexp     = b.binary(BinaryOp::Mul, neg_one, dexp, scalar());
        let c_over_h     = b.binary(BinaryOp::Div, c, h, scalar());
        let neg_c_over_h = b.binary(BinaryOp::Mul, neg_one, c_over_h, scalar());
        let fp_a         = b.binary(BinaryOp::Add, neg_inv_r, neg_dexp, scalar());
        let fp           = b.binary(BinaryOp::Add, fp_a, neg_c_over_h, scalar());

        let dvmid        = b.binary(BinaryOp::Div, f_val, fp, scalar());
        vmid             = b.binary(BinaryOp::Sub, vmid, dvmid, scalar());
    }

    b.set_outputs(vec![vmid]);
    b
}

/// Build the full diode-RC transient graph.
///
/// Returns `(graph, R_id, Is_id, C_id)`.
///
/// Inputs:
///   * `V_dc`        — the V used to compute the DC operating point seed.
///   * `V_per_step`  — `[n_steps, 1]` source-voltage waveform.
///   * `Vt`          — thermal voltage `[1]`.
///   * `h`           — timestep `[1]`.
/// Params (gradient targets):
///   * `R`, `Is`, `C` — `[1]` each.
/// Output: `Vmid_N`, the diode voltage after `n_steps` BE timesteps.
pub fn build_transient_graph(
    n_steps: usize, n_newton_dc: usize, n_newton_step: usize,
) -> (Graph, NodeId, NodeId, NodeId) {
    build_transient_graph_inner(n_steps, n_newton_dc, n_newton_step, 0)
}

/// Same as `build_transient_graph`, but the inner scan uses
/// `scan_checkpointed` with `num_checkpoints` saved carries — backward
/// memory is `O(num_checkpoints)` instead of `O(n_steps)`. Use 0 to
/// fall through to the All strategy (full trajectory cached).
pub fn build_transient_graph_checkpointed(
    n_steps: usize, n_newton_dc: usize, n_newton_step: usize,
    num_checkpoints: u32,
) -> (Graph, NodeId, NodeId, NodeId) {
    build_transient_graph_inner(n_steps, n_newton_dc, n_newton_step, num_checkpoints)
}

fn build_transient_graph_inner(
    n_steps: usize, n_newton_dc: usize, n_newton_step: usize,
    num_checkpoints: u32,
) -> (Graph, NodeId, NodeId, NodeId) {
    assert!(n_steps > 0);
    let mut g = Graph::new("rd_transient");

    let v_dc       = g.input("V_dc",       scalar());
    let v_per_step = g.input("V_per_step", vec_n(n_steps));
    let vt         = g.input("Vt",         scalar());
    let h          = g.input("h",          scalar());
    let r          = g.param("R",          scalar());
    let is_        = g.param("Is",         scalar());
    let c          = g.param("C",          scalar());

    // DC initial condition via custom_vjp.
    let vmid_0 = g.custom_fn(
        vec![v_dc, vt, r, is_],
        build_fwd_body(n_newton_dc),
        Some(build_vjp_body()),
        Some(build_jvp_body()),
    );

    // R, Is, C, Vt, h are constant across timesteps — pass them via
    // Op::Scan's bcast channel so they're filled into the body once,
    // not replicated across all `length` rows. V_per_step is a real
    // per-step xs.
    let body = build_step_body(n_newton_step);
    let bcasts = vec![r, is_, c, vt, h];
    let xs = vec![v_per_step];
    let vmid_n = if num_checkpoints == 0 || num_checkpoints == n_steps as u32 {
        g.scan_with_bcasts_and_xs(vmid_0, &bcasts, &xs, body, n_steps as u32)
    } else {
        let mut inputs = vec![vmid_0];
        inputs.extend_from_slice(&bcasts);
        inputs.extend_from_slice(&xs);
        g.add_node(
            Op::Scan {
                body: Box::new(body),
                length: n_steps as u32,
                save_trajectory: false,
                num_bcast: bcasts.len() as u32,
                num_xs: xs.len() as u32,
                num_checkpoints,
            },
            inputs, scalar(),
        )
    };

    g.set_outputs(vec![vmid_n]);
    (g, r, is_, c)
}

/// Forward + reverse-mode AD with recursive checkpointing.
/// Backward memory is `O(num_checkpoints · carry_bytes)`; without
/// checkpointing it's `O(n_steps · carry_bytes)`.
pub fn run_transient_and_grad_checkpointed(
    v_dc: f32, v_per_step: &[f32], vt: f32, h: f32,
    r: f32, is_: f32, c: f32,
    n_newton_dc: usize, n_newton_step: usize, num_checkpoints: u32,
) -> (f32, f32, f32, f32) {
    let n = v_per_step.len();
    let (fwd, r_id, is_id, c_id) = build_transient_graph_checkpointed(
        n, n_newton_dc, n_newton_step, num_checkpoints);
    let bwd = grad_with_loss(&fwd, &[r_id, is_id, c_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param("R",  &[r]);
    compiled.set_param("Is", &[is_]);
    compiled.set_param("C",  &[c]);
    let outs = compiled.run(&[
        ("V_dc",       &[v_dc][..]),
        ("V_per_step", v_per_step),
        ("Vt",         &[vt][..]),
        ("h",          &[h][..]),
        ("d_output",   &[1.0_f32][..]),
    ]);
    (outs[0][0], outs[1][0], outs[2][0], outs[3][0])
}

/// Forward only.
pub fn run_transient_forward(
    v_dc: f32, v_per_step: &[f32], vt: f32, h: f32,
    r: f32, is_: f32, c: f32,
    n_newton_dc: usize, n_newton_step: usize,
) -> f32 {
    let n = v_per_step.len();
    let (g, _r, _is, _c) = build_transient_graph(n, n_newton_dc, n_newton_step);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param("R",  &[r]);
    compiled.set_param("Is", &[is_]);
    compiled.set_param("C",  &[c]);
    let outs = compiled.run(&[
        ("V_dc",       &[v_dc][..]),
        ("V_per_step", v_per_step),
        ("Vt",         &[vt][..]),
        ("h",          &[h][..]),
    ]);
    outs[0][0]
}

/// Forward + reverse-mode AD: returns
/// `(Vmid_N, ∂Vmid_N/∂R, ∂Vmid_N/∂Is, ∂Vmid_N/∂C)`.
pub fn run_transient_and_grad(
    v_dc: f32, v_per_step: &[f32], vt: f32, h: f32,
    r: f32, is_: f32, c: f32,
    n_newton_dc: usize, n_newton_step: usize,
) -> (f32, f32, f32, f32) {
    let n = v_per_step.len();
    let (fwd, r_id, is_id, c_id) = build_transient_graph(n, n_newton_dc, n_newton_step);
    let bwd = grad_with_loss(&fwd, &[r_id, is_id, c_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param("R",  &[r]);
    compiled.set_param("Is", &[is_]);
    compiled.set_param("C",  &[c]);
    let outs = compiled.run(&[
        ("V_dc",       &[v_dc][..]),
        ("V_per_step", v_per_step),
        ("Vt",         &[vt][..]),
        ("h",          &[h][..]),
        ("d_output",   &[1.0_f32][..]),
    ]);
    (outs[0][0], outs[1][0], outs[2][0], outs[3][0])
}

// ── Pure-Rust reference ─────────────────────────────────────────────────

/// Pure-Rust diode-DC steady state — same Newton recurrence as the rlx
/// IFT body, used as the time-zero seed for the Rust transient reference.
pub fn ref_dc_op(v: f32, r: f32, is_: f32, vt: f32, n_newton: usize) -> f32 {
    let mut vmid = (v / 2.0).min(0.6);
    for _ in 0..n_newton {
        let exp_v = (vmid / vt).exp();
        let f  = (v - vmid) / r - is_ * (exp_v - 1.0);
        let fp = -1.0 / r - (is_ / vt) * exp_v;
        vmid -= f / fp;
    }
    vmid
}

/// ngspice deck for the diode-RC transient (D ∥ C — gnd, R from V).
///
/// The deck pins ngspice to BDF1 (`method=gear maxord=1`) so its
/// discretisation matches our Backward Euler outer loop. Diode model
/// uses N=1 and matches our Vt by overriding ngspice's room-temperature
/// default via `tnom` — rlx uses `Vt = 25.852 mV` exactly, which
/// corresponds to T ≈ 300 K. The default ngspice T=300.15 K is close
/// enough that f32-tolerance comparisons absorb the difference.
///
/// No `.ic` / no `uic` — ngspice's pre-tran DC analysis converges to the
/// same operating point our `ref_dc_op` computes, so both simulators
/// start in the same state.
#[cfg(feature = "ngspice")]
pub fn spice_deck(v_dc: f64, r: f64, c: f64, is_: f64) -> String {
    format!(
        "* Diode-RC transient (rlx-eda spike-diode)\n\
         .options method=gear maxord=1\n\
         .model dmod D(IS={is_:e} N=1)\n\
         V1 vin 0 {v_dc}\n\
         R1 vin vmid {r}\n\
         D1 vmid 0 dmod\n\
         C1 vmid 0 {c}\n",
    )
}

/// Pure-Rust BE+Newton transient. Same math the rlx graph implements.
pub fn ref_transient(
    v_dc: f32, v_per_step: &[f32], vt: f32, h: f32,
    r: f32, is_: f32, c: f32,
    n_newton_dc: usize, n_newton_step: usize,
) -> f32 {
    let mut vmid = ref_dc_op(v_dc, r, is_, vt, n_newton_dc);
    for &v_n in v_per_step {
        // Initial guess: previous carry (matches the rlx body).
        let mut x = vmid;
        for _ in 0..n_newton_step {
            let exp_v = (x / vt).exp();
            let f  = (v_n - x) / r - is_ * (exp_v - 1.0) - c * (x - vmid) / h;
            let fp = -1.0 / r - (is_ / vt) * exp_v - c / h;
            x -= f / fp;
        }
        vmid = x;
    }
    vmid
}
