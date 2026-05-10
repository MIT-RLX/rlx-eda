//! AC analysis of a **nonlinear** circuit: the silicon-diode-RC topology
//! from `spike-diode`, linearised at its DC operating point.
//!
//! This is the canonical Circulax workflow:
//!
//! 1. Solve the nonlinear DC operating point (Newton on the diode I-V).
//!    In a future step this graph would call out to the
//!    `custom_vjp` IFT body from `spike-diode::op_ift` so the OP is
//!    differentiable; for the first cut we precompute `Vmid*` and the
//!    small-signal conductance `g_d` in Rust and feed them as Inputs.
//! 2. At `Vmid*`, compute the small-signal conductance:
//!
//!    ```text
//!        g_d = ∂Id/∂Vmid|_{Vmid*} = (Is/Vt) · exp(Vmid*/Vt)
//!    ```
//!
//! 3. Build the small-signal MNA (3 nodes: vin, vout, i_V1) using the
//!    same 2N×2N real-block encoding as the linear-RC `build_ac_graph`.
//!    The diode contributes `g_d` in parallel with the cap; the rest
//!    of the circuit (R from V1 to vmid, V1 voltage source) is linear.
//! 4. Solve at each frequency, return `(re, im)` for vmid.
//!
//! `H(jω) = Vmid_ss / V_in_ss` is a 1-pole low-pass with effective
//! resistance `R_eff = R ∥ (1/g_d)` and capacitance `C`, giving
//! `f₃dB = 1 / (2π · R_eff · C)`.
//!
//! ## What this validates
//!
//! - The OP-linearise-then-AC pattern works end-to-end through rlx
//!   (DC OP via Newton → small-signal conductance → AC MNA solve).
//! - The 2N×2N real-block encoding handles a parallel admittance
//!   `g_d + jωC` (the cap is purely imaginary; g_d is purely real).
//! - rlx forward + AD agree with closed-form 1-pole rolloff and with
//!   ngspice's `.ac` analysis on the same diode-R-C deck.

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

const N: usize = 3;          // MNA rank: vin, vmid, i_V1
const NN: usize = 2 * N;     // real-block size
fn scalar() -> Shape { Shape::new(&[1], DType::F64) }
fn vec_nn() -> Shape { Shape::new(&[NN], DType::F64) }
fn mat_nn() -> Shape { Shape::new(&[NN, NN], DType::F64) }

fn const_scalar(g: &mut Graph, x: f64) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar())
}

fn const_vec_nn(g: &mut Graph, x: &[f64; NN]) -> NodeId {
    let mut bytes = Vec::with_capacity(NN * 8);
    for v in x { bytes.extend_from_slice(&v.to_le_bytes()); }
    g.add_node(Op::Constant { data: bytes }, vec![], vec_nn())
}

fn const_mat_nn(g: &mut Graph, x: &[f64; NN * NN]) -> NodeId {
    let mut bytes = Vec::with_capacity(NN * NN * 8);
    for v in x { bytes.extend_from_slice(&v.to_le_bytes()); }
    g.add_node(Op::Constant { data: bytes }, vec![], mat_nn())
}

/// Pure-Rust diode DC operating point — Newton on the same KCL the
/// `spike-diode` crate uses, but re-implemented at f64 to mesh with
/// the f64 AC graph.
pub fn dc_op_f64(v_dc: f64, r: f64, is_: f64, vt: f64, n_newton: usize) -> f64 {
    let mut vmid = (v_dc / 2.0).min(0.6);
    for _ in 0..n_newton {
        let exp_v = (vmid / vt).exp();
        let f  = (v_dc - vmid) / r - is_ * (exp_v - 1.0);
        let fp = -1.0 / r - (is_ / vt) * exp_v;
        vmid -= f / fp;
    }
    vmid
}

/// Small-signal diode conductance at `Vmid*`:
/// `g_d = (Is/Vt) · exp(Vmid*/Vt)`.
pub fn small_signal_conductance(vmid_star: f64, is_: f64, vt: f64) -> f64 {
    (is_ / vt) * (vmid_star / vt).exp()
}

/// Build the AC forward graph for the diode-RC circuit linearised at
/// a precomputed operating point. Returns `(graph, R_id, C_id)`.
///
/// Inputs (set per-call):
///   * `omega` — angular frequency in rad/s.
///   * `g_d`   — small-signal diode conductance (precompute via
///     `dc_op_f64` + `small_signal_conductance`).
///
/// Params: `R`, `C`. Outputs: `vmid_re`, `vmid_im`.
pub fn build_diode_rc_ac_graph() -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new("diode_rc_ac");

    let omega = g.input("omega", scalar());
    let g_d   = g.input("g_d",   scalar());
    let r     = g.param("R",     scalar());
    let c     = g.param("C",     scalar());

    let one    = const_scalar(&mut g, 1.0);
    let g_cond = g.binary(BinaryOp::Div, one, r, scalar());      // G = 1/R
    let bc     = g.binary(BinaryOp::Mul, omega, c, scalar());    // ωC

    // Pattern matrices in the 6×6 real-block layout.
    //
    // Aᵣ-side stamps:
    //   - Resistor at (vin, vmid): G on vin↔vin, vmid↔vmid; -G off-diag.
    //   - Voltage source augmentation: ones at (vin, i_V1) and
    //     (i_V1, vin).
    //   - Diode small-signal conductance: g_d at (vmid, vmid). This is
    //     the linearisation contribution from the nonlinear element.
    //
    // Aᵢ-side stamps:
    //   - Cap reactance ωC at (vmid, vmid).
    let idx = |r: usize, c: usize| r * NN + c;
    let mut p_g    = [0.0_f64; NN * NN];
    let mut p_vsrc = [0.0_f64; NN * NN];
    let mut p_gd   = [0.0_f64; NN * NN];
    let mut p_c    = [0.0_f64; NN * NN];

    for &(rr, cc, val) in &[
        (0, 0,  1.0_f64), (0, 1, -1.0),
        (1, 0, -1.0),     (1, 1,  1.0),
    ] {
        p_g[idx(rr, cc)] += val;
        p_g[idx(N + rr, N + cc)] += val;
    }
    p_vsrc[idx(0, 2)] = 1.0;
    p_vsrc[idx(2, 0)] = 1.0;
    p_vsrc[idx(N + 0, N + 2)] = 1.0;
    p_vsrc[idx(N + 2, N + 0)] = 1.0;
    p_gd[idx(1, 1)] += 1.0;
    p_gd[idx(N + 1, N + 1)] += 1.0;
    p_c[idx(N + 1, 1)] += 1.0;
    p_c[idx(1, N + 1)] += -1.0;

    let p_g_node    = const_mat_nn(&mut g, &p_g);
    let p_vsrc_node = const_mat_nn(&mut g, &p_vsrc);
    let p_gd_node   = const_mat_nn(&mut g, &p_gd);
    let p_c_node    = const_mat_nn(&mut g, &p_c);

    let g_term  = g.binary(BinaryOp::Mul, p_g_node,  g_cond, mat_nn());
    let gd_term = g.binary(BinaryOp::Mul, p_gd_node, g_d,    mat_nn());
    let c_term  = g.binary(BinaryOp::Mul, p_c_node,  bc,     mat_nn());
    let a_a = g.binary(BinaryOp::Add, p_vsrc_node, g_term,  mat_nn());
    let a_b = g.binary(BinaryOp::Add, a_a,         gd_term, mat_nn());
    let a_mat = g.binary(BinaryOp::Add, a_b,       c_term,  mat_nn());

    // Source: AC magnitude 1 V at vin (real); imag part zero.
    let mut b = [0.0_f64; NN];
    b[2] = 1.0;
    let b_vec = const_vec_nn(&mut g, &b);

    let x = g.dense_solve(a_mat, b_vec, vec_nn());

    let mut e_vmid_re = [0.0_f64; NN];
    e_vmid_re[1] = 1.0;
    let mut e_vmid_im = [0.0_f64; NN];
    e_vmid_im[N + 1] = 1.0;
    let e_re = const_vec_nn(&mut g, &e_vmid_re);
    let e_im = const_vec_nn(&mut g, &e_vmid_im);

    let masked_re = g.binary(BinaryOp::Mul, x, e_re, vec_nn());
    let vmid_re = g.reduce(masked_re, ReduceOp::Sum, vec![0], true, scalar());
    let masked_im = g.binary(BinaryOp::Mul, x, e_im, vec_nn());
    let vmid_im = g.reduce(masked_im, ReduceOp::Sum, vec![0], true, scalar());

    g.set_outputs(vec![vmid_re, vmid_im]);
    (g, r, c)
}

/// One-frequency forward at the precomputed OP: returns `(vmid_re, vmid_im)`.
pub fn run_diode_rc_ac_point(
    omega: f64, r: f64, c: f64, g_d: f64,
) -> (f64, f64) {
    let (graph, _r, _c) = build_diode_rc_ac_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);
    let outs = compiled.run_typed(&[
        ("omega", &omega.to_le_bytes(), DType::F64),
        ("g_d",   &g_d.to_le_bytes(),   DType::F64),
    ]);
    (decode_f64(&outs[0].0), decode_f64(&outs[1].0))
}

/// Sweep `pts_per_decade · log10(f_stop/f_start)` log-spaced points.
/// Returns `(freq_hz, vmid_re, vmid_im)`. Compiles once, runs per
/// frequency.
pub fn run_diode_rc_ac_sweep(
    f_start: f64, f_stop: f64, pts_per_decade: usize,
    r: f64, c: f64, is_: f64, vt: f64, v_dc: f64, n_newton_dc: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let vmid_star = dc_op_f64(v_dc, r, is_, vt, n_newton_dc);
    let g_d = small_signal_conductance(vmid_star, is_, vt);

    let log0 = f_start.log10();
    let log1 = f_stop.log10();
    let n = ((log1 - log0) * pts_per_decade as f64).round() as usize + 1;

    let (graph, _r, _c) = build_diode_rc_ac_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);

    let mut freq = Vec::with_capacity(n);
    let mut re   = Vec::with_capacity(n);
    let mut im   = Vec::with_capacity(n);
    for i in 0..n {
        let f = 10f64.powf(log0 + (log1 - log0) * (i as f64 / (n.saturating_sub(1)) as f64));
        let omega = 2.0 * std::f64::consts::PI * f;
        let outs = compiled.run_typed(&[
            ("omega", &omega.to_le_bytes(), DType::F64),
            ("g_d",   &g_d.to_le_bytes(),   DType::F64),
        ]);
        freq.push(f);
        re.push(decode_f64(&outs[0].0));
        im.push(decode_f64(&outs[1].0));
    }
    (freq, re, im)
}

/// Reverse-mode AD on `|H|²` w.r.t. `(R, C)` at one frequency.
/// Mirrors `run_ac_grad` in the linear path.
pub fn run_diode_rc_ac_grad(
    omega: f64, r: f64, c: f64, g_d: f64,
) -> (f64, f64, f64, f64) {
    let mut g = Graph::new("diode_rc_ac_loss");
    let omega_in = g.input("omega", scalar());
    let g_d_in   = g.input("g_d",   scalar());
    let r_id     = g.param("R",     scalar());
    let c_id     = g.param("C",     scalar());

    let one    = const_scalar(&mut g, 1.0);
    let g_cond = g.binary(BinaryOp::Div, one, r_id, scalar());
    let bc     = g.binary(BinaryOp::Mul, omega_in, c_id, scalar());

    let idx = |r: usize, c: usize| r * NN + c;
    let mut p_g    = [0.0_f64; NN * NN];
    let mut p_vsrc = [0.0_f64; NN * NN];
    let mut p_gd   = [0.0_f64; NN * NN];
    let mut p_c    = [0.0_f64; NN * NN];
    for &(rr, cc, val) in &[(0,0,1.0_f64),(0,1,-1.0),(1,0,-1.0),(1,1,1.0)] {
        p_g[idx(rr,cc)] += val; p_g[idx(N+rr,N+cc)] += val;
    }
    p_vsrc[idx(0,2)] = 1.0; p_vsrc[idx(2,0)] = 1.0;
    p_vsrc[idx(N+0,N+2)] = 1.0; p_vsrc[idx(N+2,N+0)] = 1.0;
    p_gd[idx(1,1)] += 1.0; p_gd[idx(N+1,N+1)] += 1.0;
    p_c[idx(N+1,1)] += 1.0; p_c[idx(1,N+1)] += -1.0;

    let p_g_node    = const_mat_nn(&mut g, &p_g);
    let p_vsrc_node = const_mat_nn(&mut g, &p_vsrc);
    let p_gd_node   = const_mat_nn(&mut g, &p_gd);
    let p_c_node    = const_mat_nn(&mut g, &p_c);

    let g_term  = g.binary(BinaryOp::Mul, p_g_node,  g_cond, mat_nn());
    let gd_term = g.binary(BinaryOp::Mul, p_gd_node, g_d_in, mat_nn());
    let c_term  = g.binary(BinaryOp::Mul, p_c_node,  bc,     mat_nn());
    let a_a = g.binary(BinaryOp::Add, p_vsrc_node, g_term,  mat_nn());
    let a_b = g.binary(BinaryOp::Add, a_a,         gd_term, mat_nn());
    let a_mat = g.binary(BinaryOp::Add, a_b,       c_term,  mat_nn());

    let mut b = [0.0_f64; NN]; b[2] = 1.0;
    let b_vec = const_vec_nn(&mut g, &b);
    let x = g.dense_solve(a_mat, b_vec, vec_nn());

    let mut e_vmid_re = [0.0_f64; NN]; e_vmid_re[1] = 1.0;
    let mut e_vmid_im = [0.0_f64; NN]; e_vmid_im[N + 1] = 1.0;
    let e_re = const_vec_nn(&mut g, &e_vmid_re);
    let e_im = const_vec_nn(&mut g, &e_vmid_im);

    let masked_re = g.binary(BinaryOp::Mul, x, e_re, vec_nn());
    let vmid_re   = g.reduce(masked_re, ReduceOp::Sum, vec![0], true, scalar());
    let masked_im = g.binary(BinaryOp::Mul, x, e_im, vec_nn());
    let vmid_im   = g.reduce(masked_im, ReduceOp::Sum, vec![0], true, scalar());

    let re_sq  = g.binary(BinaryOp::Mul, vmid_re, vmid_re, scalar());
    let im_sq  = g.binary(BinaryOp::Mul, vmid_im, vmid_im, scalar());
    let mag_sq = g.binary(BinaryOp::Add, re_sq, im_sq, scalar());
    g.set_outputs(vec![mag_sq]);

    let bwd = grad_with_loss(&g, &[r_id, c_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);
    let one_b = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("omega",    &omega.to_le_bytes(), DType::F64),
        ("g_d",      &g_d.to_le_bytes(),   DType::F64),
        ("d_output", &one_b,               DType::F64),
    ]);
    let _mag_sq = decode_f64(&outs[0].0);
    let d_dr = decode_f64(&outs[1].0);
    let d_dc = decode_f64(&outs[2].0);
    let (vmid_re, vmid_im) = run_diode_rc_ac_point(omega, r, c, g_d);
    (vmid_re, vmid_im, d_dr, d_dc)
}

// ── Analytic 1-pole references ─────────────────────────────────────────

/// `H(jω) = G / (G + g_d + jωC)` where `G = 1/R`. Closed form for the
/// linearised diode-RC at a precomputed `g_d`.
pub fn analytic_h(omega: f64, r: f64, c: f64, g_d: f64) -> (f64, f64) {
    let g_cond = 1.0 / r;
    let denom_re = g_cond + g_d;
    let denom_im = omega * c;
    let denom_mag_sq = denom_re * denom_re + denom_im * denom_im;
    let h_re =  g_cond * denom_re / denom_mag_sq;
    let h_im = -g_cond * denom_im / denom_mag_sq;
    (h_re, h_im)
}

/// `|H| = G / |G + g_d + jωC|` where `G = 1/R`.
pub fn analytic_mag(omega: f64, r: f64, c: f64, g_d: f64) -> f64 {
    let g_cond = 1.0 / r;
    let denom_re = g_cond + g_d;
    let denom_im = omega * c;
    g_cond / (denom_re * denom_re + denom_im * denom_im).sqrt()
}

/// 3-dB cut-off frequency `f₃dB = 1/(2π · R_eff · C)` where
/// `R_eff = R ∥ (1/g_d)`.
pub fn analytic_f3db(r: f64, c: f64, g_d: f64) -> f64 {
    let r_eff = 1.0 / (1.0 / r + g_d);
    1.0 / (2.0 * std::f64::consts::PI * r_eff * c)
}

// ── ngspice deck ───────────────────────────────────────────────────────

/// `.ac` deck for the diode-R-C topology at the supplied `(R, C, Is)`.
/// ngspice runs its own pre-DC analysis to find the OP, then linearises
/// the diode internally — same pattern our rlx side implements
/// explicitly via `dc_op_f64` + `small_signal_conductance`.
pub fn spice_deck(v_dc: f64, r: f64, c: f64, is_: f64) -> String {
    format!(
        "* Diode-RC AC sweep (rlx-eda spike-ac diode_rc)\n\
         .model dmod D(IS={is_:e} N=1)\n\
         V1 vin 0 DC {v_dc} AC 1\n\
         R1 vin vmid {r}\n\
         D1 vmid 0 dmod\n\
         C1 vmid 0 {c:.10e}\n",
    )
}

fn decode_f64(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    f64::from_le_bytes(buf)
}
