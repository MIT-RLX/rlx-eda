//! AC small-signal analysis spike: RC LP Bode response via complex MNA
//! solved in real arithmetic.
//!
//! ## What this validates
//!
//! - rlx can host an **AC** analysis end-to-end — frequency-domain MNA
//!   assembly, complex linear solve, and reverse-mode AD on `R`, `C`.
//! - The 2N×2N real-block encoding of complex linear systems is correct
//!   on a circuit whose closed-form Bode response is known exactly.
//! - `eda-extern-ngspice::Invoker::run_ac` returns trace data that
//!   matches the rlx-side response on the same frequency grid.
//!
//! ## Why no complex dtype in rlx
//!
//! rlx is generic-array-shaped (JAX-style) — it could grow a complex
//! dtype with Wirtinger AD, but that's a substantial rlx-side change.
//! For this spike we encode `A·x = b` over `ℂᴺ` as the equivalent
//! `2N × 2N` real system:
//!
//! ```text
//!     A = Aᵣ + j·Aᵢ,  x = xᵣ + j·xᵢ,  b = bᵣ + j·bᵢ
//!     ⇒  ⎡ Aᵣ  -Aᵢ ⎤ ⎡ xᵣ ⎤   ⎡ bᵣ ⎤
//!        ⎣ Aᵢ   Aᵣ ⎦ ⎣ xᵢ ⎦ = ⎣ bᵢ ⎦
//! ```
//!
//! This keeps rlx untouched. When AC analysis becomes hot enough to
//! justify a real complex path (sparse complex LU, large designs), the
//! migration is local — replace `build_ac_graph` with a complex-dtype
//! version. Everything else (host loop, tests, deck) stays the same.
//!
//! ## The RC LP MNA system at frequency ω
//!
//! Indices: 0=vin, 1=vout, 2=i_V1.
//!
//! ```text
//!   Aᵣ = ⎡  G  -G   1 ⎤    Aᵢ = ⎡  0   0   0 ⎤
//!        ⎢ -G   G   0 ⎥         ⎢  0  ωC   0 ⎥
//!        ⎣  1   0   0 ⎦         ⎣  0   0   0 ⎦
//!
//!   bᵣ = [0, 0, 1]ᵀ          bᵢ = [0, 0, 0]ᵀ
//! ```
//!
//! `G = 1/R`. The vsource has unit AC magnitude (ngspice convention).
//! Output `vout = xᵣ[1] + j·xᵢ[1]`; `|H(jω)|² = vout_re² + vout_im²` and
//! `∠H = atan2(vout_im, vout_re)`.

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

pub mod diode_rc;
pub use diode_rc::{
    build_diode_rc_ac_graph, run_diode_rc_ac_point, run_diode_rc_ac_sweep,
    run_diode_rc_ac_grad, dc_op_f64, small_signal_conductance,
    analytic_h as diode_rc_analytic_h,
    analytic_mag as diode_rc_analytic_mag,
    analytic_f3db as diode_rc_analytic_f3db,
    spice_deck as diode_rc_spice_deck,
};

const N: usize = 3;          // MNA rank: vin, vout, i_V1
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

/// Build the AC forward graph. Returns `(graph, R_id, C_id)`.
///
/// Inputs (set per-call): `omega` — angular frequency in rad/s.
/// Params: `R`, `C`.
/// Outputs (in this order): `vout_re`, `vout_im`.
///
/// The output is a length-2 vector reachable as two scalar outputs from
/// the runtime — we keep the length-2 vector form so adding more probed
/// nodes later is just extending the output mask.
pub fn build_ac_graph() -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new("rc_ac");

    let omega = g.input("omega", scalar());
    let r     = g.param("R",     scalar());
    let c     = g.param("C",     scalar());

    let one = const_scalar(&mut g, 1.0);
    let g_cond = g.binary(BinaryOp::Div, one, r, scalar());      // G = 1/R
    let bc     = g.binary(BinaryOp::Mul, omega, c, scalar());    // ωC

    // Constant pattern matrices in the 6×6 real-block layout. Indices
    // [0..N) hold the real block, [N..2N) the imag block.
    //
    //   real-real (Aᵣ): rows 0..3, cols 0..3
    //   real-imag (-Aᵢ): rows 0..3, cols 3..6
    //   imag-real (Aᵢ):  rows 3..6, cols 0..3
    //   imag-imag (Aᵣ):  rows 3..6, cols 3..6
    //
    // Aᵣ = G·P_G + P_vsrc, where P_G stamps the resistor (G at vin/vin,
    // vout/vout, -G off-diagonals) and P_vsrc stamps the voltage-source
    // KCL/KVL augmentation.
    let mut p_g = [0.0_f64; NN * NN];
    let mut p_vsrc = [0.0_f64; NN * NN];
    let mut p_c = [0.0_f64; NN * NN];

    // Helper: 2D index → 1D offset (row-major, NN columns).
    let idx = |r: usize, c: usize| r * NN + c;

    // Resistor stamps in Aᵣ (top-left N×N) AND in Aᵣ duplicated to
    // bottom-right N×N (the 2N×2N block carries Aᵣ on both diagonals).
    for &(rr, cc, val) in &[
        (0, 0,  1.0_f64), (0, 1, -1.0),
        (1, 0, -1.0),     (1, 1,  1.0),
    ] {
        p_g[idx(rr, cc)] += val;            // top-left
        p_g[idx(N + rr, N + cc)] += val;    // bottom-right
    }

    // Voltage-source KVL/KCL augmentation in Aᵣ (no ω dependence):
    //   row 0 ↔ vin's KCL gets the i_V1 term:  +1
    //   row 2 (the V1 equation): vin = V_src → +1·vin = 1
    p_vsrc[idx(0, 2)] = 1.0;        // top-left
    p_vsrc[idx(2, 0)] = 1.0;
    p_vsrc[idx(N + 0, N + 2)] = 1.0;  // bottom-right
    p_vsrc[idx(N + 2, N + 0)] = 1.0;

    // Capacitor stamp in Aᵢ: a single ωC at (vout, vout). The 2N×2N
    // real-block layout puts +Aᵢ in rows [N..2N) cols [0..N) and
    // -Aᵢ in rows [0..N) cols [N..2N).
    p_c[idx(N + 1, 1)] += 1.0;          // imag-real bottom-left
    p_c[idx(1, N + 1)] += -1.0;         // imag-real top-right (negated)

    let p_g_node    = const_mat_nn(&mut g, &p_g);
    let p_vsrc_node = const_mat_nn(&mut g, &p_vsrc);
    let p_c_node    = const_mat_nn(&mut g, &p_c);

    let g_term = g.binary(BinaryOp::Mul, p_g_node, g_cond, mat_nn());
    let c_term = g.binary(BinaryOp::Mul, p_c_node, bc, mat_nn());
    let a_partial = g.binary(BinaryOp::Add, p_vsrc_node, g_term, mat_nn());
    let a_mat     = g.binary(BinaryOp::Add, a_partial,  c_term, mat_nn());

    // RHS: V source magnitude = 1 V (real), no imaginary.
    //   bᵣ = [0, 0, 1, 0, 0, 0]
    //   bᵢ = [0, 0, 0, 0, 0, 0]   ← lives in the bottom half of b
    let mut b = [0.0_f64; NN];
    b[2] = 1.0;
    let b_vec = const_vec_nn(&mut g, &b);

    let x = g.dense_solve(a_mat, b_vec, vec_nn());

    // Extract vout_re = x[1], vout_im = x[N+1] = x[4].
    let mut e_vout_re = [0.0_f64; NN];
    e_vout_re[1] = 1.0;
    let mut e_vout_im = [0.0_f64; NN];
    e_vout_im[N + 1] = 1.0;
    let e_re = const_vec_nn(&mut g, &e_vout_re);
    let e_im = const_vec_nn(&mut g, &e_vout_im);

    let masked_re = g.binary(BinaryOp::Mul, x, e_re, vec_nn());
    let vout_re = g.reduce(masked_re, ReduceOp::Sum, vec![0], true, scalar());
    let masked_im = g.binary(BinaryOp::Mul, x, e_im, vec_nn());
    let vout_im = g.reduce(masked_im, ReduceOp::Sum, vec![0], true, scalar());

    g.set_outputs(vec![vout_re, vout_im]);
    (g, r, c)
}

/// One-frequency forward: returns `(vout_re, vout_im)`.
pub fn run_ac_point(omega: f64, r: f64, c: f64) -> (f64, f64) {
    let (graph, _r, _c) = build_ac_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);
    let outs = compiled.run_typed(&[("omega", &omega.to_le_bytes(), DType::F64)]);
    (decode_f64(&outs[0].0), decode_f64(&outs[1].0))
}

/// Reverse-mode AD: returns `(vout_re, vout_im, ∂|H|²/∂R, ∂|H|²/∂C)` at
/// a single frequency, where `|H|² = vout_re² + vout_im²`.
///
/// We differentiate `|H|²` rather than `|H|` because (a) it has a
/// non-singular gradient at zero magnitude and (b) chain rule to `|H|`
/// or `dB` is a single Rust multiplication outside the graph.
pub fn run_ac_grad(omega: f64, r: f64, c: f64) -> (f64, f64, f64, f64) {
    // Build a fresh graph that emits |H|² as the loss; AD on R, C.
    let mut g = Graph::new("rc_ac_loss");
    let omega_in = g.input("omega", scalar());
    let r_id     = g.param("R",     scalar());
    let c_id     = g.param("C",     scalar());

    let one = const_scalar(&mut g, 1.0);
    let g_cond = g.binary(BinaryOp::Div, one, r_id, scalar());
    let bc     = g.binary(BinaryOp::Mul, omega_in, c_id, scalar());

    let mut p_g = [0.0_f64; NN * NN];
    let mut p_vsrc = [0.0_f64; NN * NN];
    let mut p_c = [0.0_f64; NN * NN];
    let idx = |r: usize, c: usize| r * NN + c;
    for &(rr, cc, val) in &[(0, 0, 1.0_f64), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)] {
        p_g[idx(rr, cc)] += val;
        p_g[idx(N + rr, N + cc)] += val;
    }
    p_vsrc[idx(0, 2)] = 1.0;
    p_vsrc[idx(2, 0)] = 1.0;
    p_vsrc[idx(N + 0, N + 2)] = 1.0;
    p_vsrc[idx(N + 2, N + 0)] = 1.0;
    p_c[idx(N + 1, 1)] += 1.0;
    p_c[idx(1, N + 1)] += -1.0;

    let p_g_node    = const_mat_nn(&mut g, &p_g);
    let p_vsrc_node = const_mat_nn(&mut g, &p_vsrc);
    let p_c_node    = const_mat_nn(&mut g, &p_c);

    let g_term = g.binary(BinaryOp::Mul, p_g_node, g_cond, mat_nn());
    let c_term = g.binary(BinaryOp::Mul, p_c_node, bc, mat_nn());
    let a_partial = g.binary(BinaryOp::Add, p_vsrc_node, g_term, mat_nn());
    let a_mat     = g.binary(BinaryOp::Add, a_partial,  c_term, mat_nn());

    let mut b = [0.0_f64; NN];
    b[2] = 1.0;
    let b_vec = const_vec_nn(&mut g, &b);
    let x = g.dense_solve(a_mat, b_vec, vec_nn());

    let mut e_vout_re = [0.0_f64; NN];
    e_vout_re[1] = 1.0;
    let mut e_vout_im = [0.0_f64; NN];
    e_vout_im[N + 1] = 1.0;
    let e_re = const_vec_nn(&mut g, &e_vout_re);
    let e_im = const_vec_nn(&mut g, &e_vout_im);

    let masked_re = g.binary(BinaryOp::Mul, x, e_re, vec_nn());
    let vout_re   = g.reduce(masked_re, ReduceOp::Sum, vec![0], true, scalar());
    let masked_im = g.binary(BinaryOp::Mul, x, e_im, vec_nn());
    let vout_im   = g.reduce(masked_im, ReduceOp::Sum, vec![0], true, scalar());

    // |H|² = vout_re² + vout_im². `grad_with_loss` requires exactly one
    // output, so we emit only mag_sq; if a caller needs vout_re/vout_im,
    // they call `run_ac_point` for the same `(omega, r, c)`.
    let re_sq = g.binary(BinaryOp::Mul, vout_re, vout_re, scalar());
    let im_sq = g.binary(BinaryOp::Mul, vout_im, vout_im, scalar());
    let mag_sq = g.binary(BinaryOp::Add, re_sq, im_sq, scalar());
    g.set_outputs(vec![mag_sq]);

    let bwd = grad_with_loss(&g, &[r_id, c_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);
    let one_b = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("omega",    &omega.to_le_bytes(), DType::F64),
        ("d_output", &one_b,               DType::F64),
    ]);
    let _mag_sq = decode_f64(&outs[0].0);
    let d_dr = decode_f64(&outs[1].0);
    let d_dc = decode_f64(&outs[2].0);
    let (vout_re, vout_im) = run_ac_point(omega, r, c);
    (vout_re, vout_im, d_dr, d_dc)
}

/// Sweep `n_decade·log10(f_stop/f_start)` log-spaced points in
/// `[f_start, f_stop]`. Returns `(freq_hz, vout_re, vout_im)`.
pub fn run_ac_sweep(
    f_start: f64, f_stop: f64, points_per_decade: usize, r: f64, c: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let log0 = f_start.log10();
    let log1 = f_stop.log10();
    let n = ((log1 - log0) * points_per_decade as f64).round() as usize + 1;

    // Compile once and run per frequency — same pattern as the BE
    // step-graph reuse in spike-rc-transient.
    let (graph, _r, _c) = build_ac_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);

    let mut freq = Vec::with_capacity(n);
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for i in 0..n {
        let f = 10f64.powf(log0 + (log1 - log0) * (i as f64 / (n.saturating_sub(1)) as f64));
        let omega = 2.0 * std::f64::consts::PI * f;
        let outs = compiled.run_typed(&[("omega", &omega.to_le_bytes(), DType::F64)]);
        freq.push(f);
        re.push(decode_f64(&outs[0].0));
        im.push(decode_f64(&outs[1].0));
    }
    (freq, re, im)
}

// ── Analytic Bode references for the RC LP ─────────────────────────────

/// `H(jω) = 1 / (1 + jωRC)`.
pub fn analytic_h(omega: f64, r: f64, c: f64) -> (f64, f64) {
    let wrc = omega * r * c;
    let denom = 1.0 + wrc * wrc;
    (1.0 / denom, -wrc / denom)
}

/// `|H| = 1/√(1 + (ωRC)²)`.
pub fn analytic_mag(omega: f64, r: f64, c: f64) -> f64 {
    let wrc = omega * r * c;
    1.0 / (1.0 + wrc * wrc).sqrt()
}

/// `∠H = -atan(ωRC)` (radians).
pub fn analytic_phase(omega: f64, r: f64, c: f64) -> f64 {
    -(omega * r * c).atan()
}

/// `∂|H|²/∂R` at fixed ω, C.
///
/// `|H|² = 1 / (1 + (ωRC)²)` ⇒ `d|H|²/dR = −2·ω²·R·C² / (1+(ωRC)²)²`.
pub fn analytic_dmagsq_dr(omega: f64, r: f64, c: f64) -> f64 {
    let wrc = omega * r * c;
    -2.0 * omega * omega * r * c * c / (1.0 + wrc * wrc).powi(2)
}

/// `∂|H|²/∂C` at fixed ω, R.
pub fn analytic_dmagsq_dc(omega: f64, r: f64, c: f64) -> f64 {
    let wrc = omega * r * c;
    -2.0 * omega * omega * r * r * c / (1.0 + wrc * wrc).powi(2)
}

// ── SPICE deck for ngspice cross-validation ────────────────────────────

/// Build a deck for `.ac` analysis. The voltage source carries `AC 1` so
/// ngspice treats `vin` as a unit small-signal stimulus.
pub fn spice_deck(r: f64, c: f64) -> String {
    use eda_spice_emit::{Netlist, R as RPrim, SpiceEmit};
    let mut n = Netlist::new("RC LP AC sweep (rlx-eda spike)");
    // We can't use add_dc_source here — it doesn't carry the AC tag.
    // Hand-write the source line; LTspice and ngspice accept this form.
    n.add_element(format!("V1 vin 0 DC 0 AC 1"));
    RPrim { ohms: r }.emit_spice(&mut n, &["vin", "vout"], "1").unwrap();
    n.add_element(format!("C1 vout 0 {c:.10e}"));
    n.deck()
}

fn decode_f64(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    f64::from_le_bytes(buf)
}
