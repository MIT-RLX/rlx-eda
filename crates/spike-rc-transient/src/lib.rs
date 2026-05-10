//! RC low-pass transient via Backward-Euler stamps + outer Rust loop.
//!
//! Where `spike-divider-mna` solved a DC system in one shot, this spike
//! extends the same MNA pattern to a **DAE** by adding a capacitor's
//! companion conductance and a history term, and then sequences timesteps
//! in an outer Rust loop. The `Behavioral` trait we'll later define
//! distinguishes algebraic contributions (`F`) from storage contributions
//! (`Q`); BE turns `Q(y_n) - Q(y_{n-1}) = h · F(y_n)` into a linear stamp
//! `dQ/dy_n · 1/h` (the companion conductance) on the matrix and a history
//! term `dQ/dy_n · 1/h · y_{n-1}` on the rhs. For a linear cap that's
//! `gc = C/h` and `b += gc · vout_{n-1}`.
//!
//! ## The system, per timestep
//!
//! Same MNA layout as the DC divider, with `g2` replaced by `gc = C/h`:
//!
//! ```text
//!   indices: 0 = vin, 1 = vout, 2 = i_V1
//!
//!   [  g1     -g1     1 ]   [ vin_n  ]   [ 0                ]
//!   [ -g1   g1+gc     0 ] · [ vout_n ] = [ gc · vout_{n-1}  ]
//!   [   1      0      0 ]   [ i_V1_n ]   [ V_n              ]
//! ```
//!
//! Solving gives `vout_n = (g1 · V_n + gc · vout_{n-1}) / (g1 + gc)`,
//! a first-order recurrence whose closed form is
//! `vout_n = V · (1 − α^n)` with `α = RC/(h + RC)` (constant DC, zero IC).
//!
//! ## What this spike validates
//!
//! 1. **DAE shape works in rlx**: storage contributions become matrix
//!    stamps (`gc`) and rhs history terms (`gc · vout_{n-1}`); the same
//!    `DenseSolve` op handles both DC and BE.
//! 2. **Outer-loop pattern works**: a single compiled rlx graph called N
//!    times by Rust correctly threads state forward.
//! 3. **Single-step AD validates**: `∂vout_n/∂R`, `∂vout_n/∂C` from
//!    `grad_with_loss` match the closed-form analytic gradients.
//!
//! Multi-step end-to-end gradients (adjoint sensitivity / unrolled-graph
//! AD) are explicitly **out of scope** — that's the next spike.

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

fn scalar() -> Shape { Shape::new(&[1], DType::F64) }
fn vec3()   -> Shape { Shape::new(&[3], DType::F64) }
fn mat3()   -> Shape { Shape::new(&[3, 3], DType::F64) }

fn const_scalar(g: &mut Graph, x: f64) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar())
}
fn const_vec3_f64(g: &mut Graph, x: &[f64; 3]) -> NodeId {
    let mut bytes = Vec::with_capacity(24);
    for v in x { bytes.extend_from_slice(&v.to_le_bytes()); }
    g.add_node(Op::Constant { data: bytes }, vec![], vec3())
}
fn const_mat3_f64(g: &mut Graph, x: &[f64; 9]) -> NodeId {
    let mut bytes = Vec::with_capacity(72);
    for v in x { bytes.extend_from_slice(&v.to_le_bytes()); }
    g.add_node(Op::Constant { data: bytes }, vec![], mat3())
}

/// Build the per-step BE forward graph and return `(graph, R, C)`.
///
/// Inputs (set per-step at runtime): `V`, `vout_prev`, `h`.
/// Params (constant across the transient): `R`, `C`.
/// Output: `vout_n` (a `[1]` f64 scalar).
pub fn build_step_graph() -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new("rc_be_step");

    let v         = g.input("V",         scalar());
    let vout_prev = g.input("vout_prev", scalar());
    let h         = g.input("h",         scalar());
    let r         = g.param("R",         scalar());
    let c         = g.param("C",         scalar());

    let one = const_scalar(&mut g, 1.0);
    let g1  = g.binary(BinaryOp::Div, one, r, scalar()); // 1/R
    let gc  = g.binary(BinaryOp::Div, c,   h, scalar()); // C/h

    // Stamp patterns (same as DC divider, with `gc` replacing `g2`).
    let pattern_vsrc = const_mat3_f64(&mut g, &[
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
    ]);
    let pattern_r = const_mat3_f64(&mut g, &[
        1.0, -1.0, 0.0,
        -1.0, 1.0, 0.0,
        0.0,  0.0, 0.0,
    ]);
    let pattern_c = const_mat3_f64(&mut g, &[
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]);

    let r_term = g.binary(BinaryOp::Mul, pattern_r, g1, mat3());
    let c_term = g.binary(BinaryOp::Mul, pattern_c, gc, mat3());
    let a_partial = g.binary(BinaryOp::Add, pattern_vsrc, r_term, mat3());
    let a_mat     = g.binary(BinaryOp::Add, a_partial,    c_term, mat3());

    // b = e_V1 · V + e_vout · gc · vout_prev
    //   e_V1   = [0, 0, 1]   (voltage-source equation)
    //   e_vout = [0, 1, 0]   (KCL at vout, capacitor history term)
    let e_v1   = const_vec3_f64(&mut g, &[0.0, 0.0, 1.0]);
    let e_vout = const_vec3_f64(&mut g, &[0.0, 1.0, 0.0]);

    let v_part        = g.binary(BinaryOp::Mul, e_v1, v, vec3());
    let gc_vout_prev  = g.binary(BinaryOp::Mul, gc, vout_prev, scalar());
    let history_part  = g.binary(BinaryOp::Mul, e_vout, gc_vout_prev, vec3());
    let b_vec         = g.binary(BinaryOp::Add, v_part, history_part, vec3());

    let x = g.dense_solve(a_mat, b_vec, vec3());

    // Extract vout = x[1] via masked dot product (Narrow has no F64 path
    // on rlx-cpu today; we worked around this in spike-divider-mna).
    let masked = g.binary(BinaryOp::Mul, x, e_vout, vec3());
    let vout = g.reduce(masked, ReduceOp::Sum, vec![0], /*keep_dim=*/true, scalar());

    g.set_outputs(vec![vout]);
    (g, r, c)
}

/// One BE step via rlx (forward only). Builds + compiles a fresh graph
/// per call — fine for tests, terrible for transient loops; the loop
/// helpers below cache one compiled graph and reuse it across steps.
pub fn run_step_once(v: f64, vout_prev: f64, r: f64, c: f64, h: f64) -> f64 {
    let (graph, _r, _c) = build_step_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);
    let outs = compiled.run_typed(&[
        ("V",         &v.to_le_bytes(),         DType::F64),
        ("vout_prev", &vout_prev.to_le_bytes(), DType::F64),
        ("h",         &h.to_le_bytes(),         DType::F64),
    ]);
    decode_f64_scalar(&outs[0].0)
}

/// One BE step via rlx with reverse-mode AD: returns
/// `(vout_n, ∂vout_n/∂R, ∂vout_n/∂C)`.
pub fn run_step_and_grad(v: f64, vout_prev: f64, r: f64, c: f64, h: f64) -> (f64, f64, f64) {
    let (fwd, r_id, c_id) = build_step_graph();
    let bwd = grad_with_loss(&fwd, &[r_id, c_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);
    let one = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("V",         &v.to_le_bytes(),         DType::F64),
        ("vout_prev", &vout_prev.to_le_bytes(), DType::F64),
        ("h",         &h.to_le_bytes(),         DType::F64),
        ("d_output",  &one,                     DType::F64),
    ]);
    (
        decode_f64_scalar(&outs[0].0),
        decode_f64_scalar(&outs[1].0),
        decode_f64_scalar(&outs[2].0),
    )
}

/// Run a transient using the rlx step graph in an outer Rust loop.
/// Compiles the graph once; runs `n_steps` times.
///
/// `v_at_step(n)` supplies the source value at step `n` (n=1..=n_steps);
/// pass `|_| v_dc` for a constant DC excitation.
pub fn run_transient<F: FnMut(usize) -> f64>(
    n_steps: usize,
    h: f64,
    r: f64,
    c: f64,
    vout0: f64,
    mut v_at_step: F,
) -> f64 {
    let (graph, _r, _c) = build_step_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);

    let h_b = h.to_le_bytes();
    let mut vout = vout0;
    for n in 1..=n_steps {
        let v = v_at_step(n);
        let outs = compiled.run_typed(&[
            ("V",         &v.to_le_bytes(),    DType::F64),
            ("vout_prev", &vout.to_le_bytes(), DType::F64),
            ("h",         &h_b,                DType::F64),
        ]);
        vout = decode_f64_scalar(&outs[0].0);
    }
    vout
}

/// Same as `run_transient` but returns the **full waveform**:
/// `(time, vout)` with `time[0] = 0` (the IC) and `time[i] = i·h`,
/// `vout[i]` the BE-solved output at step `i`.
pub fn run_transient_trace<F: FnMut(usize) -> f64>(
    n_steps: usize,
    h: f64,
    r: f64,
    c: f64,
    vout0: f64,
    mut v_at_step: F,
) -> (Vec<f64>, Vec<f64>) {
    let (graph, _r, _c) = build_step_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);

    let h_b = h.to_le_bytes();
    let mut time = Vec::with_capacity(n_steps + 1);
    let mut vout_trace = Vec::with_capacity(n_steps + 1);
    time.push(0.0);
    vout_trace.push(vout0);

    let mut vout = vout0;
    for n in 1..=n_steps {
        let v = v_at_step(n);
        let outs = compiled.run_typed(&[
            ("V",         &v.to_le_bytes(),    DType::F64),
            ("vout_prev", &vout.to_le_bytes(), DType::F64),
            ("h",         &h_b,                DType::F64),
        ]);
        vout = decode_f64_scalar(&outs[0].0);
        time.push(n as f64 * h);
        vout_trace.push(vout);
    }
    (time, vout_trace)
}

/// Pure-Rust BE step: closed form of the 3×3 MNA solve.
pub fn ref_step(v: f64, vout_prev: f64, r: f64, c: f64, h: f64) -> f64 {
    let g1 = 1.0 / r;
    let gc = c / h;
    (g1 * v + gc * vout_prev) / (g1 + gc)
}

/// Pure-Rust transient via N applications of `ref_step`.
pub fn ref_transient<F: FnMut(usize) -> f64>(
    n_steps: usize,
    h: f64,
    r: f64,
    c: f64,
    vout0: f64,
    mut v_at_step: F,
) -> f64 {
    let mut vout = vout0;
    for n in 1..=n_steps {
        vout = ref_step(v_at_step(n), vout, r, c, h);
    }
    vout
}

// ── Closed-form analytic references ────────────────────────────────────

/// Analytic single-step BE: same as `ref_step`. Kept as a separate name to
/// match the convention in earlier spikes (`analytic_*` for symbolic
/// references).
pub fn analytic_step(v: f64, vout_prev: f64, r: f64, c: f64, h: f64) -> f64 {
    ref_step(v, vout_prev, r, c, h)
}

/// `∂vout_n/∂R` for one BE step, from differentiating the closed form.
/// `f = (g1·V + gc·vout_prev) / (g1 + gc)` with `g1 = 1/R`, `gc = C/h`.
pub fn analytic_dstep_dr(v: f64, vout_prev: f64, r: f64, c: f64, h: f64) -> f64 {
    let g1 = 1.0 / r;
    let gc = c / h;
    let s = g1 + gc;
    // df/dg1 = (V·s - (g1·V + gc·vout_prev)) / s² = gc·(V - vout_prev)/s²
    // dg1/dR = -1/R²
    -gc * (v - vout_prev) / (r * r * s * s)
}

/// `∂vout_n/∂C` for one BE step.
pub fn analytic_dstep_dc(v: f64, vout_prev: f64, r: f64, c: f64, h: f64) -> f64 {
    let g1 = 1.0 / r;
    let gc = c / h;
    let s = g1 + gc;
    // df/dgc = (vout_prev·s - (g1·V + gc·vout_prev)) / s² = g1·(vout_prev - V)/s²
    // dgc/dC = 1/h
    g1 * (vout_prev - v) / (h * s * s)
}

/// Closed-form multi-step BE for constant-DC `V`, zero IC: `V·(1 − α^N)`.
pub fn analytic_transient_dc(v_dc: f64, n_steps: usize, h: f64, r: f64, c: f64) -> f64 {
    let alpha = (r * c) / (h + r * c);
    v_dc * (1.0 - alpha.powi(n_steps as i32))
}

/// Continuum (h → 0) reference: `V · (1 − exp(−T/RC))`.
pub fn continuum_transient_dc(v_dc: f64, t: f64, r: f64, c: f64) -> f64 {
    v_dc * (1.0 - (-t / (r * c)).exp())
}

// ── Unrolled multi-step transient: end-to-end AD ────────────────────────
//
// Where `build_step_graph` + `run_transient` keep the loop in Rust (one
// compiled graph called N times), `build_unrolled_graph` chains all N
// BE steps into a **single** rlx graph and shares R, C across them. That
// lets `grad_with_loss` produce `∂vout_N/∂R, ∂vout_N/∂C` end-to-end via
// the standard reverse-mode walk — gradients flow through N stacked
// `DenseSolve`s, each emitting an implicit-function VJP `d_b = solve(Aᵀ, ...)`.
//
// The matrix A is constant across steps (depends only on R, C, h), so we
// build it once and reuse it as the input to every solve. b changes per
// step because of the history term `gc · vout_{n-1}`.

/// Build the unrolled N-step BE forward graph. Returns `(graph, R, C)`.
///
/// Inputs (set once at runtime): `V`, `vout_0`, `h`.
/// Params: `R`, `C` — shared across all steps.
/// Output: `vout_N` (a `[1]` f64 scalar).
pub fn build_unrolled_graph(n_steps: usize) -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new(format!("rc_be_unrolled_n{n_steps}"));

    let v     = g.input("V",      scalar());
    let vout0 = g.input("vout_0", scalar());
    let h     = g.input("h",      scalar());
    let r     = g.param("R",      scalar());
    let c     = g.param("C",      scalar());

    // Built once — A is constant across steps (R, C, h are constant).
    let one = const_scalar(&mut g, 1.0);
    let g1  = g.binary(BinaryOp::Div, one, r, scalar());
    let gc  = g.binary(BinaryOp::Div, c,   h, scalar());

    let pattern_vsrc = const_mat3_f64(&mut g, &[
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
    ]);
    let pattern_r = const_mat3_f64(&mut g, &[
        1.0, -1.0, 0.0,
        -1.0, 1.0, 0.0,
        0.0,  0.0, 0.0,
    ]);
    let pattern_c = const_mat3_f64(&mut g, &[
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]);
    let r_term    = g.binary(BinaryOp::Mul, pattern_r, g1, mat3());
    let c_term    = g.binary(BinaryOp::Mul, pattern_c, gc, mat3());
    let a_partial = g.binary(BinaryOp::Add, pattern_vsrc, r_term, mat3());
    let a_mat     = g.binary(BinaryOp::Add, a_partial,    c_term, mat3());

    let e_v1   = const_vec3_f64(&mut g, &[0.0, 0.0, 1.0]);
    let e_vout = const_vec3_f64(&mut g, &[0.0, 1.0, 0.0]);

    // V part of b is constant across steps (same V each step). The
    // capacitor history term `e_vout · gc · vout_{n-1}` is the only
    // per-step piece, so we can lift `e_v1·V` out of the loop.
    let v_part = g.binary(BinaryOp::Mul, e_v1, v, vec3());

    // Unroll: thread `vout` forward through N solves.
    let mut vout = vout0;
    for _ in 0..n_steps {
        let gc_vout_prev = g.binary(BinaryOp::Mul, gc, vout, scalar());
        let history     = g.binary(BinaryOp::Mul, e_vout, gc_vout_prev, vec3());
        let b           = g.binary(BinaryOp::Add, v_part, history, vec3());
        let x           = g.dense_solve(a_mat, b, vec3());
        let masked      = g.binary(BinaryOp::Mul, x, e_vout, vec3());
        vout            = g.reduce(masked, ReduceOp::Sum, vec![0], true, scalar());
    }

    g.set_outputs(vec![vout]);
    (g, r, c)
}

/// Forward only on the unrolled graph: returns `vout_N`.
pub fn run_unrolled_forward(
    v: f64, vout_0: f64, h: f64, r: f64, c: f64, n_steps: usize,
) -> f64 {
    let (g, _r, _c) = build_unrolled_graph(n_steps);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);
    let outs = compiled.run_typed(&[
        ("V",      &v.to_le_bytes(),      DType::F64),
        ("vout_0", &vout_0.to_le_bytes(), DType::F64),
        ("h",      &h.to_le_bytes(),      DType::F64),
    ]);
    decode_f64_scalar(&outs[0].0)
}

/// Forward + reverse-mode AD on the unrolled graph.
/// Returns `(vout_N, ∂vout_N/∂R, ∂vout_N/∂C)`.
pub fn run_unrolled_and_grad(
    v: f64, vout_0: f64, h: f64, r: f64, c: f64, n_steps: usize,
) -> (f64, f64, f64) {
    let (fwd, r_id, c_id) = build_unrolled_graph(n_steps);
    let bwd = grad_with_loss(&fwd, &[r_id, c_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("R", &r.to_le_bytes(), DType::F64);
    compiled.set_param_typed("C", &c.to_le_bytes(), DType::F64);
    let one = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("V",        &v.to_le_bytes(),      DType::F64),
        ("vout_0",   &vout_0.to_le_bytes(), DType::F64),
        ("h",        &h.to_le_bytes(),      DType::F64),
        ("d_output", &one,                  DType::F64),
    ]);
    (
        decode_f64_scalar(&outs[0].0),
        decode_f64_scalar(&outs[1].0),
        decode_f64_scalar(&outs[2].0),
    )
}

// ── Analytic references for the unrolled transient ─────────────────────
//
// The recurrence `vout_n = (1-α)·V + α·vout_{n-1}` with `α = RC/(h+RC)`
// has the closed form:
//
//   vout_N = V + α^N · (vout_0 − V)
//
// (zero-IC special case: vout_N = V·(1 − α^N), as in `analytic_transient_dc`).

/// `vout_N = V + α^N · (vout_0 − V)`.
pub fn analytic_transient_with_ic(
    v_dc: f64, vout_0: f64, n_steps: usize, h: f64, r: f64, c: f64,
) -> f64 {
    let alpha = (r * c) / (h + r * c);
    v_dc + alpha.powi(n_steps as i32) * (vout_0 - v_dc)
}

/// `∂vout_N/∂R = N · α^(N-1) · (vout_0 − V) · ∂α/∂R`.
pub fn analytic_dtransient_dr(
    v_dc: f64, vout_0: f64, n_steps: usize, h: f64, r: f64, c: f64,
) -> f64 {
    let n = n_steps as i32;
    let alpha = (r * c) / (h + r * c);
    // ∂α/∂R = C·h / (h + RC)²
    let d_alpha_dr = c * h / (h + r * c).powi(2);
    (n as f64) * alpha.powi(n - 1) * (vout_0 - v_dc) * d_alpha_dr
}

/// `∂vout_N/∂C = N · α^(N-1) · (vout_0 − V) · ∂α/∂C`.
pub fn analytic_dtransient_dc(
    v_dc: f64, vout_0: f64, n_steps: usize, h: f64, r: f64, c: f64,
) -> f64 {
    let n = n_steps as i32;
    let alpha = (r * c) / (h + r * c);
    let d_alpha_dc = r * h / (h + r * c).powi(2);
    (n as f64) * alpha.powi(n - 1) * (vout_0 - v_dc) * d_alpha_dc
}

fn decode_f64_scalar(bytes: &[u8]) -> f64 {
    assert!(bytes.len() >= 8, "expected ≥8 bytes for f64 scalar, got {}", bytes.len());
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    f64::from_le_bytes(buf)
}

/// SPICE deck for the same RC LP. Caller picks t_step, t_stop in the
/// `.tran` invocation; we write `IC=0` on C1 and `.ic v(vout)=0` so the
/// transient starts from a discharged cap (matches our zero-IC reference).
///
/// We force BDF1 (`.options method=gear maxord=1`), which is exactly
/// Backward Euler — same numerical method our rlx graph implements. ngspice
/// defaults to trapezoidal (O(h²)), which would diverge from our BE result
/// by O(h) per step and frustrate apples-to-apples comparison.
pub fn spice_deck(v_dc: f64, r: f64, c: f64) -> String {
    format!(
        "* RC LP transient (rlx-eda spike)\n\
         .options method=gear maxord=1\n\
         V1 vin 0 {v_dc}\n\
         R1 vin vout {r}\n\
         C1 vout 0 {c} IC=0\n\
         .ic v(vout)=0\n",
    )
}
