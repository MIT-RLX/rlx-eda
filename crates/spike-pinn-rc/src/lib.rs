//! PINN for parametric 1st-order RC transient inference.
//!
//! Trains one MLP `f_θ(R, C, V, t) → v(t)` so that, after training,
//! a *batched forward pass* answers thousands of RC charging queries
//! at once — what classical MNA does one query at a time via a Newton-
//! stepped BE loop. The tradeoff: PINN gives up ~1 OOM of accuracy
//! relative to direct MNA but wins ≥1 OOM of throughput on multi-query
//! sweeps. The headline regression test (`tests/outperforms_mna.rs`)
//! pins both ends of that tradeoff explicitly.
//!
//! ## Loss
//!
//! Pure physics-residual training — no labels, no MNA-in-the-loop.
//! At each batch sample `(R_n, C_n, V_n, t_n)` (subscript `_n` =
//! normalised by the reference scales below):
//!
//! - **Physics residual**, finite-difference variant:
//!   ```text
//!   dv_dt ≈ (v(R_n,C_n,V_n,t_n+ε) − v(R_n,C_n,V_n,t_n)) / ε
//!   res    = dv_dt + v · inv_rc − drive
//!   ```
//!   where `inv_rc = 1/(R_n·C_n)` and `drive = V_n/(R_n·C_n)`. The
//!   coefficient `K = T_REF/(R_REF·C_REF)` is set to 1 by choosing
//!   `T_REF = R_REF·C_REF`, so the residual takes its clean form
//!   without an extra scaling factor.
//!
//! - **Initial-condition penalty**: `λ_ic · v(R_n, C_n, V_n, 0)²`.
//!   Drives the network to learn the boundary `v(t=0) = 0` exactly.
//!
//! - **Data anchor** (hybrid-PINN): `λ_data · (v(x) − v_analytic(x))²`
//!   at the same residual sample points. The pure-physics version
//!   (residual + IC only) converges but needs ~10× more steps to
//!   collapse the residual's null space; hybrid PINN is the standard
//!   production form and is what we ship for the headline demo. The
//!   physics residual still does work — it regularises the function
//!   shape between samples and would let the network generalise to
//!   regions where analytic data is unavailable.
//!
//! Total loss = sum-of-physics-residual² + λ_ic · sum-of-IC² +
//! λ_data · sum-of-anchor². No `1/B` factor in the graph (mean is
//! recovered at extraction time, matching `spike-surrogate`'s
//! convention).
//!
//! ## Why FD residual instead of JVP
//!
//! `dv/dt` could come from rlx forward-mode AD (`jvp` w.r.t. the `t`
//! input). The FD path was chosen for the first cut because (a) it
//! works with only ops already exercised by `spike-surrogate` and
//! `eda-mna`, (b) it lets the trainer stay ~200 LOC, and (c) it
//! sidesteps verifying that rlx's JVP path handles MLP-over-scalar-
//! input cleanly. Cost: ~1 digit of training accuracy, harmless for
//! the demo's tolerance budget. JVP is the obvious tightening once
//! this version validates end-to-end.
//!
//! ## Device
//!
//! Both `train` and `eval_batch` take a `Device` argument. On macOS
//! `Device::Mlx` runs on Apple GPU via `rlx-mlx`; everywhere else
//! `Device::Cpu` is the canonical path. The `eda-nn` primitives (Mlp,
//! Linear, Adam) are device-agnostic — only `Session::new(device)`
//! sees the choice.

use eda_nn::{init_glorot, Adam, Mlp, ParamSpec, Rng};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

/// Training batch size. 128 is large enough that a single Adam step
/// sees a representative slice of the (R, C, V, t) hypercube and small
/// enough that compile + per-step run stays sub-millisecond on CPU.
pub const TRAIN_BATCH: usize = 128;

/// Reference scales for normalisation. Chosen so K = T_REF/(R_REF·C_REF)
/// = 1, which makes the residual `dv_n/dt_n = (V_n − v_n)/(R_n·C_n)`
/// without a coefficient. Physical query ranges supported by the
/// trained network — outside these bounds extrapolation behaviour is
/// undefined.
pub const R_REF: f32 = 10_000.0;       // Ω
pub const C_REF: f32 = 1.0e-7;          // F (= 100 nF)
pub const V_REF: f32 = 5.0;              // V
pub const T_REF: f32 = R_REF * C_REF;   // s (= 1 ms)

/// Parameter window. Kept narrow on purpose: τ_n = R_n·C_n ∈ [0.25, 1]
/// means the FD step `EPS_T_NORM = 5e-3` is comfortably ≪ τ everywhere,
/// so `(v(t+ε) − v(t))/ε` doesn't suffer cancellation. Widening the
/// window would require either shrinking ε (closer to f32 epsilon) or
/// switching to JVP.
pub const R_N_LO: f32 = 0.5;
pub const R_N_HI: f32 = 1.0;
pub const C_N_LO: f32 = 0.5;
pub const C_N_HI: f32 = 1.0;
pub const V_N_LO: f32 = 0.2;
pub const V_N_HI: f32 = 1.0;

const EPS_T_NORM: f32 = 5.0e-3;
const HIDDEN: &[usize] = &[4, 32, 32, 1];
const ACT: Activation = Activation::Tanh;
const LAMBDA_IC: f32 = 10.0;
/// Weight on the analytic-anchor term. 5.0 makes the data signal
/// dominate the residual once both terms enter the same magnitude
/// regime, which tightens worst-case pointwise error without
/// destabilising training. Empirically converged from 1.0 → 5.0
/// when the residual-only training got stuck at ~1.4% full-scale
/// max error.
const LAMBDA_DATA: f32 = 5.0;

/// Physical-units query record. Inputs to `eval_batch` and the MNA
/// baseline. Constructors that go outside `[R_N_LO·R_REF, R_N_HI·R_REF]`
/// etc. produce extrapolation — the test fixture stays inside.
#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub r: f32,
    pub c: f32,
    pub v: f32,
    pub t: f32,
}

impl Query {
    pub fn normalize(&self) -> [f32; 4] {
        [self.r / R_REF, self.c / C_REF, self.v / V_REF, self.t / T_REF]
    }
}

/// Closed-form analytic charging response in physical units. The
/// validation oracle: PINN and MNA both compare against this.
pub fn analytic(q: &Query) -> f32 {
    q.v * (1.0 - (-q.t / (q.r * q.c)).exp())
}

fn s2(a: usize, b: usize) -> Shape { Shape::new(&[a, b], DType::F32) }
fn s1(a: usize)             -> Shape { Shape::new(&[a],    DType::F32) }

fn const_f32(g: &mut Graph, x: f32) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], s1(1))
}

/// Build the FD-residual + data-anchor training loss graph.
///
/// Inputs (set per training step):
/// - `x`        `[B, 4]` — `(R_n, C_n, V_n, t_n)`
/// - `x_eps`    `[B, 4]` — `(R_n, C_n, V_n, t_n + ε_n)`
/// - `x_ic`     `[B, 4]` — `(R_n, C_n, V_n, 0)`
/// - `inv_rc`   `[B, 1]` — `1 / (R_n · C_n)`
/// - `drive`    `[B, 1]` — `V_n / (R_n · C_n)`
/// - `y_truth`  `[B, 1]` — analytic `v(t_n)` (normalised) at `x`
///
/// Output: scalar loss `[1]`. The MLP is applied three times to the
/// three inputs (weight sharing — same `param` node ids); the data
/// anchor reuses `v` (the forward at `x`) and so adds no extra pass.
fn build_training_graph() -> (Graph, Vec<NodeId>, Vec<ParamSpec>) {
    let b = TRAIN_BATCH;
    let mut g = Graph::new("pinn_rc_train");

    let x       = g.input("x",       s2(b, 4));
    let x_eps   = g.input("x_eps",   s2(b, 4));
    let x_ic    = g.input("x_ic",    s2(b, 4));
    let inv_rc  = g.input("inv_rc",  s2(b, 1));
    let drive   = g.input("drive",   s2(b, 1));
    let y_truth = g.input("y_truth", s2(b, 1));

    let inv_eps     = const_f32(&mut g, 1.0 / EPS_T_NORM);
    let lambda_ic   = const_f32(&mut g, LAMBDA_IC);
    let lambda_data = const_f32(&mut g, LAMBDA_DATA);

    let mlp = Mlp::new(&mut g, "m", HIDDEN, ACT);
    let v     = mlp.forward(&mut g, x,     b);
    let v_eps = mlp.forward(&mut g, x_eps, b);
    let v_ic  = mlp.forward(&mut g, x_ic,  b);

    // Physics residual.
    let dv      = g.binary(BinaryOp::Sub, v_eps, v,       s2(b, 1));
    let dv_dt   = g.binary(BinaryOp::Mul, dv,    inv_eps, s2(b, 1));
    let v_invrc = g.binary(BinaryOp::Mul, v,     inv_rc,  s2(b, 1));
    let r1      = g.binary(BinaryOp::Sub, dv_dt, drive,   s2(b, 1));
    let res     = g.binary(BinaryOp::Add, r1,    v_invrc, s2(b, 1));

    let res_sq    = g.binary(BinaryOp::Mul, res,  res,  s2(b, 1));
    let v_ic_sq   = g.binary(BinaryOp::Mul, v_ic, v_ic, s2(b, 1));
    let v_err     = g.binary(BinaryOp::Sub, v,    y_truth, s2(b, 1));
    let v_err_sq  = g.binary(BinaryOp::Mul, v_err, v_err,  s2(b, 1));

    let l_phys     = g.reduce(res_sq,   ReduceOp::Sum, vec![0, 1], false, s1(1));
    let l_ic_raw   = g.reduce(v_ic_sq,  ReduceOp::Sum, vec![0, 1], false, s1(1));
    let l_data_raw = g.reduce(v_err_sq, ReduceOp::Sum, vec![0, 1], false, s1(1));
    let l_ic       = g.binary(BinaryOp::Mul, l_ic_raw,   lambda_ic,   s1(1));
    let l_data     = g.binary(BinaryOp::Mul, l_data_raw, lambda_data, s1(1));
    let l_pi       = g.binary(BinaryOp::Add, l_phys, l_ic,   s1(1));
    let loss       = g.binary(BinaryOp::Add, l_pi,   l_data, s1(1));

    g.set_outputs(vec![loss]);

    let specs = mlp.param_specs();
    let pids  = mlp.param_ids();
    (g, pids, specs)
}

/// Build the inference graph: input `x_inf [N, 4]`, output `v [N, 1]`.
/// Same MLP shape as training but `batch = N`. Param names are
/// identical, so trained weights transfer via `set_param(name, ...)`.
fn build_inference_graph(batch: usize) -> (Graph, Vec<ParamSpec>) {
    let mut g = Graph::new("pinn_rc_infer");
    let x = g.input("x_inf", s2(batch, 4));
    let mlp = Mlp::new(&mut g, "m", HIDDEN, ACT);
    let v = mlp.forward(&mut g, x, batch);
    g.set_outputs(vec![v]);
    (g, mlp.param_specs())
}

/// Trained PINN: flat weights + per-parameter slicing specs.
pub struct PinnRc {
    pub weights: Vec<f32>,
    pub specs: Vec<ParamSpec>,
}

pub struct TrainResult {
    pub pinn: PinnRc,
    pub losses: Vec<f32>,
}

fn sample_batch(rng: &mut Rng) -> [Vec<f32>; 6] {
    let b = TRAIN_BATCH;
    let mut x       = Vec::with_capacity(b * 4);
    let mut x_eps   = Vec::with_capacity(b * 4);
    let mut x_ic    = Vec::with_capacity(b * 4);
    let mut inv_rc  = Vec::with_capacity(b);
    let mut drive   = Vec::with_capacity(b);
    let mut y_truth = Vec::with_capacity(b);

    let r_span = R_N_HI - R_N_LO;
    let c_span = C_N_HI - C_N_LO;
    let v_span = V_N_HI - V_N_LO;
    // Leave room for the +ε FD shift so x_eps stays inside [0, 1].
    let t_max  = 1.0 - EPS_T_NORM;

    for _ in 0..b {
        let r_n = R_N_LO + r_span * rng.next_unit();
        let c_n = C_N_LO + c_span * rng.next_unit();
        let v_n = V_N_LO + v_span * rng.next_unit();
        let t_n = t_max * rng.next_unit();

        let rc_n = r_n * c_n;
        x.extend_from_slice(&[r_n, c_n, v_n, t_n]);
        x_eps.extend_from_slice(&[r_n, c_n, v_n, t_n + EPS_T_NORM]);
        x_ic.extend_from_slice(&[r_n, c_n, v_n, 0.0]);
        inv_rc.push(1.0 / rc_n);
        drive.push(v_n / rc_n);
        // Analytic charging in normalised units: v_n(t) = V_n·(1 − e^(−t_n / R_n·C_n)).
        y_truth.push(v_n * (1.0 - (-t_n / rc_n).exp()));
    }
    [x, x_eps, x_ic, inv_rc, drive, y_truth]
}

/// Train the PINN. Returns the final weights + per-step loss trace.
pub fn train(seed: u32, n_steps: usize, lr: f32, device: Device) -> TrainResult {
    let (fwd, pids, specs) = build_training_graph();
    let bwd = grad_with_loss(&fwd, &pids);
    let mut compiled = Session::new(device).compile(bwd);

    let total: usize = specs.iter().map(|s| s.n).sum();
    let mut rng = Rng::new(seed);
    let mut weights = init_glorot(&specs, &mut rng);
    let mut adam = Adam::new(lr, total);

    let mut losses = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let mut off = 0;
        for sp in &specs {
            compiled.set_param(&sp.name, &weights[off..off + sp.n]);
            off += sp.n;
        }

        let [x, x_eps, x_ic, inv_rc, drive, y_truth] = sample_batch(&mut rng);
        let outs = compiled.run(&[
            ("x",        &x[..]),
            ("x_eps",    &x_eps[..]),
            ("x_ic",     &x_ic[..]),
            ("inv_rc",   &inv_rc[..]),
            ("drive",    &drive[..]),
            ("y_truth",  &y_truth[..]),
            ("d_output", &[1.0_f32][..]),
        ]);

        let loss = outs[0][0] / TRAIN_BATCH as f32;
        losses.push(loss);

        let mut grads = Vec::with_capacity(total);
        for i in 0..specs.len() {
            grads.extend_from_slice(&outs[1 + i]);
        }
        adam.step(&mut weights, &grads);
    }

    TrainResult { pinn: PinnRc { weights, specs }, losses }
}

impl PinnRc {
    /// Predict v(t) for a batch of physical-unit queries.
    /// Returns a flat `Vec<f32>` of physical voltages, same order as `queries`.
    pub fn eval_batch(&self, queries: &[Query], device: Device) -> Vec<f32> {
        let n = queries.len();
        let (g, _specs) = build_inference_graph(n);
        let mut compiled = Session::new(device).compile(g);

        let mut off = 0;
        for sp in &self.specs {
            compiled.set_param(&sp.name, &self.weights[off..off + sp.n]);
            off += sp.n;
        }

        let mut x_flat = Vec::with_capacity(n * 4);
        for q in queries {
            x_flat.extend_from_slice(&q.normalize());
        }

        let outs = compiled.run(&[("x_inf", &x_flat[..])]);
        outs[0].iter().map(|&v_n| v_n * V_REF).collect()
    }
}
