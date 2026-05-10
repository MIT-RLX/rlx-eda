//! Shichman-Hodges (SPICE LEVEL=1) NMOS DC current.
//!
//! ## Why this spike
//!
//! Wave-2 MOSFET work breaks into three concerns: (a) does the device
//! model itself round-trip rlx vs analytic vs ngspice; (b) does it
//! compose into a working amplifier (`spike-cmos-inverter` next); and
//! (c) does its small-signal linearization land in `spike-ac`. This
//! crate covers (a) only — a single 4-terminal NMOS biased by two
//! voltage sources, evaluating `Id` at chosen `(Vgs, Vds)` operating
//! points.
//!
//! ## The model
//!
//! Strict LEVEL=1, source-tied-to-bulk, no body effect, no
//! sub-threshold conduction:
//!
//! ```text
//!   Vov = Vgs - Vth
//!
//!   Id = ⎧ 0                                         ,  Vov ≤ 0          (cutoff)
//!        ⎨ kp · (Vov · Vds - Vds²/2) · (1 + λ·Vds)   ,  0 < Vds < Vov    (triode)
//!        ⎩ (kp/2) · Vov² · (1 + λ·Vds)               ,  Vds ≥ Vov ≥ 0    (saturation)
//! ```
//!
//! `kp` here is **the effective transconductance** `KP · W/L`. Pulling
//! `W/L` into a single parameter keeps the rlx graph a function of
//! three scalar params (Vth, kp, λ), matching the design surface that
//! gradient-based sizing actually optimizes.
//!
//! ## Smooth approximation
//!
//! rlx graphs are static DAGs — no runtime `if`/`else` — so we replace
//! the piecewise hard regions with smooth (everywhere-differentiable)
//! approximations:
//!
//! - **Cutoff smoothing:** `Vov_smooth = (1/β)·log(1 + exp(β · (Vgs - Vth)))`
//!   (softplus). β = 200 → ~5mV transition width, well below the typical
//!   bias overdrive of 100–500mV in practical operating points.
//! - **Triode/saturation smoothing:**
//!   `Vds_eff = ½·(Vds + Vov_s − √((Vds − Vov_s)² + δ))`
//!   (smooth-min). δ = 1e-4 V² → ~10mV rounding around the channel-pinch
//!   knee. Then `Id = kp · (Vov_s · Vds_eff − Vds_eff² / 2) · (1 + λ·Vds)`.
//!
//! At operating points *well inside* a region, smooth and strict agree
//! to ~1ppm. The smooth model only deviates near the cutoff/triode and
//! triode/saturation boundaries, by O(1/β) and O(√δ) respectively.
//!
//! ## What this validates
//!
//! 1. rlx-side smooth `Id` matches strict-piecewise analytic `Id` at
//!    operating points well inside cutoff, triode, and saturation.
//! 2. AD `∂Id/∂{Vth, kp, λ}` matches both the analytic gradient
//!    (closed form on the smooth formula) and centered FD on the rlx
//!    forward.
//! 3. ngspice `.op` of the same NMOS at the same `(Vgs, Vds)` reports a
//!    matching `Id` (within ngspice's hard-piecewise model + our
//!    smooth-model deviation).

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

pub mod inverter;
pub mod mc;
pub mod mc_gpu;
pub mod surrogate;
pub use inverter::{
    build_inverter_dc_graph, run_inverter_dc, run_inverter_dc_grad,
    ref_inverter_dc,
};
pub use mc_gpu::{build_mirror_graph_batched_f32, run_mc_sweep_auto};

/// Softplus sharpness for cutoff smoothing:
/// `Vov_smooth = (1/β)·log(1 + exp(β·(Vgs − Vth)))`. Higher β → tighter
/// approach to ReLU at the cost of less differentiable behavior near
/// the threshold. 200 puts the transition width at ~5 mV, negligible
/// for analog operating points.
pub const BETA: f64 = 200.0;

/// Smooth-min rounding scale (V²): `Vds_eff = ½(a + b − √((a−b)² + DELTA))`.
/// 1e-4 → ~10mV rounding at the channel-pinch knee.
pub const DELTA: f64 = 1e-4;

// ── Temperature-corner scaling ─────────────────────────────────────────
//
// We treat T as a parameter remap on `(Vth, kp)` performed *before*
// graph construction — the rlx graph stays a function of three scalars
// regardless of T. AD therefore differentiates only with respect to the
// design knobs, not corner temperature, which is the right contract for
// a thermal *corner* sweep (T is the environment, not a tuning knob).
//
// Formulas mirror the simple LEVEL=1 family ngspice uses by default:
//
//   Vth(T) = Vth0 + KT1·(T − Tnom)            // linearized; sign for nFET
//   kp(T)  = kp0 · (T_K / Tnom_K)^UTE         // from μ(T) ∝ T^UTE
//
// where T_K = T_celsius + 273.15. ngspice LEVEL=1 applies a similar
// `(T/Tnom)^-1.5` to KP automatically under `.temp`, plus a
// bandgap-based VTO shift that lands ~-0.7 mV/°C for GAMMA=0. Our KT1
// is set to -1 mV/°C — close enough that analytic and ngspice agree to
// a few percent at T ∈ {-40, 125}, exact at Tnom. The thermal-sweep
// witness uses tighter tolerance at Tnom and looser at the corners.

/// Nominal temperature for stdcell parameters, °C. Standard SPICE TNOM.
pub const T_NOM_C: f64 = 27.0;

/// Vth linear temperature coefficient, V/°C. Negative for nFET (and for
/// pFET in magnitude — sign of the Vov term depends on device polarity,
/// which the analytic NMOS model handles via its sign convention).
pub const KT1: f64 = -1.0e-3;

/// Mobility temperature exponent in `μ(T) = μ₀ · (T/Tnom)^UTE`.
/// −1.5 is the canonical phonon-scattering value used by SPICE LEVEL 1
/// out of the box and a reasonable proxy for sky130's BSIM4 default.
pub const UTE: f64 = -1.5;

/// Convert °C → Kelvin.
#[inline] pub fn celsius_to_kelvin(t_c: f64) -> f64 { t_c + 273.15 }

/// `Vth(T) = Vth0 + KT1·(T − Tnom)`. `vth0` is the threshold at `T_NOM_C`.
#[inline]
pub fn vth_at_temp(vth0: f64, t_celsius: f64) -> f64 {
    vth0 + KT1 * (t_celsius - T_NOM_C)
}

/// `kp(T) = kp0 · (T_K / Tnom_K)^UTE`. `kp0` is the transconductance at
/// `T_NOM_C`.
#[inline]
pub fn kp_at_temp(kp0: f64, t_celsius: f64) -> f64 {
    let ratio = celsius_to_kelvin(t_celsius) / celsius_to_kelvin(T_NOM_C);
    kp0 * ratio.powf(UTE)
}

fn scalar() -> Shape { Shape::new(&[1], DType::F64) }

fn const_scalar(g: &mut Graph, x: f64) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar())
}

/// Insert a smooth NMOS `Id` subgraph into `g` and return its NodeId.
/// Inputs are existing graph nodes (so this is composable into a larger
/// circuit graph). `vth, kp, lam` are typically `g.param(...)` handles.
pub fn id_subgraph(
    g: &mut Graph,
    vgs: NodeId, vds: NodeId,
    vth: NodeId, kp: NodeId, lam: NodeId,
) -> NodeId {
    let beta     = const_scalar(g, BETA);
    let inv_beta = const_scalar(g, 1.0 / BETA);
    let half     = const_scalar(g, 0.5);
    let one      = const_scalar(g, 1.0);
    let delta    = const_scalar(g, DELTA);

    // **Textbook smooth Level-1 Id.** All three rlx AD primitives this
    // graph touches (`Exp`, `Log`, `Sqrt`) have correct reverse-mode
    // gradients on f64 — verified by `rlx-runtime/tests/cpu_sqrt_grad_f64.rs`.
    // No saturation-only fallback; this graph is correct end-to-end
    // for cutoff, triode, and saturation, with O(1/β) and O(√δ)
    // deviation from the strict piecewise model only at the region
    // boundaries.
    //
    // ## Cutoff smoothing — softplus
    //
    // `Vov_smooth = (1/β) · log(1 + exp(β · (Vgs − Vth)))`.
    // β=200 puts the transition width at ~5 mV — negligible vs. typical
    // overdrive of 100–500 mV. exp(±300) is comfortably representable in
    // f64; ditto log(1+exp(300)) ≈ 300.
    //
    // ## Triode/saturation smoothing — smooth-min
    //
    // `Vds_eff = ½·(Vds + Vov_s − √((Vds − Vov_s)² + δ))`.
    // δ=1e-4 V² → ~10 mV rounding around the channel-pinch knee. In
    // saturation `Vds_eff ≈ Vov_s`; in triode `Vds_eff ≈ Vds`.
    //
    // Channel current:
    //   `Id = kp · (Vov_s · Vds_eff − Vds_eff² / 2) · (1 + λ·Vds)`.
    let vov_raw = g.binary(BinaryOp::Sub, vgs, vth, scalar());
    let scaled  = g.binary(BinaryOp::Mul, beta, vov_raw, scalar());
    let exp_v   = g.activation(Activation::Exp, scaled, scalar());
    let one_plus_exp = g.binary(BinaryOp::Add, one, exp_v, scalar());
    let log_v   = g.activation(Activation::Log, one_plus_exp, scalar());
    let vov_s   = g.binary(BinaryOp::Mul, inv_beta, log_v, scalar());

    let sum     = g.binary(BinaryOp::Add, vds, vov_s, scalar());
    let diff    = g.binary(BinaryOp::Sub, vds, vov_s, scalar());
    let diff_sq = g.binary(BinaryOp::Mul, diff, diff, scalar());
    let arg     = g.binary(BinaryOp::Add, diff_sq, delta, scalar());
    let root    = g.activation(Activation::Sqrt, arg, scalar());
    let inner   = g.binary(BinaryOp::Sub, sum, root, scalar());
    let vds_eff = g.binary(BinaryOp::Mul, half, inner, scalar());

    let term1   = g.binary(BinaryOp::Mul, vov_s, vds_eff, scalar());
    let vds_eff_sq      = g.binary(BinaryOp::Mul, vds_eff, vds_eff, scalar());
    let half_vds_eff_sq = g.binary(BinaryOp::Mul, half, vds_eff_sq, scalar());
    let bracket = g.binary(BinaryOp::Sub, term1, half_vds_eff_sq, scalar());

    let lam_vds = g.binary(BinaryOp::Mul, lam, vds, scalar());
    let clm     = g.binary(BinaryOp::Add, one, lam_vds, scalar()); // channel-length mod

    let kp_bracket = g.binary(BinaryOp::Mul, kp, bracket, scalar());
    g.binary(BinaryOp::Mul, kp_bracket, clm, scalar())
}

/// Build the standalone `Id(Vgs, Vds; Vth, kp, λ)` graph.
/// Inputs (set per call): `Vgs`, `Vds`. Params: `Vth`, `kp`, `lam`.
pub fn build_id_graph() -> (Graph, NodeId, NodeId, NodeId) {
    let mut g = Graph::new("nmos_l1_id");
    let vgs = g.input("Vgs", scalar());
    let vds = g.input("Vds", scalar());
    let vth = g.param("Vth", scalar());
    let kp  = g.param("kp",  scalar());
    let lam = g.param("lam", scalar());
    let id  = id_subgraph(&mut g, vgs, vds, vth, kp, lam);
    g.set_outputs(vec![id]);
    (g, vth, kp, lam)
}

/// Forward: smooth `Id` at (Vgs, Vds; Vth, kp, λ).
pub fn run_id(vgs: f64, vds: f64, vth: f64, kp: f64, lam: f64) -> f64 {
    let (graph, _vth, _kp, _lam) = build_id_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("Vth", &vth.to_le_bytes(), DType::F64);
    compiled.set_param_typed("kp",  &kp.to_le_bytes(),  DType::F64);
    compiled.set_param_typed("lam", &lam.to_le_bytes(), DType::F64);
    let outs = compiled.run_typed(&[
        ("Vgs", &vgs.to_le_bytes(), DType::F64),
        ("Vds", &vds.to_le_bytes(), DType::F64),
    ]);
    decode_f64(&outs[0].0)
}

/// Forward `Id` evaluated at corner temperature `t_celsius`. Equivalent
/// to `run_id` with `(Vth, kp)` replaced by `(vth_at_temp(vth0, T),
/// kp_at_temp(kp0, T))`.
pub fn run_id_at_temp(
    vgs: f64, vds: f64, vth0: f64, kp0: f64, lam: f64, t_celsius: f64,
) -> f64 {
    run_id(vgs, vds, vth_at_temp(vth0, t_celsius), kp_at_temp(kp0, t_celsius), lam)
}

/// Forward + reverse-mode AD: `(Id, ∂Id/∂Vth, ∂Id/∂kp, ∂Id/∂λ)`.
pub fn run_id_grad(vgs: f64, vds: f64, vth: f64, kp: f64, lam: f64) -> (f64, f64, f64, f64) {
    let (fwd, vth_id, kp_id, lam_id) = build_id_graph();
    let bwd = grad_with_loss(&fwd, &[vth_id, kp_id, lam_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("Vth", &vth.to_le_bytes(), DType::F64);
    compiled.set_param_typed("kp",  &kp.to_le_bytes(),  DType::F64);
    compiled.set_param_typed("lam", &lam.to_le_bytes(), DType::F64);
    let one = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("Vgs", &vgs.to_le_bytes(), DType::F64),
        ("Vds", &vds.to_le_bytes(), DType::F64),
        ("d_output", &one, DType::F64),
    ]);
    (
        decode_f64(&outs[0].0),
        decode_f64(&outs[1].0),
        decode_f64(&outs[2].0),
        decode_f64(&outs[3].0),
    )
}

// ── Strict (non-smooth) analytic reference ─────────────────────────────

/// Strict-piecewise `Id` at corner temperature `t_celsius`. Same remap as
/// `run_id_at_temp`. Used by the thermal-sweep validation pyramid.
pub fn id_strict_at_temp(
    vgs: f64, vds: f64, vth0: f64, kp0: f64, lam: f64, t_celsius: f64,
) -> f64 {
    id_strict(vgs, vds, vth_at_temp(vth0, t_celsius), kp_at_temp(kp0, t_celsius), lam)
}

/// Strict piecewise LEVEL=1 `Id`. Used as the analytic witness in tier-1
/// tests; compared against the rlx smooth `Id` at points well inside
/// each region.
pub fn id_strict(vgs: f64, vds: f64, vth: f64, kp: f64, lam: f64) -> f64 {
    let vov = vgs - vth;
    if vov <= 0.0 {
        0.0
    } else if vds < vov {
        kp * (vov * vds - 0.5 * vds * vds) * (1.0 + lam * vds)
    } else {
        0.5 * kp * vov * vov * (1.0 + lam * vds)
    }
}

/// Closed-form `∂Id/∂Vth` from the strict L1 formula.
/// Useful as an analytic witness on the AD gradient (in saturation).
pub fn analytic_did_dvth_saturation(vgs: f64, vds: f64, vth: f64, kp: f64, lam: f64) -> f64 {
    let vov = vgs - vth;
    // Id = (kp/2) · Vov² · (1 + λ·Vds);  ∂Id/∂Vth = kp · Vov · (1+λVds) · (-1)
    -kp * vov * (1.0 + lam * vds)
}

/// `∂Id/∂kp` in saturation: `Id/kp` since `Id` is linear in `kp`.
pub fn analytic_did_dkp_saturation(vgs: f64, vds: f64, vth: f64, _kp: f64, lam: f64) -> f64 {
    let vov = vgs - vth;
    0.5 * vov * vov * (1.0 + lam * vds)
}

/// `∂Id/∂λ` in saturation: `(kp/2)·Vov²·Vds`.
pub fn analytic_did_dlam_saturation(vgs: f64, vds: f64, vth: f64, kp: f64, _lam: f64) -> f64 {
    let vov = vgs - vth;
    0.5 * kp * vov * vov * vds
}

// ── SPICE deck (cross-simulator) ───────────────────────────────────────

/// Minimal NMOS DC deck: a single `M1` device with two voltage sources
/// (`Vg`, `Vd`) holding the gate and drain at chosen biases. ngspice can
/// then report `i(Vd)` (drain current as seen by the source — flipped
/// sign convention) in an `.op` analysis.
///
/// Uses the `Nmos` primitive in `eda-spice-emit` so the model card and
/// instance line follow the same conventions every other MOSFET-using
/// spike will adopt. Our `kp` parameter folds `W/L` in, so we decompose
/// it into a SPICE `KP` (per-unit-W/L) and a W/L geometry on the M card
/// chosen so `KP·W/L = kp`.
pub fn spice_deck(vgs: f64, vds: f64, vth: f64, kp: f64, lam: f64, w: f64, l: f64) -> String {
    use eda_spice_emit::{MosL1Params, Netlist, Nmos, SpiceEmit};
    let mut n = Netlist::new("NMOS L1 DC operating point (rlx-eda spike)");
    n.add_dc_source("g", "g", "0", vgs);
    n.add_dc_source("d", "d", "0", vds);
    let kp_per_wl = kp / (w / l);
    let nmos = Nmos { w, l, params: MosL1Params { vto: vth, kp: kp_per_wl, lambda: lam } };
    nmos.emit_spice(&mut n, &["d", "g", "0", "0"], "1").unwrap();
    n.deck()
}

/// Same as `spice_deck` but instructs ngspice to evaluate the operating
/// point at `t_celsius` via a `.temp <T>` directive in the preamble, and
/// declares `TNOM=27` on the model card so the device parameters
/// (`vth`, `kp`, `lam`) are interpreted as nominal-T values. ngspice
/// then applies its own LEVEL 1 temperature scaling — mobility
/// `μ(T) ∝ (T/Tnom)^-1.5` and a bandgap-based VTO shift — independently
/// of our analytic remap, so this is a real cross-engine comparison at
/// non-nominal T (not just a deck rewrite).
pub fn spice_deck_at_temp(
    vgs: f64, vds: f64, vth: f64, kp: f64, lam: f64, w: f64, l: f64, t_celsius: f64,
) -> String {
    use eda_spice_emit::{MosL1Params, Netlist, Nmos, SpiceEmit};
    let mut n = Netlist::new("NMOS L1 DC operating point at temperature (rlx-eda spike)");
    n.add_preamble(format!(".temp {t_celsius:.6}"));
    n.add_dc_source("g", "g", "0", vgs);
    n.add_dc_source("d", "d", "0", vds);
    let kp_per_wl = kp / (w / l);
    let nmos = Nmos { w, l, params: MosL1Params { vto: vth, kp: kp_per_wl, lambda: lam } };
    nmos.emit_spice(&mut n, &["d", "g", "0", "0"], "1").unwrap();
    n.deck()
}

fn decode_f64(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    f64::from_le_bytes(buf)
}
