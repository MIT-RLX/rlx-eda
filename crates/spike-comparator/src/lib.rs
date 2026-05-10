//! Behavioral comparator: smoothed-tanh in rlx, behavioral B-source in
//! ngspice/LTspice. The two engines compute the **same closed-form
//! expression** — this spike validates that rlx's AD on `Activation::Tanh`
//! composes correctly through scaling and offset, and that the resulting
//! gradient surface is what a SAR-loop optimizer would expect to see.
//!
//! ## Why behavioral first
//!
//! A regenerative-latch CMOS comparator is a deep transistor-level
//! block — preamp + latch pair + reset switches + clock — and brings
//! its own metastability / kickback / mismatch concerns. We want SAR
//! ADC top-level integration testable *now*, with a stand-in analog
//! block that's:
//!
//! - **Differentiable:** the tanh smoothing makes `dv_out/d(v+ − v−)`
//!   well-defined everywhere instead of spiking at the threshold.
//! - **Cross-simulator validatable:** ngspice's `B`-source and
//!   LTspice's `arbitrary value` source both accept the same
//!   `tanh`-based expression, so DC + transient comparison is
//!   straightforward.
//! - **Easy to swap:** when the real comparator lands, callers replace
//!   the `Comparator` block with the latched version and the SAR
//!   harness is unchanged.
//!
//! ## The transfer function
//!
//! ```text
//!     vout(v+, v−) = vol + (voh − vol) · ½·(1 + tanh(k · (v+ − v−)))
//! ```
//!
//! As `k → ∞` this approaches an ideal step from `vol` to `voh`. At
//! `k = 100/V`, the 10–90 % transition spans `~ 22 mV`, far below
//! typical analog noise margins. Smaller `k` gives a softer transition
//! useful for smoother gradients during early-stage optimization.
//!
//! Differential rejection: `vout` depends only on `v+ − v−`, so a
//! common-mode shift on both inputs leaves the output unchanged (in
//! the ideal model — real comparators have finite CMRR).

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

fn scalar() -> Shape { Shape::new(&[1], DType::F64) }
fn const_scalar(g: &mut Graph, x: f64) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar())
}

/// Behavioral comparator parameters.
///
/// Defaults: 1.8 V rails (matches `spike-cmos-gates` Vdd convention),
/// gain `k = 1000 / V` ⇒ ~2 mV 10–90 transition. That's tight enough
/// to behave as a "rail-to-rail step" in SAR-loop tests but smooth
/// enough that AD gradients are well-conditioned.
#[derive(Debug, Clone, Copy)]
pub struct Comparator {
    pub k:   f64,
    pub voh: f64,
    pub vol: f64,
}

impl Default for Comparator {
    fn default() -> Self { Self { k: 1000.0, voh: 1.8, vol: 0.0 } }
}

/// Insert the comparator's vout subgraph into `g`. All five inputs are
/// graph nodes (so this composes into a larger SAR ADC graph). Returns
/// the `vout` NodeId.
pub fn vout_subgraph(
    g: &mut Graph,
    v_plus: NodeId, v_minus: NodeId,
    k: NodeId, voh: NodeId, vol: NodeId,
) -> NodeId {
    let half = const_scalar(g, 0.5);
    let one  = const_scalar(g, 1.0);

    let diff   = g.binary(BinaryOp::Sub, v_plus, v_minus, scalar());
    let scaled = g.binary(BinaryOp::Mul, k, diff, scalar());
    let t      = g.activation(Activation::Tanh, scaled, scalar());
    let one_p  = g.binary(BinaryOp::Add, one, t, scalar());
    let half_one_p = g.binary(BinaryOp::Mul, half, one_p, scalar());

    let span   = g.binary(BinaryOp::Sub, voh, vol, scalar());
    let scaled_span = g.binary(BinaryOp::Mul, span, half_one_p, scalar());
    g.binary(BinaryOp::Add, vol, scaled_span, scalar())
}

/// Build the standalone vout graph. Inputs (set per call): `v_plus`,
/// `v_minus`. Params: `k`, `voh`, `vol`. Returns `(graph, k, voh, vol)`.
pub fn build_vout_graph() -> (Graph, NodeId, NodeId, NodeId) {
    let mut g = Graph::new("comparator_vout");
    let v_plus  = g.input("v_plus",  scalar());
    let v_minus = g.input("v_minus", scalar());
    let k   = g.param("k",   scalar());
    let voh = g.param("voh", scalar());
    let vol = g.param("vol", scalar());
    let vout = vout_subgraph(&mut g, v_plus, v_minus, k, voh, vol);
    g.set_outputs(vec![vout]);
    (g, k, voh, vol)
}

/// Forward only.
pub fn run_vout(v_plus: f64, v_minus: f64, k: f64, voh: f64, vol: f64) -> f64 {
    let (graph, _, _, _) = build_vout_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.set_param_typed("k",   &k.to_le_bytes(),   DType::F64);
    compiled.set_param_typed("voh", &voh.to_le_bytes(), DType::F64);
    compiled.set_param_typed("vol", &vol.to_le_bytes(), DType::F64);
    let outs = compiled.run_typed(&[
        ("v_plus",  &v_plus.to_le_bytes(),  DType::F64),
        ("v_minus", &v_minus.to_le_bytes(), DType::F64),
    ]);
    decode_f64(&outs[0].0)
}

/// Forward + reverse-mode AD: `(vout, ∂vout/∂k, ∂vout/∂voh, ∂vout/∂vol)`.
///
/// We differentiate w.r.t. the design parameters because those are
/// what an SAR-loop optimizer would tune. `∂vout/∂(v+ − v−)` is also
/// useful (slope at the operating point) but lives outside the param
/// set — for that, hold k/voh/vol fixed and FD on v_plus.
pub fn run_vout_grad(v_plus: f64, v_minus: f64, k: f64, voh: f64, vol: f64)
    -> (f64, f64, f64, f64)
{
    let (fwd, k_id, voh_id, vol_id) = build_vout_graph();
    let bwd = grad_with_loss(&fwd, &[k_id, voh_id, vol_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("k",   &k.to_le_bytes(),   DType::F64);
    compiled.set_param_typed("voh", &voh.to_le_bytes(), DType::F64);
    compiled.set_param_typed("vol", &vol.to_le_bytes(), DType::F64);
    let one = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("v_plus",   &v_plus.to_le_bytes(),  DType::F64),
        ("v_minus",  &v_minus.to_le_bytes(), DType::F64),
        ("d_output", &one,                   DType::F64),
    ]);
    (
        decode_f64(&outs[0].0),
        decode_f64(&outs[1].0),
        decode_f64(&outs[2].0),
        decode_f64(&outs[3].0),
    )
}

// ── Closed-form analytic references ────────────────────────────────────

/// Pure-Rust evaluation of the same closed-form. Used as the analytic
/// witness on the rlx forward.
pub fn vout_smooth(v_plus: f64, v_minus: f64, k: f64, voh: f64, vol: f64) -> f64 {
    vol + (voh - vol) * 0.5 * (1.0 + (k * (v_plus - v_minus)).tanh())
}

/// Ideal step (k → ∞). Returns `voh` if `v+ > v−`, `vol` if `v+ < v−`,
/// midpoint at exactly equal. Useful as the "intended" comparator
/// behavior — the smooth model approaches this as k grows.
pub fn vout_ideal(v_plus: f64, v_minus: f64, voh: f64, vol: f64) -> f64 {
    use std::cmp::Ordering;
    match v_plus.partial_cmp(&v_minus).unwrap_or(Ordering::Equal) {
        Ordering::Greater => voh,
        Ordering::Less    => vol,
        Ordering::Equal   => 0.5 * (voh + vol),
    }
}

/// `∂vout/∂k` at a given operating point (closed form).
/// `vout = vol + (voh − vol) · ½·(1 + tanh(k·Δv))`, so
/// `∂vout/∂k = (voh − vol) · ½ · sech²(k·Δv) · Δv`.
pub fn analytic_dvout_dk(v_plus: f64, v_minus: f64, k: f64, voh: f64, vol: f64) -> f64 {
    let dv = v_plus - v_minus;
    let t = (k * dv).tanh();
    0.5 * (voh - vol) * (1.0 - t * t) * dv
}

/// `∂vout/∂voh = ½·(1 + tanh(k·Δv))`.
pub fn analytic_dvout_dvoh(v_plus: f64, v_minus: f64, k: f64, _voh: f64, _vol: f64) -> f64 {
    0.5 * (1.0 + (k * (v_plus - v_minus)).tanh())
}

/// `∂vout/∂vol = ½·(1 − tanh(k·Δv))`.
pub fn analytic_dvout_dvol(v_plus: f64, v_minus: f64, k: f64, _voh: f64, _vol: f64) -> f64 {
    0.5 * (1.0 - (k * (v_plus - v_minus)).tanh())
}

/// `∂vout/∂(v+) = (voh − vol) · ½ · sech²(k·Δv) · k`. The slope at the
/// operating point — the comparator's effective "small-signal gain".
pub fn analytic_dvout_dvplus(v_plus: f64, v_minus: f64, k: f64, voh: f64, vol: f64) -> f64 {
    let t = (k * (v_plus - v_minus)).tanh();
    0.5 * (voh - vol) * (1.0 - t * t) * k
}

// ── SPICE deck (cross-simulator) ───────────────────────────────────────

/// Build a deck with two DC sources holding `v+` and `v−` at chosen
/// values, plus a `B`-source emitting the same closed-form expression
/// on the `out` net. ngspice and LTspice both accept the syntax.
///
/// We deliberately don't go through `eda-spice-emit::SpiceEmit` because
/// the comparator's "instance" is a single `B` line — there's no
/// 4-terminal device structure to amortize. If we add a transistor-
/// level comparator later, it'll have its own SpiceEmit impl.
pub fn spice_deck(v_plus: f64, v_minus: f64, comp: &Comparator) -> String {
    use eda_spice_emit::Netlist;
    let mut net = Netlist::new("Comparator behavioral B-source (rlx-eda spike)");
    net.add_dc_source("p", "vplus",  "0", v_plus);
    net.add_dc_source("n", "vminus", "0", v_minus);
    // ngspice/LTspice B-source: `Bname p n V=<expr>`. tanh() is in both.
    net.add_element(format!(
        "B1 out 0 V={vol} + ({voh}-{vol})*0.5*(1+tanh({k}*(v(vplus)-v(vminus))))",
        vol = comp.vol, voh = comp.voh, k = comp.k,
    ));
    net.deck()
}

fn decode_f64(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    f64::from_le_bytes(buf)
}

// ── Transistor-level companion: 5-T differential pair ─────────────────

/// CMOS differential-pair comparator — the transistor-level companion
/// to the behavioral [`Comparator`]. Five MOSFETs:
///
/// ```text
///                  vdd
///              ┌────┴────┐
///              │         │
///         Mp1 (diode)   Mp2 (mirror)
///              │         │
///              ├── node_a (mirror reference)
///              │         │
///              │         └── out
///              │
///         ┌────┴────┐
///         │         │
///        Mn1       Mn2     ← NMOS input pair, gates = vin_p / vin_n
///         │         │
///         └────┬────┘
///              │ tail
///         Mtail (gate = vbias)     ← tail current source
///              │
///             gnd
/// ```
///
/// Operation: when `vin_p > vin_n`, `Mn1` carries more of the tail
/// current. The PMOS mirror (Mp1 diode → Mp2) copies that current up
/// to the output side; `Mn2` sinks less than the mirror sources, so
/// `out` rises. `vin_p < vin_n`: opposite direction, `out` falls.
///
/// This is the simplest-possible CMOS comparator — single-ended
/// output, no clock, no regenerative latch. Adequate for SAR-ADC
/// prototyping at low speed and for `.op` cross-validation against
/// the behavioral comparator. A clocked StrongARM latch is a follow-
/// up slice once the SAR top-level needs sub-ns metastability
/// behavior.
///
/// Net order: `[vin_p, vin_n, vbias, out, vdd, gnd]`. `vbias` sets
/// the tail current; ~0.7 V at default LEVEL=1 NMOS Vto=0.5 gives a
/// ~5 µA tail.
#[derive(Debug, Clone, Copy)]
pub struct CmosComparator {
    pub n_input: eda_spice_emit::Nmos,
    pub n_tail:  eda_spice_emit::Nmos,
    pub p_load:  eda_spice_emit::Pmos,
}

impl Default for CmosComparator {
    fn default() -> Self {
        use eda_spice_emit::primitives::{Nmos, Pmos};
        Self {
            n_input: Nmos::default(),
            n_tail:  Nmos::default(),
            p_load:  Pmos::default(),
        }
    }
}

impl eda_spice_emit::SpiceEmit for CmosComparator {
    fn n_terminals(&self) -> usize { 6 }
    fn emit_spice(
        &self,
        net: &mut eda_spice_emit::Netlist,
        nets: &[&str],
        id: &str,
    ) -> Result<(), eda_spice_emit::EmitError> {
        if nets.len() != self.n_terminals() {
            return Err(eda_spice_emit::EmitError::ArityMismatch {
                block: "CmosComparator".into(),
                expected: self.n_terminals(),
                got: nets.len(),
            });
        }
        let (vp, vn, vbias, out, vdd, gnd) =
            (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5]);
        let tail = format!("{id}_tail");
        let node_a = format!("{id}_a");

        // PMOS mirror — Mp1 diode-connected (G=D=node_a), Mp2 follows.
        self.p_load.emit_spice(
            net, &[&node_a, &node_a, vdd, vdd], &format!("{id}_pa"))?;
        self.p_load.emit_spice(
            net, &[out,     &node_a, vdd, vdd], &format!("{id}_pb"))?;

        // NMOS input pair — sources tied to tail.
        self.n_input.emit_spice(
            net, &[&node_a, vp, &tail, gnd], &format!("{id}_na"))?;
        self.n_input.emit_spice(
            net, &[out,     vn, &tail, gnd], &format!("{id}_nb"))?;

        // Tail NMOS — D=tail, G=vbias, S=gnd.
        self.n_tail.emit_spice(
            net, &[&tail, vbias, gnd, gnd], &format!("{id}_nt"))?;

        Ok(())
    }
}

#[cfg(test)]
mod cmos_tests {
    use super::*;
    use eda_spice_emit::{Netlist, SpiceEmit};

    #[test]
    fn n_terminals_is_six() {
        assert_eq!(CmosComparator::default().n_terminals(), 6);
    }

    #[test]
    fn emit_produces_five_devices() {
        let mut net = Netlist::new("t");
        CmosComparator::default()
            .emit_spice(&mut net, &["vp", "vn", "vbias", "out", "vdd", "0"], "u1")
            .unwrap();
        // 2 PMOS + 2 NMOS input + 1 NMOS tail = 5 element lines.
        assert_eq!(net.body.len(), 5);
        let body = net.body.join("\n");
        // Internal nets present.
        assert!(body.contains("u1_a"),    "node_a missing");
        assert!(body.contains("u1_tail"), "tail missing");
        // PMOS mirror reference: M1pa drain == gate (diode-connected).
        assert!(body.contains("u1_pa") && body.contains("u1_pb"));
    }

    #[test]
    fn arity_mismatch_caught() {
        let mut net = Netlist::new("t");
        let err = CmosComparator::default()
            .emit_spice(&mut net, &["only_one"], "u1")
            .unwrap_err();
        assert!(matches!(err,
            eda_spice_emit::EmitError::ArityMismatch { expected: 6, got: 1, .. }));
    }
}
