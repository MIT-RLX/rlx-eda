//! Voltage divider as **type-driven blocks** — first move from spike to
//! framework.
//!
//! Where `spike-divider-layout` exposed free functions
//! (`build_resistor_cell`, `build_divider`), this spike encodes the same
//! geometry as `Resistor` and `RcDivider` Rust types implementing
//! `eda_hir::{Block, Layout<P>}`. Composition follows the trait — a
//! parent block calls a child block's `layout` to get its CellId, then
//! instantiates it.
//!
//! ## What's the same vs the free-function version
//!
//! Geometry is identical (same RES/METAL1/VIA1 pattern, same router,
//! same offsets). Test-pyramid is identical. Only the **dispatch** is
//! different: now driven by trait impls instead of named functions.
//!
//! ## What's deliberately not here yet
//!
//! `Schematic<P>` (port-bundle wiring + netlist), `Behavioral` (rlx-MIR
//! fragments) — those are the next architectural pass. This crate only
//! exercises `Block` + `Layout<P>` so the trait shapes get stress-tested
//! before they grow.

use eda_hir::{
    Block, DcBehavioral, Layout, MnaDevice, NonlinearDcBehavioral, SchemOrient, SchemSymbol,
    Schematic, SchematicIr, SymbolKind, TransientStorage,
};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Instance, LayerIndex, Library, Point, Port, PortKindId,
    Rect, Trans, Vec2,
};
use klayout_pdk::pdk;

pub mod pdks;

/// Thermal-corner parameter remap for the LEVEL=1 MOSFET parameters this
/// crate's [`Mosfet`] device exposes. The shape mirrors
/// `spike_mosfet_dc::{vth_at_temp, kp_at_temp}` so the single-device
/// witness and the circuit-level eda-mna path agree on the same physics.
///
/// T-scaling is applied as a **post-hoc remap on the params HashMap**
/// passed to `solve_dc`. Devices register their nominal-T params via
/// [`Mosfet::default_params`]; callers then walk the map through
/// [`thermal::remap_mosfet_params_for_temp`] before solving. This keeps
/// the circuit graph identical at every corner — only the scalar values
/// fed in change — so AD still differentiates only the design knobs.
pub mod thermal {
    use std::collections::HashMap;

    /// Nominal temperature for stdcell parameters, °C. Standard SPICE TNOM.
    pub const T_NOM_C: f64 = 27.0;

    /// Vth linear temperature coefficient, V/°C. Magnitude is the same
    /// for n- and p-MOS in the LEVEL=1 family; polarity-dependent sign
    /// is encoded by the device stamp's existing Vov sign convention,
    /// not here.
    pub const KT1: f64 = -1.0e-3;

    /// Mobility temperature exponent in `μ(T) = μ₀·(T/Tnom)^UTE`. -1.5
    /// is the canonical phonon-scattering value SPICE LEVEL=1 uses by
    /// default.
    pub const UTE: f64 = -1.5;

    #[inline] fn celsius_to_kelvin(t_c: f64) -> f64 { t_c + 273.15 }

    /// `Vth(T) = Vth0 + KT1·(T − Tnom)`.
    #[inline]
    pub fn vth_at_temp(vth0: f32, t_celsius: f64) -> f32 {
        (vth0 as f64 + KT1 * (t_celsius - T_NOM_C)) as f32
    }

    /// `kp(T) = kp0 · (T_K/Tnom_K)^UTE`.
    #[inline]
    pub fn kp_at_temp(kp0: f32, t_celsius: f64) -> f32 {
        let ratio = celsius_to_kelvin(t_celsius) / celsius_to_kelvin(T_NOM_C);
        (kp0 as f64 * ratio.powf(UTE)) as f32
    }

    /// Walk `params` and remap every entry whose key ends in `_Vth` or
    /// `_Kp` through the corresponding `*_at_temp` function. Other
    /// entries (`_Lambda`, `_Gamma`, `_TwoPhiF`, `_N`, `Cgs`, `Cgd`,
    /// node caps, …) are left untouched — λ has weak T-dependence,
    /// 2φF/N are body-effect knobs we don't bend with corner T, and
    /// caps are geometric.
    ///
    /// Idempotent in spirit but **not in math**: calling twice double-
    /// applies the shift. The intended usage is "build the params map
    /// once with `default_params()` then remap once for the corner".
    pub fn remap_mosfet_params_for_temp(
        params: &mut HashMap<String, f32>,
        t_celsius: f64,
    ) {
        for (key, val) in params.iter_mut() {
            if key.ends_with("_Vth") {
                *val = vth_at_temp(*val, t_celsius);
            } else if key.ends_with("_Kp") {
                *val = kp_at_temp(*val, t_celsius);
            }
        }
    }
}

/// Foundry PDKs sourced from the shared `eda-pdks` crate (which generates
/// them at build time from each foundry's `.lyp`). The `RcLikePdk` impls
/// for these structs live below — the trait is local to this crate so
/// the impls can't move.
pub mod pdks_foundry {
    pub use eda_pdks::{Gf180mcu, HAS_GF180MCU, HAS_SKY130, Sky130};
}
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, NodeId, Shape as TensorShape};
use klayout_core::Shape;
// `PolygonWireStylizer` and the routing primitives moved to `eda-pnr`
// in Phase 4 — `RcDivider::layout` below uses `eda_pnr::PnrFlow`
// instead of calling `ManhattanPlanner` / `Stylizer` directly. Any
// downstream consumer that reached for the local `PolygonWireStylizer`
// can pull `eda_pnr::PolygonWireStylizer` instead.

pdk! {
    pub RcDemo {
        dbu: 1000,
        layers: {
            RES    = (50, 0),
            METAL1 = (10, 0),
            VIA1   = (20, 0),
            DIFF   = (51, 0),
            POLY   = (52, 0),
            NWELL  = (53, 0),
            NPLUS  = (54, 0),
            PPLUS  = (55, 0),
        },
        ports: { Electrical },
    }
}

// Geometry constants — DBU (1000 DBU = 1 µm).
const RES_WIDTH: i64 = 1_000;
const PAD: i64       = 2_000;
const VIA: i64       = 500;

// ── PDK abstraction ────────────────────────────────────────────────────

/// Anything that exposes the three layers + the electrical port-kind a
/// rectangular-resistor + METAL1-routed-divider needs. Implemented by
/// `RcDemo` (in this module) and by the realistic-layer-numbers PDKs in
/// [`pdks`] (`Sky130Lite`, `Gf180Lite`). New PDKs slot in by adding one
/// `pdk! { ... }` declaration and a 4-line `impl RcLikePdk`.
///
/// This is the cross-PDK genericity story: write `Resistor` once,
/// reuse the same Rust type under many process flavors.
pub trait RcLikePdk {
    /// Resistive layer (RES / poly / poly2 / nwell, depending on PDK).
    fn res(&self) -> LayerIndex;
    /// First metal layer for contacts + routing.
    fn metal1(&self) -> LayerIndex;
    /// Via from `metal1` down to `res`.
    fn via1(&self) -> LayerIndex;
    /// Electrical-domain port-kind id from the PDK's `ports: { ... }` block.
    fn electrical_kind(&self) -> PortKindId;
}

impl RcLikePdk for RcDemo {
    fn res(&self) -> LayerIndex { self.RES }
    fn metal1(&self) -> LayerIndex { self.METAL1 }
    fn via1(&self) -> LayerIndex { self.VIA1 }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

// `RcLikePdk` impls for the auto-generated foundry PDKs. The trait is
// local; the structs come from `eda-pdks`. Both are in scope here so the
// orphan rule is satisfied. Each impl is one trivial pass-through per
// method — same shape as `RcDemo` above, just over different fields.
impl RcLikePdk for eda_pdks::Sky130 {
    fn res(&self) -> LayerIndex { self.RES }
    fn metal1(&self) -> LayerIndex { self.METAL1 }
    fn via1(&self) -> LayerIndex { self.VIA1 }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

impl RcLikePdk for eda_pdks::Gf180mcu {
    fn res(&self) -> LayerIndex { self.RES }
    fn metal1(&self) -> LayerIndex { self.METAL1 }
    fn via1(&self) -> LayerIndex { self.VIA1 }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

// ── PDK abstraction (MOSFET) ───────────────────────────────────────────

/// Layers + port-kind a 4-terminal MOSFET layout needs: an active
/// (diffusion) region with a poly stripe over it, contacts down to
/// metal1 for D/G/S/B, and an n-well region for PMOS bulk. Each Lite
/// PDK reuses its existing `metal1` / `via1` (the contact cut layer
/// works as via1 to first metal) and adds DIFF + NWELL + a dedicated
/// gate POLY where its `res()` poly is repurposed for resistors.
///
/// The MVP does not require implant (n+/p+) layers — those matter only
/// for DRC and are bolted on once we wire DRC for FET cells. Until
/// then, polarity is encoded in the cell name + bulk port.
pub trait MosfetPdk {
    /// Active (diffusion / oxide-diffusion) layer.
    fn diff(&self) -> LayerIndex;
    /// Gate poly stripe.
    fn poly(&self) -> LayerIndex;
    /// First metal — terminal contact pads.
    fn metal1(&self) -> LayerIndex;
    /// Contact cut from metal1 down to diff/poly.
    fn via1(&self) -> LayerIndex;
    /// N-well — drawn for PMOS, omitted for NMOS.
    fn nwell(&self) -> LayerIndex;
    /// N+ source/drain implant — drawn over diff for NMOS.
    fn nplus(&self) -> LayerIndex;
    /// P+ source/drain implant — drawn over diff for PMOS.
    fn pplus(&self) -> LayerIndex;
    fn electrical_kind(&self) -> PortKindId;
}

impl MosfetPdk for RcDemo {
    fn diff(&self) -> LayerIndex { self.DIFF }
    fn poly(&self) -> LayerIndex { self.POLY }
    fn metal1(&self) -> LayerIndex { self.METAL1 }
    fn via1(&self) -> LayerIndex { self.VIA1 }
    fn nwell(&self) -> LayerIndex { self.NWELL }
    fn nplus(&self) -> LayerIndex { self.NPLUS }
    fn pplus(&self) -> LayerIndex { self.PPLUS }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

// `MosfetPdk` impls for the auto-generated foundry PDKs. Sky130 and
// gf180mcu both use their poly layer for both poly resistors and FET
// gates, so `poly()` aliases `RES` (the same `LayerIndex` either way).
impl MosfetPdk for eda_pdks::Sky130 {
    fn diff(&self)            -> LayerIndex { self.DIFF }
    fn poly(&self)            -> LayerIndex { self.RES }
    fn metal1(&self)          -> LayerIndex { self.METAL1 }
    fn via1(&self)            -> LayerIndex { self.VIA1 }
    fn nwell(&self)           -> LayerIndex { self.NWELL }
    fn nplus(&self)           -> LayerIndex { self.NPLUS }
    fn pplus(&self)           -> LayerIndex { self.PPLUS }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

impl MosfetPdk for eda_pdks::Gf180mcu {
    fn diff(&self)            -> LayerIndex { self.DIFF }
    fn poly(&self)            -> LayerIndex { self.RES }
    fn metal1(&self)          -> LayerIndex { self.METAL1 }
    fn via1(&self)            -> LayerIndex { self.VIA1 }
    fn nwell(&self)           -> LayerIndex { self.NWELL }
    fn nplus(&self)           -> LayerIndex { self.NPLUS }
    fn pplus(&self)           -> LayerIndex { self.PPLUS }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

// ── Resistor primitive ─────────────────────────────────────────────────

/// A rectangular resistor on the RES layer with METAL1 contact pads at
/// each end. `length` sets the body length in DBU.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Resistor {
    /// Body length in DBU.
    pub length: i64,
    /// Diagnostic / cell-name suffix — e.g. `"R1"`. The full cell name
    /// is `"Resistor_<id>_L<length>"`.
    pub id: String,
}

impl Block for Resistor {
    fn name(&self) -> String {
        format!("Resistor_{}_L{}", self.id, self.length)
    }
}

impl<P: RcLikePdk> Layout<P> for Resistor {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let mut b = CellBuilder::new(<Self as Block>::name(self));
        let length = self.length;

        b.add_shape(pdk.res(), Rect::new(Bbox::new(
            Point::new(0, 0), Point::new(length, RES_WIDTH),
        )));

        let cy       = RES_WIDTH / 2;
        let half_pad = PAD / 2;
        let half_via = VIA / 2;
        for &x in &[0_i64, length] {
            b.add_shape(pdk.metal1(), Rect::new(Bbox::new(
                Point::new(x - half_pad, cy - half_pad),
                Point::new(x + half_pad, cy + half_pad),
            )));
            b.add_shape(pdk.via1(), Rect::new(Bbox::new(
                Point::new(x - half_via, cy - half_via),
                Point::new(x + half_via, cy + half_via),
            )));
        }

        let kind = pdk.electrical_kind();
        b.add_port(
            Port::new("a", pdk.metal1(), Point::new(-half_pad, cy), Angle90::W, PAD)
                .with_kind(kind),
        );
        b.add_port(
            Port::new("b", pdk.metal1(), Point::new(length + half_pad, cy), Angle90::E, PAD)
                .with_kind(kind),
        );

        lib.insert(b)
    }
}

/// Resistor's `NonlinearDcBehavioral` impl — same `R` Param as
/// `DcBehavioral`, but exposed as terminal currents for MNA-style
/// assembly. Linear, but uses the same trait that nonlinear devices use.
impl NonlinearDcBehavioral for Resistor {
    fn name(&self) -> String { <Self as Block>::name(self) }
    fn n_terminals(&self) -> usize { 2 }

    fn currents(&self, voltages: &[NodeId], g: &mut Graph) -> Vec<NodeId> {
        debug_assert_eq!(voltages.len(), 2);
        let s = TensorShape::new(&[1], DType::F32);
        let r_node = g.param(<Self as Block>::name(self), s.clone());
        // i_R = (v_a - v_b) / R   (flows a→b inside the device)
        let v_diff = g.binary(BinaryOp::Sub, voltages[0], voltages[1], s.clone());
        let i_r    = g.binary(BinaryOp::Div, v_diff, r_node, s.clone());
        // currents[0] = -i_R  (device takes current from terminal a)
        // currents[1] = +i_R  (device pushes current into terminal b)
        let neg_one = g.add_node(
            rlx_ir::Op::Constant { data: (-1.0_f32).to_le_bytes().to_vec() },
            vec![],
            s.clone(),
        );
        let neg_i_r = g.binary(BinaryOp::Mul, neg_one, i_r, s);
        vec![neg_i_r, i_r]
    }
}

// ── Diode primitive ────────────────────────────────────────────────────

/// Shockley-equation diode. Geometry: a small RES square + two contact
/// pads on METAL1+VIA1 (anode left, cathode right). Behavior: 2-terminal
/// nonlinear via `Is·(exp(V_ab/Vt) − 1)`.
///
/// `is_value` is the saturation current in amps — typical silicon diode
/// is `1e-15`. `Vt` is fixed at room-temperature `kT/q` for the MVP;
/// temperature-dependent diodes are a follow-on.
#[derive(Clone, PartialEq, Debug)]
pub struct Diode {
    /// Side of the square RES region, DBU. Acts as a (visual) "device
    /// area" knob; the saturation current is currently independent —
    /// real PDKs scale `Is` with area.
    pub size: i64,
    /// Saturation current (A). Carried as f32; `Is` becomes a graph
    /// `Param` keyed by `format!("{}_Is", self.name())`.
    pub is_value: f32,
    pub id: String,
}

impl std::hash::Hash for Diode {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.size.hash(h);
        // Hash f32 by bit pattern so structurally-equal Diodes hash equal.
        self.is_value.to_bits().hash(h);
        self.id.hash(h);
    }
}
impl Eq for Diode {}

impl Block for Diode {
    fn name(&self) -> String {
        // Encode is_value's bit pattern so two Diodes with different Is
        // keys still get different names → distinct graph Param slots.
        format!("Diode_{}_S{}_Is{:08x}", self.id, self.size, self.is_value.to_bits())
    }
}

/// Thermal voltage at 300 K (kT/q) — same constant the spike-diode crate uses.
const VT_DIODE: f32 = 0.025_852;

impl<P: RcLikePdk> Layout<P> for Diode {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let mut b = CellBuilder::new(<Self as Block>::name(self));
        let s = self.size;

        // RES square — represents the junction's anode side.
        b.add_shape(pdk.res(), Rect::new(Bbox::new(
            Point::new(0, 0), Point::new(s, s),
        )));

        // METAL1 contact pads at left (anode) and right (cathode) edges.
        let cy        = s / 2;
        let half_pad  = (PAD / 2).min(s / 2);
        let half_via  = VIA / 2;
        for &x in &[0_i64, s] {
            b.add_shape(pdk.metal1(), Rect::new(Bbox::new(
                Point::new(x - half_pad, cy - half_pad),
                Point::new(x + half_pad, cy + half_pad),
            )));
            b.add_shape(pdk.via1(), Rect::new(Bbox::new(
                Point::new(x - half_via, cy - half_via),
                Point::new(x + half_via, cy + half_via),
            )));
        }

        let kind = pdk.electrical_kind();
        b.add_port(
            Port::new("a", pdk.metal1(), Point::new(-half_pad, cy), Angle90::W, PAD)
                .with_kind(kind),
        );
        b.add_port(
            Port::new("b", pdk.metal1(), Point::new(s + half_pad, cy), Angle90::E, PAD)
                .with_kind(kind),
        );

        lib.insert(b)
    }
}

impl NonlinearDcBehavioral for Diode {
    fn name(&self) -> String { <Self as Block>::name(self) }
    fn n_terminals(&self) -> usize { 2 }

    fn currents(&self, voltages: &[NodeId], g: &mut Graph) -> Vec<NodeId> {
        debug_assert_eq!(voltages.len(), 2);
        let s = TensorShape::new(&[1], DType::F32);
        let is_node = g.param(format!("{}_Is", <Self as Block>::name(self)), s.clone());

        let bytes = |x: f32| x.to_le_bytes().to_vec();
        let kc = |g: &mut Graph, x: f32| g.add_node(
            rlx_ir::Op::Constant { data: bytes(x) }, vec![], s.clone(),
        );
        let vt_const = kc(g, VT_DIODE);
        let one      = kc(g, 1.0);
        let neg_one  = kc(g, -1.0);

        // V_ab = v_a - v_b ; I_D = Is·(exp(V_ab/Vt) − 1)
        let v_ab    = g.binary(BinaryOp::Sub, voltages[0], voltages[1], s.clone());
        let v_norm  = g.binary(BinaryOp::Div, v_ab, vt_const, s.clone());
        let exp_v   = g.activation(rlx_ir::op::Activation::Exp, v_norm, s.clone());
        let exp_m_1 = g.binary(BinaryOp::Sub, exp_v, one, s.clone());
        let i_d     = g.binary(BinaryOp::Mul, is_node, exp_m_1, s.clone());

        let neg_i_d = g.binary(BinaryOp::Mul, neg_one, i_d, s);
        vec![neg_i_d, i_d]
    }
}

// ── Voltage source ─────────────────────────────────────────────────────

/// Ideal independent DC voltage source. Two terminals; one branch
/// unknown (the current flowing through it).
///
/// MNA contributions:
/// - Terminal currents: `[+i_VS, −i_VS]` — source pumps `i_VS` into the
///   positive terminal, sinks the same `i_VS` from the negative one.
/// - One branch residual: `v_a − v_b − V_src = 0`.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct VoltageSource {
    /// Source value. Stored as i64-encoded mV so the type can stay
    /// `Hash + Eq` for use as a `Block`.
    pub mv: i64,
    pub id: String,
}

impl VoltageSource {
    pub fn from_volts(v: f32, id: impl Into<String>) -> Self {
        Self { mv: (v * 1000.0).round() as i64, id: id.into() }
    }
    pub fn volts(&self) -> f32 { self.mv as f32 / 1000.0 }
}

impl Block for VoltageSource {
    fn name(&self) -> String { format!("VoltageSource_{}_mV{}", self.id, self.mv) }
}

impl<P: RcLikePdk> Layout<P> for VoltageSource {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let mut b = CellBuilder::new(<Self as Block>::name(self));
        let s = 1_000_i64;
        b.add_shape(pdk.metal1(), Rect::new(Bbox::new(
            Point::new(0, 0), Point::new(s, s),
        )));
        let cy = s / 2;
        let kind = pdk.electrical_kind();
        b.add_port(Port::new("a", pdk.metal1(), Point::new(0, cy), Angle90::W, s).with_kind(kind));
        b.add_port(Port::new("b", pdk.metal1(), Point::new(s, cy), Angle90::E, s).with_kind(kind));
        lib.insert(b)
    }
}

impl MnaDevice for VoltageSource {
    fn name(&self) -> String { <Self as Block>::name(self) }
    fn n_terminals(&self) -> usize { 2 }
    fn n_branches(&self) -> usize { 1 }

    fn contributions(
        &self,
        voltages: &[NodeId],
        branches: &[NodeId],
        g: &mut Graph,
    ) -> (Vec<NodeId>, Vec<NodeId>) {
        debug_assert_eq!(voltages.len(), 2);
        debug_assert_eq!(branches.len(), 1);
        let s = TensorShape::new(&[1], DType::F32);
        let i_vs = branches[0];

        // Terminal currents: [+i_VS at a, −i_VS at b].
        let neg_one = g.add_node(
            rlx_ir::Op::Constant { data: (-1.0_f32).to_le_bytes().to_vec() },
            vec![], s.clone(),
        );
        let neg_i_vs = g.binary(BinaryOp::Mul, neg_one, i_vs, s.clone());
        let terminal_currents = vec![i_vs, neg_i_vs];

        // Branch residual: v_a − v_b − V_src = 0.
        let v_src = g.add_node(
            rlx_ir::Op::Constant { data: self.volts().to_le_bytes().to_vec() },
            vec![], s.clone(),
        );
        let v_diff = g.binary(BinaryOp::Sub, voltages[0], voltages[1], s.clone());
        let residual = g.binary(BinaryOp::Sub, v_diff, v_src, s);
        (terminal_currents, vec![residual])
    }
}

// ── Capacitor primitive ────────────────────────────────────────────────

/// Square-plate capacitor. MVP: a single METAL1 plate with two ports on
/// opposite edges. A real MIM-cap layout has two stacked plates + a
/// dielectric layer between them — that requires extending `RcLikePdk`
/// with `metal2()` / `cap_dielectric()` accessors and is the next
/// architectural step. The MVP shape is enough to validate that the
/// trait abstraction takes a second primitive type without changes.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Capacitor {
    /// Square-plate side length, DBU.
    pub plate_size: i64,
    pub id: String,
}

impl Block for Capacitor {
    fn name(&self) -> String { format!("Capacitor_{}_S{}", self.id, self.plate_size) }
}

/// Linear capacitor's transient storage: returns the `Param` node for
/// capacitance C, keyed by `<Block::name>_C` so multiple instances stay
/// distinct.
impl TransientStorage for Capacitor {
    fn name(&self) -> String { <Self as Block>::name(self) }

    fn capacitance(&self, g: &mut Graph) -> NodeId {
        g.param(
            format!("{}_C", <Self as Block>::name(self)),
            TensorShape::new(&[1], DType::F32),
        )
    }
}

impl<P: RcLikePdk> Layout<P> for Capacitor {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let mut b = CellBuilder::new(<Self as Block>::name(self));
        let s = self.plate_size;
        b.add_shape(pdk.metal1(), Rect::new(Bbox::new(
            Point::new(0, 0), Point::new(s, s),
        )));
        let cy = s / 2;
        let kind = pdk.electrical_kind();
        b.add_port(
            Port::new("a", pdk.metal1(), Point::new(0, cy), Angle90::W, s)
                .with_kind(kind),
        );
        b.add_port(
            Port::new("b", pdk.metal1(), Point::new(s, cy), Angle90::E, s)
                .with_kind(kind),
        );
        lib.insert(b)
    }
}

/// Resistor's DC behavior: expose the principal param (resistance) by
/// the block's name so each instance gets a distinct slot.
///
/// This is the **architectural payoff**: the same `Resistor` instance
/// drives both `Layout<RcDemo>::layout` (for GDS) and
/// `DcBehavioral::add_to_dc` (for rlx simulation). Layout gets the
/// `length` field for the RES geometry; simulation gets the `R` slot
/// keyed by `Block::name()`.
impl DcBehavioral for Resistor {
    fn add_to_dc(&self, g: &mut Graph) -> NodeId {
        g.param(<Self as Block>::name(self), TensorShape::new(&[1], DType::F32))
    }
}

// ── MOSFET primitive (4-terminal square-law) ───────────────────────────

/// NMOS / PMOS polarity flag. The same `Mosfet` Rust type drives both
/// flavors; polarity flips the sign convention (V_GS ↔ V_SG, V_DS ↔
/// V_SD, current direction at D / S) inside `NonlinearDcBehavioral` and
/// adds an `NWELL` shape under the device for PMOS in the layout.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum MosPolarity {
    Nmos,
    Pmos,
}

impl MosPolarity {
    pub fn name(self) -> &'static str {
        match self { MosPolarity::Nmos => "NMOS", MosPolarity::Pmos => "PMOS" }
    }
    /// `+1.0` for NMOS, `−1.0` for PMOS — used to fold polarity into a
    /// single square-law expression.
    pub fn sign(self) -> f32 {
        match self { MosPolarity::Nmos => 1.0, MosPolarity::Pmos => -1.0 }
    }
}

/// Square-law (Shichman-Hodges) MOSFET — the simplest model that
/// captures cutoff / triode / saturation regions. Wrong for sub-100 nm
/// foundries (no short-channel, no V_th roll-off, no body effect, no
/// channel-length modulation), but **correct as the framework probe**:
/// it exercises 4 terminals through `NonlinearDcBehavioral`, exercises
/// `MosfetPdk` through `Layout<P>`, and gives a closed-form oracle for
/// validating the Newton solver.
///
/// Per-instance Param slots, keyed by [`Block::name`]:
/// - `<name>_Kp` — transconductance parameter `μ·C_ox` (A/V²).
/// - `<name>_Vth` — threshold voltage **magnitude** (V); always
///   positive, polarity flips its meaning.
///
/// `W` and `L` are layout-fixed (encoded in the cell name) and folded
/// into the I-V via `K_p · W/L`, treated as constants in the graph so
/// AD targets stay on the two electrical params.
///
/// Terminal order (matches `currents()` / `Layout` ports): `[D, G, S, B]`.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Mosfet {
    pub polarity: MosPolarity,
    /// Behavioral I-V model. `SquareLaw` is the default and the
    /// `..Default::default()` value across the existing tests; switch
    /// to `EkvLite` for instances that need subthreshold conduction
    /// or smooth weak-to-strong-inversion transitions.
    pub model: MosModel,
    /// Channel width (Y dimension of diff), DBU.
    pub w: i64,
    /// Gate length (X dimension of poly over diff), DBU.
    pub l: i64,
    pub id: String,
}

/// I-V model dispatch for `Mosfet::currents()`. Both arms share the
/// same polarity-folding + body-effect + λ machinery; only the core
/// `(V_GS, V_DS) → I_D` map differs.
///
/// **`SquareLaw`** — Shichman-Hodges. Cutoff / triode / saturation
/// piecewise. Wrong for sub-100 nm but accurate enough for textbook
/// hand-analysis and the closed-form Newton oracle.
///
/// **`EkvLite`** — EKV-1.0-style smooth interpolation
/// `I_D = I_S · (ln²(1 + exp(v_p/2)) − ln²(1 + exp((v_p − v_ds)/2)))`
/// with `v_p = (V_GS_eff − V_th_eff) / (n·U_T)` and
/// `v_ds = V_DS_eff / U_T`. Captures subthreshold (exponential I_D)
/// and strong inversion in one closed-form expression — no piecewise
/// branches, so AD gradients are smooth across the transition.
/// Adds one Param: `<name>_N` (slope factor, ≈1–1.6).
///
/// ## Why no `Bsim3` / `Bsim4` / `Ekv26` / `PsP` variants
///
/// **Full industrial models are deliberately out of scope for this
/// hand-coded enum.** BSIM3v3, BSIM4, EKV-2.6 and PSP each carry 100+
/// model parameters and multi-page closed-form expressions with
/// polynomial geometry corrections (DIBL, V_th roll-off, mobility
/// reduction, velocity saturation, GIDL, gate tunneling, …). Hand-
/// transcribing a BSIM4 forward equation into `currents()` would be
/// ~1000 LOC of arithmetic ops *per equation*, with no correctness
/// path other than a paper compare — exactly the wrong spot for a
/// human in the loop.
///
/// The right path is a parser/codegen pass:
///
/// 1. **Source format** — every modern foundry ships its compact
///    model as Verilog-A (`.va`). BSIM4.8.2 is `bsim4.8.2.va`; PSP
///    is `psp103.va`; EKV-2.6 has a reference `.va` distribution.
/// 2. **Parser** — a small Verilog-A frontend (analog-block subset
///    only — `module`, `analog begin`, `branch`, `I(...) <+ ...`)
///    parses the model into an expression tree.
/// 3. **Codegen** — lower the expression tree to `rlx_ir::Graph`
///    with the same `g.binary` / `g.activation` calls used here.
///    The model's named params become `g.param` slots keyed by
///    `<instance>_<param>`, identical to the hand-coded path.
/// 4. **Integration** — a `MosModel::Compact { va_path: PathBuf }`
///    variant (or a separate `CompactMosfet` block type) wires the
///    generated graph into `currents()` without touching the
///    polarity-folding shell.
///
/// Likely lives in a new `eda-vamodel` crate (or extends
/// `eda-pdk-ingest`, since `.va` files travel with `.lyp` files in a
/// foundry PDK release). Until then `EkvLite` is the practical
/// hand-coded ceiling — captures subthreshold + strong-inversion +
/// body effect + CLM in one smooth expression, which is enough for
/// most analog-design loops the framework targets today.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, Default)]
pub enum MosModel {
    #[default]
    SquareLaw,
    EkvLite,
}

impl MosModel {
    /// Short name encoded in `Block::name()` so two `Mosfet` instances
    /// with the same W/L/id but different models keep distinct Param
    /// keys (and distinct cell names if they ever co-exist in a layout).
    pub fn short(self) -> &'static str {
        match self { MosModel::SquareLaw => "SQ", MosModel::EkvLite => "EKV" }
    }
}

// Geometry constants for MOSFET layout (DBU).
const DIFF_EXT: i64        = 1_500;   // S/D diff extension past gate.
const POLY_OVERHANG: i64   = 1_500;   // poly extends past diff each side.
                                       // Bumped from 1000 → 1500 to give 250 nm
                                       // M1 spacing between gate pad and source/
                                       // drain pads (otherwise pads share an
                                       // edge → fatal LVS short).
const NWELL_MARGIN: i64    = 2_000;   // n-well around diff for PMOS.
const IMPLANT_OVERHANG: i64 =   500;   // n+/p+ implant past diff edges.
const MOS_PAD: i64         = 1_500;   // metal1 contact pad side.
const MOS_VIA: i64         = 500;     // contact cut side.
const BODY_TAP_OFFSET: i64 = 2_500;   // body-tap port distance below diff.

impl Block for Mosfet {
    fn name(&self) -> String {
        format!("Mosfet_{}_{}_{}_W{}_L{}",
                self.polarity.name(), self.model.short(),
                self.id, self.w, self.l)
    }
}

impl Mosfet {
    /// NMOS, default `SquareLaw` model.
    pub fn nmos(w: i64, l: i64, id: impl Into<String>) -> Self {
        Self { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw,
               w, l, id: id.into() }
    }

    /// PMOS, default `SquareLaw` model.
    pub fn pmos(w: i64, l: i64, id: impl Into<String>) -> Self {
        Self { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw,
               w, l, id: id.into() }
    }

    /// Builder: switch to a different I-V model. Chains with `nmos` /
    /// `pmos`, e.g. `Mosfet::nmos(1000, 1000, "M1").with_model(MosModel::EkvLite)`.
    pub fn with_model(mut self, model: MosModel) -> Self {
        self.model = model;
        self
    }

    /// Default Param values keyed by this instance's `Block::name()`.
    /// Reduces to bare square-law (λ=γ=0); body-effect √-arg pinned at
    /// 2φF=0.7 V so γ can be scaled up later without re-seeding 2φF.
    /// `_N` (EKV slope factor) defaults to 1 so EKV-lite at strong
    /// inversion lines up exactly with square-law saturation. Use as
    /// the seed of `solve_dc`'s `params` map and override only the
    /// keys you want non-default.
    pub fn default_params(&self) -> std::collections::HashMap<String, f32> {
        let n = <Self as Block>::name(self);
        let mut m = std::collections::HashMap::new();
        m.insert(format!("{n}_Kp"),      100e-6);
        m.insert(format!("{n}_Vth"),     0.5);
        m.insert(format!("{n}_Lambda"),  0.0);
        m.insert(format!("{n}_Gamma"),   0.0);
        m.insert(format!("{n}_TwoPhiF"), 0.7);
        m.insert(format!("{n}_N"),       1.0);
        // Parasitic gate caps consumed by `attach_mosfet_with_caps`.
        // 1 fF is a reasonable order-of-magnitude for default-sized
        // long-channel NMOS — caller can override per instance.
        m.insert(format!("{n}_Cgs"),     1e-15);
        m.insert(format!("{n}_Cgd"),     1e-15);
        m
    }
}

/// Attach a `Mosfet` to a `Circuit` together with its parasitic gate
/// capacitances `C_gs` and `C_gd` (LinearCap storage devices). The
/// MOSFET I-V flows through `add_device` exactly as before; the caps
/// flow through `add_storage` and stamp BE-step companions during
/// transient simulation. At DC the caps contribute nothing — caps are
/// open at DC — so this helper is safe to use even for purely-DC
/// circuits.
///
/// `nets` is the same `[D, G, S, B]` order as `Mosfet::n_terminals()`.
/// Param keys consumed: `<mosfet_name>_Cgs` (G–S) and `<mosfet_name>_Cgd`
/// (G–D), both seeded by [`Mosfet::default_params`] at 1 fF.
///
/// Drain–bulk and source–bulk junction caps (`Cdb` / `Csb`) are
/// **not** added by this helper — those land in a follow-up slice
/// once we want to model body-related delay components. For
/// gate-driven switching transients the gate caps are what dominate.
pub fn attach_mosfet_with_caps(
    c: &mut eda_mna::Circuit,
    mos: Mosfet,
    nets: [eda_mna::NetId; 4],
) {
    let mos_name = <Mosfet as Block>::name(&mos);
    c.add_device(mos.clone(), &nets);
    c.add_storage(
        eda_mna::LinearCap::new(format!("{mos_name}_Cgs")),
        [nets[1], nets[2]],   // G, S
    );
    c.add_storage(
        eda_mna::LinearCap::new(format!("{mos_name}_Cgd")),
        [nets[1], nets[0]],   // G, D
    );
}

impl<P: MosfetPdk> Layout<P> for Mosfet {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let mut b = CellBuilder::new(<Self as Block>::name(self));

        let l = self.l;
        let w = self.w;
        let diff_xmax = l + 2 * DIFF_EXT;

        // Diff (active) — horizontal rectangle covering S/channel/D.
        b.add_shape(pdk.diff(), Rect::new(Bbox::new(
            Point::new(0, 0), Point::new(diff_xmax, w),
        )));

        // Poly stripe over the channel, overhanging diff vertically.
        let poly_x0 = DIFF_EXT;
        let poly_x1 = DIFF_EXT + l;
        let poly_y0 = -POLY_OVERHANG;
        let poly_y1 = w + POLY_OVERHANG;
        b.add_shape(pdk.poly(), Rect::new(Bbox::new(
            Point::new(poly_x0, poly_y0), Point::new(poly_x1, poly_y1),
        )));

        // PMOS: draw n-well covering diff + margin. NMOS: skip (sits in
        // p-substrate).
        if matches!(self.polarity, MosPolarity::Pmos) {
            b.add_shape(pdk.nwell(), Rect::new(Bbox::new(
                Point::new(-NWELL_MARGIN, -NWELL_MARGIN),
                Point::new(diff_xmax + NWELL_MARGIN, w + NWELL_MARGIN),
            )));
        }

        // S/D implant: nplus over diff for NMOS, pplus over diff for PMOS.
        // Slight overhang so the implant clears diff edges (DRC margin); we
        // reuse `IMPLANT_OVERHANG` rather than minting a new constant.
        let implant_layer = match self.polarity {
            MosPolarity::Nmos => pdk.nplus(),
            MosPolarity::Pmos => pdk.pplus(),
        };
        b.add_shape(implant_layer, Rect::new(Bbox::new(
            Point::new(-IMPLANT_OVERHANG, -IMPLANT_OVERHANG),
            Point::new(diff_xmax + IMPLANT_OVERHANG, w + IMPLANT_OVERHANG),
        )));

        // Helper: emit a metal1 pad + via1 cut centered at (cx, cy).
        let half_pad = MOS_PAD / 2;
        let half_via = MOS_VIA / 2;
        let mut add_contact = |layer_via: LayerIndex, cx: i64, cy: i64| {
            b.add_shape(pdk.metal1(), Rect::new(Bbox::new(
                Point::new(cx - half_pad, cy - half_pad),
                Point::new(cx + half_pad, cy + half_pad),
            )));
            b.add_shape(layer_via, Rect::new(Bbox::new(
                Point::new(cx - half_via, cy - half_via),
                Point::new(cx + half_via, cy + half_via),
            )));
        };

        // S contact (west diff), D contact (east diff), G contact (north
        // poly flag), B contact (south of diff — body tap simplified).
        let s_cx = DIFF_EXT / 2;
        let d_cx = diff_xmax - DIFF_EXT / 2;
        let cy_diff = w / 2;
        let g_cx = poly_x0 + l / 2;
        let g_cy = w + POLY_OVERHANG / 2;
        let b_cx = diff_xmax / 2;
        let b_cy = -BODY_TAP_OFFSET;

        add_contact(pdk.via1(), s_cx, cy_diff);
        add_contact(pdk.via1(), d_cx, cy_diff);
        add_contact(pdk.via1(), g_cx, g_cy);
        add_contact(pdk.via1(), b_cx, b_cy);

        let kind = pdk.electrical_kind();
        b.add_port(Port::new("d", pdk.metal1(), Point::new(d_cx, cy_diff), Angle90::E, MOS_PAD)
            .with_kind(kind));
        b.add_port(Port::new("g", pdk.metal1(), Point::new(g_cx, g_cy),    Angle90::N, MOS_PAD)
            .with_kind(kind));
        b.add_port(Port::new("s", pdk.metal1(), Point::new(s_cx, cy_diff), Angle90::W, MOS_PAD)
            .with_kind(kind));
        b.add_port(Port::new("b", pdk.metal1(), Point::new(b_cx, b_cy),    Angle90::S, MOS_PAD)
            .with_kind(kind));

        lib.insert(b)
    }
}

/// `Mosfet` I-V as `NonlinearDcBehavioral`, dispatching on `self.model`
/// (square-law vs EKV-lite).
///
/// **Shared prologue** for both models — polarity folding, body effect
/// on V_th, and channel-length modulation:
/// ```text
///   sign       = +1 (NMOS)  | -1 (PMOS)
///   V_GS_eff   = sign·(v_G - v_S)
///   V_DS_eff   = sign·(v_D - v_S)
///   V_SB_eff   = max(sign·(v_S - v_B), 0)
///   V_th_eff   = V_th0 + γ·(√(2φF + V_SB_eff) − √(2φF))
///   I_signed   = sign · I_mag0(V_GS_eff, V_DS_eff, V_th_eff) · (1 + λ·V_DS_eff)
/// ```
/// Terminal contributions (positive = pushed into the node):
/// `D: −I_signed   G: 0   S: +I_signed   B: 0`.
///
/// **Square-law `I_mag0`:**
/// ```text
///   V_ov      = max(V_GS_eff − V_th_eff, 0)
///   V_DS_clip = clamp(V_DS_eff, 0, V_ov)
///   I_mag0    = K_p · (W/L) · (V_ov · V_DS_clip − ½·V_DS_clip²)
/// ```
///
/// **EKV-lite `I_mag0`** (smooth across cutoff/triode/saturation):
/// ```text
///   v_ov   = V_GS_eff − V_th_eff               (signed; can go < 0)
///   φ_f    = ln(1 + exp( v_ov / (2·n·U_T)))
///   φ_r    = ln(1 + exp((v_ov − V_DS_eff) / (2·n·U_T)))
///   I_S    = 2 · n · K_p · U_T² · (W/L)        (specific current)
///   I_mag0 = I_S · (φ_f² − φ_r²)
/// ```
/// Subthreshold (v_ov ≪ 0): `I_mag0 ≈ I_S·exp(v_ov/(n·U_T))·(1−exp(−V_DS/U_T))`
/// — exponential-in-V_GS, the canonical weak-inversion regime.
/// Strong inversion saturation (v_ov ≫ 2·U_T, V_DS ≫ v_ov): `I_mag0 ≈
/// ½·K_p·(W/L)·v_ov²/n` — square-law saturation scaled by 1/n. Picking
/// `_N=1` makes EKV-lite match square-law saturation exactly at strong
/// inversion.
///
/// Note: f32 `exp` overflows for arguments above ≈87, which translates
/// to `V_GS−V_th ≳ 4.5 V` at U_T=0.026, n=1. Stay below that or move
/// to f64 for very-high-V_GS sweeps.
///
/// Per-instance Param keys: `_Kp`, `_Vth`, `_Lambda`, `_Gamma`,
/// `_TwoPhiF`, `_N`. Use [`Mosfet::default_params`] to seed sensible
/// values that reduce both models to their canonical baselines.
impl NonlinearDcBehavioral for Mosfet {
    fn name(&self) -> String { <Self as Block>::name(self) }
    fn n_terminals(&self) -> usize { 4 }

    fn currents(&self, voltages: &[NodeId], g: &mut Graph) -> Vec<NodeId> {
        debug_assert_eq!(voltages.len(), 4);
        let s = TensorShape::new(&[1], DType::F32);

        let v_d = voltages[0];
        let v_g = voltages[1];
        let v_s = voltages[2];
        let v_b = voltages[3];

        // Per-instance params.
        let n         = <Self as Block>::name(self);
        let kp_node   = g.param(format!("{n}_Kp"),       s.clone());
        let vth0_node = g.param(format!("{n}_Vth"),      s.clone());
        let lam_node  = g.param(format!("{n}_Lambda"),   s.clone());
        let gam_node  = g.param(format!("{n}_Gamma"),    s.clone());
        let tphi_node = g.param(format!("{n}_TwoPhiF"),  s.clone());
        let n_node    = g.param(format!("{n}_N"),        s.clone());

        // Constants.
        let bytes = |x: f32| x.to_le_bytes().to_vec();
        let kc = |g: &mut Graph, x: f32| g.add_node(
            rlx_ir::Op::Constant { data: bytes(x) }, vec![], s.clone(),
        );
        let zero    = kc(g, 0.0);
        let one     = kc(g, 1.0);
        let two     = kc(g, 2.0);
        let half    = kc(g, 0.5);
        let neg_one = kc(g, -1.0);
        let sign_c  = kc(g, self.polarity.sign());
        let wl_c    = kc(g, self.w as f32 / self.l as f32);
        // Thermal voltage at 300 K — the same constant the diode uses.
        let u_t_c   = kc(g, 0.025852);

        // V_GS_eff = sign·(v_G - v_S);  V_DS_eff = sign·(v_D - v_S)
        let vgs       = g.binary(BinaryOp::Sub, v_g, v_s, s.clone());
        let vgs_eff   = g.binary(BinaryOp::Mul, sign_c, vgs, s.clone());
        let vds       = g.binary(BinaryOp::Sub, v_d, v_s, s.clone());
        let vds_eff   = g.binary(BinaryOp::Mul, sign_c, vds, s.clone());

        // V_SB_eff = max(sign·(v_S - v_B), 0) — clamped to ≥ 0 so a
        // forward-biased body just leaves V_th at its zero-bias value.
        let v_sb_raw  = g.binary(BinaryOp::Sub, v_s, v_b, s.clone());
        let v_sb_signed = g.binary(BinaryOp::Mul, sign_c, v_sb_raw, s.clone());
        let v_sb_eff  = g.binary(BinaryOp::Max, v_sb_signed, zero, s.clone());

        // V_th_eff = V_th0 + γ·(√(2φF + V_SB_eff) − √(2φF))
        let two_phi_plus_sb = g.binary(BinaryOp::Add, tphi_node, v_sb_eff, s.clone());
        let sqrt_total      = g.activation(rlx_ir::op::Activation::Sqrt, two_phi_plus_sb, s.clone());
        let sqrt_phi_only   = g.activation(rlx_ir::op::Activation::Sqrt, tphi_node, s.clone());
        let sqrt_diff       = g.binary(BinaryOp::Sub, sqrt_total, sqrt_phi_only, s.clone());
        let body_shift      = g.binary(BinaryOp::Mul, gam_node, sqrt_diff, s.clone());
        let vth_eff         = g.binary(BinaryOp::Add, vth0_node, body_shift, s.clone());

        // Signed overdrive — used by both models, can go negative.
        let vov_signed = g.binary(BinaryOp::Sub, vgs_eff, vth_eff, s.clone());

        let i_mag0 = match self.model {
            MosModel::SquareLaw => {
                // V_ov = max(vov_signed, 0)
                let v_ov     = g.binary(BinaryOp::Max, vov_signed, zero, s.clone());
                // V_DS_clip = clamp(V_DS_eff, 0, V_ov)
                let vds_min  = g.binary(BinaryOp::Min, vds_eff, v_ov, s.clone());
                let vds_clip = g.binary(BinaryOp::Max, vds_min, zero, s.clone());
                // I_mag0 = K_p · W/L · (V_ov · V_DS_clip − ½·V_DS_clip²)
                let term1    = g.binary(BinaryOp::Mul, v_ov, vds_clip, s.clone());
                let vds_sq   = g.binary(BinaryOp::Mul, vds_clip, vds_clip, s.clone());
                let half_sq  = g.binary(BinaryOp::Mul, half, vds_sq, s.clone());
                let inner    = g.binary(BinaryOp::Sub, term1, half_sq, s.clone());
                let kp_wl    = g.binary(BinaryOp::Mul, kp_node, wl_c, s.clone());
                g.binary(BinaryOp::Mul, kp_wl, inner, s.clone())
            }
            MosModel::EkvLite => {
                // 2·n·U_T — denominator of the softplus argument.
                let two_n     = g.binary(BinaryOp::Mul, two, n_node, s.clone());
                let two_n_ut  = g.binary(BinaryOp::Mul, two_n, u_t_c, s.clone());

                // φ_f = ln(1 + exp(vov / (2·n·U_T)))
                let arg_f     = g.binary(BinaryOp::Div, vov_signed, two_n_ut, s.clone());
                let exp_f     = g.activation(rlx_ir::op::Activation::Exp, arg_f, s.clone());
                let one_p_f   = g.binary(BinaryOp::Add, one, exp_f, s.clone());
                let phi_f     = g.activation(rlx_ir::op::Activation::Log, one_p_f, s.clone());
                let phi_f_sq  = g.binary(BinaryOp::Mul, phi_f, phi_f, s.clone());

                // φ_r = ln(1 + exp((vov − V_DS_eff) / (2·n·U_T)))
                let vov_minus_ds = g.binary(BinaryOp::Sub, vov_signed, vds_eff, s.clone());
                let arg_r     = g.binary(BinaryOp::Div, vov_minus_ds, two_n_ut, s.clone());
                let exp_r     = g.activation(rlx_ir::op::Activation::Exp, arg_r, s.clone());
                let one_p_r   = g.binary(BinaryOp::Add, one, exp_r, s.clone());
                let phi_r     = g.activation(rlx_ir::op::Activation::Log, one_p_r, s.clone());
                let phi_r_sq  = g.binary(BinaryOp::Mul, phi_r, phi_r, s.clone());

                let phi_diff_sq = g.binary(BinaryOp::Sub, phi_f_sq, phi_r_sq, s.clone());

                // I_S = 2·n·K_p·U_T²·(W/L) (the EKV "specific current").
                let u_t_sq   = g.binary(BinaryOp::Mul, u_t_c, u_t_c, s.clone());
                let kp_ut_sq = g.binary(BinaryOp::Mul, kp_node, u_t_sq, s.clone());
                let i_s_pre  = g.binary(BinaryOp::Mul, two_n, kp_ut_sq, s.clone());
                let i_s_node = g.binary(BinaryOp::Mul, i_s_pre, wl_c, s.clone());

                g.binary(BinaryOp::Mul, i_s_node, phi_diff_sq, s.clone())
            }
        };

        // CLM: I_mag = I_mag0 · (1 + λ · V_DS_eff)
        let lam_vds   = g.binary(BinaryOp::Mul, lam_node, vds_eff, s.clone());
        let clm_mult  = g.binary(BinaryOp::Add, one, lam_vds, s.clone());
        let i_mag     = g.binary(BinaryOp::Mul, i_mag0, clm_mult, s.clone());

        // I_signed = sign · I_mag
        let i_signed  = g.binary(BinaryOp::Mul, sign_c, i_mag, s.clone());
        let neg_i_signed = g.binary(BinaryOp::Mul, neg_one, i_signed, s.clone());

        // [D=-I_signed, G=0, S=+I_signed, B=0]
        vec![neg_i_signed, zero, i_signed, zero]
    }
}

// ── Divider composite ──────────────────────────────────────────────────

/// Voltage divider: two `Resistor`s connected by a routed METAL1 wire.
/// Geometry parameters set the inter-resistor spacing.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct RcDivider {
    pub r1: Resistor,
    pub r2: Resistor,
    /// Horizontal gap between R1's right pad and R2's left pad, DBU.
    pub gap_x: i64,
    /// Vertical drop of R2 relative to R1, DBU. Negative ⇒ R2 below R1.
    pub gap_y: i64,
}

impl RcDivider {
    /// Default geometry: 5 µm horizontal gap, 3 µm drop — gives the
    /// router a clean L-bend.
    pub fn new(r1: Resistor, r2: Resistor) -> Self {
        Self { r1, r2, gap_x: 5_000, gap_y: -3_000 }
    }
}

impl Block for RcDivider {
    fn name(&self) -> String {
        format!("RcDivider_{}_{}", self.r1.id, self.r2.id)
    }
}

impl<P: RcLikePdk> eda_pnr::Connectivity<P> for RcDivider {
    fn connectivity(&self, lib: &Library, pdk: &P) -> eda_pnr::Netlist {
        // Children → frozen cells the netlist references by CellId.
        let r1_id = self.r1.layout(lib, pdk);
        let r2_id = self.r2.layout(lib, pdk);

        // Connectivity. The order of `connect()` calls matters for
        // external-pin promotion: the harness assigns each external
        // port the position of its net's *first* pin.
        //
        //   net `vmid`   = [R1.b, R2.a]   ← routed; vout exposed at R1.b
        //   net `vin`    = [R1.a]         ← single-pin, exposed only
        //   net `gnd`    = [R2.b]         ← single-pin, exposed only
        let mut nl = eda_pnr::Netlist::new(self.name())
            .with_default_signal_layer(pdk.metal1());
        let i_r1 = nl.add_instance("R1", r1_id);
        let i_r2 = nl.add_instance("R2", r2_id);
        nl.connect("vmid", i_r1, "b");
        nl.connect("vmid", i_r2, "a");
        nl.connect("vin", i_r1, "a");
        nl.connect("gnd", i_r2, "b");
        nl.expose("vin",  "vin",  Some(eda_hir::PinDirection::Input));
        nl.expose("vout", "vmid", Some(eda_hir::PinDirection::Output));
        nl.expose("gnd",  "gnd",  Some(eda_hir::PinDirection::Ground));
        nl
    }

    fn transforms(&self, _: &eda_pnr::Netlist, _: &Library) -> Vec<Trans> {
        vec![
            Trans::IDENTITY,
            Trans::translate(Vec2::new(self.r1.length + self.gap_x, self.gap_y)),
        ]
    }
}

impl<P: RcLikePdk> Layout<P> for RcDivider {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        eda_pnr::pnr_layout(self, lib, pdk)
    }
}

impl RcDivider {
    /// Build the divider's DC closed-form rlx graph and return
    /// `(graph, R1_param_id, R2_param_id)`. Output is `Vout`; inputs
    /// are `V` (driving voltage) and the param slots populated by each
    /// child resistor's `DcBehavioral::add_to_dc`.
    ///
    /// This is the proof that a single Rust block type drives **both**
    /// layout and simulation: `self.r1` and `self.r2` are the same
    /// `Resistor` instances that flowed through `Layout::layout`. Their
    /// `name()` is what keys the rlx-graph `Param` slots. Set
    /// `compiled.set_param("Resistor_R1_L10000", &[r_value])` at run
    /// time — the `L10000` suffix comes straight from the `length`
    /// field used in the layout.
    pub fn build_dc_graph(&self) -> (Graph, NodeId, NodeId) {
        let mut g = Graph::new(format!("{}_dc", self.name()));
        let s = TensorShape::new(&[1], DType::F32);
        let r1 = self.r1.add_to_dc(&mut g);
        let r2 = self.r2.add_to_dc(&mut g);
        let v  = g.input("V", s.clone());

        // Closed form for a 2-resistor divider: vout = V·R2 / (R1+R2).
        let v_r2 = g.binary(BinaryOp::Mul, v,  r2, s.clone());
        let r_sum = g.binary(BinaryOp::Add, r1, r2, s.clone());
        let vout  = g.binary(BinaryOp::Div, v_r2, r_sum, s);

        g.set_outputs(vec![vout]);
        (g, r1, r2)
    }

    /// Run the DC simulation forward and return Vout, along with the
    /// per-block parameter names so the caller knows how to seed them
    /// — `["Resistor_R1_L10000", "Resistor_R2_L30000"]` for the canonical
    /// case.
    pub fn dc_param_names(&self) -> [String; 2] {
        [<Resistor as Block>::name(&self.r1), <Resistor as Block>::name(&self.r2)]
    }

    /// Build the **loss graph** for inverse design: scalar squared-error
    /// `(Vout − target)²` w.r.t. the resistor params.
    ///
    /// Inputs: `V` (driving voltage), `target` (desired Vout).
    /// Params: R1, R2 (per the children's `DcBehavioral` impls).
    /// Output: scalar loss. `grad_with_loss(g, &[r1, r2])` then yields
    /// `[loss, ∂L/∂R1, ∂L/∂R2]`.
    pub fn build_loss_graph(&self) -> (Graph, NodeId, NodeId) {
        let mut g = Graph::new(format!("{}_loss", self.name()));
        let s = TensorShape::new(&[1], DType::F32);
        let r1 = self.r1.add_to_dc(&mut g);
        let r2 = self.r2.add_to_dc(&mut g);
        let v      = g.input("V", s.clone());
        let target = g.input("target", s.clone());
        // Vout = V·R2 / (R1+R2)
        let v_r2  = g.binary(BinaryOp::Mul, v,  r2, s.clone());
        let r_sum = g.binary(BinaryOp::Add, r1, r2, s.clone());
        let vout  = g.binary(BinaryOp::Div, v_r2, r_sum, s.clone());
        // Loss = (Vout − target)²
        let diff = g.binary(BinaryOp::Sub, vout, target, s.clone());
        let loss = g.binary(BinaryOp::Mul, diff, diff, s);
        g.set_outputs(vec![loss]);
        (g, r1, r2)
    }
}

// ── Optimizer trait + impls ────────────────────────────────────────────

/// Pluggable parameter-update rule. Each `step` consumes the current
/// `params` and their `grads` (matching length) and updates `params` in
/// place. Stateful optimizers (Adam, AdamW) keep their own running
/// moments inside `self`.
pub trait Optimizer {
    fn step(&mut self, params: &mut [f32], grads: &[f32]);
}

/// Plain stochastic gradient descent: `θ ← θ − lr · g`.
#[derive(Clone, Copy, Debug)]
pub struct Sgd {
    pub lr: f32,
}
impl Sgd {
    pub fn new(lr: f32) -> Self { Self { lr } }
}
impl Optimizer for Sgd {
    fn step(&mut self, p: &mut [f32], g: &[f32]) {
        debug_assert_eq!(p.len(), g.len());
        for (pi, gi) in p.iter_mut().zip(g) {
            *pi -= self.lr * gi;
        }
    }
}

/// Adam — adaptive moments with bias correction. Per-parameter running
/// mean `m` and variance `v` of gradients let it auto-rescale across
/// parameters that span orders of magnitude in scale (which is exactly
/// the "R in kΩ vs C in nF" situation).
///
/// Reference: Kingma & Ba 2014. Defaults `β₁ = 0.9`, `β₂ = 0.999`,
/// `ε = 1e-8` are the canonical values.
#[derive(Clone, Debug)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}
impl Adam {
    pub fn new(lr: f32, n_params: usize) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        }
    }
}
impl Optimizer for Adam {
    fn step(&mut self, p: &mut [f32], g: &[f32]) {
        debug_assert_eq!(p.len(), self.m.len());
        debug_assert_eq!(p.len(), g.len());
        self.t = self.t.saturating_add(1);
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..p.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g[i] * g[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            p[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// AdamW — Adam with **decoupled** weight decay. Subtracts `λ·θ`
/// directly from the parameter rather than folding it into the gradient
/// (Loshchilov & Hutter 2017). Keeps the moment estimates clean of the
/// regularization term — important when params have heterogeneous
/// magnitudes (resistance values in Ω).
#[derive(Clone, Debug)]
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}
impl AdamW {
    pub fn new(lr: f32, weight_decay: f32, n_params: usize) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        }
    }
}
impl Optimizer for AdamW {
    fn step(&mut self, p: &mut [f32], g: &[f32]) {
        debug_assert_eq!(p.len(), self.m.len());
        debug_assert_eq!(p.len(), g.len());
        self.t = self.t.saturating_add(1);
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..p.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g[i] * g[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            p[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p[i]);
        }
    }
}

// ── Inverse-design loop ────────────────────────────────────────────────

/// Bundled SGD + termination config for the original
/// `optimize_to_target` method. New code should prefer
/// [`RcDivider::optimize_to_target_with`] which takes any
/// [`Optimizer`] (Sgd / Adam / AdamW / future).
pub struct DcOptimizer {
    /// Learning rate (R-update per unit gradient).
    pub lr: f32,
    /// Cap on iterations.
    pub max_iters: usize,
    /// Stop when `loss < tol²`.
    pub tol: f32,
    /// Lower bound on each parameter (resistance ≥ this Ω).
    pub r_min: f32,
}

impl Default for DcOptimizer {
    fn default() -> Self { Self { lr: 1e6, max_iters: 1000, tol: 1e-4, r_min: 1.0 } }
}

#[derive(Debug, Clone)]
pub struct OptimResult {
    pub r1: f32,
    pub r2: f32,
    pub final_loss: f32,
    pub final_vout: f32,
    pub iters: usize,
    pub converged: bool,
}

impl RcDivider {
    /// Inverse-design via plain SGD with the bundled `DcOptimizer`
    /// config. Equivalent to calling [`Self::optimize_to_target_with`]
    /// using `Sgd::new(opt.lr)`. Kept for backward compatibility — new
    /// callers should pick an explicit optimizer.
    pub fn optimize_to_target(
        &self,
        v_in: f32,
        target_vout: f32,
        r1_init: f32,
        r2_init: f32,
        opt: &DcOptimizer,
    ) -> OptimResult {
        let mut sgd = Sgd::new(opt.lr);
        self.optimize_to_target_with(
            v_in, target_vout, r1_init, r2_init,
            &mut sgd,
            opt.max_iters, opt.tol, opt.r_min,
        )
    }

    /// Inverse-design with a caller-supplied [`Optimizer`]. Stops when
    /// `sqrt(loss) < tol` or after `max_iters`. Each parameter is
    /// clamped to `r_min` after every update so resistances stay
    /// physical.
    ///
    /// The compiled gradient graph is built once and re-run per
    /// iteration; the optimizer's own state lives across calls (Adam /
    /// AdamW carry their `m, v, t`).
    pub fn optimize_to_target_with<O: Optimizer>(
        &self,
        v_in: f32,
        target_vout: f32,
        r1_init: f32,
        r2_init: f32,
        opt: &mut O,
        max_iters: usize,
        tol: f32,
        r_min: f32,
    ) -> OptimResult {
        use rlx_opt::autodiff::grad_with_loss;
        use rlx_runtime::{Device, Session};

        let (fwd, r1_id, r2_id) = self.build_loss_graph();
        let bwd = grad_with_loss(&fwd, &[r1_id, r2_id]);
        let mut compiled = Session::new(Device::Cpu).compile(bwd);

        let [n1, n2] = self.dc_param_names();
        let mut params = [r1_init, r2_init];

        for iter in 0..max_iters {
            compiled.set_param(&n1, &[params[0]]);
            compiled.set_param(&n2, &[params[1]]);
            let outs = compiled.run(&[
                ("V",        &[v_in][..]),
                ("target",   &[target_vout][..]),
                ("d_output", &[1.0_f32][..]),
            ]);
            let loss = outs[0][0];
            let grads = [outs[1][0], outs[2][0]];

            if loss.sqrt() < tol {
                let vout = v_in * params[1] / (params[0] + params[1]);
                return OptimResult {
                    r1: params[0], r2: params[1],
                    final_loss: loss, final_vout: vout,
                    iters: iter, converged: true,
                };
            }

            opt.step(&mut params, &grads);
            params[0] = params[0].max(r_min);
            params[1] = params[1].max(r_min);
        }

        let vout = v_in * params[1] / (params[0] + params[1]);
        OptimResult {
            r1: params[0], r2: params[1],
            final_loss: (vout - target_vout).powi(2),
            final_vout: vout,
            iters: max_iters,
            converged: false,
        }
    }
}

// ── Physical synthesis: R ↔ length back-pressure ──────────────────────
//
// The block's `length` field drives layout (RES rectangle); the same
// block's `name()` keys an rlx-graph `Param` slot that the optimizer
// drives (resistance Ω). The bridge is a sheet-rho × width relationship:
//
//     R [Ω]  =  sheet_rho [Ω/sq] × length [µm] / width [µm]
//
// We pick `sheet_rho = 100 Ω/sq` and `width = 1 µm` (matching the
// optimizer test): R = length [µm] × 100 = length [DBU] × 100/1000 =
// length [DBU] / 10.

/// Physical-to-electrical: convert RES body length (DBU) to resistance (Ω).
pub fn length_to_resistance(length_dbu: i64) -> f32 {
    length_dbu as f32 / 10.0
}

/// Electrical-to-physical: convert resistance (Ω) to RES body length (DBU).
/// Rounds to the nearest DBU and clamps to a 100 nm minimum so
/// optimization can't shrink the resistor below realizable geometry.
pub fn resistance_to_length(r_ohm: f32) -> i64 {
    ((r_ohm * 10.0).round() as i64).max(100)
}

impl RcDivider {
    /// **Inverse layout**: optimize R1, R2 to hit `target_vout`, then
    /// regenerate the layout with the converged R values mapped back to
    /// physical body lengths.
    ///
    /// Returns `(OptimResult, new_RcDivider, top_CellId)`. The new
    /// `RcDivider` carries the converged `length` fields — its layout
    /// matches the simulation that achieves the target. Round-tripping
    /// through `length_to_resistance` confirms the parameters survive.
    pub fn optimize_and_relayout<P: RcLikePdk, O: Optimizer>(
        &self,
        v_in: f32,
        target_vout: f32,
        opt: &mut O,
        max_iters: usize,
        tol: f32,
        r_min: f32,
        lib: &Library,
        pdk: &P,
    ) -> (OptimResult, RcDivider, CellId) {
        let r1_init = length_to_resistance(self.r1.length);
        let r2_init = length_to_resistance(self.r2.length);

        let res = self.optimize_to_target_with(
            v_in, target_vout, r1_init, r2_init,
            opt, max_iters, tol, r_min,
        );

        let new_div = RcDivider::new(
            Resistor { length: resistance_to_length(res.r1), id: self.r1.id.clone() },
            Resistor { length: resistance_to_length(res.r2), id: self.r2.id.clone() },
        );
        let top = new_div.layout(lib, pdk);
        (res, new_div, top)
    }
}

// ── Convenience: build a fresh layout in one call ──────────────────────

/// Build a fresh `Library` + `RcDemo` PDK and lay out a `RcDivider` with
/// the given resistor lengths. Returns `(library, pdk, top_cell_id)`.
pub fn make_divider_layout(r1_len: i64, r2_len: i64) -> (Library, RcDemo, CellId) {
    let lib = RcDemo::new_library("rc_demo");
    let pdk = RcDemo::register(&lib);
    let div = RcDivider::new(
        Resistor { length: r1_len, id: "R1".into() },
        Resistor { length: r2_len, id: "R2".into() },
    );
    let top = div.layout(&lib, &pdk);
    (lib, pdk, top)
}

// ── Schematic<P> impls ─────────────────────────────────────────────────
//
// Mirror of the `Layout<P>` impls above: each block declares its
// schematic glyph (and value text) the same way it declares its
// layout. A composite block (`RcDivider`) glues children together by
// translating each child's `SchematicIr` and merging.
//
// The `P: RcLikePdk` bound is the same as `Layout<RcDemo>` etc. — we
// don't read PDK-specific glyph variants in the MVP, but the slot is
// reserved for foundry PDKs where (e.g.) NMOS_LV vs NMOS_HV draw
// different symbols.

impl<P: RcLikePdk> Schematic<P> for Resistor {
    fn schematic(&self, _pdk: &P) -> SchematicIr {
        // 2-terminal resistor at origin, vertical orientation by
        // default — the parent block transforms the IR to wherever it
        // should sit. The label is the block's id (`"R1"`, `"R2"`);
        // value text encodes the layout-derived resistance.
        let r_ohm = length_to_resistance(self.length);
        let mut ir = SchematicIr::new();
        // Leaf-block schematics leave pin_nets empty: the parent block
        // (which knows the routing) populates them after composing the
        // child's IR via translate / merge.
        ir.add_symbol(
            SchemSymbol::new(self.id.clone(), SymbolKind::Resistor, (0.0, 0.0), SchemOrient::Vertical)
                .with_value(format!("{:.1} Ω", r_ohm))
        );
        ir
    }
}

impl<P: RcLikePdk> Schematic<P> for Capacitor {
    fn schematic(&self, _pdk: &P) -> SchematicIr {
        let mut ir = SchematicIr::new();
        ir.add_symbol(SchemSymbol::new(
            self.id.clone(),
            SymbolKind::Capacitor,
            (0.0, 0.0),
            SchemOrient::Vertical,
        ));
        ir
    }
}

impl<P: RcLikePdk> Schematic<P> for Diode {
    fn schematic(&self, _pdk: &P) -> SchematicIr {
        let mut ir = SchematicIr::new();
        ir.add_symbol(
            SchemSymbol::new(self.id.clone(), SymbolKind::Diode, (0.0, 0.0), SchemOrient::Vertical)
                .with_value(format!("Is = {:.0e} A", self.is_value))
        );
        ir
    }
}

/// Mosfet picks its glyph from `polarity` (`SymbolKind::Nmos` /
/// `SymbolKind::Pmos`). Value text encodes the layout-fixed W/L; per-
/// instance electrical params (Kp/Vth/λ/γ/2φF) live on the simulation
/// graph and aren't re-encoded here — the renderer would clutter
/// otherwise. PDK bound is `MosfetPdk` so the impl lines up with the
/// `Layout<P: MosfetPdk>` impl above; foundry PDKs that differentiate
/// (e.g. NMOS_LV vs NMOS_HV) can branch on `pdk` later.
impl<P: MosfetPdk> Schematic<P> for Mosfet {
    fn schematic(&self, _pdk: &P) -> SchematicIr {
        let mut ir = SchematicIr::new();
        let kind = match self.polarity {
            MosPolarity::Nmos => SymbolKind::Nmos,
            MosPolarity::Pmos => SymbolKind::Pmos,
        };
        ir.add_symbol(
            SchemSymbol::new(self.id.clone(), kind, (0.0, 0.0), SchemOrient::Vertical)
                .with_value(format!("W/L = {}/{}", self.w, self.l))
        );
        ir
    }
}

impl<P: RcLikePdk> Schematic<P> for RcDivider {
    fn schematic(&self, pdk: &P) -> SchematicIr {
        // Layout (schematic units, y up). Two design rules drive the
        // placement:
        //   1. Both grounds share the same y so the schematic has a
        //      visible "ground rail" at the bottom. Without this, the
        //      V symbol (short body) and the R1+R2 stack (tall) finish
        //      at different y-coordinates and look misaligned.
        //   2. The vin net label sits on the wire midpoint, not at a
        //      port endpoint — net labels conventionally label a wire,
        //      not a pin.
        //
        //   R1 vertical at (6, 6) → pins (6, 8) top, (6, 4) bot
        //   R2 vertical at (6, 2) → pins (6, 4) top, (6, 0) bot
        //   V  vertical at (0, 6) → pins (0, 8) top, (0, 4) bot
        //   GND_V below V  at (0, -2) (single pin at (0, -1))
        //   GND_R below R2 at (6, -2) (single pin at (6, -1))
        //
        // The 5-unit wire on V's bottom (y=4 → y=-1) mirrors the
        // R1+R2 stack on the right, so the diagram reads as two
        // balanced columns sitting on a shared ground.
        // Children's IRs — translated, then their pin_nets are filled
        // in by the parent here (the resistor itself doesn't know what
        // nets its terminals end up on; that's the parent's contract).
        let mut r1_ir = <Resistor as Schematic<P>>::schematic(&self.r1, pdk).translate(6.0, 6.0);
        for sym in &mut r1_ir.symbols {
            // Resistor pins(): [(-2, 0), (2, 0)] in horizontal-default
            // frame → [top, bot] when vertical. R1 top → vin, R1 bot → vout.
            sym.pin_nets = vec![Some("vin".into()), Some("vout".into())];
        }
        let mut r2_ir = <Resistor as Schematic<P>>::schematic(&self.r2, pdk).translate(6.0, 2.0);
        for sym in &mut r2_ir.symbols {
            sym.pin_nets = vec![Some("vout".into()), Some("gnd".into())];
        }

        let mut ir = SchematicIr::new().with_title("Voltage divider");

        // V source: pin 0 (positive) on vin, pin 1 (negative) on gnd.
        ir.add_symbol(
            SchemSymbol::new("V", SymbolKind::Vsource, (0.0, 6.0), SchemOrient::Vertical)
                .with_value("5 V")
                .with_pin_nets([Some("vin"), Some("gnd")])
        );
        // Both grounds at y=-2 (shared rail). Single-pin symbol → "gnd".
        ir.add_symbol(
            SchemSymbol::new("GND", SymbolKind::Ground, (0.0, -2.0), SchemOrient::default())
                .with_pin_nets([Some("gnd")])
        );
        ir.add_symbol(
            SchemSymbol::new("GND", SymbolKind::Ground, (6.0, -2.0), SchemOrient::default())
                .with_pin_nets([Some("gnd")])
        );

        ir = ir.merge(r1_ir).merge(r2_ir);

        // Wires:
        //   vin: V top (0, 8) → R1 top (6, 8)
        //   gnd_v: V bot (0, 4) → GND_V pin (0, -1) — 5-unit drop
        //   gnd_r: R2 bot (6, 0) → GND_R pin (6, -1)
        //   vout: short stub from the R1/R2 junction (6, 4) right.
        ir.add_wire([(0.0, 8.0), (6.0, 8.0)],   Some("vin".into()));
        ir.add_wire([(6.0, 4.0), (7.6, 4.0)],   Some("vout".into()));
        ir.add_wire([(0.0, 4.0), (0.0, -1.0)],  None);
        ir.add_wire([(6.0, 0.0), (6.0, -1.0)],  None);

        // Top-level ports. `vin` sits at the wire midpoint so the
        // label reads as a net label, not as a pin endpoint. `vout`
        // and `gnd` sit at their natural pins.
        ir.add_port("vin",  (3.0, 8.0));
        ir.add_port("vout", (7.6, 4.0));
        ir.add_port("gnd",  (6.0, -1.0));

        ir
    }
}
