//! `spike-lna` — RF counterpart of `spike-divider-block` and
//! `spike-waveguide-block`. An inductively-degenerated cascode CMOS LNA
//! at ~2.4 GHz, with closed-form 2-port S-parameters on the rlx graph
//! and a multi-PDK floorplan layout.
//!
//! Three pieces:
//!
//! 1. The [`RfPdk`] trait — minimum layer + port-kind contract every
//!    CMOS process needs to host the LNA's primitives. Mirrors
//!    `RcLikePdk` / `MosfetPdk` in `spike-divider-block` and
//!    `OpticalPdk` in `spike-waveguide-block`.
//!
//! 2. Two layout primitives — [`Mosfet`] (single-finger NMOS) and
//!    [`SpiralInductor`] (square-spiral on the PDK's `metal_top` layer).
//!    Both `Block` + `Layout<P: RfPdk>`. Inductance is computed from
//!    the Mohan square-spiral closed form so geometry-side and
//!    behavioral-side stay consistent.
//!
//! 3. The [`Lna`] block — a 2-stage cascode with three inductors
//!    (`Lg` gate, `Ls` source-degeneration, `Ld` drain-tank) wrapping
//!    two stacked `Mosfet`s. Implements [`RfScattering`], which builds
//!    `S₁₁(ω)` and `|S₂₁|(ω₀)` on the rlx graph from the Razavi §5.3.3
//!    inductive-degeneration small-signal model.
//!
//! ## Why this is the canonical RF spike
//!
//! The inductively-degenerated cascode LNA is *the* "hello world" of
//! RFIC — Razavi ch. 5, Lee ch. 11, and every 2.4 GHz / 5.8 GHz
//! receiver paper since the late 1990s reduces the input matching
//! problem to:
//!
//! ```text
//!   Z_in(ω) = jω(Lg + Ls) + 1/(jω·Cgs) + (gm·Ls)/Cgs
//!   match:  Re Z_in = Z₀         Im Z_in = 0
//!     →     gm·Ls/Cgs = Z₀       ω₀² (Lg + Ls) Cgs = 1
//! ```
//!
//! That makes "drive `|S₁₁(2.4 GHz)|²` to zero by tuning `Lg`" a
//! one-parameter inverse-design problem with a closed-form optimum —
//! the RF analog of "drop a notch onto `λ_target` by tuning `n_eff_A`"
//! in `spike-waveguide-block::Mzi`.

use eda_hir::{Block, Layout, PinDirection};
use eda_pnr::{Connectivity, Netlist};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, LayerIndex, Library, Path, Point, Port, PortKindId, Rect,
    Shape, Trans, Vec2,
};
use klayout_pdk::pdk;
use rlx_ir::{op::BinaryOp, DType, Graph, NodeId, Op, Shape as TensorShape};

// ── PDK abstraction ────────────────────────────────────────────────────

/// Layers + port-kind a CMOS LNA needs: a FET gate stack
/// (`diff` / `poly` / `nplus` / `pplus` / `nwell`), first-metal
/// routing (`metal1`, `via1`), and a top-metal layer for the spiral
/// inductors (`metal_top`).
///
/// `metal_top` is conceptually the thick, low-resistance upper metal
/// (M5 / M6 in real foundries). PDKs that don't yet expose a distinct
/// upper metal alias `metal_top` to `metal1`; the trait still serves
/// as the single seam consumers code against, so swapping in a real
/// top metal is a one-line change to the trait impl when `eda-pdks`
/// grows the layer.
pub trait RfPdk {
    /// Active / oxide-diffusion layer (FET source/drain region).
    fn diff(&self) -> LayerIndex;
    /// Gate poly stripe.
    fn poly(&self) -> LayerIndex;
    /// First metal — terminal contact pads, gate / drain / source leads.
    fn metal1(&self) -> LayerIndex;
    /// Top metal — spiral-inductor windings. Aliased to `metal1` on
    /// PDKs that don't yet expose a distinct upper metal.
    fn metal_top(&self) -> LayerIndex;
    /// Contact cut from `metal1` down to diff/poly.
    fn via1(&self) -> LayerIndex;
    /// N-well — drawn under PMOS body (unused by NMOS-only LNA but
    /// part of the trait surface for symmetry with `MosfetPdk`).
    fn nwell(&self) -> LayerIndex;
    /// N+ implant over diff for NMOS source/drain.
    fn nplus(&self) -> LayerIndex;
    /// P+ implant — for PMOS / substrate ties.
    fn pplus(&self) -> LayerIndex;
    /// Electrical-domain port kind.
    fn electrical_kind(&self) -> PortKindId;
}

// ── Local demo PDK ─────────────────────────────────────────────────────
//
// `RfDemo` exposes a distinct `METAL_TOP` layer so unit tests can show
// the inductor windings actually living on a layer separate from the
// FET routing (the foundry impls below alias `metal_top` to `metal1`
// until `eda-pdks` exposes upper metals).

pdk! {
    pub RfDemo {
        dbu: 1000,
        layers: {
            DIFF      = (65, 20),
            POLY      = (66, 20),
            METAL1    = (68, 20),
            METAL_TOP = (72, 20),
            VIA1      = (66, 44),
            NWELL     = (64, 20),
            NPLUS     = (93, 44),
            PPLUS     = (94, 20),
        },
        ports: { Electrical },
    }
}

impl RfPdk for RfDemo {
    fn diff(&self)            -> LayerIndex { self.DIFF }
    fn poly(&self)            -> LayerIndex { self.POLY }
    fn metal1(&self)          -> LayerIndex { self.METAL1 }
    fn metal_top(&self)       -> LayerIndex { self.METAL_TOP }
    fn via1(&self)            -> LayerIndex { self.VIA1 }
    fn nwell(&self)           -> LayerIndex { self.NWELL }
    fn nplus(&self)           -> LayerIndex { self.NPLUS }
    fn pplus(&self)           -> LayerIndex { self.PPLUS }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

// ── Foundry PDK impls ──────────────────────────────────────────────────
//
// Each gated by the matching feature so consumers only pay for the
// foundries they enabled. `metal_top` aliases `METAL1` because
// `eda-pdks` currently surfaces only the lowest metal — when upper
// metals land in the generated struct, only these aliases change.

#[cfg(feature = "sky130")]
impl RfPdk for eda_pdks::Sky130 {
    fn diff(&self)            -> LayerIndex { self.DIFF }
    fn poly(&self)            -> LayerIndex { self.RES }
    fn metal1(&self)          -> LayerIndex { self.METAL1 }
    fn metal_top(&self)       -> LayerIndex { self.METAL1 }
    fn via1(&self)            -> LayerIndex { self.VIA1 }
    fn nwell(&self)           -> LayerIndex { self.NWELL }
    fn nplus(&self)           -> LayerIndex { self.NPLUS }
    fn pplus(&self)           -> LayerIndex { self.PPLUS }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

#[cfg(feature = "gf180mcu")]
impl RfPdk for eda_pdks::Gf180mcu {
    fn diff(&self)            -> LayerIndex { self.DIFF }
    fn poly(&self)            -> LayerIndex { self.RES }
    fn metal1(&self)          -> LayerIndex { self.METAL1 }
    fn metal_top(&self)       -> LayerIndex { self.METAL1 }
    fn via1(&self)            -> LayerIndex { self.VIA1 }
    fn nwell(&self)           -> LayerIndex { self.NWELL }
    fn nplus(&self)           -> LayerIndex { self.NPLUS }
    fn pplus(&self)           -> LayerIndex { self.PPLUS }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

// ── Mosfet block ───────────────────────────────────────────────────────

/// Single-finger NMOS layout primitive.
///
/// Geometry: a `length × (width+overhang)` poly stripe straddling a
/// `width`-tall active rectangle, with NPLUS implant covering the
/// active and three `metal1` contact pads (drain / source / gate).
/// Body tie elided — the LNA composite stamps a substrate ring
/// around the whole block when needed.
///
/// `width_dbu` is the channel width `W`; `length_dbu` is the gate
/// length `L`. The aspect ratio (`W/L`) drives the small-signal `gm`
/// and `Cgs` parameters that the [`RfScattering`] model exposes — but
/// the behavioral side is *not* derived from the geometry here.
/// Geometry just stamps shapes; the small-signal model takes `gm` /
/// `Cgs` as independent rlx params (matching how a designer hands
/// PEX-extracted values to a layout-extraction step).
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Mosfet {
    /// Channel width `W` in DBU.
    pub width_dbu: i64,
    /// Gate length `L` in DBU.
    pub length_dbu: i64,
    /// Stable identifier — distinguishes `M1` / `M2` in cascode pairs.
    pub id: String,
}

impl Block for Mosfet {
    fn name(&self) -> String {
        format!("Mosfet_{}_W{}_L{}", self.id, self.width_dbu, self.length_dbu)
    }
}

const PAD: i64 = 2_000; // 2 µm square contact pads
const GATE_OVERHANG: i64 = 1_000; // poly extends 1 µm past diff each side

impl<P: RfPdk> Layout<P> for Mosfet {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let w = self.width_dbu;
        let l = self.length_dbu;
        let half_w = w / 2;
        let mut cb = CellBuilder::new(<Self as Block>::name(self));

        // Active diffusion: extends past the gate on each side to make
        // room for source/drain contacts.
        let diff_pad = PAD + 500;
        let diff_rect = Rect::new(Bbox::new(
            Point::new(-l / 2 - diff_pad, -half_w),
            Point::new(l / 2 + diff_pad, half_w),
        ));
        cb.add_shape(pdk.diff(), Shape::Box(diff_rect));
        cb.add_shape(pdk.nplus(), Shape::Box(diff_rect));

        // Gate poly stripe — vertical, straddles the diffusion.
        cb.add_shape(
            pdk.poly(),
            Shape::Box(Rect::new(Bbox::new(
                Point::new(-l / 2, -half_w - GATE_OVERHANG),
                Point::new(l / 2, half_w + GATE_OVERHANG),
            ))),
        );

        // S/D/G metal1 contact pads.
        let drain_x = l / 2 + diff_pad - PAD / 2;
        let source_x = -drain_x;
        let gate_y = half_w + GATE_OVERHANG + PAD;

        for x in [drain_x, source_x] {
            cb.add_shape(
                pdk.metal1(),
                Shape::Box(Rect::new(Bbox::new(
                    Point::new(x - PAD / 2, -PAD / 2),
                    Point::new(x + PAD / 2, PAD / 2),
                ))),
            );
            cb.add_shape(
                pdk.via1(),
                Shape::Box(Rect::new(Bbox::new(
                    Point::new(x - PAD / 4, -PAD / 4),
                    Point::new(x + PAD / 4, PAD / 4),
                ))),
            );
        }
        cb.add_shape(
            pdk.metal1(),
            Shape::Box(Rect::new(Bbox::new(
                Point::new(-PAD / 2, gate_y - PAD / 2),
                Point::new(PAD / 2, gate_y + PAD / 2),
            ))),
        );

        let elec = pdk.electrical_kind();
        cb.add_port(
            Port::new("drain", pdk.metal1(), Point::new(drain_x, 0), Angle90::E, PAD)
                .with_kind(elec),
        );
        cb.add_port(
            Port::new("source", pdk.metal1(), Point::new(source_x, 0), Angle90::W, PAD)
                .with_kind(elec),
        );
        cb.add_port(
            Port::new("gate", pdk.metal1(), Point::new(0, gate_y), Angle90::N, PAD)
                .with_kind(elec),
        );

        lib.insert(cb)
    }
}

// ── Spiral inductor ────────────────────────────────────────────────────

/// Square planar spiral inductor on the PDK's top metal layer.
///
/// Geometry: a single axis-aligned spiral conductor of `n_turns`
/// turns, wound clockwise from outer to inner on `metal_top`, with
/// `outer_dbu` outer side length, `width_dbu` trace width, and
/// `spacing_dbu` between turns. The inner terminal is brought back
/// out under the spiral via a `metal1` underpass with a `via1` stack
/// at the crossover, as a real planar spiral requires — without it
/// every "ring" would be electrically isolated and the device
/// wouldn't conduct.
///
/// Two electrical ports:
///   - `p1` at the outer end on `metal_top`
///   - `p2` at the underpass exit on `metal1`, just outside the
///     outer ring
///
/// ## Inductance model — Mohan's modified Wheeler (square)
///
/// ```text
///   L ≈ K1 · μ₀ · n² · d_avg / (1 + K2 · ρ)
///   d_avg = (d_out + d_in) / 2          ρ = (d_out − d_in) / (d_out + d_in)
///   K1 = 2.34 (square)                  K2 = 2.75 (square)
/// ```
///
/// Returned by [`SpiralInductor::inductance_nh`] in nanohenries from
/// dimensions in DBU. The behavioral [`RfScattering`] impl on `Lna`
/// uses these values to *initialize* `Lg` / `Ls` / `Ld` so the
/// behavioral and layout sides start consistent — Adam tuning then
/// drifts the params away from the geometric value.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SpiralInductor {
    /// Outer side length in DBU.
    pub outer_dbu: i64,
    /// Trace width in DBU.
    pub width_dbu: i64,
    /// Turn-to-turn spacing in DBU.
    pub spacing_dbu: i64,
    /// Number of turns.
    pub n_turns: u32,
    /// Stable identifier.
    pub id: String,
}

impl Block for SpiralInductor {
    fn name(&self) -> String {
        format!(
            "SpiralInductor_{}_D{}_W{}_S{}_N{}",
            self.id, self.outer_dbu, self.width_dbu, self.spacing_dbu, self.n_turns
        )
    }
}

impl SpiralInductor {
    /// Inner side length `d_in` in DBU.
    pub fn inner_dbu(&self) -> i64 {
        let pitch = self.width_dbu + self.spacing_dbu;
        (self.outer_dbu - 2 * (self.n_turns as i64) * pitch + 2 * self.spacing_dbu).max(0)
    }

    /// Mohan square-spiral inductance in nanohenries.
    pub fn inductance_nh(&self) -> f32 {
        // µ₀ = 4π × 10⁻⁷ H/m → in nH/m: 4π × 10² ≈ 1256.64.
        const MU0_NH_PER_M: f32 = 4.0 * std::f32::consts::PI * 1.0e2;
        const K1: f32 = 2.34;
        const K2: f32 = 2.75;
        const DBU_TO_M: f32 = 1.0e-9; // 1 DBU = 1 nm

        let d_out = self.outer_dbu as f32 * DBU_TO_M;
        let d_in = self.inner_dbu() as f32 * DBU_TO_M;
        let d_avg = 0.5 * (d_out + d_in);
        let rho = if d_out + d_in > 0.0 {
            (d_out - d_in) / (d_out + d_in)
        } else {
            0.0
        };
        let n = self.n_turns as f32;
        K1 * MU0_NH_PER_M * n * n * d_avg / (1.0 + K2 * rho)
    }
}

impl<P: RfPdk> Layout<P> for SpiralInductor {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let mut cb = CellBuilder::new(<Self as Block>::name(self));
        let w = self.width_dbu;
        let pitch = w + self.spacing_dbu;
        let n = self.n_turns.max(1) as i64;
        let mt = pdk.metal_top();
        let m1 = pdk.metal1();
        let v1 = pdk.via1();

        // Centerline of the outermost turn — `outer_dbu` is the outer
        // side length, so the trace's centerline sits `w/2` inside.
        let c0 = (self.outer_dbu - w) / 2;

        // Build the spiral centerline as a polyline winding clockwise
        // from outer top-left to the inner top-left of the last turn.
        // Each non-final turn does top → right → bottom → (truncated)
        // left → inward step; the final turn closes the loop with a
        // full left edge and the inner end becomes the underpass start.
        let mut pts: Vec<Point> = Vec::with_capacity(n as usize * 5 + 1);
        let mut c = c0;
        pts.push(Point::new(-c, c));
        for k in 0..n {
            pts.push(Point::new(c, c));   // top edge
            pts.push(Point::new(c, -c));  // right edge
            pts.push(Point::new(-c, -c)); // bottom edge
            if k == n - 1 {
                pts.push(Point::new(-c, c));
            } else {
                let next_c = c - pitch;
                pts.push(Point::new(-c, next_c));      // truncated left
                pts.push(Point::new(-next_c, next_c)); // inward step
                c = next_c;
            }
        }
        cb.add_shape(mt, Shape::Path(Path::new(pts.clone(), w)));

        // Underpass on metal1: bridge the inner end back out across the
        // spiral on a different layer so it doesn't short the turns.
        // The bridge runs straight up the left edge from the inner end,
        // outside the spiral, to a port above the outer ring.
        let inner_end = *pts.last().expect("spiral has at least one point");
        let outside_y = c0 + pitch + w; // safely above outermost ring
        let bridge_pts = vec![
            inner_end,
            Point::new(-c0 - pitch - w, inner_end.y),
            Point::new(-c0 - pitch - w, outside_y),
        ];
        cb.add_shape(m1, Shape::Path(Path::new(bridge_pts, w)));

        // Via stack at the metal_top → metal1 transition (inner end).
        let via_half = w / 2;
        cb.add_shape(
            v1,
            Shape::Box(Rect::new(Bbox::new(
                Point::new(inner_end.x - via_half, inner_end.y - via_half),
                Point::new(inner_end.x + via_half, inner_end.y + via_half),
            ))),
        );

        let elec = pdk.electrical_kind();
        // p1: outer end on metal_top.
        cb.add_port(
            Port::new("p1", mt, Point::new(-c0, c0), Angle90::N, w).with_kind(elec),
        );
        // p2: inner end brought out via the metal1 underpass.
        cb.add_port(
            Port::new(
                "p2",
                m1,
                Point::new(-c0 - pitch - w, outside_y),
                Angle90::N,
                w,
            )
            .with_kind(elec),
        );

        lib.insert(cb)
    }
}

// ── LNA composite ──────────────────────────────────────────────────────

/// Inductively-degenerated cascode CMOS LNA at a single design
/// frequency `f₀`.
///
/// Two stacked NMOS transistors (`m1` common-source input device, `m2`
/// cascode), three spiral inductors:
///
/// ```text
///   RF_in ── Lg ── G(M1) ─┬─ M1 (gm, Cgs) ─┬─ S(M2) ─ M2 ─ D(M2) ── RF_out
///                                          │                     │
///                                         Ls (degeneration)     Ld + R_L
///                                          │                     │
///                                         GND                   VDD
/// ```
///
/// Behavioral params (rlx): `gm` (S), `cgs` (F), `lg` / `ls` / `ld`
/// (H), `rl` (Ω). Layout params: per-Mosfet W/L, per-SpiralInductor
/// geometry. The two surfaces are independent on purpose — designers
/// in real RF flows hand-tune the small-signal numbers from PEX, not
/// from layout dimensions.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Lna {
    pub m1: Mosfet,
    pub m2: Mosfet,
    pub lg: SpiralInductor,
    pub ls: SpiralInductor,
    pub ld: SpiralInductor,
    /// Stable name — woven into every rlx Param key so multiple LNAs
    /// in one graph stay distinct.
    pub id: String,
}

impl Lna {
    /// Default 2.4 GHz Wi-Fi-band LNA — values picked so the canonical
    /// match condition is satisfied for `Z₀ = 50 Ω`:
    /// `gm·Ls/Cgs = 50`, `ω₀²·(Lg + Ls)·Cgs = 1`.
    ///
    /// `gm = 50 mS`, `Cgs = 250 fF`  → `Ls = 250 pH`
    /// `ω₀ = 2π · 2.4 GHz`           → `Lg + Ls ≈ 17.6 nH`, `Lg ≈ 17.4 nH`
    /// Drain tank: `Ld = 10 nH`, `R_L = 500 Ω` (modest gain target).
    pub fn lna_24ghz(id: impl Into<String>) -> Self {
        let id: String = id.into();
        Self {
            m1: Mosfet { width_dbu: 200_000, length_dbu: 180, id: format!("{id}_m1") },
            m2: Mosfet { width_dbu: 200_000, length_dbu: 180, id: format!("{id}_m2") },
            lg: SpiralInductor {
                outer_dbu: 280_000, width_dbu: 4_000, spacing_dbu: 2_000,
                n_turns: 6, id: format!("{id}_lg"),
            },
            ls: SpiralInductor {
                outer_dbu: 80_000, width_dbu: 4_000, spacing_dbu: 2_000,
                n_turns: 2, id: format!("{id}_ls"),
            },
            ld: SpiralInductor {
                outer_dbu: 220_000, width_dbu: 4_000, spacing_dbu: 2_000,
                n_turns: 5, id: format!("{id}_ld"),
            },
            id,
        }
    }

    pub fn gm_param_name(&self)  -> String { format!("{}.gm",  self.id) }
    pub fn cgs_param_name(&self) -> String { format!("{}.cgs", self.id) }
    pub fn lg_param_name(&self)  -> String { format!("{}.lg",  self.id) }
    pub fn ls_param_name(&self)  -> String { format!("{}.ls",  self.id) }
    pub fn ld_param_name(&self)  -> String { format!("{}.ld",  self.id) }
    pub fn rl_param_name(&self)  -> String { format!("{}.rl",  self.id) }
}

// ── Layout via eda-pnr ─────────────────────────────────────────────────
//
// The LNA's `Layout::layout` declares the topology as a PNR
// [`Netlist`] and hands placement + routing to `eda-pnr`:
//
// * **Place**: a hand-picked floorplan (M1+M2 cascode at the centre,
//   Lg above, Ls below, Ld off to the right, IO pads ringing the
//   block) drives a `ManualPlacer` so the visual layout matches what
//   a designer drew.
// * **Route**: `ManhattanRouter` with `WireStyle::Polygon` actually
//   *wires* the cascode (M1.drain ↔ M2.source), the gate path
//   (rf_in pad ↔ Lg ↔ M1.gate), the drain tank (M2.drain ↔ Ld ↔
//   rf_out pad ↔ vdd pad), the source degeneration (M1.source ↔ Ls
//   ↔ gnd pad), and the bias path (vbias pad ↔ M2.gate). Phase 1
//   the floorplan rendered five disconnected boxes; this version
//   routes seven nets between them.
//
// Real LNAs put inductors on dedicated thick top metal far from the
// FETs — this layout is a representative floorplan, not a tape-out.

const FET_PITCH_Y: i64 = 100_000;    // vertical spacing between M1 and M2 centres
const FET_TO_INDUCTOR: i64 = 80_000; // gap between FET stack and surrounding inductors
const PAD_OFFSET: i64 = 30_000;      // distance from inductor edge to pad

/// IO bond-pad: a `metal1` square with a single `io` port at the
/// centre. Instantiated five times in `Lna::layout` (one per
/// external pin) so the netlist's `connect` calls give the router
/// real pin endpoints to land wires on.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Pad {
    /// Pad side length in DBU.
    pub size_dbu: i64,
    pub id: String,
}

impl Block for Pad {
    fn name(&self) -> String { format!("Pad_{}_S{}", self.id, self.size_dbu) }
}

impl<P: RfPdk> Layout<P> for Pad {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let mut cb = CellBuilder::new(<Self as Block>::name(self));
        let half = self.size_dbu / 2;
        cb.add_shape(
            pdk.metal1(),
            Shape::Box(Rect::new(Bbox::new(
                Point::new(-half, -half),
                Point::new(half, half),
            ))),
        );
        cb.add_port(
            Port::new("io", pdk.metal1(), Point::new(0, 0), Angle90::E, self.size_dbu)
                .with_kind(pdk.electrical_kind()),
        );
        lib.insert(cb)
    }
}

impl Block for Lna {
    fn name(&self) -> String {
        format!("Lna_{}", self.id)
    }
}

impl<P: RfPdk> Connectivity<P> for Lna {
    fn connectivity(&self, lib: &Library, pdk: &P) -> Netlist {
        let m1_cell = self.m1.layout(lib, pdk);
        let m2_cell = self.m2.layout(lib, pdk);
        let lg_cell = self.lg.layout(lib, pdk);
        let ls_cell = self.ls.layout(lib, pdk);
        let ld_cell = self.ld.layout(lib, pdk);

        let pad_size = PAD * 2;
        let pad_block = |name: &str| Pad { size_dbu: pad_size, id: format!("{}_{name}", self.id) };
        let pad_rf_in  = pad_block("rf_in").layout(lib, pdk);
        let pad_rf_out = pad_block("rf_out").layout(lib, pdk);
        let pad_vdd    = pad_block("vdd").layout(lib, pdk);
        let pad_gnd    = pad_block("gnd").layout(lib, pdk);
        let pad_vbias  = pad_block("vbias").layout(lib, pdk);

        let mut nl = Netlist::new(<Self as Block>::name(self))
            .with_default_signal_layer(pdk.metal1());
        // Pads are positions-fixed: AD placement (when callers
        // upgrade to it) shouldn't move I/O pads, only internal
        // FETs / inductors. Today's `ManualPlacer` produces the
        // same result either way, but the `fixed = true` marker
        // is the authoritative declaration.
        let i_m1     = nl.add_instance("M1",     m1_cell);
        let i_m2     = nl.add_instance("M2",     m2_cell);
        let i_lg     = nl.add_instance("Lg",     lg_cell);
        let i_ls     = nl.add_instance("Ls",     ls_cell);
        let i_ld     = nl.add_instance("Ld",     ld_cell);
        let i_p_in   = nl.add_fixed_instance("PAD_RFIN",  pad_rf_in);
        let i_p_out  = nl.add_fixed_instance("PAD_RFOUT", pad_rf_out);
        let i_p_vdd  = nl.add_fixed_instance("PAD_VDD",   pad_vdd);
        let i_p_gnd  = nl.add_fixed_instance("PAD_GND",   pad_gnd);
        let i_p_bias = nl.add_fixed_instance("PAD_VBIAS", pad_vbias);

        // Connectivity (Razavi §5.3.3 inductively-degenerated cascode):
        //   rf_in ── Lg ── G(M1)
        //                  M1.drain ── M2.source
        //                              M2.drain ── Ld ── rf_out, vdd
        //                  M1.source ── Ls ── gnd
        //                  M2.gate ── vbias
        nl.connect("rf_in",   i_p_in,  "io");
        nl.connect("rf_in",   i_lg,    "p1");
        nl.connect("gate1",   i_lg,    "p2");
        nl.connect("gate1",   i_m1,    "gate");
        nl.connect("cascode", i_m1,    "drain");
        nl.connect("cascode", i_m2,    "source");
        nl.connect("drain",   i_m2,    "drain");
        nl.connect("drain",   i_ld,    "p1");
        nl.connect("drain",   i_p_out, "io");
        nl.connect("vdd",     i_ld,    "p2");
        nl.connect("vdd",     i_p_vdd, "io");
        nl.connect("source",  i_m1,    "source");
        nl.connect("source",  i_ls,    "p1");
        nl.connect("gnd",     i_ls,    "p2");
        nl.connect("gnd",     i_p_gnd, "io");
        nl.connect("vbias",   i_m2,    "gate");
        nl.connect("vbias",   i_p_bias,"io");

        nl.expose("rf_in",  "rf_in",  Some(PinDirection::Input));
        nl.expose("rf_out", "drain",  Some(PinDirection::Output));
        nl.expose("vdd",    "vdd",    Some(PinDirection::Power));
        nl.expose("gnd",    "gnd",    Some(PinDirection::Ground));
        nl.expose("vbias",  "vbias",  Some(PinDirection::Input));
        nl
    }

    fn transforms(&self, _: &Netlist, _: &Library) -> Vec<Trans> {
        let m1_y: i64 = 0;
        let m2_y: i64 = FET_PITCH_Y;
        let lg_y = m2_y + FET_PITCH_Y + self.lg.outer_dbu / 2 + FET_TO_INDUCTOR;
        let ls_y = m1_y - FET_PITCH_Y - self.ls.outer_dbu / 2 - FET_TO_INDUCTOR;
        let ld_x = self.ld.outer_dbu / 2 + FET_TO_INDUCTOR + 50_000;
        let pad_rf_in_pos  = Vec2::new(-self.lg.outer_dbu / 2 - PAD_OFFSET, lg_y);
        let pad_rf_out_pos = Vec2::new(ld_x + self.ld.outer_dbu / 2 + PAD_OFFSET, m2_y);
        let pad_vdd_pos    = Vec2::new(ld_x, m2_y + self.ld.outer_dbu / 2 + PAD_OFFSET);
        let pad_gnd_pos    = Vec2::new(0, ls_y - self.ls.outer_dbu / 2 - PAD_OFFSET);
        let pad_vbias_pos  = Vec2::new(-self.m2.width_dbu / 2 - PAD_OFFSET, m2_y);
        vec![
            Trans::translate(Vec2::new(0, m1_y)),
            Trans::translate(Vec2::new(0, m2_y)),
            Trans::translate(Vec2::new(0, lg_y)),
            Trans::translate(Vec2::new(0, ls_y)),
            Trans::translate(Vec2::new(ld_x, m2_y)),
            Trans::translate(pad_rf_in_pos),
            Trans::translate(pad_rf_out_pos),
            Trans::translate(pad_vdd_pos),
            Trans::translate(pad_gnd_pos),
            Trans::translate(pad_vbias_pos),
        ]
    }
}

impl<P: RfPdk> Layout<P> for Lna {
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        eda_pnr::pnr_layout(self, lib, pdk)
    }
}

// ── RF scattering trait ────────────────────────────────────────────────
//
// 2-port S-parameter contract — registers the RF block's small-signal
// params on the rlx graph and returns `(re, im)` nodes for `S₁₁(ω)`.
// `S₂₁` is scalar at resonance for the MVP (the canonical
// "matched-LNA gain" number); a frequency-swept `S₂₁` becomes a
// follow-up when the LC drain-tank model needs it.
//
// Shape choice: `(NodeId, NodeId)` for `(re, im)` pairs mirrors
// `OpticalScattering::s21` in `spike-waveguide-block`. Frequency is a
// runtime `Op::Input` so one compiled session sweeps ω.

/// Helper for inserting `f32` constants. Mirrors `const_f32` in
/// `spike-waveguide-block`.
pub(crate) fn const_f32(g: &mut Graph, val: f32, shape: TensorShape) -> NodeId {
    g.add_node(
        Op::Constant { data: val.to_le_bytes().to_vec() },
        vec![],
        shape,
    )
}

/// 2-port S-parameter / matched-gain contract for an RF block.
pub trait RfScattering: Block {
    /// Reference impedance `Z₀` (Ω) the S-parameters are normalized
    /// against — 50 Ω in essentially every RF context.
    fn z0(&self) -> f32 { 50.0 }

    /// Register this block's small-signal Params and return
    /// `(S₁₁_re, S₁₁_im)` at runtime `freq_hz`.
    ///
    /// Closed-form Razavi §5.3.3:
    ///
    /// ```text
    ///   Z_in(ω) = jω(Lg+Ls) + 1/(jωCgs) + (gm·Ls)/Cgs
    ///   S₁₁    = (Z_in − Z₀) / (Z_in + Z₀)
    /// ```
    fn s11(&self, freq_hz: NodeId, g: &mut Graph) -> (NodeId, NodeId);

    /// Matched-LNA gain magnitude at the current frequency input —
    /// scalar real, dimensionless. Razavi eq. 5.79:
    ///
    /// ```text
    ///   |S₂₁| ≈ gm · R_L / (2 · ω · Cgs · Z₀)
    /// ```
    ///
    /// Valid at the input-match frequency; off-resonance the formula
    /// loses the `Q_in` peaking and a full LC-tank model is needed.
    fn s21_matched_mag(&self, freq_hz: NodeId, g: &mut Graph) -> NodeId;
}

impl RfScattering for Lna {
    fn s11(&self, freq_hz: NodeId, g: &mut Graph) -> (NodeId, NodeId) {
        let s = TensorShape::new(&[1], DType::F32);
        let gm = g.param(self.gm_param_name(), s.clone());
        let cgs = g.param(self.cgs_param_name(), s.clone());
        let lg = g.param(self.lg_param_name(), s.clone());
        let ls = g.param(self.ls_param_name(), s.clone());

        // ω = 2π · f.
        let two_pi = const_f32(g, std::f32::consts::TAU, s.clone());
        let omega = g.binary(BinaryOp::Mul, two_pi, freq_hz, s.clone());

        // R_in = (gm · Ls) / Cgs   (real part of Z_in).
        let gm_ls = g.binary(BinaryOp::Mul, gm, ls, s.clone());
        let r_in = g.binary(BinaryOp::Div, gm_ls, cgs, s.clone());

        // X_in = ω·(Lg + Ls) − 1/(ω·Cgs)   (imag part).
        let lg_plus_ls = g.binary(BinaryOp::Add, lg, ls, s.clone());
        let omega_l = g.binary(BinaryOp::Mul, omega, lg_plus_ls, s.clone());
        let omega_cgs = g.binary(BinaryOp::Mul, omega, cgs, s.clone());
        let one = const_f32(g, 1.0, s.clone());
        let one_over_wc = g.binary(BinaryOp::Div, one, omega_cgs, s.clone());
        let x_in = g.binary(BinaryOp::Sub, omega_l, one_over_wc, s.clone());

        // S₁₁ = (Z_in − Z₀) / (Z_in + Z₀).
        // (a+jb)/(c+jd) = ((a·c + b·d) + j(b·c − a·d)) / (c² + d²).
        let z0 = const_f32(g, self.z0(), s.clone());
        let a = g.binary(BinaryOp::Sub, r_in, z0, s.clone());
        let b = x_in;
        let c = g.binary(BinaryOp::Add, r_in, z0, s.clone());
        let d = x_in;

        let ac = g.binary(BinaryOp::Mul, a, c, s.clone());
        let bd = g.binary(BinaryOp::Mul, b, d, s.clone());
        let bc = g.binary(BinaryOp::Mul, b, c, s.clone());
        let ad = g.binary(BinaryOp::Mul, a, d, s.clone());

        let num_re = g.binary(BinaryOp::Add, ac, bd, s.clone());
        let num_im = g.binary(BinaryOp::Sub, bc, ad, s.clone());

        let c2 = g.binary(BinaryOp::Mul, c, c, s.clone());
        let d2 = g.binary(BinaryOp::Mul, d, d, s.clone());
        let denom = g.binary(BinaryOp::Add, c2, d2, s.clone());

        let re = g.binary(BinaryOp::Div, num_re, denom, s.clone());
        let im = g.binary(BinaryOp::Div, num_im, denom, s);
        (re, im)
    }

    fn s21_matched_mag(&self, freq_hz: NodeId, g: &mut Graph) -> NodeId {
        let s = TensorShape::new(&[1], DType::F32);
        let gm = g.param(self.gm_param_name(), s.clone());
        let cgs = g.param(self.cgs_param_name(), s.clone());
        let rl = g.param(self.rl_param_name(), s.clone());

        let two_pi = const_f32(g, std::f32::consts::TAU, s.clone());
        let omega = g.binary(BinaryOp::Mul, two_pi, freq_hz, s.clone());

        // num = gm · R_L
        let num = g.binary(BinaryOp::Mul, gm, rl, s.clone());
        // den = 2 · ω · Cgs · Z₀
        let two_z0 = const_f32(g, 2.0 * self.z0(), s.clone());
        let omega_cgs = g.binary(BinaryOp::Mul, omega, cgs, s.clone());
        let den = g.binary(BinaryOp::Mul, omega_cgs, two_z0, s.clone());
        g.binary(BinaryOp::Div, num, den, s)
    }
}

impl Lna {
    /// Build an inverse-design **input-match loss graph** —
    /// `|S₁₁(f)|²` at a runtime operating frequency. Drive this to
    /// zero with Adam over `lg` (`Lg` is the canonical knob; `gm`,
    /// `Cgs`, `Ls` are usually fixed by sizing + bias upstream).
    ///
    /// Inputs: `freq_hz`. Outputs: scalar `|S₁₁|²`.
    /// Differentiable through every Param the graph registers
    /// (`gm`, `cgs`, `lg`, `ls`).
    pub fn build_match_loss_graph(&self) -> Graph {
        let mut g = Graph::new(format!("{}_match_loss", self.id));
        let s = TensorShape::new(&[1], DType::F32);
        let f = g.input("freq_hz", s.clone());
        let (re, im) = self.s11(f, &mut g);
        let re2 = g.binary(BinaryOp::Mul, re, re, s.clone());
        let im2 = g.binary(BinaryOp::Mul, im, im, s.clone());
        let mag2 = g.binary(BinaryOp::Add, re2, im2, s);
        g.set_outputs(vec![mag2]);
        g
    }

    /// Build a forward graph returning `[Re S₁₁, Im S₁₁, |S₂₁|]` at
    /// runtime `freq_hz`. Used by the demo bin for sweeping. Each
    /// behavioral Param is registered exactly once — the trait
    /// methods `s11` / `s21_matched_mag` are convenience entries that
    /// each register their own Params; building one composite graph
    /// directly avoids the duplicate-name registration that happens
    /// when both methods touch the same scalar (`gm`, `Cgs`).
    pub fn build_forward_graph(&self) -> Graph {
        let mut g = Graph::new(format!("{}_forward", self.id));
        let s = TensorShape::new(&[1], DType::F32);
        let f = g.input("freq_hz", s.clone());

        let gm = g.param(self.gm_param_name(), s.clone());
        let cgs = g.param(self.cgs_param_name(), s.clone());
        let lg = g.param(self.lg_param_name(), s.clone());
        let ls = g.param(self.ls_param_name(), s.clone());
        let _ld = g.param(self.ld_param_name(), s.clone()); // registered for completeness
        let rl = g.param(self.rl_param_name(), s.clone());

        let two_pi = const_f32(&mut g, std::f32::consts::TAU, s.clone());
        let omega = g.binary(BinaryOp::Mul, two_pi, f, s.clone());

        // S₁₁.
        let gm_ls = g.binary(BinaryOp::Mul, gm, ls, s.clone());
        let r_in = g.binary(BinaryOp::Div, gm_ls, cgs, s.clone());
        let lg_plus_ls = g.binary(BinaryOp::Add, lg, ls, s.clone());
        let omega_l = g.binary(BinaryOp::Mul, omega, lg_plus_ls, s.clone());
        let omega_cgs = g.binary(BinaryOp::Mul, omega, cgs, s.clone());
        let one = const_f32(&mut g, 1.0, s.clone());
        let one_over_wc = g.binary(BinaryOp::Div, one, omega_cgs, s.clone());
        let x_in = g.binary(BinaryOp::Sub, omega_l, one_over_wc, s.clone());
        let z0 = const_f32(&mut g, self.z0(), s.clone());
        let a = g.binary(BinaryOp::Sub, r_in, z0, s.clone());
        let c = g.binary(BinaryOp::Add, r_in, z0, s.clone());
        let ac = g.binary(BinaryOp::Mul, a, c, s.clone());
        let bd = g.binary(BinaryOp::Mul, x_in, x_in, s.clone());
        let bc = g.binary(BinaryOp::Mul, x_in, c, s.clone());
        let ad = g.binary(BinaryOp::Mul, a, x_in, s.clone());
        let num_re = g.binary(BinaryOp::Add, ac, bd, s.clone());
        let num_im = g.binary(BinaryOp::Sub, bc, ad, s.clone());
        let c2 = g.binary(BinaryOp::Mul, c, c, s.clone());
        let denom = g.binary(BinaryOp::Add, c2, bd, s.clone());
        let re = g.binary(BinaryOp::Div, num_re, denom, s.clone());
        let im = g.binary(BinaryOp::Div, num_im, denom, s.clone());

        // |S₂₁| = gm·R_L / (2·ω·Cgs·Z₀).
        let num_s21 = g.binary(BinaryOp::Mul, gm, rl, s.clone());
        let two_z0 = const_f32(&mut g, 2.0 * self.z0(), s.clone());
        let den_s21 = g.binary(BinaryOp::Mul, omega_cgs, two_z0, s.clone());
        let s21_mag = g.binary(BinaryOp::Div, num_s21, den_s21, s);

        g.set_outputs(vec![re, im, s21_mag]);
        g
    }
}
