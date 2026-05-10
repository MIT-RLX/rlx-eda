//! Composition hierarchy — `Device → Cell → Macro → Tile → Core → Die →
//! Reticle → Wafer → Lot`.
//!
//! These traits sit on top of [`Block`](crate::Block) (the identity-and-equality
//! base contract every composable thing satisfies) and refine it with the
//! obligations specific to each level of the standard semiconductor
//! abstraction stack. The codebase's existing `StdCell` (in `eda-stdcells`)
//! and `Tile` (in `eda-tile`) sit at the Cell and Tile rungs respectively;
//! `Tile` keeps its richer abutment contract in its own crate, so it is
//! intentionally **not** redefined here.
//!
//! ## Why all rungs at once
//!
//! Per the philosophy in `lib.rs`, traits earn their place — most rungs
//! below have only a minimal contract until a real consumer arrives. They
//! exist now so that downstream crates have a stable place to land
//! (`impl Macro<P> for SarSlice`, `impl Die<P> for TinyConvDie`, …)
//! without having to invent a new vocabulary at each step.
//!
//! ## What is *not* here
//!
//! - **`Block`** — already defined in [`crate`].
//! - **`Tile`** — defined in `eda-tile` with a richer pitch/rails/edge-port
//!   contract; would only lose information if collapsed into a marker here.
//! - **Device-as-trait** — already covered by the capability traits
//!   (`MnaDevice`, `NonlinearDcBehavioral`, `DcBehavioral`, …) in
//!   [`crate`]. A separate `Device` marker would just rename them; we add
//!   it below only as a hierarchy-level marker (`Block`-bounded), not as a
//!   replacement for the capability traits.

use crate::{Block, Layout};
use klayout_core::{Bbox, Point, Vec2};

/// Direction of a pin / port / pad relative to its containing block.
///
/// Used uniformly across [`Pin`] and [`IoPad`] so that hierarchy
/// rungs share one vocabulary for connectivity direction. Power /
/// ground are treated as their own directions (rather than as
/// `InOut`) because downstream tools (PDN extraction, IR-drop,
/// thermal) almost always want to fan them out separately from
/// signal pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinDirection {
    /// Signal flows into the block.
    Input,
    /// Signal flows out of the block.
    Output,
    /// Bidirectional signal (tri-state bus, analog node).
    InOut,
    /// Power-supply pin (VDD, VDDA, …). Carried separately from
    /// `InOut` so PDN-aware tools can find supplies without parsing
    /// names.
    Power,
    /// Ground / return pin (VSS, VSSA, …). Same rationale as
    /// `Power`.
    Ground,
}

/// A named pin on a [`Cell`], [`Macro`], or [`Core`]. Carries name +
/// direction only — geometry (which layer, where on the boundary)
/// belongs to the layout side and is reachable through `Layout::layout`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pin {
    /// Pin name as it appears in the netlist / Liberty file.
    pub name: String,
    /// Signal-flow direction relative to the containing block.
    pub direction: PinDirection,
}

/// A bond pad on a [`Die`]. Position is the pad-centre in DBU, in the
/// die's local coordinate frame (i.e. relative to the die outline,
/// not to a wafer or reticle).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IoPad {
    /// Pad name (matches the corresponding [`Core`] / top-level signal).
    pub name: String,
    /// Direction of the pad — `Power` / `Ground` for supply pads,
    /// `Input` / `Output` / `InOut` for signal pads.
    pub direction: PinDirection,
    /// Pad-centre position in DBU, in the die's local frame.
    pub centre: Point,
}

/// One die instance placed at `origin` (DBU, in the parent frame —
/// reticle field for [`Reticle::fields`], wafer surface for
/// [`Wafer::step_pattern`]).
///
/// Identifies the die by `die_name` rather than carrying a typed
/// reference: the same record is reused at the reticle and wafer
/// rungs, which would otherwise have to be generic over the same `P`
/// twice.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiePlacement {
    /// Name of the [`Die`] being placed (its `Block::name`).
    pub die_name: String,
    /// Origin of the die's local frame in the parent frame, in DBU.
    pub origin: Point,
}

/// Hierarchy rung 1 — **atomic primitive** (transistor, resistor,
/// capacitor, diode).
///
/// The simulation-side capability traits (`MnaDevice`,
/// `NonlinearDcBehavioral`, `DcBehavioral`, `TransientStorage`,
/// `TransientDelay`) carry the actual electrical contract; this trait
/// is the hierarchy-level marker so a `Device` can be referred to as
/// such by composition code without committing to a particular
/// electrical capability.
///
/// Implementors typically already implement one or more capability
/// traits (e.g. `Mosfet: NonlinearDcBehavioral + Device`).
pub trait Device: Block {
    /// Names of the device's electrical terminals, in the order the
    /// capability trait (`MnaDevice::contributions`,
    /// `NonlinearDcBehavioral::currents`, …) consumes terminal
    /// voltages. For a MOSFET that is conventionally
    /// `["d", "g", "s", "b"]`; for a 2-terminal R / C / L it is
    /// `["a", "b"]`.
    fn terminal_names(&self) -> Vec<String>;
}

/// Hierarchy rung 2 — **cell**: a small, self-contained composition of
/// devices that implements one logic / analog function (an inverter, a
/// flip-flop, a current mirror).
///
/// Foundry standard cells (`StdCell` in `eda-stdcells`) are the
/// canonical consumer; user-defined primitive cells implement this
/// directly. Cells are the smallest unit that floorplanning
/// considers — below this rung, devices are placed implicitly by
/// their containing cell's layout.
pub trait Cell<P>: Block + Layout<P> {
    /// External pins, in declaration order (matches the order in
    /// the cell's Liberty / netlist record).
    fn pins(&self) -> Vec<Pin>;
}

/// Hierarchy rung 3 — **macro**: a composed block of cells with a
/// declared boundary and named pins, but without the abutment
/// contract that [`Tile`](../../eda_tile/trait.Tile.html) imposes.
///
/// The natural home for things like `spike-divider-block`,
/// `spike-sar-adc`, `spike-waveguide-block`. Standard EDA terminology
/// calls this rung "block" (as in "hard macro" / "soft block"); we
/// use `Macro` to avoid collision with the crate-wide [`Block`] base
/// trait.
pub trait Macro<P>: Block + Layout<P> {
    /// External pins of this macro, in declaration order.
    fn pins(&self) -> Vec<Pin>;
    /// Macro boundary (placement / routing boundary) in DBU, in the
    /// macro's local frame. Used by floorplanning to reserve area
    /// and by routers to decide where signals may enter / exit.
    fn boundary(&self) -> Bbox;
}

// Hierarchy rung 4 — `Tile` is defined in `eda-tile` with a richer
// pitch / rails / edge-port contract. Not redefined here.

/// Hierarchy rung 5 — **core**: a composed block of tiles plus the
/// glue (clocking, reset, IO interface) needed to expose a coherent
/// interface to the surrounding die.
///
/// Speculative: no in-tree consumer yet. Contract is intentionally
/// thin (just an IO-pin list) until a real consumer pins down the
/// shape.
pub trait Core<P>: Block + Layout<P> {
    /// Top-level IO pins exposed by the core (clock, reset, data
    /// buses).
    ///
    /// Distinct from a [`Die`]'s `io_pads` — those are the physical
    /// bond pads; these are the logical ports that a die would route
    /// to its pads. A single core IO pin may map to several pads
    /// (differential pairs, multi-bit buses).
    fn io_pins(&self) -> Vec<Pin>;
}

/// Hierarchy rung 6 — **die**: one chip instance.
///
/// Carries the outline rectangle (drawn in DBU), the bond-pad list,
/// and the scribe-line clearance that wafer stepping must respect.
/// The die is the largest object that still has a single GDS layout —
/// reticle and above describe how dies are physically composed onto
/// silicon, not how they are drawn.
pub trait Die<P>: Block + Layout<P> {
    /// Die outline (chip boundary) in DBU, in the die's local frame.
    /// Used as the placement footprint at the reticle / wafer rungs.
    fn outline(&self) -> Bbox;
    /// Bond pads on this die, in the die's local frame. Order is the
    /// die's own pad-ring order — package / wirebond tools rely on
    /// it being stable.
    fn io_pads(&self) -> Vec<IoPad>;
    /// Scribe-line clearance (saw-street half-width) in DBU. Wafer-
    /// level stepping adds at least this much spacing between
    /// neighbouring die instances so the dicing saw has somewhere to
    /// cut without entering active area.
    fn scribe_clearance_dbu(&self) -> i64;
}

/// Hierarchy rung 7 — **reticle**: one lithography step field.
///
/// Holds one or more die placements that are exposed together in a
/// single stepper shot. Reticles aren't `Layout<P>` because the
/// artifact at this rung is a mask set, not a `CellId`; geometry is
/// reachable through the placed dies' own `Layout` impls.
///
/// Speculative: no in-tree consumer yet.
pub trait Reticle: Block {
    /// Reticle field size (the printable area of one stepper shot)
    /// in DBU.
    fn field_size(&self) -> Vec2;
    /// Die placements within the reticle field, with origins in DBU
    /// in the field's local frame (typically the field's lower-left
    /// corner).
    fn fields(&self) -> Vec<DiePlacement>;
}

/// Hierarchy rung 8 — **wafer**: a substrate disc holding many
/// stepped-out reticle exposures.
///
/// The unit at which yield, edge exclusion, and wafer-map analyses
/// live. Like [`Reticle`], not `Layout<P>` — the wafer-level
/// "layout" is a step pattern, not a drawn cell.
///
/// Speculative: no in-tree consumer yet.
pub trait Wafer: Block {
    /// Wafer diameter in mm (200, 300, 450, …).
    fn diameter_mm(&self) -> f64;
    /// Edge exclusion (no dies inside this annular rim) in mm.
    /// Standard 300 mm wafers exclude the outer ~3 mm.
    fn edge_exclusion_mm(&self) -> f64;
    /// Die placements across the wafer surface, with origins in DBU
    /// from the wafer centre. Consumers (yield maps, thermal-across-
    /// wafer studies, wafer-level reliability extractions) iterate
    /// these.
    fn step_pattern(&self) -> Vec<DiePlacement>;
}

/// Hierarchy rung 9 — **lot**: a process-run batch of wafers.
///
/// The natural home for cross-wafer statistical / yield analysis:
/// PCM (Process Control Monitor) data, retest logs, and lot-level
/// corner-shift parameters all key off the lot.
///
/// Speculative: no in-tree consumer yet.
pub trait Lot: Block {
    /// Number of wafers in the lot. Conventionally 25 (one FOUP).
    fn wafer_count(&self) -> usize;
    /// Foundry process-run identifier (lot number / run ID) — an
    /// opaque string used to key external metadata (PCM data, retest
    /// logs, e-test reports).
    fn process_run_id(&self) -> String;
}
