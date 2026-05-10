//! `eda-pnr` — Place-and-Route as a first-class layer in rlx-eda.
//!
//! Until this crate landed, every spike that needed wires between
//! sub-cells (resistors, FETs, inductors, …) hand-coded the same
//! shape: place children at hard-coded `Trans`es, then call
//! `klayout-route::ManhattanPlanner.plan(...)` per-edge with a
//! polygon stylizer copy-pasted from `spike-divider-block`. The
//! LNA / MZI floorplans skipped routing entirely — the "place"
//! half worked but pads + inductors were never actually wired up.
//!
//! This crate lifts that skeleton into a shared layer, with one
//! design move worth flagging upfront:
//!
//! ## AD-first
//!
//! Per-instance positions live as `rlx_ir::Param` nodes from the
//! start. The router's half-perimeter wirelength (HPWL) — the
//! standard placement objective in EDA — is a smooth-max-formed
//! loss on the rlx graph, differentiable wrt every instance's
//! (x, y). That makes placement a gradient-descent target the same
//! way the LNA's `Lg` and the MZI's `n_eff_A` are: hand
//! `eda_pnr::ad::hpwl_loss_graph(&netlist)` to `grad_with_loss` and
//! Adam, and the placer minimizes wirelength like any other ML
//! objective. Same path for any future loss (timing, congestion,
//! IR drop) — point, build a graph, optimize.
//!
//! Concrete pieces:
//!
//! 1. [`Netlist`] — declarative connectivity. Instances reference
//!    already-laid-out child cells; [`Net`]s name the connections
//!    between them; [`ExternalPin`]s expose nets at the top level.
//!
//! 2. [`Placer`] (`ManualPlacer`, `GridPlacer`) — decides each
//!    instance's `Trans`. `ManualPlacer` is what existing spikes
//!    do today, just lifted out of their `Layout::layout` impls.
//!    `GridPlacer` arranges instances by bbox in rows when the
//!    caller doesn't care.
//!
//! 3. [`Router`] (`ManhattanRouter`) — wraps
//!    `klayout-route::ManhattanPlanner` + a stylizer. Resolves each
//!    net's pins to absolute ports via the placement, runs the
//!    Manhattan planner, and emits routed shapes. Multi-pin nets
//!    fan out as a star from the first pin.
//!
//! 4. [`PnrFlow::run`] — runs the placer, then the router, then
//!    stamps every instance + every wire + every external pin into
//!    a freshly-built top `CellBuilder` and inserts it into the
//!    library, returning the resulting `CellId`.
//!
//! 5. [`ad`] — AD-enabled placement: a [`ad::DifferentiablePlacement`]
//!    wraps each instance's `(x, y)` as rlx Params,
//!    [`ad::hpwl_loss_graph`] returns the wirelength loss, and
//!    [`ad::DifferentiablePlacement::materialize`] snaps the
//!    optimized float positions back to integer DBU so the
//!    standard `PnrFlow::run` can stamp the layout.
//!
//! 6. [`MatchGroup`] — analog matching constraints (differential
//!    pairs, common-centroid groups, interdigitated arrays) declared
//!    on the [`Netlist`] and folded into the AD loss as a quadratic
//!    penalty on instance positions; see
//!    [`ad::combined_loss_graph_with_symmetry`].
//!
//! ## Where this fits in the stack
//!
//! ```text
//!   eda-hir         (Block / Layout / Schematic traits)
//!     │
//!     ▼
//!   eda-pnr         (Netlist + Placer + Router + PnrFlow + AD)    ← this crate
//!     │
//!     ▼
//!   klayout-route   (Manhattan / A* / global-route primitives)
//!     │
//!     ▼
//!   klayout-core    (Library / Cell / Port / Path / Shape / Trans)
//! ```
//!
//! ## What's intentionally not here yet (Phase 2+)
//!
//! - **Optimizing placers beyond AD HPWL** — simulated annealing,
//!   force-directed placement, congestion-aware terms.
//! - **Schematic→Netlist** — `eda-hir::Schematic<P>` returns a
//!   *visual* `SchematicIr`, not a hierarchical netlist. A Phase 2
//!   trait `Connectivity<P>` will return a [`Netlist`] directly.
//! - **Power grid synthesis** — VDD/VSS rings, M-N stripes,
//!   IR-drop checks. `klayout-route::power_grid` already has the
//!   primitives; PNR will wrap them.
//! - **Multi-layer routing** — current `ManhattanRouter` keeps
//!   every net on a single layer. Crossings need vias + a layer
//!   plan, which lives in `klayout-route::multilayer`.

pub mod ad;
pub mod connectivity;
pub mod flow;
pub mod netlist;
pub mod place;
pub mod route;

pub use connectivity::{pnr_layout, pnr_layout_with, Connectivity};
pub use flow::PnrFlow;
pub use netlist::{
    ExternalPin, MatchGroup, MatchKind, Net, NetInstance, Netlist, PinRef, SymmetryAxis,
};
pub use place::{GridPlacer, ManualPlacer, Placement, Placer};
pub use route::{
    ManhattanRouter, MultiPinStrategy, PolygonWireStylizer, RoutedDesign, Router, Wire, WireStyle,
};
