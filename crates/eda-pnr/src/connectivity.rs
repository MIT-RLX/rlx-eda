//! `Connectivity<P>` — a sibling of `eda_hir::Layout<P>` that lets
//! a `Block` declare *what's connected to what* once, and get the
//! layout-side wiring for free via `eda_pnr::pnr_layout`.
//!
//! Until this trait landed, every composite block whose layout
//! ran through PnrFlow had two near-identical bodies:
//!
//! 1. A `Layout::layout` that built the netlist + transforms +
//!    PnrFlow.run, returning the resulting CellId.
//! 2. A `Schematic::schematic` that emitted a parallel
//!    visual-schematic `SchematicIr`.
//!
//! `Connectivity<P>` separates the connectivity from the placement
//! decision: implementors return a [`Netlist`] (instances + nets +
//! external pins) and a list of [`Trans`] for the placer. The free
//! function [`pnr_layout`] runs the standard `PnrFlow`. Callers who
//! want a custom router can use [`pnr_layout_with`] or fall back
//! to a hand-rolled Layout impl.

use klayout_core::{CellId, Library, Trans};

use eda_hir::{Block, Layout};

use crate::flow::PnrFlow;
use crate::netlist::Netlist;
use crate::place::ManualPlacer;
use crate::route::{ManhattanRouter, Router};

/// Composite-block connectivity declaration. Implementors return
/// the `Netlist` + the per-instance `Trans` placement; the PNR
/// flow handles routing + top-cell stamping.
///
/// This is opt-in: a block that already has a hand-written
/// `Layout::layout` doesn't need to implement `Connectivity`. New
/// composite blocks should prefer `Connectivity` so the placement
/// decision and the connectivity declaration stay separated.
pub trait Connectivity<P>: Block {
    /// Build the netlist after laying out every child cell. The
    /// implementor is responsible for calling `child.layout(lib,
    /// pdk)` for each instance, then registering them via
    /// [`Netlist::add_instance`] / [`Netlist::add_fixed_instance`].
    fn connectivity(&self, lib: &Library, pdk: &P) -> Netlist;

    /// One `Trans` per instance returned by `connectivity()`,
    /// in the same order. For AD-driven placement, this returns
    /// the *seed* placement — `pnr_layout` always uses a
    /// `ManualPlacer` over these transforms; if you want
    /// SA / HPWL-AD optimization, override
    /// `Layout::layout` directly and skip the helper.
    fn transforms(&self, netlist: &Netlist, lib: &Library) -> Vec<Trans>;
}

/// Run the canonical [`PnrFlow`] (`ManualPlacer` + default
/// [`ManhattanRouter`]) on a [`Connectivity`] implementor and
/// return the produced top `CellId`. The standard implementation
/// of `Layout::layout` for composite blocks reduces to a one-line
/// forwarder:
///
/// ```ignore
/// impl<P: MyPdk> Layout<P> for MyComposite {
///     fn layout(&self, lib: &Library, pdk: &P) -> CellId {
///         eda_pnr::pnr_layout(self, lib, pdk)
///     }
/// }
/// ```
pub fn pnr_layout<P, T>(t: &T, lib: &Library, pdk: &P) -> CellId
where
    T: Connectivity<P> + Layout<P>,
{
    pnr_layout_with(t, lib, pdk, ManhattanRouter::default())
}

/// Same as [`pnr_layout`] but takes a custom router (e.g.
/// [`MultiPinStrategy::Steiner`](crate::MultiPinStrategy) for
/// blocks with dense ≥3-pin nets, or a future multi-layer
/// router).
pub fn pnr_layout_with<P, T, R>(t: &T, lib: &Library, pdk: &P, router: R) -> CellId
where
    T: Connectivity<P> + Layout<P>,
    R: Router,
{
    let netlist = t.connectivity(lib, pdk);
    let transforms = t.transforms(&netlist, lib);
    let placer = ManualPlacer::new(transforms);
    PnrFlow::new(placer, router).run(&netlist, lib).top
}
