//! End-to-end PNR flow: place, route, stamp into a top cell.
//!
//! [`PnrFlow::run`] is the single entry point most consumers want.
//! Hand it a [`Netlist`], a [`Library`] holding every child cell's
//! geometry, and the placer + router you want — it returns the
//! `CellId` of the produced top cell, which you can then
//! instantiate, render, GDS-export, etc.
//!
//! The flow is deliberately linear:
//!
//! 1. `placer.place(netlist, lib)`           — decide each instance's `Trans`
//! 2. `router.route(netlist, place, lib)`    — emit wires for every net
//! 3. Build a fresh `CellBuilder`            — instances + wires + external pins
//! 4. `lib.insert(top)` → `CellId`           — return the new top cell
//!
//! Failed nets (from the router) propagate as a `RoutedDesign` field
//! on the result so callers can inspect what didn't make it without
//! a panic.

use klayout_core::{CellBuilder, CellId, Instance, Library, Port};

use crate::netlist::Netlist;
use crate::place::{Placement, Placer};
use crate::route::{RoutedDesign, Router};

pub struct PnrFlow<P: Placer, R: Router> {
    pub placer: P,
    pub router: R,
}

impl<P: Placer, R: Router> PnrFlow<P, R> {
    pub fn new(placer: P, router: R) -> Self { Self { placer, router } }

    /// Run the full flow and return everything: the produced top
    /// `CellId`, the [`Placement`] (so the caller can reuse it for
    /// reporting / chart overlays), and the [`RoutedDesign`] (so
    /// failed nets are surfaceable).
    pub fn run(&self, netlist: &Netlist, lib: &Library) -> PnrResult {
        let placement = self.placer.place(netlist, lib);
        let routed = self.router.route(netlist, &placement, lib);

        let mut top = CellBuilder::new(netlist.name.clone());

        // Stamp child instances at their placed transforms. The
        // public `Instance::new` is the only field we touch — the
        // hierarchical name is recorded in the netlist for
        // diagnostics / extraction; klayout-core's Instance doesn't
        // expose a name slot today.
        for (inst, t) in netlist.instances.iter().zip(placement.transforms.iter()) {
            let _ = &inst.name; // anchor; name is netlist-side metadata
            top.add_instance(Instance::new(inst.cell, *t));
        }

        // Stamp every routed wire shape on its declared layer.
        for wire in &routed.wires {
            for shape in &wire.shapes {
                top.add_shape(wire.layer, shape.clone());
            }
        }

        // Promote each external pin: find its net, pull the first
        // pin's absolute port, and re-emit at the top level. The
        // first-pin choice is deterministic across rebuilds because
        // `Net::pins` order is preserved by `Netlist::connect`.
        for ext in &netlist.external_pins {
            if let Some(net) = netlist.nets.iter().find(|n| n.name == ext.net) {
                if let Some(first) = net.pins.first() {
                    if let Some(inst) = netlist.instances.get(first.instance) {
                        let cell = lib.get(inst.cell);
                        if let Some(port) = cell.port(&first.port) {
                            let abs = port.transform(placement.transforms[first.instance]);
                            let mut new_port = Port::new(
                                ext.name.clone(),
                                abs.layer,
                                abs.center,
                                abs.angle,
                                abs.width,
                            );
                            new_port = new_port.with_kind(abs.kind);
                            top.add_port(new_port);
                        }
                    }
                }
            }
        }

        let top_id = lib.insert(top);
        PnrResult { top: top_id, placement, routed }
    }
}

/// Result of [`PnrFlow::run`] — top cell + the intermediate
/// placement and routing artifacts (the latter so callers can report
/// failed nets, visualize the placement, drive AD losses, etc.).
#[derive(Clone, Debug)]
pub struct PnrResult {
    pub top: CellId,
    pub placement: Placement,
    pub routed: RoutedDesign,
}
