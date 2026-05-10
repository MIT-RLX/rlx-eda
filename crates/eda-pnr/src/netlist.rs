//! Connectivity declaration — the input to placement and routing.
//!
//! A [`Netlist`] is the layout-domain analog of a flat schematic
//! netlist: a set of [`NetInstance`]s (each pointing at an
//! already-laid-out child `CellId`), a set of [`Net`]s naming which
//! pins are connected, and a set of [`ExternalPin`]s exposing
//! selected nets at the top level.
//!
//! Why a fresh type rather than reusing `eda-hir::SchematicIr`:
//! `SchematicIr` is *visual* — symbol positions in arbitrary
//! schematic-paper coordinates, polyline wires for rendering. PNR
//! needs symbolic connectivity decoupled from any layout decision.
//! A future `Connectivity<P>: Block` trait can produce a [`Netlist`]
//! directly; until that lands callers populate one with the builder
//! API.

use klayout_core::{CellId, LayerIndex};

/// One placed instance — a reference to a child cell that's already
/// in the library.
///
/// `fixed = true` pins the instance's position so AD placement
/// treats it as a constant rather than registering an `(x, y)`
/// Param pair. The intended use is I/O bond pads, hard macros, or
/// anything else whose location is dictated by the floorplan and
/// should not move under gradient descent.
#[derive(Clone, Debug)]
pub struct NetInstance {
    /// Stable, hierarchically-flat name. Used as the `Instance`
    /// name in the produced top cell and as a diagnostic identifier.
    pub name: String,
    /// The already-laid-out cell. Caller obtains this by calling
    /// each child block's `Layout::layout(lib, pdk)` *before*
    /// building the netlist.
    pub cell: CellId,
    /// `true` ⇒ AD placement holds this instance's position fixed.
    /// The `Trans` supplied to the `Placer` for this instance is
    /// authoritative; `eda-pnr::ad` reads it as a graph `Constant`
    /// rather than registering Params.
    pub fixed: bool,
}

/// A single connection — the set of pins that must end up on the
/// same electrical / optical / RF net.
///
/// `weight` scales this net's HPWL contribution in
/// [`crate::ad::combined_loss_graph`]. The default `1.0` makes
/// every net count equally; higher values prioritize keeping
/// timing-critical paths short, lower values let bias / power
/// nets give way to signal congestion. Standard timing-driven
/// placement uses slack-derived weights (negative-slack-weighted
/// HPWL, Eisenmann/Johannes 1998).
#[derive(Clone, Debug)]
pub struct Net {
    pub name: String,
    /// Pins this net touches. A net with `< 2` pins is silently
    /// dropped from routing (an internal-only or unconnected node).
    pub pins: Vec<PinRef>,
    /// Override [`Netlist::default_signal_layer`] for this net —
    /// e.g. heavy power buses on a top metal while signals stay on
    /// metal1.
    pub layer: Option<LayerIndex>,
    /// HPWL multiplier in AD placement. Default `1.0`.
    pub weight: f32,
}

/// `(instance, port-name)` pair. The instance index points into
/// [`Netlist::instances`]; the port name is resolved against the
/// instance's `Cell::port(name)` at routing time.
#[derive(Clone, Debug)]
pub struct PinRef {
    pub instance: usize,
    pub port: String,
}

/// External pin exposed at the top level. Backed by the first pin
/// of the net it references — that pin's absolute (placed) position
/// becomes the external port's location.
#[derive(Clone, Debug)]
pub struct ExternalPin {
    pub name: String,
    /// Net the external pin is electrically tied to. Must exist in
    /// [`Netlist::nets`].
    pub net: String,
    pub direction: Option<eda_hir::PinDirection>,
}

/// Axis a [`MatchKind::Mirror`] reflects across, or that
/// [`MatchKind::Interdigitated`] arrays along.
///
/// `Vertical` = the axis line itself is vertical (constant x), so
/// instances mirror in their x-coord and share their y-coord. This
/// is the differential-pair convention: NMOS pair sitting at the
/// same y, equidistant left/right of a vertical centerline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetryAxis {
    /// Mirror axis is a vertical line `x = axis_coord` ⇒ x-coords
    /// reflect, y-coords are equal.
    Vertical,
    /// Mirror axis is a horizontal line `y = axis_coord` ⇒ y-coords
    /// reflect, x-coords are equal.
    Horizontal,
}

/// Geometric matching constraint. Realised by the AD placer via
/// [`crate::ad::symmetry_loss_graph`] as a quadratic penalty on
/// instance positions; the regular [`crate::PnrFlow`] ignores it.
///
/// Indices reference [`Netlist::instances`].
#[derive(Clone, Debug)]
pub enum MatchKind {
    /// Two instances mirrored across `axis` at coordinate `axis_coord`.
    /// The canonical differential pair: `M1` and `M2` placed at
    /// `(c - d, y)` and `(c + d, y)` for some `d`. AD penalty:
    /// `(x_a + x_b − 2·c)² + (y_a − y_b)²` (axes swapped for
    /// horizontal). The optimizer is free to pick `d` — only the
    /// reflection symmetry is constrained.
    Mirror {
        a: usize,
        b: usize,
        axis: SymmetryAxis,
        axis_coord: f32,
    },
    /// `instances` must share centroid `center`. Standard
    /// common-centroid matching for ratioed devices (current
    /// mirrors, capacitor arrays). AD penalty:
    /// `(mean(x_i) − cx)² + (mean(y_i) − cy)²`.
    ///
    /// Note this constrains only the centroid, not the relative
    /// arrangement; combine with one or more [`MatchKind::Mirror`]
    /// groups (or an [`MatchKind::Interdigitated`] group) to also
    /// pin the pattern.
    CommonCentroid {
        instances: Vec<usize>,
        center: (f32, f32),
    },
    /// `instances` form an arithmetic progression along `axis`:
    /// position k sits at `origin + k · pitch`. The other axis is
    /// constrained to be equal across all members. Standard
    /// row-of-fingers / interdigitated array layout.
    Interdigitated {
        instances: Vec<usize>,
        axis: SymmetryAxis,
        origin: f32,
        pitch: f32,
    },
}

/// One named matching constraint. `weight` scales the penalty term
/// in the combined loss; default `1.0`. See [`MatchKind`] for the
/// per-kind quadratic form.
#[derive(Clone, Debug)]
pub struct MatchGroup {
    pub name: String,
    pub kind: MatchKind,
    pub weight: f32,
}

/// The complete declarative netlist. Build via [`Netlist::new`] +
/// [`Netlist::add_instance`] / [`Netlist::connect`] / [`Netlist::expose`].
#[derive(Clone, Debug)]
pub struct Netlist {
    pub name: String,
    pub instances: Vec<NetInstance>,
    pub nets: Vec<Net>,
    pub external_pins: Vec<ExternalPin>,
    /// Geometric matching constraints — differential pairs,
    /// common-centroid groups, interdigitated arrays. Empty by
    /// default; populated via [`Netlist::match_mirror`] /
    /// [`Netlist::match_common_centroid`] /
    /// [`Netlist::match_interdigitated`].
    pub match_groups: Vec<MatchGroup>,
    /// Default routing layer for every [`Net`] without an explicit
    /// override. Required when the netlist has any signal nets.
    pub default_signal_layer: Option<LayerIndex>,
}

impl Netlist {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instances: Vec::new(),
            nets: Vec::new(),
            external_pins: Vec::new(),
            match_groups: Vec::new(),
            default_signal_layer: None,
        }
    }

    pub fn with_default_signal_layer(mut self, layer: LayerIndex) -> Self {
        self.default_signal_layer = Some(layer);
        self
    }

    /// Add an instance, returning the index callers thread into
    /// [`PinRef::instance`] for subsequent [`Netlist::connect`] calls.
    pub fn add_instance(&mut self, name: impl Into<String>, cell: CellId) -> usize {
        let idx = self.instances.len();
        self.instances.push(NetInstance { name: name.into(), cell, fixed: false });
        idx
    }

    /// Add an instance whose AD-placement position is fixed (a
    /// graph `Constant` rather than a `Param`). Used for I/O pads,
    /// hard macros, anything whose floorplan slot the placer must
    /// not move.
    pub fn add_fixed_instance(&mut self, name: impl Into<String>, cell: CellId) -> usize {
        let idx = self.instances.len();
        self.instances.push(NetInstance { name: name.into(), cell, fixed: true });
        idx
    }

    /// Mark an existing instance's position as fixed.
    pub fn set_fixed(&mut self, instance: usize, fixed: bool) {
        if let Some(inst) = self.instances.get_mut(instance) {
            inst.fixed = fixed;
        }
    }

    /// Add `(instance, port)` to the given net, creating the net if
    /// it doesn't exist. Idempotent on re-adding the same pin.
    pub fn connect(&mut self, net_name: impl Into<String>, instance: usize, port: impl Into<String>) {
        let net_name: String = net_name.into();
        let port: String = port.into();
        let pin = PinRef { instance, port: port.clone() };
        if let Some(net) = self.nets.iter_mut().find(|n| n.name == net_name) {
            if !net.pins.iter().any(|p| p.instance == instance && p.port == port) {
                net.pins.push(pin);
            }
        } else {
            self.nets.push(Net {
                name: net_name,
                pins: vec![pin],
                layer: None,
                weight: 1.0,
            });
        }
    }

    /// Override an existing net's HPWL weight. No-op if the net
    /// doesn't exist (matches the lenient style of `set_net_layer`).
    pub fn set_net_weight(&mut self, net_name: &str, weight: f32) {
        if let Some(net) = self.nets.iter_mut().find(|n| n.name == net_name) {
            net.weight = weight;
        }
    }

    /// Override the routing layer for one net.
    pub fn set_net_layer(&mut self, net_name: &str, layer: LayerIndex) {
        if let Some(net) = self.nets.iter_mut().find(|n| n.name == net_name) {
            net.layer = Some(layer);
        }
    }

    /// Declare a differential-pair / mirror constraint: instances
    /// `a` and `b` must be reflections across `axis_coord` along
    /// `axis`. Indices must point into [`Netlist::instances`];
    /// out-of-range indices are silently ignored at AD-graph build
    /// time so duplicate / stale declarations don't crash flow.
    pub fn match_mirror(
        &mut self,
        name: impl Into<String>,
        a: usize,
        b: usize,
        axis: SymmetryAxis,
        axis_coord: f32,
    ) {
        self.match_groups.push(MatchGroup {
            name: name.into(),
            kind: MatchKind::Mirror { a, b, axis, axis_coord },
            weight: 1.0,
        });
    }

    /// Declare a common-centroid constraint: the supplied instances'
    /// centroid must equal `center`. Requires at least 2 instances;
    /// shorter groups are dropped at graph-build time.
    pub fn match_common_centroid(
        &mut self,
        name: impl Into<String>,
        instances: Vec<usize>,
        center: (f32, f32),
    ) {
        self.match_groups.push(MatchGroup {
            name: name.into(),
            kind: MatchKind::CommonCentroid { instances, center },
            weight: 1.0,
        });
    }

    /// Declare an interdigitated row: instances spaced at `pitch`
    /// along `axis`, equal on the other axis, starting at `origin`.
    /// The order of `instances` defines the row order (index `k`
    /// targets `origin + k · pitch`).
    pub fn match_interdigitated(
        &mut self,
        name: impl Into<String>,
        instances: Vec<usize>,
        axis: SymmetryAxis,
        origin: f32,
        pitch: f32,
    ) {
        self.match_groups.push(MatchGroup {
            name: name.into(),
            kind: MatchKind::Interdigitated { instances, axis, origin, pitch },
            weight: 1.0,
        });
    }

    /// Override the penalty weight on a previously-declared match
    /// group. No-op if the name doesn't resolve.
    pub fn set_match_weight(&mut self, name: &str, weight: f32) {
        if let Some(g) = self.match_groups.iter_mut().find(|g| g.name == name) {
            g.weight = weight;
        }
    }

    /// Expose `net_name` as an external pin called `pin_name`. The
    /// pin location is resolved at flow time from the first pin
    /// declared on the net.
    pub fn expose(
        &mut self,
        pin_name: impl Into<String>,
        net_name: impl Into<String>,
        direction: Option<eda_hir::PinDirection>,
    ) {
        self.external_pins.push(ExternalPin {
            name: pin_name.into(),
            net: net_name.into(),
            direction,
        });
    }
}

// Builder/connectivity behavior is exercised end-to-end via
// `tests/two_resistor.rs`, which builds a real Library, lays out
// child cells, and asserts on the produced top cell's shapes /
// ports. A separate unit test file would need to fabricate
// `CellId` values out of thin air — `CellId` is opaque on purpose.
