//! Routing — turn a placed [`Netlist`] into wire shapes.
//!
//! [`Router`] is one method: given a netlist + its [`Placement`] +
//! the library (so it can resolve port positions on placed child
//! cells), emit one or more [`Wire`]s per routable net plus a list
//! of nets that didn't make it.
//!
//! [`ManhattanRouter`] is the only built-in for now. It wraps
//! `klayout-route::ManhattanPlanner` (90° one-bend planner) plus a
//! choice of stylizer ([`WireStyle::Path`] for canonical
//! `Shape::Path` output, [`WireStyle::Polygon`] for axis-aligned
//! rectangles that DRC and LVS extractors actually see).
//!
//! ## Multi-pin nets
//!
//! Multi-pin nets fan out as a star from `pins[0]` to every other
//! pin. That's not minimum-spanning-tree-optimal — the
//! to-do-Phase-2 list is using `klayout_route::rsmt` for proper
//! Steiner trees — but it's the same shape `spike-divider-block`
//! has been doing by hand, lifted into one place.

use klayout_core::{LayerIndex, Library, Path, PathCap, Point, Rect, Shape};
use klayout_route::{rsmt, ManhattanPlanner, Obstacles, Planner, Stylizer, WirePathStylizer};
use smallvec::SmallVec;

use crate::netlist::Netlist;
use crate::place::Placement;

/// One routed net's contribution to the layout.
#[derive(Clone, Debug)]
pub struct Wire {
    pub net: String,
    pub layer: LayerIndex,
    pub shapes: Vec<Shape>,
}

/// What the router produced. `failed_nets` lists nets that couldn't
/// be routed — typically because they had `< 2` pins (internal /
/// unconnected) or because no signal layer was available.
#[derive(Clone, Debug, Default)]
pub struct RoutedDesign {
    pub wires: Vec<Wire>,
    pub failed_nets: Vec<String>,
}

pub trait Router {
    fn route(&self, netlist: &Netlist, place: &Placement, lib: &Library) -> RoutedDesign;
}

/// How a planned `Path` becomes shapes on the wire layer.
#[derive(Clone, Copy, Debug)]
pub enum WireStyle {
    /// Emit a single `Shape::Path`. Canonical, fast, but
    /// `klayout_geom::Region`, `klayout_drc::*`, and
    /// `klayout_connect::extract_*` only see polygons / boxes —
    /// they ignore Paths. Fine for visual rendering, not for
    /// extraction.
    Path,
    /// Emit one axis-aligned rectangle per Manhattan segment.
    /// Visible to DRC + LVS. The same stylizer
    /// `spike-divider-block` has been carrying inline since day 1;
    /// lifted into [`PolygonWireStylizer`] in this module.
    Polygon,
}

/// How `≥ 3`-pin nets are decomposed into routable segments.
///
/// Phase 1 of `eda-pnr` shipped only `Star`; Phase 4 adds `Steiner`
/// via [`klayout_route::rsmt`]. 2-pin nets always go through the
/// `ManhattanPlanner` directly; this enum only kicks in for nets
/// with three or more pins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiPinStrategy {
    /// Fan out from `pins[0]` to every other pin via a Manhattan
    /// 1-bend planner. Cheap, but inflates wirelength on dense
    /// multi-pin nets.
    Star,
    /// Rectilinear Steiner Minimal Tree (RSMT). Wraps
    /// `klayout_route::rsmt` and stamps each edge as a Manhattan
    /// segment. Optimal in total wirelength up to constant
    /// factors, well within seconds for hundreds of pins.
    Steiner,
}

/// 90°-Manhattan router. Single-bend per 2-pin segment, single
/// layer per net (overrideable via [`crate::Net::layer`]).
/// Multi-pin nets follow [`MultiPinStrategy`] — default Star,
/// switch to Steiner for tighter wirelength on dense nets.
#[derive(Clone, Debug)]
pub struct ManhattanRouter {
    pub style: WireStyle,
    pub multi_pin: MultiPinStrategy,
}

impl Default for ManhattanRouter {
    fn default() -> Self {
        Self { style: WireStyle::Polygon, multi_pin: MultiPinStrategy::Star }
    }
}

impl ManhattanRouter {
    pub fn new(style: WireStyle) -> Self {
        Self { style, multi_pin: MultiPinStrategy::Star }
    }

    pub fn with_multi_pin(mut self, strat: MultiPinStrategy) -> Self {
        self.multi_pin = strat;
        self
    }
}

impl Router for ManhattanRouter {
    fn route(&self, netlist: &Netlist, place: &Placement, lib: &Library) -> RoutedDesign {
        let mut wires = Vec::new();
        let mut failed = Vec::new();

        for net in &netlist.nets {
            if net.pins.len() < 2 {
                failed.push(net.name.clone());
                continue;
            }
            let layer = match net.layer.or(netlist.default_signal_layer) {
                Some(l) => l,
                None => {
                    failed.push(net.name.clone());
                    continue;
                }
            };

            // Resolve every pin to an absolute `Port` in the top
            // cell's coordinate frame.
            let mut abs_ports = Vec::with_capacity(net.pins.len());
            let mut net_failed = false;
            for pin in &net.pins {
                let inst = match netlist.instances.get(pin.instance) {
                    Some(i) => i,
                    None => { net_failed = true; break; }
                };
                let cell = lib.get(inst.cell);
                let port = match cell.port(&pin.port) {
                    Some(p) => p.transform(place.transforms[pin.instance]),
                    None => { net_failed = true; break; }
                };
                abs_ports.push(port);
            }
            if net_failed {
                failed.push(net.name.clone());
                continue;
            }

            // Multi-pin nets dispatch on `multi_pin`. 2-pin nets
            // always go through the planner directly.
            let mut shapes = Vec::new();
            if abs_ports.len() == 2 || self.multi_pin == MultiPinStrategy::Star {
                for tgt in abs_ports.iter().skip(1) {
                    let path =
                        ManhattanPlanner.plan(&abs_ports[0], tgt, &Obstacles::default());
                    push_styled(self.style, layer, path, &mut shapes);
                }
            } else {
                // Steiner: build RSMT over pin centers, stamp each
                // edge as a 2-point Manhattan path with the
                // widest-port width (matches `ManhattanPlanner`'s
                // own width policy).
                let pins: Vec<Point> = abs_ports.iter().map(|p| p.center).collect();
                let tree = rsmt(&pins);
                let width = abs_ports
                    .iter()
                    .map(|p| p.width)
                    .max()
                    .unwrap_or(1);
                for &(a, b) in &tree.edges {
                    let pa = tree.nodes[a];
                    let pb = tree.nodes[b];
                    if pa == pb { continue; }
                    let mut pts: SmallVec<[Point; 4]> = SmallVec::new();
                    // RSMT edges may have one common axis, or join
                    // two Steiner points along an L-shape. Insert
                    // a corner if neither x nor y is shared.
                    pts.push(pa);
                    if pa.x != pb.x && pa.y != pb.y {
                        pts.push(Point::new(pb.x, pa.y));
                    }
                    pts.push(pb);
                    let path = Path {
                        points: pts,
                        width,
                        begin_ext: 0,
                        end_ext: 0,
                        cap: PathCap::Flat,
                    };
                    push_styled(self.style, layer, path, &mut shapes);
                }
            }
            wires.push(Wire { net: net.name.clone(), layer, shapes });
        }

        RoutedDesign { wires, failed_nets: failed }
    }
}

fn push_styled(style: WireStyle, layer: LayerIndex, path: Path, out: &mut Vec<Shape>) {
    let stylized: Vec<(LayerIndex, Shape)> = match style {
        WireStyle::Path => WirePathStylizer.stylize(layer, path),
        WireStyle::Polygon => PolygonWireStylizer.stylize(layer, path),
    };
    for (_, s) in stylized { out.push(s); }
}

/// Polygonising stylizer — fattens each axis-aligned segment of a
/// `Path` into a `Shape::Box` rectangle on the wire layer plus an
/// elbow rectangle covering each corner. Lifted from
/// `spike-divider-block::PolygonWireStylizer` so every PNR consumer
/// gets the DRC-visible wire geometry without the local copy.
///
/// Limitation: assumes axis-aligned segments (the
/// `ManhattanPlanner` invariant). General polygonization for
/// arbitrary-angle paths becomes a sibling stylizer when we add
/// non-Manhattan routing.
#[derive(Clone, Copy, Debug)]
pub struct PolygonWireStylizer;

impl Stylizer for PolygonWireStylizer {
    fn stylize(&self, layer: LayerIndex, path: Path) -> Vec<(LayerIndex, Shape)> {
        let half = path.width / 2;
        let mut out = Vec::new();
        for w in path.points.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (xmin, xmax) = (a.x.min(b.x) - half, a.x.max(b.x) + half);
            let (ymin, ymax) = (a.y.min(b.y) - half, a.y.max(b.y) + half);
            out.push((
                layer,
                Shape::Box(Rect::new(klayout_core::Bbox::new(
                    Point::new(xmin, ymin),
                    Point::new(xmax, ymax),
                ))),
            ));
        }
        out
    }
}
