//! `eda-extract` — geometry → electrical netlist → SPICE deck.
//!
//! The bridge from a laid-out `CellId` to a runnable simulation deck.
//! `klayout-connect::extract_flat` already turns shapes on a conductor
//! layer into connected components; this crate adds two things on top:
//!
//! 1. **Device recognition** — walk the top cell's `Instance`s, and for
//!    each child cell, ask a [`DeviceRecognizer`] what electrical device
//!    it represents (R, C, L, …) and what its terminal port names are.
//!    Each terminal port's absolute placed position then resolves to a
//!    net by bbox containment.
//! 2. **Top-port relabelling** — `extract_flat` auto-names anonymous
//!    nets `net_0`, `net_1`, …. We rename a net to a top-level port's
//!    name when that port lies inside the net, so the emitted deck
//!    uses meaningful names (`vin`, `vout`, `gnd`) instead of numeric
//!    placeholders.
//!
//! The output [`ExtractedDesign`] is the layout-extracted analog of an
//! `eda-pnr::Netlist`: the same connectivity, but back-derived from
//! geometry rather than declared up front. [`to_spice_deck`] turns it
//! into an `eda_spice_emit::Netlist` ready for ngspice / LTspice.
//!
//! ## What this crate doesn't do (yet)
//!
//! - **Hierarchical extraction.** v1 walks the *top cell's* instances
//!   and treats each child as an opaque device. Recursing into composite
//!   children is a follow-up.
//! - **Multi-layer / via stitching.** Routes that hop layers via vias
//!   need the `Conductor`/`Via` config from `klayout-connect`. v1 is
//!   single conductor layer.
//! - **Device parameter back-extraction.** R / C / FET W/L come from
//!   the [`DeviceRecognizer`] (which reads them from the spike's own
//!   knobs), not from measured geometry. A future variant will compute
//!   R from RES rectangle aspect × sheet ρ.

use eda_spice_emit::Netlist as SpiceNetlist;
use klayout_connect::{extract_hierarchical, ExtractConfig, Net as ConnNet};
use klayout_core::{Bbox, CellId, Library, Point};
use thiserror::Error;

pub use klayout_connect::{Conductor, Via};

/// Electrical device class. Mirrors the SPICE element-prefix convention:
/// R/C/L/V/I cover everything the v1 demo touches; `Other` keeps the
/// door open for diodes (`D`), MOSFETs (`M`), subcircuits (`X`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    R,
    C,
    L,
    V,
    I,
    Other(String),
}

impl DeviceKind {
    /// SPICE element-line prefix (`R`, `C`, …).
    pub fn prefix(&self) -> &str {
        match self {
            DeviceKind::R => "R",
            DeviceKind::C => "C",
            DeviceKind::L => "L",
            DeviceKind::V => "V",
            DeviceKind::I => "I",
            DeviceKind::Other(s) => s.as_str(),
        }
    }
}

/// What a [`DeviceRecognizer`] returns for one child cell. The
/// `terminals` list is the canonical port-name order — emission writes
/// element lines as `<prefix><id> <net@terminals[0]> <net@terminals[1]> … <value>`.
#[derive(Clone, Debug)]
pub struct DeviceSpec {
    pub kind: DeviceKind,
    pub value: f64,
    pub terminals: Vec<String>,
}

/// Maps a child cell name to its electrical model. `None` means "skip
/// this instance" — useful for purely geometric children (alignment
/// markers, label cells) that don't show up in SPICE.
///
/// A blanket impl makes any `Fn(&str) -> Option<DeviceSpec>` work as a
/// recognizer, so callers can pass an inline closure for one-off use.
pub trait DeviceRecognizer {
    fn recognize(&self, cell_name: &str) -> Option<DeviceSpec>;
}

impl<F> DeviceRecognizer for F
where
    F: Fn(&str) -> Option<DeviceSpec>,
{
    fn recognize(&self, cell_name: &str) -> Option<DeviceSpec> {
        self(cell_name)
    }
}

/// Sky130 standard-device recognizer. Returns one `DeviceSpec` per
/// supported sky130 subckt cell name (`sky130_fd_pr__nfet_01v8`,
/// `sky130_fd_pr__pfet_01v8`, …), or `None` for unknown cells.
///
/// **Parameter handling.** `DeviceSpec::value` is a single `f64` —
/// fine for R/C/L/V, awkward for MOSFETs that carry W and L. v1
/// returns `value: 0.0` and lets the caller's `lvs_compare(...,
/// tol=0.0)` ignore the parameter check (0 == 0). When dimension-
/// preserving LVS lands, this returns `W·L` as the value so a
/// dimension swap on either side flags as a value mismatch.
///
/// **Terminal order.** `["d", "g", "s", "b"]` — ngspice convention
/// for `M*` element lines. Match this on the schematic side.
pub fn sky130_recognizer() -> impl DeviceRecognizer {
    |cell_name: &str| -> Option<DeviceSpec> {
        let mosfet = ["d", "g", "s", "b"];
        match cell_name {
            // 1.8V FETs (the LELO_EX devices).
            "sky130_fd_pr__nfet_01v8"
            | "sky130_fd_pr__nfet_01v8_lvt"
            | "sky130_fd_pr__pfet_01v8"
            | "sky130_fd_pr__pfet_01v8_lvt"
            | "sky130_fd_pr__pfet_01v8_hvt"
            // 5V cascoded FETs.
            | "sky130_fd_pr__nfet_05v0_nvt"
            | "sky130_fd_pr__nfet_g5v0d10v5"
            | "sky130_fd_pr__pfet_g5v0d10v5"
            // Generic stand-in name for synthetic test layouts.
            | "nfet_01v8" | "pfet_01v8"
            => Some(DeviceSpec {
                kind: DeviceKind::Other("M".into()),
                value: 0.0,
                terminals: mosfet.iter().map(|s| s.to_string()).collect(),
            }),
            // Sky130 poly resistors (the analog spike workhorses).
            // 2-terminal: positive, negative.
            "sky130_fd_pr__res_high_po_0p35"
            | "sky130_fd_pr__res_high_po_0p69"
            | "sky130_fd_pr__res_high_po_1p41"
            | "sky130_fd_pr__res_high_po_2p85"
            | "sky130_fd_pr__res_high_po_5p73"
            => Some(DeviceSpec {
                kind: DeviceKind::R,
                value: 0.0,
                terminals: vec!["a".into(), "b".into()],
            }),
            // MIM cap.
            "sky130_fd_pr__cap_mim_m3_1" | "sky130_fd_pr__cap_mim_m3_2"
            => Some(DeviceSpec {
                kind: DeviceKind::C,
                value: 0.0,
                terminals: vec!["a".into(), "b".into()],
            }),
            _ => None,
        }
    }
}

/// One electrical net after extraction + relabelling.
#[derive(Clone, Debug)]
pub struct ExtractedNet {
    pub name: String,
    pub bbox: Bbox,
    /// Merged polygons making up this net (top-cell frame). Used by
    /// [`net_containing`] for point-in-polygon containment so an
    /// L-shaped net's corner gap doesn't accidentally claim ports
    /// that lie in the bbox-but-not-on-metal region.
    pub polygons: Vec<klayout_core::Polygon>,
}

/// One device terminal pin, resolved to the net it lands on.
#[derive(Clone, Debug)]
pub struct ExtractedTerminal {
    /// Port name on the child cell (e.g. `"a"`, `"b"`).
    pub port: String,
    /// Absolute placed position of the port (after the instance trans).
    pub at: Point,
    /// Net the port lands on, by bbox containment.
    pub net: String,
}

/// One recognized device instance.
#[derive(Clone, Debug)]
pub struct ExtractedDevice {
    /// Top-cell instance index; doubles as the SPICE designator suffix
    /// (`R0`, `R1`, …). Stable across runs because it tracks
    /// insertion order.
    pub instance_index: usize,
    pub cell_name: String,
    pub kind: DeviceKind,
    pub value: f64,
    pub terminals: Vec<ExtractedTerminal>,
}

/// Top-level cell port, mapped to the extracted net it lands on.
#[derive(Clone, Debug)]
pub struct ExtractedTopPort {
    pub name: String,
    pub at: Point,
    pub net: String,
}

/// The full extraction result.
#[derive(Clone, Debug)]
pub struct ExtractedDesign {
    pub top_cell_name: String,
    pub nets: Vec<ExtractedNet>,
    pub devices: Vec<ExtractedDevice>,
    pub top_ports: Vec<ExtractedTopPort>,
}

#[derive(Debug, Error)]
pub enum ExtractError {
    /// A device terminal port was on a child cell, but no extracted net
    /// contained the port's absolute (placed) position. Means the
    /// layout has an unwired pin — exactly the bug this crate exists
    /// to catch (the LNA / MZI floorplans hit this for every pad).
    #[error(
        "instance #{instance_index} ({cell_name}): port '{port}' at ({x}, {y}) not on any net (unwired pin)"
    )]
    UnwiredPin {
        instance_index: usize,
        cell_name: String,
        port: String,
        x: i64,
        y: i64,
    },
    /// A child cell was instantiated in the top, the recognizer claimed
    /// it had a port (`terminals`), but the cell itself doesn't expose
    /// a port by that name. Recognizer bug.
    #[error("instance #{instance_index} ({cell_name}): cell has no port named '{port}'")]
    MissingPort {
        instance_index: usize,
        cell_name: String,
        port: String,
    },
    /// A top-level port doesn't lie on any extracted net. Either the
    /// port was placed off the conductor layer or the layout dropped a
    /// pad. Either way it's a layout bug.
    #[error("top port '{name}' at ({x}, {y}) not on any net")]
    DanglingTopPort { name: String, x: i64, y: i64 },
}

pub type Result<T> = std::result::Result<T, ExtractError>;

/// Run extraction on `top` and recognize devices via `recognizer`.
///
/// `config` goes straight to [`klayout_connect::extract_hierarchical`] —
/// the hierarchical variant is mandatory because device pads typically
/// live inside child cells (a `Resistor` cell carries its own METAL1
/// pads), and `extract_flat` would only see the top cell's own shapes
/// (the routing wire, but not the pads it connects to). After the
/// connected-components pass, every device terminal port and every
/// top-level port gets resolved to one of the extracted nets by bbox
/// containment. Nets containing a top-level port get renamed to that
/// port's name (so emitted decks read `vin / vout / gnd` rather than
/// `net_0/1/2`).
pub fn extract<R: DeviceRecognizer>(
    lib: &Library,
    top: CellId,
    config: &ExtractConfig,
    recognizer: &R,
) -> Result<ExtractedDesign> {
    let conn = extract_hierarchical(lib, top, config);

    let mut nets: Vec<ExtractedNet> = conn
        .nets()
        .iter()
        .map(|n: &ConnNet| ExtractedNet {
            name: n.name.to_string(),
            bbox: n.bbox,
            polygons: n.polygons.clone(),
        })
        .collect();

    let top_cell = lib.get(top);
    let top_cell_name = top_cell.name().to_string();

    // Top-port → net: run before device terminals so device terminals
    // see the friendly post-relabel names.
    let mut top_ports: Vec<ExtractedTopPort> = Vec::new();
    for port in top_cell.ports() {
        let pt = port.center;
        let Some(net_idx) = net_containing(&nets, pt) else {
            return Err(ExtractError::DanglingTopPort {
                name: port.name.to_string(),
                x: pt.x,
                y: pt.y,
            });
        };
        // Relabel only when the net still has its auto-name. A
        // user-provided label (from a Text shape) wins over an
        // inferred port name — labels are explicit intent.
        let net_name = if nets[net_idx].name.starts_with("net_") {
            nets[net_idx].name = port.name.to_string();
            port.name.to_string()
        } else {
            nets[net_idx].name.clone()
        };
        top_ports.push(ExtractedTopPort {
            name: port.name.to_string(),
            at: pt,
            net: net_name,
        });
    }

    let mut devices: Vec<ExtractedDevice> = Vec::new();
    for (idx, inst) in top_cell.instances().iter().enumerate() {
        let child = lib.get(inst.cell);
        let child_name = child.name().to_string();
        let Some(spec) = recognizer.recognize(&child_name) else {
            continue;
        };

        let mut terminals = Vec::with_capacity(spec.terminals.len());
        for term in &spec.terminals {
            let port = child.port(term).ok_or_else(|| ExtractError::MissingPort {
                instance_index: idx,
                cell_name: child_name.clone(),
                port: term.clone(),
            })?;
            let abs = port.transform(inst.trans);
            let pt = abs.center;
            let Some(net_idx) = net_containing(&nets, pt) else {
                return Err(ExtractError::UnwiredPin {
                    instance_index: idx,
                    cell_name: child_name.clone(),
                    port: term.clone(),
                    x: pt.x,
                    y: pt.y,
                });
            };
            terminals.push(ExtractedTerminal {
                port: term.clone(),
                at: pt,
                net: nets[net_idx].name.clone(),
            });
        }

        devices.push(ExtractedDevice {
            instance_index: idx,
            cell_name: child_name,
            kind: spec.kind,
            value: spec.value,
            terminals,
        });
    }

    Ok(ExtractedDesign { top_cell_name, nets, devices, top_ports })
}

/// Find the net whose actual merged geometry contains `p`. Falls back
/// to bbox-only when a net lacks polygon detail (e.g. caller built
/// `ExtractedNet` directly without going through `extract`). The
/// polygon-first path means an L-shaped net's bbox corner gap doesn't
/// accidentally claim a port that geometrically belongs to a
/// different net — the bug `lelo_ex` ran into pre-fix.
fn net_containing(nets: &[ExtractedNet], p: Point) -> Option<usize> {
    // Polygon containment: prefer real geometry. Iterate in net
    // declaration order; first match wins (matches the prior bbox
    // semantics for tie-breaking).
    for (i, n) in nets.iter().enumerate() {
        if n.polygons.is_empty() {
            continue;
        }
        // Quick reject by bbox before paying for the ray cast.
        let bb = n.bbox;
        if !(bb.min.x <= p.x && p.x <= bb.max.x && bb.min.y <= p.y && p.y <= bb.max.y) {
            continue;
        }
        if n.polygons.iter().any(|poly| point_in_polygon(p, poly)) {
            return Some(i);
        }
    }
    // No net carries polygons — fall back to bbox containment so
    // pre-polygon-aware callers still work.
    if nets.iter().all(|n| n.polygons.is_empty()) {
        return nets.iter().position(|n| {
            let bb = n.bbox;
            bb.min.x <= p.x && p.x <= bb.max.x && bb.min.y <= p.y && p.y <= bb.max.y
        });
    }
    None
}

/// Standard ray-casting point-in-polygon (even-odd rule). Edge points
/// count as inside — matches KLayout's edge-inclusive convention so a
/// port placed exactly on a wire boundary is treated as on-net.
fn point_in_polygon(p: Point, poly: &klayout_core::Polygon) -> bool {
    let hull = &poly.hull;
    if hull.is_empty() {
        return false;
    }
    // First check holes — a point inside a hole is *not* on the net.
    for hole in &poly.holes {
        if point_in_ring(p, hole) {
            return false;
        }
    }
    point_in_ring(p, hull)
}

fn point_in_ring(p: Point, ring: &[Point]) -> bool {
    let n = ring.len();
    if n < 3 {
        // Degenerate: a point on a line/single-vertex ring matches
        // only when geometrically equal.
        return ring.iter().any(|&v| v.x == p.x && v.y == p.y);
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let a = ring[i];
        let b = ring[j];
        // Edge-on-point: count as inside.
        if on_segment(p, a, b) {
            return true;
        }
        let crosses = (a.y > p.y) != (b.y > p.y);
        if crosses {
            // Avoid f64 — use exact i128 arithmetic for the slope.
            let ax = a.x as i128;
            let ay = a.y as i128;
            let bx = b.x as i128;
            let by = b.y as i128;
            let py = p.y as i128;
            let px = p.x as i128;
            let lhs = (bx - ax) * (py - ay);
            let rhs = (by - ay) * (px - ax);
            // Compare orientation in a sign-correct way: (b.y - a.y)
            // can be negative, so we compare sign-aligned products.
            let bya_pos = by > ay;
            if bya_pos {
                if lhs > rhs { inside = !inside; }
            } else if lhs < rhs {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn on_segment(p: Point, a: Point, b: Point) -> bool {
    // Colinearity via cross product, then bounding-box check.
    let cross = (b.x - a.x) as i128 * (p.y - a.y) as i128
              - (b.y - a.y) as i128 * (p.x - a.x) as i128;
    if cross != 0 { return false; }
    p.x >= a.x.min(b.x) && p.x <= a.x.max(b.x)
        && p.y >= a.y.min(b.y) && p.y <= a.y.max(b.y)
}

/// Emit `design` as a SPICE deck (just the device lines — caller adds
/// sources / `.op` / `.tran`). Net `gnd` (a common top-port name) is
/// rewritten to `0` so SPICE sees the standard ground node.
pub fn to_spice_deck(design: &ExtractedDesign, title: &str) -> SpiceNetlist {
    let mut net = SpiceNetlist::new(title);
    for dev in &design.devices {
        let mut line = format!("{}{}", dev.kind.prefix(), dev.instance_index);
        for term in &dev.terminals {
            line.push(' ');
            line.push_str(&spice_net_name(&term.net));
        }
        line.push(' ');
        line.push_str(&format!("{:.10e}", dev.value));
        net.add_element(line);
    }
    net
}

fn spice_net_name(n: &str) -> String {
    match n {
        "gnd" | "GND" | "0" => "0".to_string(),
        other => other.to_string(),
    }
}

// ── EM segment auto-extraction ───────────────────────────────────

/// Bridge from extracted nets + per-net peak currents to
/// `eda_em::Segment` triples. Each net becomes one segment whose
/// width is the net's bbox smaller-dimension (a reasonable
/// approximation for axis-aligned manhattan routes — captures the
/// narrow-strap regime EM typically catches).
///
/// `nets` come from `klayout_connect::Netlist::nets()`,
/// `peak_current_for_net` returns peak `|i(net)|` over the transient
/// (or 0.0 to skip a net entirely), `dbu` is the library's `dbu()`
/// for the µm conversion, and `layer_for_net` resolves each net to an
/// `eda_em::Layer`. Today's `klayout_connect::Net` doesn't carry the
/// originating layer (the bbox is union across via stitches) — the
/// caller resolves it from net naming or the conductor config.
///
/// **What this approximates:** one segment per net with W = smaller
/// bbox dim. A net with a long L-shape on M1 + a short tail on M2
/// gets one M1 segment with the M1 width — under-counts the M2 leg's
/// EM risk. Correct treatment is per-shape, which requires
/// shape-level extraction (the `klayout-connect::extract_flat` API
/// that v1 used populated `Net::shapes`; the new hierarchical path
/// drops it). When that's restored, swap this for shape-level.
pub fn em_segments_from_nets(
    nets: &[ConnNet],
    dbu: i64,
    peak_current_for_net: impl Fn(&str) -> f64,
    layer_for_net: impl Fn(&ConnNet) -> Option<eda_em::Layer>,
) -> Vec<eda_em::Segment> {
    let mut out = Vec::new();
    for net in nets {
        let i_a = peak_current_for_net(&net.name);
        if i_a == 0.0 {
            continue;
        }
        let Some(em_layer) = layer_for_net(net) else { continue };
        let bb = net.bbox;
        if bb.is_empty() { continue; }
        let w_dbu = (bb.max.x - bb.min.x).min(bb.max.y - bb.min.y);
        if w_dbu <= 0 { continue; }
        let width_um = w_dbu as f64 / dbu as f64;
        out.push(eda_em::Segment {
            net: net.name.to_string(),
            layer: em_layer,
            width_um,
            current_a: i_a,
        });
    }
    out
}

// ── LVS: schematic-vs-layout-extracted netlist diff ──────────────

/// One mismatch found by [`lvs_compare`].
#[derive(Clone, Debug, PartialEq)]
pub enum LvsMismatch {
    /// Schematic and layout disagree on device count.
    DeviceCountDiffers { schematic: usize, layout: usize },
    /// Same index, different device class (R vs C, …).
    DeviceKindDiffers { index: usize, schematic: DeviceKind, layout: DeviceKind },
    /// Same kind + index, value differs by more than the tolerance.
    DeviceValueDiffers {
        index: usize,
        kind: DeviceKind,
        schematic: f64,
        layout: f64,
        rel_err: f64,
    },
    /// A schematic terminal connects to a different net than the
    /// layout terminal at the same `(device_index, port)` slot.
    /// `schematic_net` and `layout_net` are the two net names.
    TerminalNetDiffers {
        index: usize,
        port: String,
        schematic_net: String,
        layout_net: String,
    },
}

/// One declared schematic device for LVS purposes — same shape as
/// [`DeviceSpec`] plus the per-terminal net it connects to. The
/// `instance_index` matches the layout side's stable index, so the
/// comparison is positional (`schematic[i]` vs `layout[i]`).
#[derive(Clone, Debug)]
pub struct SchematicDevice {
    pub instance_index: usize,
    pub kind: DeviceKind,
    pub value: f64,
    /// `(port, net)` pairs in canonical terminal order.
    pub terminals: Vec<(String, String)>,
}

/// Compare a schematic-declared netlist against a layout-extracted
/// one. Returns every mismatch found; empty `Vec` ⇒ LVS clean.
///
/// `value_tol` is the relative tolerance for value-equality
/// (`|s - l| / max(|s|, |l|, ε) ≤ value_tol`). Set `0.0` for exact
/// match, `1e-6` for "should be the same number".
///
/// Net names are compared after rewriting `gnd`/`GND` → `0` (matching
/// what [`to_spice_deck`] does), so the schematic and layout sides
/// don't need to agree on which casing they use for ground.
pub fn lvs_compare(
    schematic: &[SchematicDevice],
    layout: &[ExtractedDevice],
    value_tol: f64,
) -> Vec<LvsMismatch> {
    let mut out = Vec::new();
    if schematic.len() != layout.len() {
        out.push(LvsMismatch::DeviceCountDiffers {
            schematic: schematic.len(),
            layout: layout.len(),
        });
        // Continue per-index up to the shorter list — useful for
        // localising the divergence even when counts differ.
    }
    for (i, (s, l)) in schematic.iter().zip(layout.iter()).enumerate() {
        if s.kind != l.kind {
            out.push(LvsMismatch::DeviceKindDiffers {
                index: i,
                schematic: s.kind.clone(),
                layout: l.kind.clone(),
            });
            continue;
        }
        let denom = s.value.abs().max(l.value.abs()).max(f64::EPSILON);
        let rel = (s.value - l.value).abs() / denom;
        if rel > value_tol {
            out.push(LvsMismatch::DeviceValueDiffers {
                index: i,
                kind: s.kind.clone(),
                schematic: s.value,
                layout: l.value,
                rel_err: rel,
            });
        }
        // Per-terminal net check. We compare positionally — the
        // schematic side declares ports in canonical order, layout
        // side returns them in the same order from the recognizer.
        for ((sport, snet), lterm) in s.terminals.iter().zip(l.terminals.iter()) {
            let snet_norm = spice_net_name(snet);
            let lnet_norm = spice_net_name(&lterm.net);
            if snet_norm != lnet_norm {
                out.push(LvsMismatch::TerminalNetDiffers {
                    index: i,
                    port: sport.clone(),
                    schematic_net: snet_norm,
                    layout_net: lnet_norm,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod containment_tests {
    use super::*;
    use klayout_core::{Bbox, Point, Polygon};

    fn rect_poly(x0: i64, y0: i64, x1: i64, y1: i64) -> Polygon {
        Polygon::rect(Bbox::new(Point::new(x0, y0), Point::new(x1, y1)))
    }

    /// L-shape via two abutting rectangles. The bbox of the union
    /// covers the corner gap; geometric containment must reject it.
    fn l_shape() -> Vec<Polygon> {
        vec![
            rect_poly(0, 0, 100, 20),  // horizontal arm at the bottom
            rect_poly(0, 0,  20, 100), // vertical arm on the left
        ]
    }

    #[test]
    fn point_inside_l_arm_is_in_polygon() {
        let polys = l_shape();
        // Point on the horizontal arm.
        assert!(polys.iter().any(|p| point_in_polygon(Point::new(50, 10), p)));
        // Point on the vertical arm.
        assert!(polys.iter().any(|p| point_in_polygon(Point::new(10, 50), p)));
    }

    #[test]
    fn point_in_l_corner_gap_is_not_in_polygon() {
        // (50, 50) is inside the bbox [0,0]..[100,100] but neither
        // arm of the L covers it. Polygon containment must say NO.
        let polys = l_shape();
        assert!(!polys.iter().any(|p| point_in_polygon(Point::new(50, 50), p)));
    }

    #[test]
    fn net_containing_picks_correct_l_net_over_bbox_overlap() {
        // Two nets whose bboxes both contain (50, 50):
        //   netA: a tiny dot at (50, 50)
        //   netB: an L-shape whose corner gap is (50, 50)
        //
        // Bbox-only resolution would match whichever comes first;
        // polygon containment must pick netA (whose geometry
        // actually contains the point).
        let nets = vec![
            ExtractedNet {
                name: "L".into(),
                bbox: Bbox::new(Point::new(0, 0), Point::new(100, 100)),
                polygons: l_shape(),
            },
            ExtractedNet {
                name: "DOT".into(),
                bbox: Bbox::new(Point::new(45, 45), Point::new(55, 55)),
                polygons: vec![rect_poly(45, 45, 55, 55)],
            },
        ];
        let idx = net_containing(&nets, Point::new(50, 50)).expect("found");
        assert_eq!(nets[idx].name, "DOT", "expected DOT, got {}", nets[idx].name);
    }

    #[test]
    fn falls_back_to_bbox_when_no_polygons_present() {
        // Caller built ExtractedNet directly without polygons.
        // Containment must still work via bbox.
        let nets = vec![ExtractedNet {
            name: "X".into(),
            bbox: Bbox::new(Point::new(0, 0), Point::new(10, 10)),
            polygons: vec![],
        }];
        assert_eq!(net_containing(&nets, Point::new(5, 5)).map(|i| &*nets[i].name), Some("X"));
    }
}

#[cfg(test)]
mod lvs_tests {
    use super::*;

    fn term(port: &str, net: &str) -> (String, String) {
        (port.into(), net.into())
    }

    fn ext_term(port: &str, net: &str) -> ExtractedTerminal {
        ExtractedTerminal {
            port: port.into(),
            at: Point { x: 0, y: 0 },
            net: net.into(),
        }
    }

    #[test]
    fn matching_netlists_have_no_mismatches() {
        let schem = vec![SchematicDevice {
            instance_index: 0, kind: DeviceKind::R, value: 10_000.0,
            terminals: vec![term("a", "vin"), term("b", "vmid")],
        }];
        let layout = vec![ExtractedDevice {
            instance_index: 0, cell_name: "res".into(),
            kind: DeviceKind::R, value: 10_000.0,
            terminals: vec![ext_term("a", "vin"), ext_term("b", "vmid")],
        }];
        let mm = lvs_compare(&schem, &layout, 1e-9);
        assert!(mm.is_empty());
    }

    #[test]
    fn device_count_mismatch_is_reported() {
        let schem = vec![SchematicDevice {
            instance_index: 0, kind: DeviceKind::R, value: 1.0, terminals: vec![],
        }];
        let layout = vec![];
        let mm = lvs_compare(&schem, &layout, 0.0);
        assert!(matches!(mm[0], LvsMismatch::DeviceCountDiffers { schematic: 1, layout: 0 }));
    }

    #[test]
    fn value_within_tolerance_passes() {
        let schem = vec![SchematicDevice {
            instance_index: 0, kind: DeviceKind::R, value: 10_000.0, terminals: vec![],
        }];
        let layout = vec![ExtractedDevice {
            instance_index: 0, cell_name: "r".into(),
            kind: DeviceKind::R, value: 10_001.0, terminals: vec![],
        }];
        // 0.01% rel err, tol 1e-3 ⇒ pass.
        assert!(lvs_compare(&schem, &layout, 1e-3).is_empty());
        // tol 1e-5 ⇒ fail.
        assert!(matches!(
            lvs_compare(&schem, &layout, 1e-5)[0],
            LvsMismatch::DeviceValueDiffers { .. },
        ));
    }

    #[test]
    fn ground_alias_does_not_flag_terminal() {
        // Schematic says "gnd", layout extracted "0" — same net.
        let schem = vec![SchematicDevice {
            instance_index: 0, kind: DeviceKind::R, value: 1.0,
            terminals: vec![term("a", "gnd"), term("b", "x")],
        }];
        let layout = vec![ExtractedDevice {
            instance_index: 0, cell_name: "r".into(),
            kind: DeviceKind::R, value: 1.0,
            terminals: vec![ext_term("a", "0"), ext_term("b", "x")],
        }];
        assert!(lvs_compare(&schem, &layout, 0.0).is_empty());
    }

    #[test]
    fn terminal_swap_is_reported() {
        let schem = vec![SchematicDevice {
            instance_index: 0, kind: DeviceKind::R, value: 1.0,
            terminals: vec![term("a", "vin"), term("b", "vmid")],
        }];
        let layout = vec![ExtractedDevice {
            instance_index: 0, cell_name: "r".into(),
            kind: DeviceKind::R, value: 1.0,
            // swapped
            terminals: vec![ext_term("a", "vmid"), ext_term("b", "vin")],
        }];
        let mm = lvs_compare(&schem, &layout, 0.0);
        assert_eq!(mm.len(), 2);
        assert!(matches!(mm[0], LvsMismatch::TerminalNetDiffers { .. }));
    }
}
