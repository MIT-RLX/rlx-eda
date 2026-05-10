//! Layout renderer: walk a `Cell` tree, draw shapes per layer, emit SVG.
//!
//! Strategy: flatten by recursive descent, applying each instance's
//! transform. Each leaf shape is mapped from DBU (y-up, integer) to SVG
//! user units (y-down, float) via a single linear map driven by
//! [`crate::Style::units_per_dbu`].
//!
//! Layer color is picked from a fixed palette keyed by `(layer, datatype)`
//! so the same GDS pair gets the same color across renders. Layers that
//! have a `LayerInfo::name` show up in the legend; unnamed layers fall
//! back to `"L<layer>/<datatype>"`.

use std::collections::{BTreeMap, HashSet};

use klayout_core::{
    Bbox, Cell, CellId, HAlign, LayerIndex, LayerInfo, Library, Path, Point, Polygon, Rect, Shape,
    Text, Trans, VAlign,
};

use crate::svg::{palette_color, xml_escape, SvgDoc};
use crate::Style;

#[cfg(feature = "gds")]
mod foundry_io {
    //! Re-exports of `klayout-io` so callers can write GDS / OASIS
    //! straight from the same module that produces SVG. This means
    //! one import (`eda_viz::layout`) covers the full "render or hand
    //! to a foundry tool" surface.
    //!
    //! These are zero-overhead re-exports — the actual implementation
    //! lives in the sibling `klayout-io` crate. Their presence here
    //! is documentation: GDS export is a first-class output of the
    //! rlx-eda layout flow, not an "import this other crate" detour.

    pub use klayout_io::{
        read_gds_bytes, read_gds_path, write_gds_bytes, write_gds_path,
        read_oasis_bytes, read_oasis_path, write_oasis_bytes, write_oasis_path,
    };
}

#[cfg(feature = "gds")]
pub use foundry_io::*;

/// Render the cell rooted at `top` to a complete SVG document.
pub fn render_to_svg(lib: &Library, top: CellId, style: &Style) -> String {
    let cell = lib.get(top);
    let bbox = cell.full_bbox(lib);
    if bbox.is_empty() {
        // Empty cell — emit a tiny placeholder so the file still parses.
        return SvgDoc::new(0.0, 0.0, 1.0, 1.0).finish();
    }

    // Collect shapes per layer first so each layer renders as a single
    // <g> — keeps z-order predictable and lets us size the legend
    // strip above the cell from the actual layer count.
    let mut by_layer: BTreeMap<LayerIndex, Vec<(Shape, Trans)>> = BTreeMap::new();
    let mut ports: Vec<(Point, &'static str /* stroke color */)> = Vec::new();
    let mut instances: Vec<InstanceInfo> = Vec::new();
    let hidden: HashSet<LayerIndex> = style.hidden_layers.iter().copied().collect();
    let mut on_path: HashSet<CellId> = HashSet::new();
    collect(
        lib, top, &cell, Trans::IDENTITY,
        &mut by_layer, &mut ports, &mut instances,
        style.show_ports, &hidden, &mut on_path,
    );

    let pad = style.pad_dbu;
    let (vx, mut vy, vw, mut vh) = viewbox(bbox, pad, style.units_per_dbu);

    // Reserve a strip above the content for the legend so it never sits
    // on top of geometry. Strip height grows with the number of drawn
    // layers; width matches the legend's intrinsic width.
    let legend_row_h = 12.0;
    let legend_pad = 4.0;
    let legend_strip_h = if style.show_legend && !by_layer.is_empty() {
        legend_pad * 2.0 + legend_row_h * by_layer.len() as f64 + 4.0
    } else {
        0.0
    };
    vy -= legend_strip_h;
    vh += legend_strip_h;

    let mut doc = SvgDoc::new(vx, vy, vw, vh);

    if let Some(bg) = &style.background {
        let attrs = format!("fill=\"{bg}\" stroke=\"none\"");
        doc.rect(vx, vy, vw, vh, &attrs);
    }

    let mut legend: Vec<(LayerIndex, LayerInfo, String)> = Vec::new();

    // Assign palette slots in BTreeMap order — drawn layers get
    // sequential colors, so adjacent GDS numbers always look distinct.
    // BTreeMap iteration is by `LayerIndex`, which is the stable order
    // layers were registered in the `Library`.
    for (slot, (lyr, items)) in by_layer.iter().enumerate() {
        let info = lib.layer_info(*lyr);
        let color: String = match style.layer_palette.as_ref()
            .and_then(|p| p.get(info.layer, info.datatype))
        {
            Some(c) => c.to_string(),
            None    => palette_color(slot as u32).to_string(),
        };
        legend.push((*lyr, info.clone(), color.clone()));

        let mut group_attrs = format!(
            "fill=\"{color}\" fill-opacity=\"{:.3}\" stroke=\"{color}\" stroke-width=\"{:.3}\"",
            style.fill_opacity, style.stroke_width,
        );
        if style.emit_css_classes {
            // Stable, CSS-friendly class — `layer-{layer}-{datatype}`.
            // External stylesheets can override fill/stroke per layer.
            group_attrs.push_str(&format!(
                " class=\"layer layer-{}-{}\"", info.layer, info.datatype,
            ));
        }
        doc.group_open(&group_attrs);
        if style.tooltips {
            // SVG <title> child becomes a hover tooltip in browsers.
            let label = if info.name.is_empty() {
                format!("L{}/{}", info.layer, info.datatype)
            } else {
                format!("{} ({}/{})", info.name, info.layer, info.datatype)
            };
            doc.raw(&format!("<title>{}</title>", xml_escape(&label)));
        }
        for (shape, t) in items {
            draw_shape(&mut doc, shape, *t, style.units_per_dbu);
        }
        doc.group_close();
    }

    if style.show_instance_labels && !instances.is_empty() {
        draw_instance_labels(&mut doc, &instances, style.units_per_dbu);
    }

    if !style.highlights.is_empty() {
        draw_highlights(&mut doc, &style.highlights, style.units_per_dbu);
    }

    if style.show_ports {
        let port_attrs = "fill=\"black\" stroke=\"none\"";
        doc.group_open(port_attrs);
        for (p, _) in &ports {
            let (x, y) = map(*p, style.units_per_dbu);
            doc.circle(x, y, (style.stroke_width * 2.0).max(1.0), "");
        }
        doc.group_close();
    }

    if style.show_legend {
        draw_legend(&mut doc, &legend, vx, vy, style);
    }

    doc.finish()
}

/// Convenience: render and write to `path`. Returns the SVG string too
/// in case the caller wants to feed it to `png::svg_to_png` without
/// re-reading from disk.
pub fn write_svg(
    lib: &Library,
    top: CellId,
    style: &Style,
    path: &std::path::Path,
) -> std::io::Result<String> {
    let s = render_to_svg(lib, top, style);
    std::fs::write(path, &s)?;
    Ok(s)
}

// ── internals ──────────────────────────────────────────────────────────

/// One placed (or replicated) instance, recorded so the renderer can
/// label it / wrap shapes in an `<a>`-able group.
pub(crate) struct InstanceInfo {
    pub cell_name: String,
    /// Bbox of the instance in the *top* frame.
    pub bbox: Bbox,
}

#[allow(clippy::too_many_arguments)]
fn collect(
    lib: &Library,
    cell_id: CellId,
    cell: &Cell,
    t: Trans,
    out: &mut BTreeMap<LayerIndex, Vec<(Shape, Trans)>>,
    ports: &mut Vec<(Point, &'static str)>,
    instances: &mut Vec<InstanceInfo>,
    capture_ports: bool,
    hidden: &HashSet<LayerIndex>,
    on_path: &mut HashSet<CellId>,
) {
    // Cycle guard — silently break the recursion if a Library somehow
    // contains a cell that references itself transitively. (klayout-rs
    // doesn't make this easy, but a buggy LIR transform could.)
    if !on_path.insert(cell_id) {
        return;
    }
    for layer in cell.layers() {
        if hidden.contains(&layer) { continue; }
        for shape in cell.shapes_on(layer) {
            out.entry(layer).or_default().push((shape.clone(), t));
        }
    }
    if capture_ports {
        for p in cell.ports() {
            ports.push((t.apply(p.center), "black"));
        }
    }
    for inst in cell.instances() {
        let child = lib.get(inst.cell);
        let child_t = t.compose(inst.trans);
        // Wrappers (cells with only sub-instances and no direct
        // shapes) have an empty `local_bbox` (i64::MIN/MAX sentinel)
        // — applying a transform to that produces garbage and crashes
        // downstream label placement. Fall back to `full_bbox(lib)`
        // for those so the instance gets a real bbox.
        let local = child.local_bbox();
        let bbox_src = if local.is_empty() { child.full_bbox(lib) } else { local };
        let child_bbox = child_t.apply_bbox(bbox_src);
        instances.push(InstanceInfo {
            cell_name: child.name().as_str().to_string(),
            bbox: child_bbox,
        });
        collect(lib, inst.cell, &child, child_t, out, ports, instances,
                capture_ports, hidden, on_path);
        if let Some(rep) = &inst.repetition {
            stamp_repetition(lib, inst.cell, &child, t, &inst.trans, rep,
                             out, ports, instances,
                             capture_ports, hidden, on_path);
        }
    }
    on_path.remove(&cell_id);
}

#[allow(clippy::too_many_arguments)]
fn stamp_repetition(
    lib: &Library,
    child_id: CellId,
    child: &Cell,
    parent_t: Trans,
    base_t: &Trans,
    rep: &klayout_core::Repetition,
    out: &mut BTreeMap<LayerIndex, Vec<(Shape, Trans)>>,
    ports: &mut Vec<(Point, &'static str)>,
    instances: &mut Vec<InstanceInfo>,
    capture_ports: bool,
    hidden: &HashSet<LayerIndex>,
    on_path: &mut HashSet<CellId>,
) {
    use klayout_core::{Repetition, Vec2};
    let stamp = |off: Vec2,
                     out: &mut BTreeMap<LayerIndex, Vec<(Shape, Trans)>>,
                     ports: &mut Vec<(Point, &'static str)>,
                     instances: &mut Vec<InstanceInfo>,
                     on_path: &mut HashSet<CellId>| {
        let mut t = *base_t;
        t.disp = klayout_core::Vec2::new(t.disp.x + off.x, t.disp.y + off.y);
        let composed = parent_t.compose(t);
        let bbox = composed.apply_bbox(child.local_bbox());
        instances.push(InstanceInfo {
            cell_name: child.name().as_str().to_string(),
            bbox,
        });
        collect(lib, child_id, child, composed, out, ports, instances,
                capture_ports, hidden, on_path);
    };
    match rep {
        Repetition::Regular { row, col, n_rows, n_cols } => {
            for r in 0..*n_rows {
                for c in 0..*n_cols {
                    if r == 0 && c == 0 { continue; } // base instance already placed
                    let off = klayout_core::Vec2::new(
                        col.x * c as i64 + row.x * r as i64,
                        col.y * c as i64 + row.y * r as i64,
                    );
                    stamp(off, out, ports, instances, on_path);
                }
            }
        }
        Repetition::Irregular { offsets } => {
            for off in offsets {
                stamp(*off, out, ports, instances, on_path);
            }
        }
    }
}

fn draw_shape(doc: &mut SvgDoc, shape: &Shape, t: Trans, upd: f64) {
    match shape {
        Shape::Box(rect) => draw_rect(doc, rect, t, upd),
        Shape::Polygon(p) => draw_polygon(doc, p, t, upd),
        Shape::Path(p) => draw_path(doc, p, t, upd),
        Shape::Text(text) => draw_text(doc, text, t, upd),
    }
}

fn draw_text(doc: &mut SvgDoc, text: &Text, t: Trans, upd: f64) {
    let anchor = t.apply(text.anchor);
    let (x, y) = map(anchor, upd);
    // KLayout's Text::size is in DBU; if zero, fall back to a screen
    // size that's readable but not overwhelming.
    let size = if text.size > 0 {
        (text.size as f64) * upd
    } else {
        8.0
    };
    let halign = match text.halign {
        HAlign::Left   => "start",
        HAlign::Center => "middle",
        HAlign::Right  => "end",
    };
    // SVG y-axis flip means Top/Bottom swap relative to layout.
    let baseline = match text.valign {
        VAlign::Top    => "ideographic",
        VAlign::Middle => "central",
        VAlign::Bottom => "alphabetic",
    };
    let attrs = format!(
        "font-family=\"sans-serif\" font-size=\"{size:.2}\" \
         text-anchor=\"{halign}\" dominant-baseline=\"{baseline}\" \
         fill=\"currentColor\" stroke=\"none\"",
    );
    doc.text(x, y, &attrs, &text.string);
}

fn draw_rect(doc: &mut SvgDoc, rect: &Rect, t: Trans, upd: f64) {
    let b = t.apply_bbox(rect.bbox);
    let (x1, y1) = map(b.min, upd);
    let (x2, y2) = map(b.max, upd);
    let x = x1.min(x2);
    let y = y1.min(y2);
    let w = (x2 - x1).abs();
    let h = (y2 - y1).abs();
    doc.rect(x, y, w, h, "");
}

fn draw_polygon(doc: &mut SvgDoc, p: &Polygon, t: Trans, upd: f64) {
    let pts: Vec<(f64, f64)> = p.hull.iter().map(|pt| map(t.apply(*pt), upd)).collect();
    doc.polygon(&pts, "");
    // Holes: SVG <polygon> doesn't natively represent holes. For a
    // visual approximation, draw each hole as a same-color but lighter
    // overlay; future work could switch to <path> with even-odd fill.
    for hole in &p.holes {
        let hpts: Vec<(f64, f64)> = hole.iter().map(|pt| map(t.apply(*pt), upd)).collect();
        doc.polygon(&hpts, "fill=\"white\" fill-opacity=\"1\" stroke=\"none\"");
    }
}

fn draw_path(doc: &mut SvgDoc, p: &Path, t: Trans, upd: f64) {
    if p.points.is_empty() { return; }
    let pts: Vec<(f64, f64)> = p.points.iter().map(|pt| map(t.apply(*pt), upd)).collect();
    let width = (p.width as f64) * upd;
    let attrs = format!(
        "fill=\"none\" stroke-width=\"{width:.3}\" stroke-linecap=\"butt\" \
         stroke-linejoin=\"miter\""
    );
    doc.polyline(&pts, &attrs);
}

fn draw_instance_labels(doc: &mut SvgDoc, instances: &[InstanceInfo], upd: f64) {
    // Render labels ABOVE each instance bbox (in layout y-up: that's
    // bbox.max.y + small offset; in SVG y-down: above the top edge of
    // the rendered group). White stroke-paint underneath the fill
    // creates a halo so the text stays readable on top of any layer
    // color.
    let attrs = "font-family=\"sans-serif\" font-size=\"7\" \
                 text-anchor=\"middle\" dominant-baseline=\"alphabetic\" \
                 fill=\"#222\" stroke=\"white\" stroke-width=\"2.5\" \
                 paint-order=\"stroke\" stroke-linejoin=\"round\"";
    for inst in instances {
        // Defensive guard — cells whose bbox couldn't be resolved
        // (empty / sentinel) would overflow the y-flip below. Skip
        // those rather than panic; the cell still renders, it just
        // doesn't get an instance label.
        if inst.bbox.is_empty() { continue; }
        let cx = (inst.bbox.min.x + inst.bbox.max.x) as f64 * 0.5 * upd;
        // SVG y-down: top of the bbox in screen space is `-max.y`,
        // and we sit a few pixels above that.
        let top_y = -(inst.bbox.max.y) as f64 * upd - 3.0;
        // Strip the cell-name's structural prefix so the user sees a
        // short, readable label. The convention `Resistor_R1_L10000`
        // → `R1` reads off the second underscore-separated field
        // when present; otherwise we render the whole name.
        let display = short_instance_label(&inst.cell_name);
        doc.text(cx, top_y, attrs, &display);
    }
}

/// Heuristic: `Resistor_R1_L10000` → `R1`, `Mosfet_M3_W2_L1` → `M3`,
/// `Inverter_INV0` → `INV0`. Falls back to the full name when the
/// pattern doesn't match.
fn short_instance_label(name: &str) -> String {
    let mut parts = name.split('_');
    let _kind = parts.next();
    match parts.next() {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => name.to_string(),
    }
}

fn draw_highlights(doc: &mut SvgDoc, highlights: &[crate::Highlight], upd: f64) {
    // Each highlight: thick translucent fill + outlined border + label.
    // Drawn AFTER all shape layers so it sits on top of geometry.
    for h in highlights {
        let b = h.bbox;
        let (x1, y1) = map(b.min, upd);
        let (x2, y2) = map(b.max, upd);
        let x = x1.min(x2);
        let y = y1.min(y2);
        let w = (x2 - x1).abs();
        let height = (y2 - y1).abs();
        // Translucent fill — high enough fill-opacity to draw the eye
        // but low enough to keep underlying geometry visible.
        let fill = format!(
            "fill=\"{}\" fill-opacity=\"0.30\" stroke=\"{}\" stroke-width=\"1.5\"",
            h.color, h.color,
        );
        doc.rect(x, y, w, height, &fill);
        if !h.label.is_empty() {
            // Label sits just above the bbox, top-aligned.
            doc.text(
                x, y - 2.0,
                &format!(
                    "font-family=\"sans-serif\" font-size=\"6\" \
                     fill=\"{}\" stroke=\"none\"",
                    h.color,
                ),
                &h.label,
            );
        }
    }
}

fn draw_legend(
    doc: &mut SvgDoc,
    legend: &[(LayerIndex, LayerInfo, String)],
    vx: f64,
    vy: f64,
    _style: &Style,
) {
    if legend.is_empty() { return; }
    let pad = 4.0;
    let row_h = 12.0;
    let sw = 14.0;
    let font = 9.0;
    let height = pad * 2.0 + row_h * legend.len() as f64;
    let width = 130.0;

    // Place legend in the top-left of the reserved strip above the
    // cell. `vy` is the strip's top edge.
    let lx = vx + pad;
    let ly = vy + pad;
    doc.rect(
        lx, ly, width, height,
        "fill=\"white\" stroke=\"#333\" stroke-width=\"0.5\"",
    );
    for (i, (_, info, color)) in legend.iter().enumerate() {
        let row_y = ly + pad + (i as f64) * row_h;
        doc.rect(
            lx + pad, row_y, sw, row_h - 2.0,
            &format!("fill=\"{color}\" stroke=\"{color}\" stroke-width=\"0.5\""),
        );
        let label = if info.name.is_empty() {
            format!("L{}/{}", info.layer, info.datatype)
        } else {
            format!("{} ({}/{})", info.name, info.layer, info.datatype)
        };
        doc.text(
            lx + pad + sw + 4.0,
            row_y + row_h - 4.0,
            &format!("font-size=\"{font}\" font-family=\"sans-serif\" fill=\"#222\""),
            &label,
        );
    }
}

fn viewbox(bbox: Bbox, pad_dbu: i64, upd: f64) -> (f64, f64, f64, f64) {
    let min_x = (bbox.min.x - pad_dbu) as f64 * upd;
    let max_x = (bbox.max.x + pad_dbu) as f64 * upd;
    // y flip: layout y-up → svg y-down.
    let min_y_layout = (bbox.min.y - pad_dbu) as f64 * upd;
    let max_y_layout = (bbox.max.y + pad_dbu) as f64 * upd;
    let min_y_svg = -max_y_layout;
    let max_y_svg = -min_y_layout;
    (min_x, min_y_svg, max_x - min_x, max_y_svg - min_y_svg)
}

fn map(p: Point, upd: f64) -> (f64, f64) {
    (p.x as f64 * upd, -(p.y as f64) * upd)
}

