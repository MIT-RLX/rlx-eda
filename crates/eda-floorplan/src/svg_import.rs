//! Import an SVG file as a `klayout-core::Cell` of polygons on a single
//! PDK layer.
//!
//! Lets external vector art (foundry-supplied alignment / seal stamps,
//! project banners, hand-drawn floorplan annotations) drop into a layout
//! without a manual KLayout import step. Each `<path>` in the source
//! SVG is flattened into one or more polygons via `tiny-skia-path`'s
//! segment iterator (curves subdivided to a configurable tolerance);
//! every polygon lands on the caller-supplied layer.
//!
//! The result is a `Block`-shaped cell — caller adds it to a parent
//! cell via `Instance::new(cell, trans)` exactly like any other
//! laid-out child.

use std::path::Path as FsPath;

use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, LayerIndex, Library, Point, Polygon, Port, PortKindId,
    Rect, Shape,
};

/// Subdivision count for cubic / quadratic Bezier flattening. 16 keeps
/// curve segments visually smooth at typical layout zooms without
/// blowing up the polygon vertex count.
const CURVE_FLATTEN_STEPS: usize = 16;

pub struct SvgImportOptions {
    /// User-units → DBU scale. Default `1000.0` puts a 1.0 SVG unit
    /// at 1 µm under sky130's `dbu = 1000`.
    pub dbu_per_user_unit: f64,
    /// Y-axis convention. SVG is y-down; layout is y-up. `true` flips
    /// y so the imported geometry isn't upside-down on the chip.
    pub flip_y: bool,
    /// Optional electrical port kind for the two ports the imported
    /// cell exposes (left / right edge midpoints). When `None` the
    /// ports get [`PortKindId::ANY`].
    pub port_kind: Option<PortKindId>,
    /// Add a thin frame rectangle on the same layer so the bounding
    /// box is unambiguous in the GDS even when interior fill is
    /// sparse.
    pub draw_frame: bool,
}

impl Default for SvgImportOptions {
    fn default() -> Self {
        Self {
            dbu_per_user_unit: 1_000.0,
            flip_y: true,
            port_kind: None,
            draw_frame: false,
        }
    }
}

/// Load an SVG from disk and stamp every path's filled region as
/// polygons on `layer`. Returns the new top cell's `CellId`.
pub fn import_svg(
    lib: &Library,
    path: &FsPath,
    cell_name: impl Into<String>,
    layer: LayerIndex,
    opts: &SvgImportOptions,
) -> std::io::Result<CellId> {
    let svg = std::fs::read_to_string(path)?;
    import_svg_str(lib, &svg, cell_name, layer, opts)
}

/// Same as [`import_svg`] but takes the SVG XML directly. Useful when
/// the SVG is generated on the fly or read from a non-file source.
pub fn import_svg_str(
    lib: &Library,
    svg: &str,
    cell_name: impl Into<String>,
    layer: LayerIndex,
    opts: &SvgImportOptions,
) -> std::io::Result<CellId> {
    let usvg_opts = usvg::Options::default();
    let tree = usvg::Tree::from_str(svg, &usvg_opts)
        .map_err(|e| std::io::Error::other(format!("svg parse: {e}")))?;

    let mut cb = CellBuilder::new(cell_name.into());

    let mut bbox_min = (i64::MAX, i64::MAX);
    let mut bbox_max = (i64::MIN, i64::MIN);

    walk_group(lib, &mut cb, tree.root(), layer, opts, &mut bbox_min, &mut bbox_max);

    if opts.draw_frame && bbox_min.0 < bbox_max.0 {
        cb.add_shape(
            layer,
            Shape::Box(Rect::new(Bbox::new(
                Point::new(bbox_min.0, bbox_min.1),
                Point::new(bbox_max.0, bbox_max.1),
            ))),
        );
    }

    // Two ports on the left and right edge midpoints so the imported
    // cell is electrically connectable as a stamp / decal — caller can
    // wire it into a power rail or leave it floating.
    if bbox_min.0 < bbox_max.0 {
        let mid_y = (bbox_min.1 + bbox_max.1) / 2;
        let edge_w = ((bbox_max.0 - bbox_min.0) / 50).max(1_000);
        let kind = opts.port_kind.unwrap_or(PortKindId::ANY);
        cb.add_port(
            Port::new("p_w", layer, Point::new(bbox_min.0, mid_y), Angle90::W, edge_w)
                .with_kind(kind),
        );
        cb.add_port(
            Port::new("p_e", layer, Point::new(bbox_max.0, mid_y), Angle90::E, edge_w)
                .with_kind(kind),
        );
    }

    Ok(lib.insert(cb))
}

fn walk_node(
    lib: &Library,
    cb: &mut CellBuilder,
    node: &usvg::Node,
    layer: LayerIndex,
    opts: &SvgImportOptions,
    bbox_min: &mut (i64, i64),
    bbox_max: &mut (i64, i64),
) {
    match node {
        usvg::Node::Group(g) => walk_group(lib, cb, g, layer, opts, bbox_min, bbox_max),
        usvg::Node::Path(p) => {
            // Skip outline-only paths — fill is what we lift to silicon.
            if p.fill().is_none() {
                return;
            }
            stamp_path(cb, p, layer, opts, bbox_min, bbox_max);
        }
        // Image / Text nodes are ignored: an SVG `<image>` is raster
        // content (already a bitmap, no polygons to lift), and `<text>`
        // would need font glyph extraction — both are out of scope for
        // this importer. Caller can rasterize separately if needed.
        _ => {}
    }
}

fn walk_group(
    lib: &Library,
    cb: &mut CellBuilder,
    g: &usvg::Group,
    layer: LayerIndex,
    opts: &SvgImportOptions,
    bbox_min: &mut (i64, i64),
    bbox_max: &mut (i64, i64),
) {
    for child in g.children() {
        walk_node(lib, cb, child, layer, opts, bbox_min, bbox_max);
    }
}

fn stamp_path(
    cb: &mut CellBuilder,
    path: &usvg::Path,
    layer: LayerIndex,
    opts: &SvgImportOptions,
    bbox_min: &mut (i64, i64),
    bbox_max: &mut (i64, i64),
) {
    let abs = path.abs_transform();
    let data = path.data();

    let to_dbu = |x: f32, y: f32| -> Point {
        let nx = abs.sx * x + abs.kx * y + abs.tx;
        let ny = abs.ky * x + abs.sy * y + abs.ty;
        let ny = if opts.flip_y { -ny } else { ny };
        Point::new(
            (nx as f64 * opts.dbu_per_user_unit) as i64,
            (ny as f64 * opts.dbu_per_user_unit) as i64,
        )
    };

    // First pass: collect every subpath (each MoveTo opens a new
    // subpath; Close terminates one). Multiple subpaths within one
    // `<path>` are how SVG encodes glyph counters / negative space —
    // an outer hull and one or more inner holes that the fill rule
    // (nonzero / even-odd) cuts out. Stamping each as a solid polygon
    // (the old behaviour) painted over the holes; here we collect first,
    // then group below.
    let mut subpaths: Vec<Vec<Point>> = Vec::new();
    let mut current: Vec<Point> = Vec::new();
    let mut last_xy: (f32, f32) = (0.0, 0.0);

    let push_curve = |current: &mut Vec<Point>, x0: f32, y0: f32,
                      f: &dyn Fn(f32) -> (f32, f32), to_dbu: &dyn Fn(f32, f32) -> Point| {
        for s in 1..=CURVE_FLATTEN_STEPS {
            let t = s as f32 / CURVE_FLATTEN_STEPS as f32;
            let (x, y) = f(t);
            let _ = (x0, y0);
            current.push(to_dbu(x, y));
        }
    };

    for seg in data.segments() {
        match seg {
            tiny_skia_path::PathSegment::MoveTo(p) => {
                if current.len() >= 3 { subpaths.push(std::mem::take(&mut current)); }
                else { current.clear(); }
                current.push(to_dbu(p.x, p.y));
                last_xy = (p.x, p.y);
            }
            tiny_skia_path::PathSegment::LineTo(p) => {
                current.push(to_dbu(p.x, p.y));
                last_xy = (p.x, p.y);
            }
            tiny_skia_path::PathSegment::QuadTo(p1, p2) => {
                let (x0, y0) = last_xy;
                let f = move |t: f32| {
                    let u = 1.0 - t;
                    (
                        u * u * x0 + 2.0 * u * t * p1.x + t * t * p2.x,
                        u * u * y0 + 2.0 * u * t * p1.y + t * t * p2.y,
                    )
                };
                push_curve(&mut current, x0, y0, &f, &to_dbu);
                last_xy = (p2.x, p2.y);
            }
            tiny_skia_path::PathSegment::CubicTo(p1, p2, p3) => {
                let (x0, y0) = last_xy;
                let f = move |t: f32| {
                    let u = 1.0 - t;
                    (
                        u * u * u * x0
                            + 3.0 * u * u * t * p1.x
                            + 3.0 * u * t * t * p2.x
                            + t * t * t * p3.x,
                        u * u * u * y0
                            + 3.0 * u * u * t * p1.y
                            + 3.0 * u * t * t * p2.y
                            + t * t * t * p3.y,
                    )
                };
                push_curve(&mut current, x0, y0, &f, &to_dbu);
                last_xy = (p3.x, p3.y);
            }
            tiny_skia_path::PathSegment::Close => {
                if current.len() >= 3 { subpaths.push(std::mem::take(&mut current)); }
                else { current.clear(); }
            }
        }
    }
    if current.len() >= 3 { subpaths.push(current); }

    if subpaths.is_empty() { return; }

    // Second pass: even-odd fill rule via containment depth. Each
    // subpath's depth = number of *other* subpaths whose interior
    // contains its first vertex; depth even ⇒ hull, depth odd ⇒ hole
    // belonging to the smallest containing hull. This handles the
    // typical glyph case (one outer hull + one inner counter) and
    // nested counters too. (Strict nonzero would need signed-area
    // accounting; for fonts + logos even-odd matches what the source
    // SVG renderer actually shows.)
    let n = subpaths.len();
    let mut depth: Vec<usize> = vec![0; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let probe = subpaths[i][0];
        // Pick the smallest-bbox containing parent so nested counters
        // attach to the right hull.
        let mut best: Option<(usize, i64)> = None;
        for j in 0..n {
            if i == j { continue; }
            if !point_in_polygon(probe, &subpaths[j]) { continue; }
            depth[i] += 1;
            let area = bbox_area(&subpaths[j]);
            if best.map_or(true, |(_, a)| area < a) {
                best = Some((j, area));
            }
        }
        parent[i] = best.map(|(j, _)| j);
    }

    // Assign holes (odd depth) to their nearest even-depth ancestor.
    let mut holes_for: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        if depth[i] % 2 == 1 {
            // Walk parent chain up to the first even-depth subpath.
            let mut p = parent[i];
            while let Some(j) = p {
                if depth[j] % 2 == 0 { holes_for.entry(j).or_default().push(i); break; }
                p = parent[j];
            }
        }
    }

    for i in 0..n {
        if depth[i] % 2 != 0 { continue; }
        for pt in subpaths[i].iter() {
            bbox_min.0 = bbox_min.0.min(pt.x);
            bbox_min.1 = bbox_min.1.min(pt.y);
            bbox_max.0 = bbox_max.0.max(pt.x);
            bbox_max.1 = bbox_max.1.max(pt.y);
        }
        let holes: Vec<smallvec::SmallVec<[Point; 8]>> = holes_for
            .get(&i)
            .map(|v| v.iter().map(|&h| subpaths[h].clone().into()).collect())
            .unwrap_or_default();
        cb.add_shape(
            layer,
            Shape::Polygon(Polygon {
                hull: subpaths[i].clone().into(),
                holes,
            }),
        );
    }
}

/// Even-odd point-in-polygon. `poly` is a closed loop (last → first
/// edge implied).
fn point_in_polygon(p: Point, poly: &[Point]) -> bool {
    if poly.len() < 3 { return false; }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        let crosses = (pi.y > p.y) != (pj.y > p.y)
            && (p.x as i128)
                < ((pj.x as i128 - pi.x as i128) * (p.y as i128 - pi.y as i128)
                    / ((pj.y - pi.y).max(1) as i128)
                    + pi.x as i128);
        if crosses { inside = !inside; }
        j = i;
    }
    inside
}

fn bbox_area(poly: &[Point]) -> i64 {
    let (mut x0, mut y0) = (i64::MAX, i64::MAX);
    let (mut x1, mut y1) = (i64::MIN, i64::MIN);
    for p in poly {
        x0 = x0.min(p.x); y0 = y0.min(p.y);
        x1 = x1.max(p.x); y1 = y1.max(p.y);
    }
    (x1 - x0).max(0) * (y1 - y0).max(0)
}
