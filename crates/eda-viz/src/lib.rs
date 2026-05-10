//! `eda-viz` — render circuits to SVG (and optionally PNG).
//!
//! Two complementary renderers live in this crate:
//!
//! - [`layout`] turns a `(Library, CellId)` into an SVG by flattening the
//!   cell tree and drawing every `Shape` on a per-layer color. This is
//!   the visualizer for whatever a `Layout<P>` impl produced.
//! - [`schematic`] draws a hand-built symbolic circuit (resistor zigzag,
//!   capacitor plates, diode triangle, …). It does not yet read a
//!   `Schematic<P>` IR — that trait is still pending in `eda-hir` — so
//!   callers compose the [`schematic::Schematic`] data structure
//!   themselves. When a `Schematic<P>` trait lands, a thin adapter goes
//!   on top of this renderer.
//!
//! ## SVG vs PNG
//!
//! SVG is the core output (text, no extra deps). The `png` cargo feature
//! pulls `resvg` + `tiny-skia` and adds [`png::svg_to_png`]. Default
//! builds stay light.
//!
//! ## Coordinate handling
//!
//! Layout coordinates are i64 DBU with y pointing up. SVG user space is
//! float with y pointing down. The renderer applies a single
//! `units_per_dbu` scale and y-flip when mapping; everything inside an
//! `<svg>` element is in already-mapped float units.

pub mod layout;
pub mod palette;
pub mod schematic;
pub mod svg;
pub mod waveform;

#[cfg(feature = "png")]
pub mod png;

pub use palette::{LayerPalette, LypColorKind};

use klayout_core::Bbox;

/// Convert `klayout_drc::Violation`s into [`Highlight`]s ready to drop
/// into `Style::highlights`. Requires the `drc` feature.
#[cfg(feature = "drc")]
pub fn highlights_from_drc(
    violations: &[klayout_drc::Violation],
    color: &str,
) -> Vec<Highlight> {
    violations.iter()
        .map(|v| Highlight {
            bbox: v.bbox,
            color: color.to_string(),
            label: v.rule.to_string(),
        })
        .collect()
}

/// Write `svg` to `path` as a gzip-compressed `.svgz`. Requires the
/// `svgz` feature.
#[cfg(feature = "svgz")]
pub fn write_svgz(svg: &str, path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    // Default compression level (6) — good balance for text. SVG
    // typically compresses 5-10x.
    let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
    enc.write_all(svg.as_bytes())?;
    enc.finish()?;
    Ok(())
}

/// One DRC/LVS error / annotation overlaid on the layout.
///
/// Renders as a thick translucent rectangle on top of the geometry,
/// with the `label` text in the corner. The verifier crates
/// (`klayout-drc`, `klayout-validate`) produce errors-with-bbox; feed
/// each into `Style::highlights` to surface them on the rendered
/// layout instead of in a separate report.
#[derive(Clone, Debug)]
pub struct Highlight {
    pub bbox: Bbox,
    /// Any valid SVG color string (e.g. `"#e74c3c"`, `"red"`).
    pub color: String,
    /// Short label drawn near the highlight (DRC rule name, etc.).
    /// Empty string suppresses the label.
    pub label: String,
}

/// Render style options for layout rendering.
#[derive(Clone, Debug)]
pub struct Style {
    /// SVG user-units per DBU. Defaults to `0.01` — at the divider's
    /// `dbu = 1000`, that's 10 user units per micron, which puts a
    /// 10 µm resistor at 100 px in a default-rendered viewer.
    pub units_per_dbu: f64,
    /// Padding (DBU) added around the cell bbox before viewBox.
    pub pad_dbu: i64,
    /// Stroke width (user units) for shape outlines and schematic wires.
    pub stroke_width: f64,
    /// Fill opacity for layer polygons / rects in `[0.0, 1.0]`.
    pub fill_opacity: f64,
    /// Background color (any valid SVG color string), or `None` for
    /// transparent.
    pub background: Option<String>,
    /// Show port markers (small triangles) on the layout.
    pub show_ports: bool,
    /// Render a small layer-color legend in the corner.
    pub show_legend: bool,
    /// Optional layer-color override sourced from a foundry `.lyp`
    /// file. When set, the renderer looks up `(layer, datatype)` in
    /// the palette before falling back to the built-in palette.
    pub layer_palette: Option<LayerPalette>,
    /// Verifier errors / annotations rendered on top of the geometry.
    /// Empty by default — populate from DRC / LVS results.
    pub highlights: Vec<Highlight>,
    /// Emit `<title>` children on shape groups so SVG viewers show
    /// layer / cell info on hover. Cheap (a few bytes per group); off
    /// by default to keep snapshot-test output minimal.
    pub tooltips: bool,
    /// Layers to *exclude* from rendering. Common debugging workflow:
    /// hide METAL1 to inspect RES underneath.
    pub hidden_layers: Vec<klayout_core::LayerIndex>,
    /// Show per-instance text labels (cell names) at each instance's
    /// bbox center. Off by default — readable layouts only have a
    /// handful of named instances.
    pub show_instance_labels: bool,
    /// Emit `class="layer-N"` etc. on rendered groups so external CSS
    /// can override the inline palette.
    pub emit_css_classes: bool,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            units_per_dbu: 0.01,
            pad_dbu: 1_000,
            stroke_width: 0.5,
            fill_opacity: 0.45,
            background: Some("white".into()),
            show_ports: true,
            show_legend: true,
            layer_palette: None,
            highlights: Vec::new(),
            tooltips: false,
            hidden_layers: Vec::new(),
            show_instance_labels: false,
            emit_css_classes: false,
        }
    }
}
