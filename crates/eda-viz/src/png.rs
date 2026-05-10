//! SVG → PNG via `resvg` + `tiny-skia`. Compiled in only when the `png`
//! cargo feature is enabled, so the default build keeps a minimal dep
//! tree.
//!
//! ## Bundled fonts
//!
//! Two fonts are embedded so PNG output is byte-stable across hosts:
//!
//! - **DejaVu Sans** (`assets/DejaVuSans.ttf`, public-domain DejaVu
//!   changes over Bitstream Vera) — the default for general text:
//!   layer legends, schematic labels, axis titles.
//! - **Latin Modern Math** (`assets/LatinModernMath.otf`, GUST Font
//!   License) — the dashboard's math renderer requests this family
//!   for formulas; it has proper math italic, real radicals, and
//!   full Greek + math-symbol coverage.
//!
//! SVGs that don't reference `Latin Modern Math` fall through to
//! DejaVu Sans automatically.

use std::fmt;

const BUNDLED_SANS: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");
const BUNDLED_MATH: &[u8] = include_bytes!("../assets/LatinModernMath.otf");

#[derive(Debug)]
pub enum PngError {
    /// SVG failed to parse.
    Parse(String),
    /// Could not allocate the output pixmap (typically a zero-sized SVG
    /// or a scale that produces a 0×0 image).
    EmptyImage,
    /// Encoding the rasterized pixels to PNG failed.
    Encode(String),
}

impl fmt::Display for PngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PngError::Parse(e) => write!(f, "svg parse: {e}"),
            PngError::EmptyImage => write!(f, "empty pixmap (svg has zero size?)"),
            PngError::Encode(e) => write!(f, "png encode: {e}"),
        }
    }
}

impl std::error::Error for PngError {}

/// Rasterize `svg` to a PNG byte buffer at the given scale factor.
/// `scale = 1.0` matches the SVG's intrinsic `width`/`height`; values
/// above 1 supersample for sharper output.
///
/// Loads the system font database so text inside the SVG (layer
/// legend, schematic labels) rasterizes. `usvg::Options::default()`
/// alone ships an empty fontdb and silently drops `<text>` nodes.
pub fn svg_to_png(svg: &str, scale: f32) -> Result<Vec<u8>, PngError> {
    let mut opts = usvg::Options::default();
    {
        let db = opts.fontdb_mut();
        db.load_font_data(BUNDLED_SANS.to_vec());
        db.load_font_data(BUNDLED_MATH.to_vec());
        // Map generic CSS families to our bundled faces. SVGs that say
        // `font-family="sans-serif"` (or omit family) → DejaVu;
        // explicit `font-family="Latin Modern Math"` resolves to the
        // math face by name.
        db.set_sans_serif_family("DejaVu Sans");
        db.set_serif_family("Latin Modern Math");
        db.set_monospace_family("DejaVu Sans");
    }
    opts.font_family = "DejaVu Sans".to_string();
    let tree = usvg::Tree::from_str(svg, &opts).map_err(|e| PngError::Parse(e.to_string()))?;

    let size = tree.size().to_int_size();
    let w = ((size.width() as f32) * scale).ceil() as u32;
    let h = ((size.height() as f32) * scale).ceil() as u32;
    let mut pixmap = tiny_skia::Pixmap::new(w, h).ok_or(PngError::EmptyImage)?;

    let transform = tiny_skia::Transform::from_scale(scale, scale);
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    pixmap.encode_png().map_err(|e| PngError::Encode(e.to_string()))
}
