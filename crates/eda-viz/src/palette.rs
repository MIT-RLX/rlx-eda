//! Layer-color palette — typically sourced from a foundry `.lyp`
//! KLayout layer-property file, so rendered SVGs use the same colors
//! engineers see when opening the GDS in KLayout.
//!
//! ## Why this lives in eda-viz, not eda-pdk-ingest
//!
//! `eda-pdk-ingest` parses `.lyp` files for layer (name, GDS pair,
//! frame) tuples, which is what the layout PDK macro generation
//! needs. *Colors* are a rendering concern; tying them to the PDK
//! ingester would force the layout pipeline to depend on render
//! settings. So the palette is a separate, optional piece the
//! renderer reads.
//!
//! ## Source of colors
//!
//! Two construction paths:
//!
//! - [`LayerPalette::set`] — caller writes hex strings directly. Useful
//!   when there's no `.lyp` available, or for one-off demos.
//! - [`LayerPalette::from_lyp`] — parse a foundry `.lyp` file via
//!   `eda-pdk-ingest` and lift its `<fill-color>` (or `<frame-color>`
//!   as a fallback) into the palette so screenshots match KLayout.

use std::collections::HashMap;

/// Which colour KLayout's lyp file uses on shapes. Most people want
/// the fill — that's what KLayout paints solid-shape interiors with.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum LypColorKind {
    /// `<fill-color>` — KLayout's interior fill. The default.
    #[default]
    Fill,
    /// `<frame-color>` — KLayout's outline. Use when shapes render
    /// hollow (e.g. drawing only borders).
    Frame,
}

/// Maps `(layer, datatype)` → SVG color string.
#[derive(Clone, Debug, Default)]
pub struct LayerPalette {
    by_gds: HashMap<(u16, u16), String>,
}

impl LayerPalette {
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the color for a specific GDS pair.
    pub fn set(&mut self, layer: u16, datatype: u16, color: impl Into<String>) -> &mut Self {
        self.by_gds.insert((layer, datatype), color.into());
        self
    }

    /// Look up the color for a `(layer, datatype)` pair, if set.
    pub fn get(&self, layer: u16, datatype: u16) -> Option<&str> {
        self.by_gds.get(&(layer, datatype)).map(|s| s.as_str())
    }

    pub fn is_empty(&self) -> bool { self.by_gds.is_empty() }
    pub fn len(&self) -> usize { self.by_gds.len() }

    /// Build a palette from a KLayout `.lyp` file, taking
    /// `<fill-color>` (or `<frame-color>` per `kind`) for every layer
    /// that has one. Layers without the requested colour are skipped
    /// silently — the renderer's built-in fallback handles them.
    ///
    /// Returns the parser error if the lyp's `<source>` tags are
    /// malformed; a successfully-parsed lyp with no colour annotations
    /// returns an empty palette rather than an error.
    pub fn from_lyp(xml: &str, kind: LypColorKind)
        -> Result<Self, eda_pdk_ingest::LypError>
    {
        let layers = eda_pdk_ingest::parse_lyp(xml)?;
        let mut pal = Self::new();
        for p in layers {
            let chosen = match kind {
                LypColorKind::Fill  => p.fill_color.or(p.frame_color),
                LypColorKind::Frame => p.frame_color.or(p.fill_color),
            };
            if let Some(c) = chosen {
                pal.set(p.layer, p.datatype, c);
            }
        }
        Ok(pal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LYP: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
 <properties>
  <frame-color>#ff0000</frame-color>
  <fill-color>#aa0000</fill-color>
  <name>poly 66/20</name>
  <source>66/20@1</source>
 </properties>
 <properties>
  <frame-color>#0000ff</frame-color>
  <name>met1 68/20</name>
  <source>68/20@1</source>
 </properties>
 <properties>
  <name>uncoloured 99/0</name>
  <source>99/0@1</source>
 </properties>
</layer-properties>
"#;

    #[test]
    fn from_lyp_fill_prefers_fill_color() {
        let p = LayerPalette::from_lyp(LYP, LypColorKind::Fill).unwrap();
        assert_eq!(p.get(66, 20), Some("#aa0000"));
        // No fill-color → falls back to frame-color.
        assert_eq!(p.get(68, 20), Some("#0000ff"));
        // No colour at all → not in palette; renderer uses its default.
        assert_eq!(p.get(99, 0), None);
    }

    #[test]
    fn from_lyp_frame_prefers_frame_color() {
        let p = LayerPalette::from_lyp(LYP, LypColorKind::Frame).unwrap();
        assert_eq!(p.get(66, 20), Some("#ff0000"));
        assert_eq!(p.get(68, 20), Some("#0000ff"));
    }
}
