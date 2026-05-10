//! Layer-3 conformance: place geometry on a PDK layer, extract a Region,
//! and run a single `klayout_drc::width` rule. Asserts that:
//!
//! * a 1 µm wide box on the layer passes a 0.5 µm min-width check
//!   (`width` returns an empty violation region), and
//! * a 0.4 µm wide box on the same layer fails the same check
//!   (violation region is non-empty).
//!
//! This is a synthetic rule, not a foundry rule deck — Layer-3 in the
//! testing ladder is the integration smoke test: "the PDK + KLayout DRC
//! pipeline composes." Once a real PDK ships with a deck (e.g. via
//! klayout-drc rule files), foundry-specific decks belong here next to
//! these smoke tests.
//!
//! One CMOS PDK (Sky130) and one photonic PDK (Cornerstone si220) are
//! exercised — enough to demonstrate the pattern is foundry-neutral.

#![allow(unused_imports)]

use klayout_core::{Bbox, CellBuilder, CellId, LayerIndex, LayerInfo, Library, Point, Rect, Shape};
use klayout_drc::width;
use klayout_geom::Region;

/// Place a single axis-aligned box of the given (x, y) extent at the
/// origin on `layer`, freeze the cell, and return its id.
fn place_box(lib: &Library, layer: LayerIndex, name: &str, w: i64, h: i64) -> CellId {
    let mut cb = CellBuilder::new(name);
    let rect = Rect::new(Bbox::new(Point::new(0, 0), Point::new(w, h)));
    cb.add_shape(layer, Shape::Box(rect));
    lib.insert(cb)
}

/// Run a min-width check on the region of `cell`'s shapes on `layer`,
/// and return the violation polygon count.
fn width_violations(lib: &Library, cell: CellId, layer: LayerIndex, min: i64) -> usize {
    let region = Region::from_cell_layer(lib, cell, layer);
    width(&region, min).polygons().len()
}

#[cfg(feature = "sky130")]
#[test]
fn sky130_res_layer_drc_smoke() {
    if !eda_pdks::HAS_SKY130 {
        eprintln!("skipping: sky130 lyp not present at build time");
        return;
    }
    let lib = eda_pdks::Sky130::new_library("drc_sky130");
    let pdk = eda_pdks::Sky130::register(&lib);

    // 1 µm wide × 5 µm long → passes a 0.5 µm min-width.
    let clean = place_box(&lib, pdk.RES, "clean", 1_000, 5_000);
    assert_eq!(
        width_violations(&lib, clean, pdk.RES, 500),
        0,
        "Sky130: 1µm box flagged as narrow under 0.5µm min-width",
    );

    // 0.4 µm wide × 5 µm long → fails the same check.
    let narrow = place_box(&lib, pdk.RES, "narrow", 400, 5_000);
    let violations = width_violations(&lib, narrow, pdk.RES, 500);
    assert!(
        violations >= 1,
        "Sky130: 0.4µm box should violate 0.5µm min-width, got {} violations",
        violations,
    );
}

#[cfg(feature = "cornerstone-si220")]
#[test]
fn cornerstone_si220_wg_layer_drc_smoke() {
    if !eda_pdks::HAS_CORNERSTONE_SI220 {
        eprintln!("skipping: cornerstone-si220 lyp not present at build time");
        return;
    }
    let lib = eda_pdks::CornerstoneSi220::new_library("drc_cs");
    let pdk = eda_pdks::CornerstoneSi220::register(&lib);

    // Same synthetic 0.5 µm min-width rule, applied to the photonic WG.
    let clean = place_box(&lib, pdk.WG, "clean", 1_000, 5_000);
    assert_eq!(
        width_violations(&lib, clean, pdk.WG, 500),
        0,
        "CornerstoneSi220: 1µm WG box flagged as narrow under 0.5µm min-width",
    );

    let narrow = place_box(&lib, pdk.WG, "narrow", 400, 5_000);
    let violations = width_violations(&lib, narrow, pdk.WG, 500);
    assert!(
        violations >= 1,
        "CornerstoneSi220: 0.4µm WG box should violate 0.5µm min-width, got {}",
        violations,
    );
}
