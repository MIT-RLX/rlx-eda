//! GDS / OASIS roundtrip via the re-exports in `eda_viz::layout`.
//!
//! Builds a small two-layer cell, writes it through one of the
//! foundry formats, reads it back, and confirms the layer histogram
//! survives. Catches regressions where the `gds` feature gates
//! something the wrong way, or where the re-export drifts from
//! `klayout-io`.

#![cfg(feature = "gds")]

use eda_viz::layout;
use klayout_core::{Bbox, CellBuilder, LayerInfo, Library, Point, Rect};

fn build_two_layer_lib() -> (Library, klayout_core::CellId) {
    let lib = Library::new("roundtrip", 1000);
    let res = lib.layer(LayerInfo::named("RES", 50, 0));
    let met = lib.layer(LayerInfo::named("METAL1", 10, 0));
    let mut b = CellBuilder::new("two_layer");
    b.add_shape(res, Rect::new(Bbox::new(Point::new(0, 0), Point::new(10_000, 1_000))));
    b.add_shape(met, Rect::new(Bbox::new(Point::new(0, 2_000), Point::new(10_000, 3_000))));
    b.add_shape(met, Rect::new(Bbox::new(Point::new(0, 4_000), Point::new(10_000, 5_000))));
    let top = lib.insert(b);
    (lib, top)
}

fn count_shapes(lib: &Library, top: klayout_core::CellId) -> std::collections::BTreeMap<(u16, u16), usize> {
    use std::collections::BTreeMap;
    let cell = lib.get(top);
    let mut hist: BTreeMap<(u16, u16), usize> = BTreeMap::new();
    for layer in cell.layers() {
        let info = lib.layer_info(layer);
        let key = (info.layer, info.datatype);
        let n = cell.shapes_on(layer).count();
        if n > 0 {
            *hist.entry(key).or_default() += n;
        }
    }
    hist
}

#[test]
fn gds_roundtrip_preserves_layer_histogram() {
    let (lib, top) = build_two_layer_lib();
    let before = count_shapes(&lib, top);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("two_layer.gds");
    layout::write_gds_path(&lib, &path).expect("write gds");

    let lib2 = layout::read_gds_path(&path).expect("read gds");
    // Round-tripped library has one cell at the top — find it.
    let top2 = lib2
        .top_cells()
        .into_iter()
        .next()
        .expect("at least one top cell");
    let after = count_shapes(&lib2, top2);

    assert_eq!(before, after, "layer histogram diverged across GDS roundtrip");
    // Sanity: we have something to compare.
    assert_eq!(before.values().sum::<usize>(), 3);
}

#[test]
fn oasis_roundtrip_preserves_layer_histogram() {
    let (lib, top) = build_two_layer_lib();
    let before = count_shapes(&lib, top);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("two_layer.oas");
    layout::write_oasis_path(&lib, &path).expect("write oasis");

    let lib2 = layout::read_oasis_path(&path).expect("read oasis");
    let top2 = lib2
        .top_cells()
        .into_iter()
        .next()
        .expect("at least one top cell");
    let after = count_shapes(&lib2, top2);

    assert_eq!(before, after, "layer histogram diverged across OASIS roundtrip");
}
