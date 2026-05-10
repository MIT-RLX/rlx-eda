//! `StdCell::layout` smoke test — does NOT require the foundry library
//! to be checked out. Builds a tiny in-memory `Library` with a fake
//! cell named like a foundry cell, then checks that `StdCell::layout`
//! finds it and that `build_composite` instances it at the right
//! placement.

use eda_hir::Layout;
use eda_pdks::Sky130;
use eda_stdcells::cell::{build_composite, StdCell};
use klayout_core::{Bbox, CellBuilder, Library, Point, Rect};

/// Build an empty test cell with a single distinguishing shape so the
/// content-addressable `Library::insert` doesn't dedup it against
/// other "empty" cells. (Klayout's content hash deliberately omits
/// the name — two cells with identical geometry but different names
/// dedup to the first one inserted.)
fn distinct_test_cell(name: &str, layer: klayout_core::LayerIndex, mark: i64) -> CellBuilder {
    let mut b = CellBuilder::new(name);
    b.add_shape(layer, Rect::new(Bbox::new(Point::new(0, 0), Point::new(mark, mark))));
    b
}

/// Build the PDK by registering its layers in the supplied library.
/// The pdk! macro generates `register(lib: &Library) -> Self`.
fn fake_pdk(lib: &Library) -> Sky130 {
    Sky130::register(lib)
}

#[test]
fn layout_resolves_cell_by_name_in_target_library() {
    let lib = Library::new("test", 1000);
    let pdk = fake_pdk(&lib);
    let cell_id = lib.insert(distinct_test_cell(
        "sky130_fd_sc_hd__nand2_1", pdk.METAL1, 100,
    ));

    let cell = StdCell::new("sky130_fd_sc_hd__nand2_1", "u1");
    let resolved = cell.layout(&lib, &pdk);

    assert_eq!(resolved, cell_id, "StdCell::layout should resolve to the inserted CellId");
}

#[test]
#[should_panic(expected = "foundry cell")]
fn layout_panics_when_cell_missing() {
    let lib = Library::new("test", 1000);
    let pdk = fake_pdk(&lib);
    let cell = StdCell::new("sky130_fd_sc_hd__no_such_cell", "u1");
    let _ = cell.layout(&lib, &pdk);
}

#[test]
fn build_composite_instances_each_child_at_its_origin() {
    let lib = Library::new("test", 1000);
    let pdk = fake_pdk(&lib);
    let _nand_id = lib.insert(distinct_test_cell(
        "sky130_fd_sc_hd__nand2_1", pdk.METAL1, 100,
    ));
    let _inv_id  = lib.insert(distinct_test_cell(
        "sky130_fd_sc_hd__inv_1",   pdk.METAL1, 200,
    ));

    let mut parent = CellBuilder::new("composite_under_test");
    build_composite(
        &mut parent,
        &lib,
        &pdk,
        &[
            (StdCell::new("sky130_fd_sc_hd__nand2_1", "u1"), Point::new(0, 0)),
            (StdCell::new("sky130_fd_sc_hd__inv_1", "u2"),    Point::new(2_000, 0)),
        ],
    );

    let frozen = parent.freeze(&lib);
    assert_eq!(
        frozen.instances().len(),
        2,
        "expected one Instance per composite child"
    );
}
