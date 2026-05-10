//! Mock sc_hd library tests — run only with `--features mock-stdcells`.
//! These let us exercise the cell-placement pipeline without the
//! foundry GDS checkout.

#![cfg(feature = "mock-stdcells")]

use eda_hir::Layout;
use eda_pdks::Sky130;
use eda_stdcells::{
    cell::{build_composite, StdCell},
    mock::{mock_cell_names, populate_mock_sc_hd},
};
use klayout_core::{CellBuilder, Library, Point};

#[test]
fn populate_inserts_every_advertised_cell() {
    let lib = Library::new("mock-test", 1000);
    let pdk = Sky130::register(&lib);

    let cells = populate_mock_sc_hd(&lib, &pdk);
    let names: Vec<_> = mock_cell_names().collect();

    assert_eq!(
        cells.len(),
        names.len(),
        "populate should produce one StdCellRef per advertised name"
    );
    for name in &names {
        assert!(cells.contains_key(*name), "missing {name}");
        // Every populated cell should also be findable in the lib by name.
        assert!(
            lib.by_name(name).is_some(),
            "cell {name} not registered in target library"
        );
    }
}

#[test]
fn mock_cells_carry_liberty_metadata_with_nonzero_area() {
    let lib = Library::new("mock-meta-test", 1000);
    let pdk = Sky130::register(&lib);
    let cells = populate_mock_sc_hd(&lib, &pdk);

    let nand = cells.get("sky130_fd_sc_hd__nand2_1").expect("nand2_1 mock");
    let meta = nand.metadata.as_ref().expect("metadata");
    assert!(meta.area_um2_x1000 > 0);
    // Sanity: nand2 should be larger than inv (a 1-input gate).
    let inv = cells.get("sky130_fd_sc_hd__inv_1").unwrap();
    assert!(
        nand.metadata.as_ref().unwrap().area_um2_x1000
            > inv.metadata.as_ref().unwrap().area_um2_x1000
    );
}

#[test]
fn stdcell_layout_resolves_against_mock_library() {
    // The whole point of the mock: StdCell::layout, written for
    // foundry cells, works unchanged against mocks.
    let lib = Library::new("mock-stdcell-test", 1000);
    let pdk = Sky130::register(&lib);
    let _ = populate_mock_sc_hd(&lib, &pdk);

    let cell = StdCell::new("sky130_fd_sc_hd__nand2_1", "u_test");
    let cid = cell.layout(&lib, &pdk);
    assert_eq!(cid, lib.by_name("sky130_fd_sc_hd__nand2_1").unwrap());
}

#[test]
fn build_composite_against_mock_library_produces_correct_instance_count() {
    let lib = Library::new("mock-composite-test", 1000);
    let pdk = Sky130::register(&lib);
    let _ = populate_mock_sc_hd(&lib, &pdk);

    // Tiny "row 0 of a MAC tile" approximation: weight register
    // (4 dfxtp_1) + multiplier first half (4 and2_1).
    let mut row = CellBuilder::new("mock_row0");
    let cells: Vec<_> = (0..4)
        .map(|i| {
            (
                StdCell::new("sky130_fd_sc_hd__dfxtp_1", format!("wreg_{i}")),
                Point::new(i * 2_300, 0),
            )
        })
        .chain((0..4).map(|i| {
            (
                StdCell::new("sky130_fd_sc_hd__and2_1", format!("pp_{i}")),
                Point::new(4 * 2_300 + i * 1_290, 0),
            )
        }))
        .collect();
    build_composite(&mut row, &lib, &pdk, &cells);

    let frozen = row.freeze(&lib);
    assert_eq!(frozen.instances().len(), 8);
}
