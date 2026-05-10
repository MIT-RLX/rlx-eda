//! `ScHdLibrary::sum_area_um2_x1000` tests. Uses the mock library
//! so the test runs without the foundry checkout.

#![cfg(feature = "mock-stdcells")]

use eda_stdcells::{populate_mock_sc_hd, ScHdLibrary};
use klayout_core::Library;

fn lib_with_mocks() -> ScHdLibrary {
    let lib = Library::new("area-sum", 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let cells = populate_mock_sc_hd(&lib, &pdk);
    ScHdLibrary { library: lib, cells }
}

#[test]
fn sum_area_returns_per_cell_total() {
    let l = lib_with_mocks();
    // Single instance of nand2_1 (mock area = 6256 µm²·×1000).
    let sum = l
        .sum_area_um2_x1000(&[("sky130_fd_sc_hd__nand2_1", 1)])
        .expect("nand2_1 has metadata");
    assert_eq!(sum, 6_256);
}

#[test]
fn sum_area_scales_linearly_with_count() {
    let l = lib_with_mocks();
    let one = l.sum_area_um2_x1000(&[("sky130_fd_sc_hd__inv_1", 1)]).unwrap();
    let ten = l.sum_area_um2_x1000(&[("sky130_fd_sc_hd__inv_1", 10)]).unwrap();
    assert_eq!(ten, one * 10);
}

#[test]
fn sum_area_aggregates_across_multiple_cell_types() {
    let l = lib_with_mocks();
    let inv_area = l.sum_area_um2_x1000(&[("sky130_fd_sc_hd__inv_1", 1)]).unwrap();
    let nand_area = l.sum_area_um2_x1000(&[("sky130_fd_sc_hd__nand2_1", 1)]).unwrap();
    let combined = l
        .sum_area_um2_x1000(&[
            ("sky130_fd_sc_hd__inv_1", 3),
            ("sky130_fd_sc_hd__nand2_1", 2),
        ])
        .unwrap();
    assert_eq!(combined, 3 * inv_area + 2 * nand_area);
}

#[test]
fn sum_area_returns_none_for_unknown_cell() {
    let l = lib_with_mocks();
    let sum = l.sum_area_um2_x1000(&[("sky130_fd_sc_hd__no_such_cell", 1)]);
    assert!(sum.is_none());
}

#[test]
fn sum_area_handles_empty_inventory() {
    let l = lib_with_mocks();
    let sum = l.sum_area_um2_x1000(&[]).expect("empty inventory");
    assert_eq!(sum, 0);
}
