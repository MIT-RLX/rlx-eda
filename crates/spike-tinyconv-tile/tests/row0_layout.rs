//! End-to-end test that `Mac8x8Tile::layout` (Digital topology)
//! places row 0 against a mock sc_hd library. When the foundry GDS
//! is checked out, swap `populate_mock_sc_hd` for
//! `ScHdLibrary::load(...)` and the same assertions apply.

use eda_hir::Layout;
use eda_stdcells::mock::populate_mock_sc_hd;
use klayout_core::Library;
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};

fn lib_with_mocks() -> (Library, eda_pdks::Sky130) {
    let lib = Library::new("row0", 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let _ = populate_mock_sc_hd(&lib, &pdk);
    (lib, pdk)
}

fn digital_tile() -> Mac8x8Tile {
    Mac8x8Tile::with_topology("u0", TileParams::default(), MacTopology::Digital)
}

/// Cumulative instance counts at the end of each row (inclusive).
/// Used to slice `cell.instances()` per-row in assertions.
const ROW_END: [usize; 4] = [
    /* row 0 */ 8 + 32 + 24,                  //  64
    /* row 1 */ 8 + 32 + 24 + 32 + 32 + 16,    // 144
    /* row 2 */ 8 + 32 + 24 + 32 + 32 + 16 + 16 + 16, // 176
    /* row 3 */ 8 + 32 + 24 + 32 + 32 + 16 + 16 + 16 + 16 + 10, // 202
];

#[test]
fn all_rows_place_202_instances() {
    let (lib, pdk) = lib_with_mocks();
    let cid = digital_tile().layout(&lib, &pdk);
    let cell = lib.get(cid);
    assert_eq!(cell.instances().len(), ROW_END[3]);
    assert_eq!(cell.instances().len(), 202);
}

#[test]
fn row0_first_eight_instances_are_weight_register() {
    let (lib, pdk) = lib_with_mocks();
    let cid = digital_tile().layout(&lib, &pdk);
    let cell = lib.get(cid);

    let dff_id = lib
        .by_name("sky130_fd_sc_hd__dfxtp_1")
        .expect("mock dfxtp_1 in lib");
    for (i, inst) in cell.instances().iter().take(8).enumerate() {
        assert_eq!(inst.cell, dff_id, "instance {i} should be dfxtp_1");
    }
}

#[test]
fn row0_instances_step_by_per_cell_widths() {
    let (lib, pdk) = lib_with_mocks();
    let cid = digital_tile().layout(&lib, &pdk);
    let cell = lib.get(cid);

    let origins: Vec<_> = cell.instances().iter().map(|i| i.trans.disp).collect();

    // Row 0 dfxtp_1 segment.
    assert_eq!(origins[0].x, 0);
    assert_eq!(origins[1].x, 2_300);
    assert_eq!(origins[7].x, 7 * 2_300);

    // and2_1 segment starts after 8 × 2300 = 18_400.
    assert_eq!(origins[8].x, 18_400);
    assert_eq!(origins[9].x, 18_400 + 1_290);

    // fa_1 segment starts after 18_400 + 32 × 1290 = 59_680.
    assert_eq!(origins[40].x, 59_680);
    assert_eq!(origins[41].x, 59_680 + 3_680);

    // Every row 0 instance sits on y = 0.
    for o in &origins[..ROW_END[0]] {
        assert_eq!(o.y, 0);
    }
}

#[test]
fn each_row_origins_at_sc_hd_pitch() {
    let (lib, pdk) = lib_with_mocks();
    let cid = digital_tile().layout(&lib, &pdk);
    let cell = lib.get(cid);
    let origins: Vec<_> = cell.instances().iter().map(|i| i.trans.disp).collect();

    // Row 0: indices 0..ROW_END[0], y = 0
    // Row 1: indices ROW_END[0]..ROW_END[1], y = 2720
    // Row 2: indices ROW_END[1]..ROW_END[2], y = 5440
    // Row 3: indices ROW_END[2]..ROW_END[3], y = 8160
    let row_y = [0_i64, 2_720, 5_440, 8_160];
    let mut start = 0;
    for (row_idx, end) in ROW_END.iter().enumerate() {
        let y_expected = row_y[row_idx];
        for o in &origins[start..*end] {
            assert_eq!(o.y, y_expected, "row {row_idx} should have y={y_expected}");
        }
        start = *end;
    }
}

#[test]
fn each_row_x_starts_at_zero() {
    let (lib, pdk) = lib_with_mocks();
    let cid = digital_tile().layout(&lib, &pdk);
    let cell = lib.get(cid);
    let origins: Vec<_> = cell.instances().iter().map(|i| i.trans.disp).collect();

    let mut start = 0;
    for end in ROW_END.iter() {
        assert_eq!(origins[start].x, 0, "first instance per row should start at x=0");
        start = *end;
    }
}

#[test]
fn every_row_fits_inside_pitch_x() {
    let (lib, pdk) = lib_with_mocks();
    let cid = digital_tile().layout(&lib, &pdk);
    let cell = lib.get(cid);
    let origins: Vec<_> = cell.instances().iter().map(|i| i.trans.disp).collect();

    // Track per-row max(x + cell_width) by walking instances in
    // their placement order. We don't have widths from the
    // klayout instance API, so we just bound the *origin* of the
    // last instance in each row against pitch_x. (The width budget
    // was confirmed analytically when the row constants were
    // chosen — row 1 sums to ~195.84 µm < 220 µm pitch.)
    let pitch_x = 220_000_i64;
    let mut start = 0;
    for end in ROW_END.iter() {
        let last_x = origins[end - 1].x;
        assert!(
            last_x < pitch_x,
            "last instance origin {last_x} ≥ pitch_x {pitch_x}"
        );
        start = *end;
    }
    let _ = start;
}

#[test]
#[should_panic(expected = "foundry cell")]
fn layout_panics_when_mock_cells_missing() {
    // No mocks populated — confirm the helpful panic fires.
    let lib = Library::new("missing-mocks", 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let _ = digital_tile().layout(&lib, &pdk);
}
