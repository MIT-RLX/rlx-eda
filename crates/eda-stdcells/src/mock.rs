//! Mock stand-in cells for the v1 `sky130_fd_sc_hd` subset.
//!
//! Lets the downstream layout pipeline (`Mac8x8Tile::layout` row
//! population, `tile_grid` end-to-end runs) execute without the
//! ~200 MB foundry checkout. Each mock carries the foundry-canonical
//! cell name and a bbox sized to match the real cell's area; that is
//! enough for placement, abutment, and tile-area accounting to work
//! correctly. Mocks are emphatically **not** routable, **not**
//! timing-correct, and **not** shippable — anything that uses these
//! must declare so in its name (`mock_*`) or in a banner comment.
//!
//! The seven cells covered are exactly the seven called out in the
//! PLAN.md "Digital MAC tile floorplan" cell-counts table. Adding
//! a new cell is one entry in `MOCK_CELLS`.
//!
//! Areas come from the published sky130_fd_sc_hd Liberty data
//! (rounded to 0.001 µm²); cell heights are the standard sc_hd
//! 2.72 µm row pitch.

#![cfg(feature = "mock-stdcells")]

use std::collections::HashMap;

use klayout_core::{Bbox, CellBuilder, Library, Point, Rect};
use spike_divider_block::MosfetPdk;

use crate::cell::StdCellRef;
use crate::liberty::LibertyMetadata;

/// One row in the mock cell table.
struct MockSpec {
    /// Foundry-canonical cell name.
    name: &'static str,
    /// Cell width in DBU (sky130: 1000 DBU/µm).
    width_dbu: i64,
    /// Cell area in µm² × 1000 (matches `LibertyMetadata` units).
    area_um2_x1000: u64,
}

/// sc_hd row height = 2.72 µm.
const SC_HD_HEIGHT_DBU: i64 = 2_720;

/// The v1 sc_hd subset that `Mac8x8Tile::layout` instantiates.
/// Widths are derived from `area / 2.72` rounded to 10 nm.
const MOCK_CELLS: &[MockSpec] = &[
    MockSpec { name: "sky130_fd_sc_hd__inv_1",    width_dbu: 1_840, area_um2_x1000:  5_005 },
    MockSpec { name: "sky130_fd_sc_hd__buf_1",    width_dbu: 1_840, area_um2_x1000:  5_005 },
    MockSpec { name: "sky130_fd_sc_hd__nand2_1",  width_dbu: 2_300, area_um2_x1000:  6_256 },
    MockSpec { name: "sky130_fd_sc_hd__nor2_1",   width_dbu: 2_300, area_um2_x1000:  6_256 },
    MockSpec { name: "sky130_fd_sc_hd__and2_1",   width_dbu: 1_290, area_um2_x1000:  3_509 },
    MockSpec { name: "sky130_fd_sc_hd__fa_1",     width_dbu: 3_680, area_um2_x1000: 10_010 },
    MockSpec { name: "sky130_fd_sc_hd__dfxtp_1",  width_dbu: 2_300, area_um2_x1000:  6_256 },
    MockSpec { name: "sky130_fd_sc_hd__mux2_1",   width_dbu: 3_680, area_um2_x1000: 10_010 },
];

/// Insert the v1 mock sc_hd subset into `lib`. Each mock cell gets a
/// single rect on `pdk.metal1()` sized to (width, sc_hd height); the
/// rect ensures the content hash is unique so `Library::insert`
/// doesn't dedup mocks against each other (the dedup-by-content
/// behaviour we hit in `tests/stdcell_layout.rs`).
///
/// Returns the `name → StdCellRef` map ready to drop into
/// `ScHdLibrary { library, cells }`. Caller is responsible for
/// owning the library.
pub fn populate_mock_sc_hd<P: MosfetPdk>(
    lib: &Library,
    pdk: &P,
) -> HashMap<String, StdCellRef> {
    let m1 = pdk.metal1();
    let mut cells = HashMap::with_capacity(MOCK_CELLS.len());
    for (idx, spec) in MOCK_CELLS.iter().enumerate() {
        let mut b = CellBuilder::new(spec.name);
        // Body rect — defines the cell area. Two mocks may share width
        // (e.g. inv_1 and buf_1 both ~1.84 µm); the unique-marker rect
        // below ensures content-hash distinctness so `Library::insert`
        // doesn't dedup them. (Klayout's hash deliberately omits the
        // cell name; we hit this in `tests/stdcell_layout.rs` too.)
        b.add_shape(
            m1,
            Rect::new(Bbox::new(
                Point::new(0, 0),
                Point::new(spec.width_dbu, SC_HD_HEIGHT_DBU),
            )),
        );
        // Unique-marker rect: 1×1 nm, x-offset = MOCK_CELLS index.
        // Invisible at any practical zoom; guarantees unique content
        // hash across mocks.
        let m = idx as i64;
        b.add_shape(
            m1,
            Rect::new(Bbox::new(Point::new(m, 0), Point::new(m + 1, 1))),
        );
        let cell_id = lib.insert(b);
        cells.insert(
            spec.name.to_string(),
            StdCellRef {
                name: spec.name.to_string(),
                cell_id,
                metadata: Some(LibertyMetadata {
                    cell_name: spec.name.to_string(),
                    area_um2_x1000: spec.area_um2_x1000,
                    pins: Vec::new(), // mock; pin info added when needed
                }),
            },
        );
    }
    cells
}

/// List of mock cell names — useful for tests that want to assert
/// "every cell I expect is present." Stable in order.
pub fn mock_cell_names() -> impl Iterator<Item = &'static str> {
    MOCK_CELLS.iter().map(|s| s.name)
}
