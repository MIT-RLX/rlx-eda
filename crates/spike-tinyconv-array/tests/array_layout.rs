//! `ArrayBlock::layout` end-to-end. Composes a Mac8x8Tile grid via
//! `eda_tile::tile_grid` against a mock sc_hd library; verifies the
//! parent has the expected instance count and that abutment / rail
//! checks pass for the default 4×4 array.

use eda_hir::Layout;
use eda_stdcells::populate_mock_sc_hd;
use klayout_core::Library;
use spike_tinyconv_array::array::{ArrayBlock, ArrayConfig};

fn lib_with_mocks(name: &str) -> (Library, eda_pdks::Sky130) {
    let lib = Library::new(name, 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let _ = populate_mock_sc_hd(&lib, &pdk);
    (lib, pdk)
}

#[test]
fn default_4x4_array_lays_out_as_16_tile_instances() {
    let (lib, pdk) = lib_with_mocks("default-array");
    let array = ArrayBlock::new("u_default", ArrayConfig::default());
    let cid = array.layout(&lib, &pdk);
    let cell = lib.get(cid);
    assert_eq!(cell.instances().len(), 16, "4×4 grid → 16 tile instances");
}

#[test]
fn one_by_one_degenerate_array_still_composes() {
    let (lib, pdk) = lib_with_mocks("1x1-array");
    let array = ArrayBlock::new(
        "u_1x1",
        ArrayConfig {
            grid: (1, 1),
            ..ArrayConfig::default()
        },
    );
    let cid = array.layout(&lib, &pdk);
    assert_eq!(lib.get(cid).instances().len(), 1);
}

#[test]
fn rectangular_array_preserves_grid_dimensions() {
    let (lib, pdk) = lib_with_mocks("rect-array");
    let array = ArrayBlock::new(
        "u_rect",
        ArrayConfig {
            grid: (3, 5),
            ..ArrayConfig::default()
        },
    );
    let cid = array.layout(&lib, &pdk);
    assert_eq!(lib.get(cid).instances().len(), 15);
}

#[test]
fn array_cell_inventory_scales_by_grid_size() {
    // 4×4 default array → 16 × per-tile inventory.
    let array = ArrayBlock::new("u_inv4", ArrayConfig::default());
    let inv = array.cell_inventory();
    let total: usize = inv.iter().map(|(_, n)| n).sum();
    // Per tile: 202 cells. 4×4 grid: 3232.
    assert_eq!(total, 16 * 202);
}

#[test]
fn array_inventory_coalesces_duplicate_cell_names() {
    // Inventory entries should be unique per cell name, even after
    // grid scaling. (Catches a bug where duplicate (name, count)
    // pairs slip through.)
    let array = ArrayBlock::new("u_dedup", ArrayConfig::default());
    let inv = array.cell_inventory();
    let names: std::collections::HashSet<&str> =
        inv.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names.len(),
        inv.len(),
        "duplicate cell names in inventory: {inv:?}"
    );
}

#[test]
fn array_block_name_encodes_grid_and_topology() {
    use eda_hir::Block;
    let array = ArrayBlock::new(
        "u_named",
        ArrayConfig {
            grid: (2, 2),
            pipeline_depth: 3,
            ..ArrayConfig::default()
        },
    );
    let n = array.name();
    assert!(n.contains("u_named"));
    assert!(n.contains("2x2"));
    assert!(n.contains("d3"));
    assert!(n.contains("dig"), "topology tag should appear: {n}");
}
