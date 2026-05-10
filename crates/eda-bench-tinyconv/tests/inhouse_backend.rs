//! `InhouseBackend::measure_physical` end-to-end test using the
//! mock sky130 library. When the foundry library is checked out,
//! swap `populate_mock_sc_hd` for `ScHdLibrary::load(...)` and the
//! same assertions apply.

use eda_bench_tinyconv::{
    backends::{inhouse::InhouseBackend, Backend},
    metrics::Physical,
};
use eda_stdcells::{populate_mock_sc_hd, ScHdLibrary};
use klayout_core::Library;
use spike_tinyconv_array::array::{ArrayBlock, ArrayConfig};
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};

fn lib_with_mocks() -> ScHdLibrary {
    let lib = Library::new("inhouse-test", 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let cells = populate_mock_sc_hd(&lib, &pdk);
    ScHdLibrary { library: lib, cells }
}

fn digital_tile() -> Mac8x8Tile {
    Mac8x8Tile::with_topology("u_inhouse", TileParams::default(), MacTopology::Digital)
}

#[test]
fn measure_physical_returns_liberty_derived_area() {
    let backend = InhouseBackend::new(digital_tile(), lib_with_mocks());
    let p: Physical = backend.measure_physical().expect("measure runs");
    let area = p.area_um2.expect("area was filled");

    // Per-cell mock areas (µm²·×1000):
    //   dfxtp_1  6256, and2_1  3509, fa_1  10010, inv_1  5005
    // Inventory: 40 dfxtp_1 + 64 and2_1 + 88 fa_1 + 10 inv_1
    //   = 250_240 + 224_576 + 880_880 + 50_050 = 1_405_746 (× 1000 µm²)
    //   = 1_405.746 µm²
    let expected = 1_405.746;
    assert!(
        (area - expected).abs() < 1e-3,
        "got {area} µm², expected {expected}"
    );
}

#[test]
fn measure_physical_leaves_other_fields_none() {
    // OpenRCX (parasitics), ngspice (power/timing), and HotSpot
    // (thermal) all stay unwired — those fields must be honestly
    // `None`, not `Some(0.0)`.
    let backend = InhouseBackend::new(digital_tile(), lib_with_mocks());
    let p = backend.measure_physical().unwrap();
    assert!(p.max_freq_mhz.is_none());
    assert!(p.wns_ns.is_none());
    assert!(p.dynamic_power_mw.is_none());
    assert!(p.leakage_power_mw.is_none());
    assert!(p.parasitic_cap_ff.is_none());
    assert!(p.peak_temp_c.is_none());
    assert!(p.energy_pj_per_inference.is_none());
}

#[test]
fn measure_physical_errors_when_library_missing_a_cell() {
    // Build a library that omits one of the cells the inventory
    // references. measure_physical should surface this as a
    // Toolchain error (with a descriptive message), not panic.
    let lib = Library::new("missing-cell", 1000);
    let _pdk = eda_pdks::Sky130::register(&lib);
    // Skip populate_mock_sc_hd entirely — empty library.
    let library = ScHdLibrary {
        library: lib,
        cells: std::collections::HashMap::new(),
    };
    let backend = InhouseBackend::new(digital_tile(), library);
    let err = backend.measure_physical().unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("missing"),
        "expected 'missing' in error message, got: {s}"
    );
}

#[test]
fn backend_name_is_inhouse() {
    let backend = InhouseBackend::new(digital_tile(), lib_with_mocks());
    assert_eq!(backend.name(), "inhouse");
}

#[test]
fn from_array_scales_area_by_grid_size() {
    // Default 4×4 grid → 16 tiles. Each tile = 1405.746 µm² of
    // foundry-cell area. Total = 22 491.936 µm².
    let array = ArrayBlock::new("u_arr", ArrayConfig::default());
    let backend = InhouseBackend::from_array(&array, lib_with_mocks());
    let p = backend.measure_physical().unwrap();
    let area = p.area_um2.unwrap();
    let per_tile = 1_405.746;
    let expected = 16.0 * per_tile;
    assert!(
        (area - expected).abs() < 1e-2,
        "got {area} µm², expected {expected} (16 tiles × {per_tile})"
    );
    assert_eq!(backend.scope, "array");
}

#[test]
fn loss_weights_with_inhouse_baseline_picks_up_real_area() {
    use eda_bench_tinyconv::optimization::LossWeights;

    let library = lib_with_mocks();
    let tile = digital_tile();
    let inv = tile.cell_inventory();

    let weights = LossWeights::default().with_inhouse_baseline(&library, &inv);
    let baseline = weights
        .area_baseline_um2
        .expect("inventory should sum to a known area");

    // Tile-scope total = 1405.746 µm² (40 dfxtp_1 + 64 and2_1 +
    // 88 fa_1 + 10 inv_1, mock Liberty).
    assert!(
        (baseline - 1_405.746).abs() < 1e-3,
        "expected 1405.746 µm² baseline, got {baseline}"
    );
}

#[test]
fn loss_weights_with_inhouse_baseline_no_op_on_missing_cell() {
    use eda_bench_tinyconv::optimization::LossWeights;

    // Empty library — sum returns None, so the helper leaves
    // `area_baseline_um2` unchanged from its default (None).
    let lib = Library::new("empty", 1000);
    let _pdk = eda_pdks::Sky130::register(&lib);
    let library = ScHdLibrary {
        library: lib,
        cells: std::collections::HashMap::new(),
    };
    let inv = digital_tile().cell_inventory();

    let weights = LossWeights::default().with_inhouse_baseline(&library, &inv);
    assert!(weights.area_baseline_um2.is_none());
}

#[test]
fn from_tile_and_from_array_use_consistent_per_cell_areas() {
    // Sanity: per-tile area × grid_count == array area (no
    // per-cell rounding drift). Catches bugs where ArrayBlock's
    // inventory aggregation might forget a cell type.
    let library = lib_with_mocks();
    let tile_lib_clone = lib_with_mocks(); // separate Library for from_tile
    let tile = digital_tile();
    let tile_area = InhouseBackend::from_tile(&tile, tile_lib_clone)
        .measure_physical()
        .unwrap()
        .area_um2
        .unwrap();
    let array_3x4 = ArrayBlock::new(
        "u_consist",
        ArrayConfig {
            grid: (3, 4),
            ..ArrayConfig::default()
        },
    );
    let array_area = InhouseBackend::from_array(&array_3x4, library)
        .measure_physical()
        .unwrap()
        .area_um2
        .unwrap();
    assert!(
        (array_area - 12.0 * tile_area).abs() < 1e-3,
        "3×4 array area should be 12 × tile area: array={array_area} tile×12={}",
        12.0 * tile_area,
    );
}
