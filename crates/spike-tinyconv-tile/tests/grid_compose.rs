//! End-to-end test of the abutment + tile_grid pipeline. The tile
//! exposes pitch / rails / edge_ports declaratively; the layout
//! skeleton emits matching shapes; tile_grid composes a uniform
//! grid and validates the contracts at compose time.

use eda_hir::Layout;
use eda_stdcells::mock::populate_mock_sc_hd;
use eda_tile::{tile_grid, GridError, PdnCheck, Tile};
use klayout_core::Library;
use spike_divider_block::MosfetPdk;
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};

/// Build a fresh test library populated with sky130 + mock sc_hd
/// cells. Every test that calls `tile.layout(...)` needs these
/// pre-populated, since the v1 layout body instantiates row 0
/// cells by name and panics if any are missing.
fn lib_with_mocks(name: &str) -> (Library, eda_pdks::Sky130) {
    let lib = Library::new(name, 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let _ = populate_mock_sc_hd(&lib, &pdk);
    (lib, pdk)
}

fn digital_tile() -> Mac8x8Tile {
    Mac8x8Tile::with_topology("u0", TileParams::default(), MacTopology::Digital)
}

#[test]
fn layout_skeleton_inserts_a_cell_with_rails_and_row0_cells() {
    let (lib, pdk) = lib_with_mocks("skeleton-test");
    let tile = digital_tile();

    let cid = tile.layout(&lib, &pdk);
    let cell = lib.get(cid);

    // Rails on met1 — 3 VDD + 2 GND rails + 2 boundary markers = 7
    // shapes minimum on metal1.
    let rail_shapes: Vec<_> = cell.shapes_on(pdk.metal1()).collect();
    assert!(
        rail_shapes.len() >= 5,
        "expected at least 5 rail shapes (3 VDD + 2 GND), got {}",
        rail_shapes.len()
    );

    // All 4 rows of cell instances:
    //   row 0:  8 dfxtp_1 + 32 and2_1 + 24 fa_1            =  64
    //   row 1: 32 and2_1  + 32 fa_1   + 16 dfxtp_1         =  80
    //   row 2: 16 dfxtp_1 + 16 fa_1                        =  32
    //   row 3: 16 fa_1    + 10 inv_1                       =  26
    //   total                                              = 202
    assert_eq!(
        cell.instances().len(),
        202,
        "rows 0-3 should place a total of 202 cell instances"
    );
}

#[test]
fn tile_grid_composes_a_4x4_array_with_correct_instance_count() {
    let (lib, pdk) = lib_with_mocks("grid-test");
    let tile = digital_tile();

    let cid = tile_grid(&tile, 4, 4, &lib, &pdk, None).expect("compose");
    let parent = lib.get(cid);
    assert_eq!(parent.instances().len(), 16, "4×4 grid → 16 instances");
}

#[test]
fn tile_grid_passes_pdn_check_when_array_within_budget() {
    let (lib, pdk) = lib_with_mocks("pdn-pass-test");
    let tile = digital_tile();

    // 480 nm strap × 1 mA/µm = 0.48 mA budget. 4 tiles × 0.1 mA =
    // 0.4 mA → under budget → pass.
    let result = tile_grid(
        &tile, 4, 4, &lib, &pdk,
        Some(PdnCheck { per_tile_peak_ma: 0.1, jmax_ma_per_um: 1.0 }),
    );
    assert!(result.is_ok(), "expected PDN pass, got {result:?}");
}

#[test]
fn tile_grid_fails_pdn_check_when_array_over_budget() {
    let (lib, pdk) = lib_with_mocks("pdn-fail-test");
    let tile = digital_tile();

    // 16 tiles × 0.1 mA = 1.6 mA → over the 0.48 mA budget → fail.
    let result = tile_grid(
        &tile, 16, 16, &lib, &pdk,
        Some(PdnCheck { per_tile_peak_ma: 0.1, jmax_ma_per_um: 1.0 }),
    );
    match result {
        Err(GridError::Pdn(_)) => {}
        other => panic!("expected GridError::Pdn, got {other:?}"),
    }
}

#[test]
fn tile_grid_handles_1x1_degenerate_case() {
    let (lib, pdk) = lib_with_mocks("1x1-test");
    let tile = digital_tile();

    let cid = tile_grid(&tile, 1, 1, &lib, &pdk, None).expect("1×1");
    assert_eq!(lib.get(cid).instances().len(), 1);
}

#[test]
fn instance_origins_step_by_pitch() {
    let (lib, pdk) = lib_with_mocks("origins-test");
    let tile = digital_tile();
    let pitch = <Mac8x8Tile as Tile<eda_pdks::Sky130>>::pitch(&tile);

    let cid = tile_grid(&tile, 3, 2, &lib, &pdk, None).expect("compose");
    let parent = lib.get(cid);
    let origins: Vec<_> = parent
        .instances()
        .iter()
        .map(|i| i.trans.disp)
        .collect();

    // Row-major: (0,0) (P,0) (2P,0) (0,P) (P,P) (2P,P).
    let expected = [
        (0, 0),
        (pitch.x, 0),
        (2 * pitch.x, 0),
        (0, pitch.y),
        (pitch.x, pitch.y),
        (2 * pitch.x, pitch.y),
    ];
    assert_eq!(origins.len(), expected.len());
    for (got, want) in origins.iter().zip(expected.iter()) {
        assert_eq!((got.x, got.y), *want);
    }
}
