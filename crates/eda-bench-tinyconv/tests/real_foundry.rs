//! End-to-end test against real sky130_fd_sc_hd foundry data.
//! Soft-skips if the GDS / Liberty aren't checked out (mirrors the
//! `eda-stdcells/tests/load.rs` skip pattern), so contributors
//! without the volare install see clean passes.
//!
//! When the foundry data is present, this is the proof-of-life
//! that the whole pipeline (`cell_inventory` → `sum_area_um2_x1000`
//! → `InhouseBackend::measure_physical` → markdown report) runs
//! against actual sky130 numbers, not mocks.

use eda_bench_tinyconv::backends::{inhouse::InhouseBackend, Backend};
use eda_stdcells::{default_sc_hd_path, ScHdLibrary};
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};

/// Probe for a sky130_fd_sc_hd Liberty file alongside the GDS.
/// Returns `None` if no `.lib` is found nearby — the load still
/// works without metadata, just no area numbers.
fn discover_lib_for(gds: &std::path::Path) -> Option<std::path::PathBuf> {
    // Liberty lives at `<gds dir>/../lib/sky130_fd_sc_hd__tt_025C_1v80.lib`
    // in the canonical volare layout.
    let lib_dir = gds.parent()?.parent()?.join("lib");
    let candidate = lib_dir.join("sky130_fd_sc_hd__tt_025C_1v80.lib");
    candidate.is_file().then_some(candidate)
}

#[test]
fn real_foundry_inhouse_backend_reports_area() {
    let gds = default_sc_hd_path();
    if !gds.is_file() {
        eprintln!("skipping: sky130_fd_sc_hd GDS not present at {gds:?}");
        return;
    }
    let lib = discover_lib_for(&gds);
    if lib.is_none() {
        eprintln!("skipping: sky130_fd_sc_hd Liberty .lib not found near {gds:?}");
        return;
    }

    let library = match ScHdLibrary::load(&gds, lib.as_deref()) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("skipping: foundry load failed: {e}");
            return;
        }
    };

    eprintln!(
        "loaded foundry library: {} cells indexed",
        library.cells.len()
    );

    // Debug: are any Liberty entries surfacing area > 0 at all?
    let cells_with_meta = library
        .cells
        .values()
        .filter(|r| r.metadata.is_some())
        .count();
    let cells_with_nonzero_area = library
        .cells
        .values()
        .filter_map(|r| r.metadata.as_ref())
        .filter(|m| m.area_um2_x1000 > 0)
        .count();
    eprintln!(
        "metadata stats: {cells_with_meta} cells have metadata, \
         {cells_with_nonzero_area} of those have nonzero area"
    );
    // Print 3 sample cells with metadata.
    for (name, r) in library.cells.iter().take(3) {
        eprintln!(
            "  sample: {name} → metadata = {:?}",
            r.metadata.as_ref().map(|m| (
                m.cell_name.as_str(),
                m.area_um2_x1000,
                m.pins.len()
            ))
        );
    }

    // Confirm the v1 inventory's cells are all present in the
    // foundry library.
    let tile = Mac8x8Tile::with_topology(
        "u_real",
        TileParams::default(),
        MacTopology::Digital,
    );
    let inv = tile.cell_inventory();
    for (name, _) in &inv {
        assert!(
            library.cells.contains_key(*name),
            "foundry library missing cell {name}"
        );
        let cell_ref = &library.cells[*name];
        assert!(
            cell_ref.metadata.is_some(),
            "cell {name} missing Liberty metadata"
        );
    }

    let backend = InhouseBackend::from_tile(&tile, library);
    let physical = backend
        .measure_physical()
        .expect("foundry-backed measurement runs");
    let area = physical.area_um2.expect("area was filled");

    eprintln!("real-foundry inhouse-tile area: {area:.3} µm²");
    assert!(area > 0.0, "real foundry area should be positive");
    assert!(
        area < 100_000.0,
        "real foundry area shouldn't be insanely large"
    );
}
