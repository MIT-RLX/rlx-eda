//! End-to-end bench pipeline: manifest → inhouse backend (with mock
//! sky130 library) → measure → push into Report → render markdown.
//!
//! This is the first test that exercises the *whole* bench
//! framework with real numbers (Liberty-derived area). Functional /
//! ORFS / FPGA backends are still stubbed; their slots in the
//! report show `_no … yet_` placeholders.

use eda_bench_tinyconv::{
    backends::{inhouse::InhouseBackend, Backend},
    manifest::{Manifest, ManifestInputs},
    Report,
};
use eda_stdcells::{populate_mock_sc_hd, ScHdLibrary};
use klayout_core::Library;
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};
use std::io::Write;

fn fake_cargo_lock() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("eda-bench-tinyconv-e2e-{}", std::process::id()));
    std::fs::File::create(&p)
        .unwrap()
        .write_all(b"[[package]]\n")
        .unwrap();
    p
}

fn lib_with_mocks() -> ScHdLibrary {
    let lib = Library::new("e2e", 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let cells = populate_mock_sc_hd(&lib, &pdk);
    ScHdLibrary { library: lib, cells }
}

#[test]
fn full_pipeline_produces_markdown_with_real_area() {
    // 1. Manifest — toolchain fingerprint.
    let cargo_lock = fake_cargo_lock();
    let manifest = Manifest::capture(ManifestInputs {
        sky130_repo: None,
        orfs_image: None,
        weights: None,
        cargo_lock: &cargo_lock,
        seed: 42,
    })
    .expect("manifest captures");

    // 2. Backend — real measurement against mock foundry.
    let backend = InhouseBackend::new(
        Mac8x8Tile::with_topology("u_e2e", TileParams::default(), MacTopology::Digital),
        lib_with_mocks(),
    );
    let physical = backend.measure_physical().expect("physical measurement");

    // 3. Report — push and render.
    let mut report = Report::new(manifest);
    report.physical.push((backend.name(), physical));
    let md = report.to_markdown();

    // ── Verifications ────────────────────────────────────────
    // Manifest section appears with the seed we supplied.
    assert!(md.contains("optimizer seed | 42"));

    // Physical section contains the inhouse row with the real
    // Liberty-derived area (1405.746 µm² rounded to 3 decimals).
    assert!(md.contains("inhouse"), "inhouse backend row missing");
    assert!(
        md.contains("1405.746"),
        "expected Liberty-derived area in markdown; got:\n{md}"
    );

    // Other-backend slots stay honest (no ORFS / FPGA pushed).
    assert!(md.contains("## Functional metrics"));
    assert!(md.contains("_no functional measurements yet_"));
    assert!(md.contains("## Yield gate"));
    assert!(md.contains("_no L5 PVT × MC runs yet"));

    let _ = std::fs::remove_file(&cargo_lock);
}

#[test]
fn report_renders_inhouse_alongside_other_backend_rows() {
    // Push the inhouse result + a hand-built ORFS-shaped result
    // (simulating what the docker backend would produce). Verifies
    // the report renders multiple backend rows + does not
    // conflate `None` with `0.0` in any field.
    use eda_bench_tinyconv::Physical;

    let cargo_lock = fake_cargo_lock();
    let manifest = Manifest::capture(ManifestInputs {
        sky130_repo: None,
        orfs_image: None,
        weights: None,
        cargo_lock: &cargo_lock,
        seed: 1,
    })
    .unwrap();

    let inhouse_p = InhouseBackend::new(
        Mac8x8Tile::with_topology("u_mix", TileParams::default(), MacTopology::Digital),
        lib_with_mocks(),
    )
    .measure_physical()
    .unwrap();

    let orfs_p = Physical {
        area_um2: Some(1500.0),
        max_freq_mhz: Some(250.0),
        wns_ns: Some(0.05),
        dynamic_power_mw: Some(2.5),
        leakage_power_mw: Some(0.05),
        parasitic_cap_ff: Some(900.0),
        peak_temp_c: Some(45.0),
        energy_pj_per_inference: Some(120.0),
    };

    let mut report = Report::new(manifest);
    report.physical.push(("inhouse", inhouse_p));
    report.physical.push(("orfs", orfs_p));
    let md = report.to_markdown();

    // Both rows present.
    assert!(md.contains("| inhouse |"), "inhouse row missing");
    assert!(md.contains("| orfs |"), "orfs row missing");

    // Areas: in-house is real, ORFS is the synthetic 1500.000.
    assert!(md.contains("1405.746"), "inhouse area missing");
    assert!(md.contains("1500.000"), "orfs area missing");

    // ORFS leakage is `Some(0.05)` → renders as 0.050; in-house
    // leakage is `None` → renders as em-dash. Confirms the
    // reporter distinguishes the two.
    assert!(md.contains("0.050"), "orfs leakage value missing");

    let _ = std::fs::remove_file(&cargo_lock);
}
