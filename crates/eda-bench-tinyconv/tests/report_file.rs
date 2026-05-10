//! `Report::write_markdown` — file I/O smoke test.
//!
//! Bench-driven CI uses this to drop a per-run markdown artifact
//! at `target/bench/<git-sha>/report.md`; this test exercises the
//! same path against a tempdir.

use eda_bench_tinyconv::{
    backends::{inhouse::InhouseBackend, Backend},
    manifest::{Manifest, ManifestInputs},
    Report,
};
use eda_stdcells::{populate_mock_sc_hd, ScHdLibrary};
use klayout_core::Library;
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};
use std::io::Write;

fn fake_cargo_lock(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("eda-bench-tinyconv-rwf-{}-{tag}", std::process::id()));
    std::fs::File::create(&p)
        .unwrap()
        .write_all(b"[[package]]\n")
        .unwrap();
    p
}

fn build_report(tag: &str) -> Report {
    let cargo_lock = fake_cargo_lock(tag);
    let manifest = Manifest::capture(ManifestInputs {
        sky130_repo: None,
        orfs_image: None,
        weights: None,
        cargo_lock: &cargo_lock,
        seed: 7,
    })
    .unwrap();
    let _ = std::fs::remove_file(&cargo_lock);

    let lib = Library::new("rwf", 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let cells = populate_mock_sc_hd(&lib, &pdk);
    let library = ScHdLibrary { library: lib, cells };

    let backend = InhouseBackend::new(
        Mac8x8Tile::with_topology("u_rwf", TileParams::default(), MacTopology::Digital),
        library,
    );
    let physical = backend.measure_physical().unwrap();

    let mut r = Report::new(manifest);
    r.physical.push((backend.name(), physical));
    r
}

#[test]
fn write_markdown_creates_file_at_path() {
    let report = build_report("file-at-path");

    let mut path = std::env::temp_dir();
    path.push(format!("rwf-out-{}.md", std::process::id()));

    report.write_markdown(&path).expect("write succeeds");

    let contents = std::fs::read_to_string(&path).expect("file readable");
    assert!(contents.contains("# TinyConv-MNIST bench report"));
    assert!(contents.contains("1405.746"), "real area number absent");
    assert!(contents.contains("optimizer seed | 7"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn write_markdown_creates_parent_directories() {
    let report = build_report("nested-dirs");

    let mut path = std::env::temp_dir();
    path.push(format!("rwf-nested-{}", std::process::id()));
    path.push("a");
    path.push("b");
    path.push("report.md");

    report
        .write_markdown(&path)
        .expect("write creates parent dirs");
    assert!(path.exists());

    // Cleanup nested tempdir.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap().parent().unwrap());
}
