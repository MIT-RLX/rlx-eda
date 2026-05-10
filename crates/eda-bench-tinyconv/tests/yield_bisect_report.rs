//! Tests for `YieldGate::evaluate`, `bisect::bisect`, and
//! `Report::to_markdown`. All pure-function — no toolchain required.

use eda_bench_tinyconv::{
    bisect::{bisect, Divergence},
    manifest::{Manifest, ManifestInputs},
    metrics::{
        functional::{Functional, Level, YieldGate},
        Physical,
    },
    report::Report,
};
use std::io::Write;

fn fn_run(top1: f64, divergence: Option<usize>) -> Functional {
    Functional {
        level: Level::L5PvtMc,
        top1_acc: top1,
        per_class_acc: [top1; 10],
        divergence_first_layer: divergence,
        n_images: 10_000,
    }
}

#[test]
fn yield_gate_passes_when_pass_rate_meets_threshold() {
    let g = YieldGate::RELEASE; // 0.97 / 0.99
    // 99 of 100 runs at 0.98 (pass), 1 at 0.95 (fail) → 0.99 pass rate.
    let mut runs: Vec<Functional> = (0..99).map(|_| fn_run(0.98, None)).collect();
    runs.push(fn_run(0.95, None));
    assert!(g.evaluate(&runs));
    assert!((g.pass_rate(&runs) - 0.99).abs() < 1e-9);
}

#[test]
fn yield_gate_fails_when_pass_rate_below_threshold() {
    let g = YieldGate::RELEASE;
    // 98 / 100 → 0.98, below 0.99 threshold.
    let mut runs: Vec<Functional> = (0..98).map(|_| fn_run(0.98, None)).collect();
    runs.push(fn_run(0.95, None));
    runs.push(fn_run(0.95, None));
    assert!(!g.evaluate(&runs));
}

#[test]
fn yield_gate_fails_on_empty_input() {
    assert!(!YieldGate::RELEASE.evaluate(&[]));
    assert_eq!(YieldGate::RELEASE.pass_rate(&[]), 0.0);
}

#[test]
fn bisect_emits_one_record_per_divergent_backend() {
    let runs = vec![
        ("inhouse", fn_run(0.97, Some(3))),
        ("orfs", fn_run(0.97, None)),
        ("fpga", fn_run(0.97, Some(1))),
    ];
    let divs = bisect(&runs);
    assert_eq!(divs.len(), 2);
    assert_eq!(divs[0].backend, "inhouse");
    assert_eq!(divs[0].layer, 3);
    assert_eq!(divs[1].backend, "fpga");
    assert_eq!(divs[1].layer, 1);
    // v1 fields stay empty until activation checkpoints land.
    for d in &divs {
        assert_eq!(d.image, None);
        assert_eq!(d.tile, None);
        assert_eq!(d.max_abs_err, 0.0);
    }
}

#[test]
fn bisect_returns_empty_when_no_divergence() {
    let runs = vec![("inhouse", fn_run(0.97, None))];
    assert!(bisect(&runs).is_empty());
}

fn fake_manifest() -> Manifest {
    // Minimal manifest from a tempfile — exercises the real capture
    // path so the report rendering sees a populated struct.
    //
    // Process-id-only filenames race across parallel tests in this
    // file (`empty_inputs_gracefully` and `full_markdown_sections`
    // both call here, both use the same PID). Add a per-call counter
    // so each invocation gets a unique tempfile path.
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let mut p = std::env::temp_dir();
    p.push(format!("eda-bench-report-test-{}-{n}", std::process::id()));
    std::fs::File::create(&p)
        .unwrap()
        .write_all(b"[[package]]\n")
        .unwrap();
    let m = Manifest::capture(ManifestInputs {
        sky130_repo: None,
        orfs_image: None,
        weights: None,
        cargo_lock: &p,
        seed: 17,
    })
    .expect("capture");
    let _ = std::fs::remove_file(&p);
    m
}

fn empty_physical() -> Physical {
    Physical {
        area_um2: Some(4321.5),
        max_freq_mhz: Some(250.0),
        wns_ns: Some(0.12),
        dynamic_power_mw: Some(1.7),
        leakage_power_mw: None,
        parasitic_cap_ff: Some(880.0),
        peak_temp_c: None,
        energy_pj_per_inference: None,
    }
}

#[test]
fn report_renders_full_markdown_sections() {
    let mut r = Report::new(fake_manifest());
    r.physical.push(("orfs", empty_physical()));
    r.functional.push(("inhouse", fn_run(0.978, Some(2))));
    r.functional.push(("orfs", fn_run(0.981, None)));
    r.l5_runs.extend((0..100).map(|i| fn_run(if i < 99 { 0.98 } else { 0.95 }, None)));

    let md = r.to_markdown();

    // Headers
    assert!(md.contains("# TinyConv-MNIST bench report"));
    assert!(md.contains("## Reproducibility manifest"));
    assert!(md.contains("## Physical metrics"));
    assert!(md.contains("## Functional metrics"));
    assert!(md.contains("## Yield gate"));
    assert!(md.contains("## Functional divergence (bisection)"));

    // Manifest contents
    assert!(md.contains("optimizer seed | 17"));

    // Physical row
    assert!(md.contains("orfs"));
    assert!(md.contains("4321.500"));
    // Null fields render as em-dash
    assert!(md.contains(" — "));

    // Yield gate computed correctly: 99/100 ≥ 0.99 → PASS
    assert!(md.contains("PASS"));

    // Bisection picked up the inhouse divergence at layer 2
    assert!(md.contains("| inhouse |"));
    assert!(md.contains("| 2 |"));
}

#[test]
fn report_handles_empty_inputs_gracefully() {
    let r = Report::new(fake_manifest());
    let md = r.to_markdown();
    assert!(md.contains("_no physical measurements yet_"));
    assert!(md.contains("_no functional measurements yet_"));
    assert!(md.contains("_no L5 PVT × MC runs yet"));
    assert!(md.contains("_no divergences detected_"));
}
