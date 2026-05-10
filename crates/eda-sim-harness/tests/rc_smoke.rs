//! Phase-1 end-to-end smoke test.
//!
//! A trivial RC testbench (no PDK needed — just a 1Ω resistor and a 1pF
//! cap with a step input) exercises the full harness loop:
//!
//!   build_netlist → corner sweep → ngspice → .meas parse → spec check
//!   → reporter → HTML/MD/PNG on disk.
//!
//! Two corners (typical, hot) prove the cross-corner aggregation. The
//! spec is "v(out) at 5τ ≥ 0.99 V" which both corners pass.
//!
//! Skipped when ngspice isn't on PATH so this can run in CI matrices
//! that lack a SPICE engine. The whole point of the harness is to be
//! exercised against real ngspice though, so this is the canonical
//! validation.

use std::path::PathBuf;

use eda_sim_harness::{
    Analysis, Cache, CacheMode, Corner, CornerSet, Harness, Measurement, Spec, SpecBundle,
    SpecCheck, Testbench,
};
use eda_spice_emit::Netlist;

fn ngspice_present() -> bool {
    std::env::var_os("PATH").map_or(false, |path| {
        std::env::split_paths(&path)
            .any(|d| d.join("ngspice").is_file())
    })
}

struct RcTb;

impl Testbench for RcTb {
    fn name(&self) -> &str { "rc_step" }

    fn build_netlist(&self, corner: &Corner) -> Netlist {
        let mut nl = Netlist::new("rc step response");
        // Step from 0→vdd at t=0. PULSE with PER huge ⇒ single pulse.
        nl.add_element(format!(
            "Vin in 0 PULSE(0 {vdd:.6e} 0 1e-12 1e-12 1 1e30)",
            vdd = corner.vdd,
        ));
        nl.add_element("R1 in out 1k".to_string());
        nl.add_element("C1 out 0 1n".to_string());
        nl.add_element(".ic v(out)=0".to_string());
        nl
    }

    fn measurements(&self) -> Vec<Measurement> {
        vec![
            // Final value at t = 5τ = 5µs. Should approach vdd.
            Measurement::tran("vout_settled", "find v(out) at=5u", Some("V")),
            // Peak value over the run (sanity).
            Measurement::tran("vout_peak", "max v(out) from=0 to=5u", Some("V")),
        ]
    }

    fn analysis(&self) -> Analysis {
        Analysis::Tran { t_step: 50e-9, t_stop: 5e-6, uic: true }
    }
}

fn specs() -> SpecBundle {
    SpecBundle {
        specs: vec![
            Spec {
                name: "vout_settled".into(),
                min: Some(0.95),
                typ: Some(0.993),
                max: Some(1.30),
                unit: Some("V".into()),
            },
            Spec {
                name: "vout_peak".into(),
                min: Some(0.95),
                typ: None,
                max: Some(1.30),
                unit: Some("V".into()),
            },
        ],
    }
}

#[test]
fn rc_step_runs_and_reports_pass() {
    if !ngspice_present() {
        eprintln!("ngspice not found on PATH; skipping rc_step harness smoke test");
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let outdir: PathBuf = tmp.path().join("results");
    let cachedir: PathBuf = tmp.path().join("cache");

    let corners = CornerSet::new()
        .push(Corner::typical("tt", 1.0))
        .push(Corner::etc("hot", "tt", 1.0, 85.0));

    let tb = RcTb;
    let outcomes = Harness::new(&tb)
        .corners(corners)
        .specs(specs())
        .cache(Cache::new(cachedir.clone(), CacheMode::Auto))
        .output_dir(&outdir)
        .run()
        .expect("harness run");

    assert_eq!(outcomes.len(), 2);

    for o in &outcomes {
        assert!(!o.from_cache, "first run should not be cached");
        // 5τ should put us at ~99.3 % of vdd.
        let v = o.measures.get("vout_settled").and_then(|v| v.as_number())
            .unwrap_or_else(|| panic!("vout_settled missing for {}: stdout=\n{}\n--- deck ---\n{}", o.corner.label, o.stdout, o.deck));
        assert!((0.95..1.05).contains(&v),
            "vout_settled out of range for {}: {v}\n--- deck ---\n{}\n--- stdout ---\n{}", o.corner.label, o.deck, o.stdout);
        for (_, c) in &o.spec_checks {
            assert!(matches!(c, SpecCheck::Pass { .. }),
                "spec failed unexpectedly: {:?} stdout=\n{}", c, o.stdout);
        }
    }

    // Reporter writes everything.
    let bundle = specs();
    let reporter = eda_sim_harness::Reporter::new("rc_step", &bundle, &outcomes, &outdir);
    let written = reporter.write_all().expect("reporter");
    assert!(written.iter().any(|p| p.file_name().unwrap() == "README.md"));
    assert!(written.iter().any(|p| p.file_name().unwrap() == "rc_step_summary.html"));
    assert!(written.iter().any(|p| p.file_name().unwrap().to_string_lossy().ends_with("typical.html")));
    // PNGs land for transient analyses.
    assert!(written.iter().any(|p| p.extension().and_then(|s| s.to_str()) == Some("png")),
        "expected at least one PNG in {:?}", written);

    // Cache reuse: re-run with Auto should hit the cache for both corners.
    let outcomes2 = Harness::new(&tb)
        .corners(CornerSet::new()
            .push(Corner::typical("tt", 1.0))
            .push(Corner::etc("hot", "tt", 1.0, 85.0)))
        .specs(specs())
        .cache(Cache::new(cachedir.clone(), CacheMode::Auto))
        .output_dir(&outdir)
        .run()
        .expect("second harness run");
    for o in &outcomes2 {
        assert!(o.from_cache, "second run should hit cache for {}", o.corner.label);
    }
}
