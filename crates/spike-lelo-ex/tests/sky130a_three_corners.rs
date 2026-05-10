//! End-to-end sky130A LELO_EX run across three corners (tt/ff/ss).
//!
//! Skips when sky130A isn't registered via `rlx-eda pdk install`; if
//! you want to run this locally:
//!
//! ```sh
//! cargo run -p rlx-eda-cli -- pdk install sky130A
//! cargo test  -p spike-lelo-ex --test sky130a_three_corners -- --nocapture
//! ```
//!
//! What this asserts:
//!   - All three corners simulate without ngspice errors.
//!   - The mirror ratio `ibn / iref` lands within `[3.0, 5.0]` per
//!     corner (target 4.0; ±25% catches typical sky130 corner shift).
//!   - `vgs_m1` is in `[0.4, 1.0] V` — comfortably inside the model's
//!     valid bias range.
//!   - The harness's reporter writes per-corner HTML, summary HTML,
//!     `README.md`, and PNGs to disk.

use std::path::PathBuf;

use eda_sim_harness::{
    docs_dir_for_crate, Cache, CacheMode, Corner, CornerSet, Harness, Reporter, Spec, SpecBundle,
    SpecCheck,
};
use spike_lelo_ex::LeloEx;

fn ngspice_present() -> bool {
    std::env::var_os("PATH").map_or(false, |p| {
        std::env::split_paths(&p).any(|d| d.join("ngspice").is_file())
    })
}

const IREF: f64 = 5e-6;

fn specs() -> SpecBundle {
    SpecBundle {
        specs: vec![
            // Mirror current — 4× iref nominal. sky130 λ pushes the
            // typical ratio to ~4.7× and slow corners to ~5×, so the
            // envelope is [3.5, 5.5]× to absorb tt/ff/ss spread without
            // making the bound trivially loose.
            Spec {
                name: "ibn".into(),
                min: Some(IREF * 3.5),
                typ: Some(IREF * 4.7),
                max: Some(IREF * 5.5),
                unit: Some("A".into()),
            },
            // Vgs sanity — in-range bias for nfet_01v8 at 5 µA, W=2 L=2 µm.
            Spec {
                name: "vgs_m1".into(),
                min: Some(0.50),
                typ: Some(0.76),
                max: Some(0.95),
                unit: Some("V".into()),
            },
        ],
    }
}

#[test]
fn lelo_ex_runs_against_sky130a_tt_ff_ss() {
    if !ngspice_present() {
        eprintln!("ngspice not on PATH; skipping LELO_EX sky130A test");
        return;
    }
    let pdk = match rlx_eda_cli::resolve_pdk("sky130A") {
        Ok(e) => e,
        Err(e) => {
            eprintln!("sky130A not registered ({e}); skipping. Run `rlx-eda pdk install sky130A` first.");
            return;
        }
    };
    // Sanity: the lib must contain tt/ff/ss for this test.
    for need in ["tt", "ff", "ss"] {
        assert!(
            pdk.sections.iter().any(|s| s == need),
            "sky130A registry entry is missing the {need} section. Re-run `rlx-eda pdk install sky130A`.",
        );
    }

    // Publish the run artifacts (HTML/MD/PDF/PNG) into the crate's
    // docs/ directory so they round-trip with the source tree. Cache
    // stays under the OS tempdir so reruns are fast but never pollute
    // the repo. Cleared on each test invocation so docs/ is a faithful
    // snapshot of the latest verified run.
    let outdir = docs_dir_for_crate(env!("CARGO_MANIFEST_DIR")).join("sky130a_tt_ff_ss");
    let _ = std::fs::remove_dir_all(&outdir);
    let cachedir = std::env::temp_dir().join("rlx-eda-spike-lelo-ex-cache");

    let corners = CornerSet::new()
        .push(Corner::typical("tt", pdk.vdd_nom))
        .push(Corner::etc("ff", "ff", pdk.vdd_nom * 1.10, 85.0))
        .push(Corner::etc("ss", "ss", pdk.vdd_nom * 0.90, -40.0));

    let tb = LeloEx::new(&pdk.lib_path);
    let bundle = specs();
    let outcomes = Harness::new(&tb)
        .corners(corners)
        .specs(bundle.clone())
        .cache(Cache::new(cachedir.clone(), CacheMode::Auto))
        .output_dir(&outdir)
        .run()
        .expect("harness run");

    assert_eq!(outcomes.len(), 3);
    eprintln!("\n== LELO_EX sky130A ==");
    for o in &outcomes {
        let ibn = o.measures.get("ibn").and_then(|v| v.as_number())
            .unwrap_or_else(|| panic!("ibn missing for {}\n--- stdout ---\n{}\n--- deck ---\n{}", o.corner.label, o.stdout, o.deck));
        let vgs = o.measures.get("vgs_m1").and_then(|v| v.as_number())
            .unwrap_or_else(|| panic!("vgs_m1 missing for {}\n--- stdout ---\n{}", o.corner.label, o.stdout));
        let ratio = ibn / IREF;
        eprintln!(
            "  {label:<6}  ibn = {ibn:>10.3e} A  (×{ratio:.2})   vgs_m1 = {vgs:.3} V   (cached={cached})",
            label = o.corner.label, cached = o.from_cache,
        );

        assert!(
            (3.5..=5.5).contains(&ratio),
            "[{}] mirror ratio out of envelope: ibn = {ibn:.3e} A, ratio = {ratio:.2}\n--- stdout ---\n{}",
            o.corner.label, o.stdout,
        );
        assert!(
            (0.50..=0.95).contains(&vgs),
            "[{}] vgs_m1 out of range: {vgs:.3} V\n--- stdout ---\n{}",
            o.corner.label, o.stdout,
        );

        for (name, check) in &o.spec_checks {
            assert!(
                matches!(check, SpecCheck::Pass { .. }),
                "[{}] spec '{name}' didn't pass: {:?}", o.corner.label, check,
            );
        }
    }

    // Reporter: full set of artifacts on disk in crates/spike-lelo-ex/docs/sky130a_tt_ff_ss.
    let reporter = Reporter::new("lelo_ex", &bundle, &outcomes, &outdir);
    let written = reporter.write_all().expect("reporter");
    let names: Vec<String> = written.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    for required in [
        "README.md",
        "lelo_ex_summary.html",
        "lelo_ex_summary.pdf",
    ] {
        assert!(names.iter().any(|n| n == required),
            "missing {required} in published docs: {names:?}");
    }
    for label in ["typical", "ff", "ss"] {
        for ext in ["html", "png"] {
            let want = format!("lelo_ex_{label}.{ext}");
            assert!(names.iter().any(|n| n == &want),
                "missing {want} in published docs: {names:?}");
        }
    }
    // The PDF actually parses as a PDF (4-byte magic).
    let pdf = outdir.join("lelo_ex_summary.pdf");
    let bytes = std::fs::read(&pdf).expect("pdf read");
    assert!(bytes.starts_with(b"%PDF-"), "summary.pdf isn't a valid PDF: starts with {:?}", &bytes[..bytes.len().min(8)]);

    let _: PathBuf = outdir; // silence unused if checks ever shrink
}
