//! Schematic vs. Layout regression — runs the typical corner under both
//! `View::Schematic` and `View::Layout`, asserting that the layout view
//! produces a measurable but bounded delta from the schematic baseline.
//!
//! This is the cicsim `make all`-equivalent: simulate every corner on
//! both netlists, then diff. Fails if the layout has *no* delta (would
//! mean the parasitics aren't actually getting included) or *too much*
//! delta (regression — a routing change suddenly shifted the mirror).

use eda_sim_harness::{
    docs_dir_for_crate, Cache, CacheMode, Corner, CornerSet, Harness, Reporter, Spec, SpecBundle,
    View,
};
use spike_lelo_ex::LeloEx;

const IREF: f64 = 5e-6;

fn ngspice_present() -> bool {
    std::env::var_os("PATH").map_or(false, |p| {
        std::env::split_paths(&p).any(|d| d.join("ngspice").is_file())
    })
}

#[test]
fn lelo_ex_sch_vs_lay_typical() {
    if !ngspice_present() {
        eprintln!("ngspice not on PATH; skipping");
        return;
    }
    let pdk = match rlx_eda_cli::resolve_pdk("sky130A") {
        Ok(e) => e,
        Err(e) => { eprintln!("sky130A not registered ({e}); skipping"); return; }
    };

    let outdir = docs_dir_for_crate(env!("CARGO_MANIFEST_DIR")).join("sky130a_sch_vs_lay");
    let _ = std::fs::remove_dir_all(&outdir);
    let cachedir = std::env::temp_dir().join("rlx-eda-spike-lelo-ex-sch-vs-lay");

    // One typical corner expanded into both views.
    let corners = CornerSet::new()
        .push(Corner::typical("tt", pdk.vdd_nom))
        .expand_views();
    assert_eq!(corners.corners.len(), 2);
    assert!(corners.corners.iter().any(|c| c.view == View::Schematic));
    assert!(corners.corners.iter().any(|c| c.view == View::Layout));

    let tb = LeloEx::new(&pdk.lib_path);
    let bundle = SpecBundle {
        specs: vec![Spec {
            name: "ibn".into(),
            min: Some(IREF * 3.5),
            typ: Some(IREF * 4.7),
            max: Some(IREF * 5.5),
            unit: Some("A".into()),
        }],
    };
    let outcomes = Harness::new(&tb)
        .corners(corners)
        .specs(bundle.clone())
        .cache(Cache::new(cachedir, CacheMode::Auto))
        .output_dir(&outdir)
        .run()
        .expect("harness run");

    let sch = outcomes.iter().find(|o| o.corner.view == View::Schematic)
        .expect("Sch corner not found");
    let lay = outcomes.iter().find(|o| o.corner.view == View::Layout)
        .expect("Lay corner not found");

    let sch_ibn = sch.measures.get("ibn").and_then(|v| v.as_number())
        .unwrap_or_else(|| panic!("Sch ibn missing\nstdout=\n{}", sch.stdout));
    let lay_ibn = lay.measures.get("ibn").and_then(|v| v.as_number())
        .unwrap_or_else(|| panic!("Lay ibn missing\nstdout=\n{}", lay.stdout));

    let delta = lay_ibn - sch_ibn;
    let pct = (delta / sch_ibn) * 100.0;
    eprintln!("\n== LELO_EX Sch vs Lay (typical corner) ==");
    eprintln!("  Sch ibn = {:.4e} A", sch_ibn);
    eprintln!("  Lay ibn = {:.4e} A   (Δ = {:.4e} A, {:+.3} %)", lay_ibn, delta, pct);

    // Real layout extraction shifts mirror current. Assert non-trivial
    // delta (≥ 0.05 %) so we know parasitics ARE participating, and
    // bounded (≤ 5 %) so a regression in `layout_drain_r` calibration
    // doesn't slip through silently.
    let abs_pct = pct.abs();
    assert!(
        abs_pct >= 0.05,
        "Sch vs Lay delta too small ({pct:.4} %): parasitics didn't contribute. Check Rdrain wiring.",
    );
    assert!(
        abs_pct <= 5.0,
        "Sch vs Lay delta too large ({pct:.4} %): something inflated parasitics. Check layout_drain_r.",
    );

    // Reporter writes per-view artifacts with `_Sch_` / `_Lay_` infix.
    let reporter = Reporter::new("lelo_ex", &bundle, &outcomes, &outdir);
    let written = reporter.write_all().expect("reporter");
    let names: Vec<String> = written.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    for required in [
        "lelo_ex_Sch_typical.html",
        "lelo_ex_Lay_typical.html",
        "lelo_ex_Sch_typical.deck.spice",
        "lelo_ex_Lay_typical.deck.spice",
        "lelo_ex_summary.html",
        "lelo_ex_summary.pdf",
        "README.md",
    ] {
        assert!(names.iter().any(|n| n == required), "missing {required}\ngot: {names:?}");
    }

    // The Sch and Lay decks must differ — Lay should contain Rdrain.
    let sch_deck = std::fs::read_to_string(outdir.join("lelo_ex_Sch_typical.deck.spice")).unwrap();
    let lay_deck = std::fs::read_to_string(outdir.join("lelo_ex_Lay_typical.deck.spice")).unwrap();
    assert!(!sch_deck.contains("Rdrain"), "Sch deck unexpectedly contains Rdrain");
    assert!(lay_deck.contains("Rdrain"), "Lay deck missing Rdrain — view branching broken");
}
