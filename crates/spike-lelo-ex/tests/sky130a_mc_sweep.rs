//! 8-run Monte Carlo sweep of the LELO_EX 1:4 mirror against sky130A's
//! `.lib mc` section.
//!
//! Verifies:
//!   - The harness injects a unique `set rndseed=<n>` per corner, so
//!     successive MC draws produce DIFFERENT `ibn` values (σ > 0).
//!   - Mean mirror ratio sits inside the typ-corner spec envelope.
//!   - Aggregate stats from `eda_waveform::mc::{collect_stats, check_spec}`
//!     are well-formed (matching the harness's per-spec `SpecCheck`).
//!   - All artifacts (HTML/MD/PDF/PNG) publish to
//!     `crates/spike-lelo-ex/docs/sky130a_mc/`.

use eda_sim_harness::{
    docs_dir_for_crate, Cache, CacheMode, Corner, CornerSet, Harness, MeasurementValue, Reporter,
    Spec, SpecBundle,
};
use eda_waveform::mc as wmc;
use spike_lelo_ex::LeloEx;

const IREF: f64 = 5e-6;
const N_MC: usize = 8;

fn ngspice_present() -> bool {
    std::env::var_os("PATH").map_or(false, |p| {
        std::env::split_paths(&p).any(|d| d.join("ngspice").is_file())
    })
}

fn specs() -> SpecBundle {
    SpecBundle {
        specs: vec![
            Spec {
                name: "ibn".into(),
                min: Some(IREF * 3.0),
                typ: Some(IREF * 4.7),
                max: Some(IREF * 6.0),
                unit: Some("A".into()),
            },
            Spec {
                name: "vgs_m1".into(),
                min: Some(0.40),
                typ: Some(0.76),
                max: Some(1.10),
                unit: Some("V".into()),
            },
        ],
    }
}

#[test]
fn lelo_ex_mc_sweep_against_sky130a() {
    if !ngspice_present() {
        eprintln!("ngspice not on PATH; skipping MC sweep");
        return;
    }
    let pdk = match rlx_eda_cli::resolve_pdk("sky130A") {
        Ok(e) => e,
        Err(e) => {
            eprintln!("sky130A not registered ({e}); skipping. Run `rlx-eda pdk install sky130A` first.");
            return;
        }
    };
    assert!(pdk.sections.iter().any(|s| s == "mc"),
        "sky130A registry entry missing the `mc` section. Re-run pdk install.");

    let outdir = docs_dir_for_crate(env!("CARGO_MANIFEST_DIR")).join("sky130a_mc");
    let _ = std::fs::remove_dir_all(&outdir);
    let cachedir = std::env::temp_dir().join("rlx-eda-spike-lelo-ex-mc-cache");

    let corners = CornerSet::new()
        // typical reference run for context.
        .push(Corner::typical("tt", pdk.vdd_nom))
        // 8 MC draws against the `mc` lib section.
        .add_mc("mc", pdk.vdd_nom, N_MC, /*seed_base=*/ 1);

    let tb = LeloEx::new(&pdk.lib_path);
    let bundle = specs();
    let outcomes = Harness::new(&tb)
        .corners(corners)
        .specs(bundle.clone())
        .cache(Cache::new(cachedir, CacheMode::Auto))
        .output_dir(&outdir)
        .run()
        .expect("harness MC run");

    assert_eq!(outcomes.len(), 1 + N_MC);

    // Pull `ibn` from each MC outcome (skip the typical baseline at idx 0).
    let mc_runs: Vec<wmc::Run<f64>> = outcomes.iter().skip(1).map(|o| {
        let v = match o.measures.get("ibn") {
            Some(MeasurementValue::Number(v)) => v,
            _ => panic!(
                "missing ibn for {}\n--- stdout ---\n{}",
                o.corner.label, o.stdout,
            ),
        };
        wmc::Run { label: o.corner.label.clone(), metric: v }
    }).collect();

    let stats = wmc::collect_stats(&mc_runs, wmc::Worst::Min)
        .expect("collect_stats over MC runs");

    eprintln!("\n== LELO_EX sky130A MC sweep ({} draws) ==", N_MC);
    eprintln!("  mean   = {:>10.3e} A   (×{:.3})", stats.mean, stats.mean / IREF);
    eprintln!("  std    = {:>10.3e} A", stats.std);
    eprintln!("  min    = {:>10.3e} A   (worst: {})", stats.min, stats.worst.as_ref().unwrap().0);
    eprintln!("  max    = {:>10.3e} A", stats.max);
    eprintln!("  median = {:>10.3e} A", stats.median);
    for r in &mc_runs {
        let ratio = r.metric / IREF;
        eprintln!("    {:<8}  ibn = {:>10.3e} A  (×{:.3})", r.label, r.metric, ratio);
    }

    // σ must be strictly positive — otherwise the rndseed isn't actually
    // varying the draws and our MC infrastructure is silently broken.
    assert!(
        stats.std > 1e-9,
        "MC σ = {:.3e} ≈ 0; either rndseed isn't taking effect or mc lib_section is degenerate",
        stats.std,
    );

    // Mean ratio centered near the typ corner (4.7×) within ±25 %.
    let mean_ratio = stats.mean / IREF;
    assert!(
        (3.5..=6.0).contains(&mean_ratio),
        "MC mean mirror ratio {mean_ratio:.3} out of envelope [3.5, 6.0]",
    );

    // Yield against the ibn spec.
    let yield_ = wmc::check_spec(&mc_runs, |v| v >= IREF * 3.0 && v <= IREF * 6.0);
    eprintln!(
        "  yield: {}/{} pass ({:.0}%) against ibn ∈ [3.0, 6.0]× iref",
        yield_.n_pass, yield_.n_total, yield_.yield_frac * 100.0,
    );
    assert_eq!(yield_.n_total, N_MC);

    // Reporter publishes the MC sweep into docs/.
    let reporter = Reporter::new("lelo_ex_mc", &bundle, &outcomes, &outdir);
    let written = reporter.write_all().expect("reporter");
    let names: Vec<String> = written.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    for required in ["README.md", "lelo_ex_mc_summary.html", "lelo_ex_mc_summary.pdf"] {
        assert!(names.iter().any(|n| n == required), "docs missing {required}: {names:?}");
    }
    // Per-corner HTML for at least the first 3 mc runs.
    for i in 0..3 {
        let want = format!("lelo_ex_mc_mc_{:03}.html", i);
        assert!(names.iter().any(|n| n == &want), "missing {want}: {names:?}");
    }
}
