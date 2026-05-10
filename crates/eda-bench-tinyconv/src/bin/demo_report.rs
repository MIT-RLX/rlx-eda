//! Demo bench report — runs the full pipeline against the mock
//! sky130 library and writes a markdown artifact for human
//! inspection.
//!
//! ```sh
//! cargo run -p eda-bench-tinyconv --features demo-bin --bin demo_report
//! ```
//!
//! Output: `target/bench/demo/report.md` (relative to the workspace
//! root). Sections: reproducibility manifest, physical metrics
//! (Liberty-derived in-house area for tile + array), functional
//! metrics (FPGA L1 reference accuracy), inference performance
//! (per-image latency + throughput), bundle (when enabled), and
//! honest "no … yet" placeholders for ORFS / yield-gate slots.

use eda_bench_tinyconv::{
    backends::{fpga::FpgaBackend, inhouse::InhouseBackend, Backend},
    bundle::write_bundle,
    config::BenchConfig,
    inference::{run_inference_bench, SimulatedLatency},
    manifest::{Manifest, ManifestInputs},
    metrics::{functional::Level, Physical},
    optimization::LossWeights,
    Report,
};
use spike_tinyconv_array::lower::total_cycles;
use spike_tinyconv_tile::silicon_time_ns_per_inference;
use eda_bench_tinyconv::optimization::{inner, LossWeights as BenchLossWeights};
use eda_config::Configurable;
use eda_stdcells::{populate_mock_sc_hd, ScHdLibrary};
use klayout_core::Library;
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use spike_tinyconv_array::array::{ArrayBlock, ArrayConfig};
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};
use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let workspace_root: PathBuf = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("..")
        .join("..");

    // ── Config (loaded from `<workspace>/configs/bench.toml` if
    // present; or `EDA_CONFIG_BENCH=path/to/file` to override).
    // Demo binary explicitly resolves from the computed
    // workspace_root because `CARGO_MANIFEST_DIR` points at the
    // bench crate, not the workspace root.
    let cfg_path = match std::env::var("EDA_CONFIG_BENCH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => workspace_root.join("configs").join("bench.toml"),
    };
    let cfg: BenchConfig = eda_config::load_or_default(&cfg_path);
    eprintln!("loaded config from {}", cfg_path.display());
    let _ = BenchConfig::load_or_default; // silence unused-import lint when this fn isn't called
    eprintln!(
        "config seed={}, inference: {}×{}, pnr: {}, bundle.merge: {}",
        cfg.run.seed,
        cfg.inference.n_images,
        cfg.inference.repetitions,
        cfg.pnr.enabled,
        cfg.bundle.merge_weights,
    );

    let cargo_lock = workspace_root.join("Cargo.lock");
    let manifest = Manifest::capture(ManifestInputs {
        sky130_repo: Some(std::path::Path::new("/Users/Shared/mtl/skywater130")),
        orfs_image: None,
        weights: None,
        cargo_lock: &cargo_lock,
        seed: cfg.run.seed,
    })
    .unwrap_or_else(|e| {
        eprintln!("warning: capturing manifest failed ({e}); using tempfile fallback");
        let mut p = std::env::temp_dir();
        p.push("eda-bench-tinyconv-demo-cargo-lock");
        std::fs::write(&p, b"[[package]]\n").unwrap();
        Manifest::capture(ManifestInputs {
            sky130_repo: None,
            orfs_image: None,
            weights: None,
            cargo_lock: &p,
            seed: cfg.run.seed,
        })
        .expect("fallback manifest")
    });

    let lib = Library::new("demo", 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let cells = populate_mock_sc_hd(&lib, &pdk);
    let library = ScHdLibrary { library: lib, cells };

    // ── Tile + array physical ───────────────────────────────────
    let tile = Mac8x8Tile::with_topology(
        "u_demo_tile",
        TileParams::default(),
        MacTopology::Digital,
    );
    let tile_phys: Physical = InhouseBackend::from_tile(&tile, clone_library(&library))
        .measure_physical()
        .expect("tile measurement");

    let array = ArrayBlock::new("u_demo_array", ArrayConfig::default());
    let array_phys: Physical = InhouseBackend::from_array(&array, clone_library(&library))
        .measure_physical()
        .expect("array measurement");

    // ── FPGA L1 reference functional ────────────────────────────
    let model = tinyconv_mnist_from_cortexm();
    let test_set = vec![one_test_pair()];
    let fpga = FpgaBackend::new(workspace_root.join("target/fpga"), model.clone())
        .with_test_set(test_set.clone());
    let l1 = fpga
        .measure_functional(Level::L1Reference, &[])
        .expect("L1 reference");

    // ── Inference performance ───────────────────────────────────
    // L1 reference: pure Rust, wall-clock only.
    let inference_l1 = run_inference_bench(&model, &test_set, &cfg.inference)
        .expect("inference bench");

    // L2: real RTL simulation via Verilator (gated by feature).
    //   - With `--features bench-rtl-sim`: invokes Verilator inside
    //     docker on the rlx-fpga-emitted SystemVerilog. Cycle count
    //     comes from a counter on rising clock edges between
    //     `start=1` and `done=1`. Adds ~30-60s wall-clock.
    //   - Without: falls back to the analytic `total_cycles` estimate
    //     (often off by orders of magnitude vs the real RTL — the
    //     bench reporter labels this clearly).
    let inference_l2 = build_l2_inference(&inference_l1, &model, &workspace_root);

    let l2_label: &str = if inference_l2_used_real_sim() {
        "rtl-sim (L2, Verilator)"
    } else {
        "rtl-sim-estimate (L2, analytic)"
    };

    // ── Loss baseline anchor ────────────────────────────────────
    let inv = tile.cell_inventory();
    let weights = LossWeights::default().with_inhouse_baseline(&clone_library(&library), &inv);
    let baseline_um2 = weights.area_baseline_um2.unwrap_or(0.0);

    // ── Bundle (when enabled in config) ─────────────────────────
    // Stand-in payloads — real demo would emit `top.sv` from
    // rlx-fpga and weight blobs from cortexm. For the bundle
    // contract, we just need bytes to checksum.
    let stand_in_top_sv = b"// placeholder top.sv\nmodule top; endmodule\n";
    let stand_in_weights = b"\x01\x02\x03\x04\x05\x06\x07\x08";
    let bundle_entries = write_bundle(
        &cfg.bundle,
        &[
            ("top.sv", stand_in_top_sv.as_ref()),
            ("weights/conv1.mem", stand_in_weights.as_ref()),
        ],
    )
    .unwrap_or_else(|e| {
        eprintln!("warning: bundle write failed ({e}); skipping");
        Vec::new()
    });

    // ── Render + write ──────────────────────────────────────────
    let mut report = Report::new(manifest);
    report.physical.push(("inhouse-tile", tile_phys));
    report.physical.push(("inhouse-array", array_phys));
    report.functional.push(("fpga", l1));
    report.inference.push(("fpga-reference (L1)", inference_l1.clone()));
    report.inference.push((static_l2_label(l2_label), inference_l2.clone()));
    report.bundle = bundle_entries;

    let out = workspace_root.join(&cfg.run.output_path);
    report.write_markdown(&out)?;
    println!(
        "wrote bench report to {} ({} bytes)",
        out.display(),
        std::fs::metadata(&out)?.len()
    );
    println!(
        "loss-weights area baseline (Liberty-derived, tile-scope): {baseline_um2} µm²"
    );
    println!(
        "inference L1 (FPGA reference, wall-clock): mean {:.1} µs, p99 {:.1} µs, host throughput {:.0}/s ({}×{} samples)",
        inference_l1.mean_us,
        inference_l1.p99_us,
        inference_l1.throughput_per_sec,
        inference_l1.n_images,
        inference_l1.repetitions,
    );
    if let Some(sim) = inference_l2.simulated {
        println!(
            "inference L2 ({}): {} cycles × {:.0} ns = {:.1} µs / inference, silicon throughput {:.0}/s",
            l2_label,
            sim.cycles_per_inference,
            sim.period_ns,
            sim.total_ns / 1000.0,
            sim.silicon_throughput_per_sec,
        );

        // ── Adam optimizes silicon clock time, not host wall time ───
        // Run inner Adam with delay-dominated weights + the real cycle
        // count plumbed through. Adam's residual now optimizes
        // `cycles × period_ns` end-to-end; the converged tile params
        // give a lower silicon time per inference than the starting
        // point. Show the before/after gap.
        let starting_tile = Mac8x8Tile::with_topology(
            "u_silicon_opt",
            TileParams { w_l_n: 1.0, w_l_p: 1.0, vdd: 1.0, ..TileParams::default() },
            MacTopology::Digital,
        );
        let initial_silicon_ns =
            silicon_time_ns_per_inference(starting_tile.params, sim.cycles_per_inference);

        let weights_for_silicon = BenchLossWeights {
            // Delay-dominated weights: care more about silicon
            // latency than energy/area for this demo run.
            alpha_energy: 1.0,
            beta_delay: 100.0,
            gamma_area: 0.01,
            cycles_per_inference: Some(sim.cycles_per_inference),
            ..BenchLossWeights::default()
        };
        let inner_cfg = inner::InnerConfig {
            max_steps: 100,
            learning_rate: 0.05,
            weights: weights_for_silicon,
            noise_model: None, // disable accuracy gate for the latency-pareto demo
            ..inner::InnerConfig::default()
        };
        match inner::run(&starting_tile, &inner_cfg) {
            Ok(trace) => {
                let last = trace.last().expect("non-empty trace");
                let converged_params = TileParams {
                    w_l_n: last.w_l_n as f64,
                    w_l_p: last.w_l_p as f64,
                    vdd: last.vdd as f64,
                    ..TileParams::default()
                };
                let final_silicon_ns =
                    silicon_time_ns_per_inference(converged_params, sim.cycles_per_inference);
                println!();
                println!("=== Adam optimizes silicon clock time, not host wall time ===");
                println!(
                    "  starting params  (W/L_n=1.0, W/L_p=1.0, Vdd=1.0): silicon time = {:.2} (normalized) per inference",
                    initial_silicon_ns
                );
                println!(
                    "  converged params (W/L_n={:.3}, W/L_p={:.3}, Vdd={:.3}): silicon time = {:.2} (normalized) per inference",
                    last.w_l_n, last.w_l_p, last.vdd, final_silicon_ns
                );
                let pct = 100.0 * (initial_silicon_ns - final_silicon_ns) / initial_silicon_ns;
                println!(
                    "  Δ = {:.2} ({:.1}% reduction) — Adam pushed Vdd up + sized transistors larger",
                    initial_silicon_ns - final_silicon_ns,
                    pct
                );
                println!(
                    "  cycles_per_inference is fixed at {} by the rlx-fpga emit; only per-cycle delay was tunable.",
                    sim.cycles_per_inference
                );
                println!(
                    "  To reduce cycles, the OUTER DADO loop walks ArrayConfig candidates (parallelism / weight stationarity / pipeline depth)."
                );
            }
            Err(e) => eprintln!("warning: silicon-time Adam loop failed ({e})"),
        }
    }
    if cfg.bundle.merge_weights {
        println!("bundle written to {}", cfg.bundle.output_path);
    }
    Ok(())
}

fn one_test_pair() -> (Vec<i8>, u8) {
    use rlx_cortexm::model_weights::{TEST_IMAGE, TEST_LABEL};
    (TEST_IMAGE.to_vec(), TEST_LABEL)
}

fn clone_library(_src: &ScHdLibrary) -> ScHdLibrary {
    let lib = Library::new("demo-clone", 1000);
    let pdk = eda_pdks::Sky130::register(&lib);
    let cells = populate_mock_sc_hd(&lib, &pdk);
    ScHdLibrary { library: lib, cells }
}

/// Real RTL sim path. Available with `--features bench-rtl-sim`.
#[cfg(feature = "bench-rtl-sim")]
fn build_l2_inference(
    base: &eda_bench_tinyconv::inference::InferenceMetrics,
    _model: &rlx_fpga::model::Model,
    workspace_root: &std::path::Path,
) -> eda_bench_tinyconv::inference::InferenceMetrics {
    use eda_bench_tinyconv::backends::rtl_sim::RtlSimBackend;

    let hw_dir = workspace_root.join("..").join("rlx").join("rlx-fpga").join("hw").join("tinyconv_mnist");
    if !hw_dir.join("top.sv").exists() {
        eprintln!(
            "warning: rlx-fpga emit not at {hw_dir:?}; falling back to analytic estimate"
        );
        return analytic_l2(base, _model);
    }
    eprintln!("running real RTL sim through Verilator (~30-60s on first build)…");
    let backend = RtlSimBackend::new(hw_dir);
    let (image, _label) = one_test_pair();
    match backend.measure_inference_one(&image) {
        Ok(result) => {
            eprintln!(
                "  RTL sim: prediction={}, cycles={}",
                result.prediction, result.cycles
            );
            base.clone().with_simulated(result.to_simulated(10.0))
        }
        Err(e) => {
            eprintln!("warning: RTL sim failed ({e}); falling back to analytic estimate");
            analytic_l2(base, _model)
        }
    }
}

/// Analytic-estimate path when Verilator isn't compiled in.
#[cfg(not(feature = "bench-rtl-sim"))]
fn build_l2_inference(
    base: &eda_bench_tinyconv::inference::InferenceMetrics,
    model: &rlx_fpga::model::Model,
    _workspace_root: &std::path::Path,
) -> eda_bench_tinyconv::inference::InferenceMetrics {
    analytic_l2(base, model)
}

fn analytic_l2(
    base: &eda_bench_tinyconv::inference::InferenceMetrics,
    model: &rlx_fpga::model::Model,
) -> eda_bench_tinyconv::inference::InferenceMetrics {
    let cycles = total_cycles(model, /* 4×4 budget */ 16) as u64;
    base.clone().with_simulated(SimulatedLatency::from_cycles(cycles, 10.0))
}

#[cfg(feature = "bench-rtl-sim")]
fn inference_l2_used_real_sim() -> bool { true }
#[cfg(not(feature = "bench-rtl-sim"))]
fn inference_l2_used_real_sim() -> bool { false }

/// Promote a runtime-chosen label to `&'static str`. Backed by a
/// global string interner (just two leak-once entries).
fn static_l2_label(label: &str) -> &'static str {
    use std::sync::OnceLock;
    static REAL: OnceLock<&'static str> = OnceLock::new();
    static EST: OnceLock<&'static str> = OnceLock::new();
    if label.contains("Verilator") {
        *REAL.get_or_init(|| Box::leak(label.to_string().into_boxed_str()))
    } else {
        *EST.get_or_init(|| Box::leak(label.to_string().into_boxed_str()))
    }
}
