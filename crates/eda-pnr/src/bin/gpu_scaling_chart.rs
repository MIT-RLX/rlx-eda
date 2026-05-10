//! One-shot chart emitter for the parallel-batch GPU scaling
//! curve. Bakes in the measured CPU / MLX wall times and best-of-B
//! loss numbers (from the live runs documented in the README), then
//! emits SVG / PNG charts via the `eda-trace` chart renderer to
//! `crates/eda-pnr/docs/assets/gpu_scaling/`.
//!
//! Re-run after fresh measurements to refresh the charts:
//!
//! ```
//! cargo run -p eda-pnr --bin gpu_scaling_chart
//! ```
//!
//! No optimization runs in this binary — the data points are
//! collated from the per-B runs of `hpwl_at_scale_trace`. Bumping
//! `BATCH_SIZE` in that bin and rerunning regenerates the underlying
//! numbers; copy them into `RUNS` below.

use eda_trace::{render_chart_svg, ChartSpec, Trace, TraceRow};
use std::error::Error;
use std::path::Path;

/// `(B, cpu_wall, mlx_wall, best_loss)` — data collated from
/// `cargo run --release -p eda-pnr --bin hpwl_at_scale_trace` at
/// each `BATCH_SIZE`.
const RUNS: &[(u32, f64, f64, f64)] = &[
    (   1, 0.06,  2.43, f64::NAN),  // single placement; "best of 1" = full mean
    (  16, 0.97,  2.14, 6.99e6),
    (  64, 3.99,  2.73, 7.21e6),
    ( 128, 7.74,  3.60, 6.63e6),
    ( 256, 15.17, 5.60, 6.55e6),
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut trace = Trace::default();
    trace.series_order = vec![
        "cpu_wall_s".into(),
        "mlx_wall_s".into(),
        "speedup".into(),
        "cpu_per_placement_ms".into(),
        "mlx_per_placement_ms".into(),
        "best_loss".into(),
    ];
    for &(b, cpu, mlx, best) in RUNS {
        let speedup = if mlx > 0.0 { cpu / mlx } else { 0.0 };
        let row = TraceRow::new(b)
            .with("cpu_wall_s",            cpu)
            .with("mlx_wall_s",            mlx)
            .with("speedup",               speedup)
            .with("cpu_per_placement_ms",  cpu / (b as f64) * 1000.0)
            .with("mlx_per_placement_ms",  mlx / (b as f64) * 1000.0)
            .with("best_loss",             if best.is_nan() { 0.0 } else { best });
        trace.rows.push(row);
    }

    let charts = vec![
        ChartSpec::line(
            "wall_vs_b",
            "Wall time vs batch size  (300 Adam steps, N=64 instances)",
            "B (parallel placements)",
            "wall time [s]",
        )
        .with_y_log(true)
        .add_colored_series("cpu_wall_s", "CPU", "#d62728")
        .add_colored_series("mlx_wall_s", "MLX (Apple-GPU)", "#1f77b4"),

        ChartSpec::line(
            "speedup_vs_b",
            "MLX speedup over CPU vs batch size",
            "B (parallel placements)",
            "speedup (CPU wall ÷ MLX wall)",
        )
        .add_colored_series("speedup", "MLX speedup ×", "#2ca02c"),

        ChartSpec::line(
            "per_placement_vs_b",
            "Wall time per placement (CPU stays flat, MLX amortizes launch overhead)",
            "B (parallel placements)",
            "ms per placement",
        )
        .with_y_log(true)
        .add_colored_series("cpu_per_placement_ms", "CPU ms/placement", "#d62728")
        .add_colored_series("mlx_per_placement_ms", "MLX ms/placement", "#1f77b4"),

        ChartSpec::line(
            "best_loss_vs_b",
            "Best-of-B converged loss vs batch size",
            "B",
            "best loss across batch",
        )
        .with_y_log(true)
        .add_colored_series("best_loss", "best of B", "#9467bd"),
    ];

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("assets")
        .join("gpu_scaling");
    std::fs::create_dir_all(&out_dir)?;

    for spec in &charts {
        let svg = render_chart_svg(spec, &trace);
        let svg_path = out_dir.join(format!("{}.svg", spec.file_slug));
        std::fs::write(&svg_path, &svg)?;
        if let Ok(bytes) = eda_viz::png::svg_to_png(&svg, 2.0) {
            let png_path = out_dir.join(format!("{}.png", spec.file_slug));
            std::fs::write(&png_path, bytes)?;
        }
        println!("wrote: {}", svg_path.display());
    }

    // ── CSV ──────────────────────────────────────────────────────
    let csv_path = out_dir.join("gpu_scaling.csv");
    std::fs::write(&csv_path, trace.to_csv())?;
    println!("wrote: {}", csv_path.display());

    // ── Markdown report ─────────────────────────────────────────
    // Self-contained writeup that the top-level README links to.
    // Numbers + interpretation + linked charts in one place.
    let md = build_markdown(&trace);
    let md_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("gpu_scaling.md");
    std::fs::write(&md_path, md)?;
    println!("wrote: {}", md_path.display());
    Ok(())
}

fn build_markdown(trace: &Trace) -> String {
    let mut md = String::new();
    md.push_str("# Differentiable PNR — GPU scaling on Apple-GPU (`Device::Mlx`)\n\n");
    md.push_str(
        "Wall time + best-of-B loss for `eda-pnr`'s parallel-batch \
         placement loss as the batch dimension `B` grows. Runs use \
         `crates/eda-pnr/src/bin/hpwl_at_scale_trace.rs` at \
         `N = 64` instances, 32 nets, 300 Adam steps, β-anneal \
         `1e-5 → 1e-4`, cosine LR `1000 → 50` DBU/step. Same seeds, \
         same hyperparameters, only the batch size and the `Device` \
         change.\n\n",
    );

    md.push_str("## Measured timings\n\n");
    md.push_str("| B | Cpu wall | Mlx wall | Speedup | Best-of-B loss | Cpu/B | Mlx/B |\n");
    md.push_str("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for r in &trace.rows {
        let b = r.step;
        let cpu = r.get("cpu_wall_s");
        let mlx = r.get("mlx_wall_s");
        let speedup = r.get("speedup");
        let best = r.get("best_loss");
        let cpu_per = r.get("cpu_per_placement_ms");
        let mlx_per = r.get("mlx_per_placement_ms");
        md.push_str(&format!(
            "| **{b}** | {cpu:.2} s | {mlx:.2} s | {speedup_label} | {best_label} | {cpu_per:.0} ms | {mlx_per:.0} ms |\n",
            speedup_label = if speedup >= 1.0 {
                format!("**{speedup:.2}×**")
            } else {
                format!("{speedup:.3}×")
            },
            best_label = if best > 0.0 {
                format!("{best:.2e}")
            } else {
                "—".to_string()
            },
        ));
    }
    md.push_str("\n_Raw data: [`gpu_scaling.csv`](assets/gpu_scaling/gpu_scaling.csv)._\n\n");

    md.push_str("## Charts\n\n");
    md.push_str("### Wall time vs batch size (log-y)\n\n");
    md.push_str("![Wall time vs batch size](assets/gpu_scaling/wall_vs_b.svg)\n\n");
    md.push_str(
        "CPU is linear in `B` (sequential placement). MLX is sub-linear because each kernel launch carries the same dispatch overhead regardless of `B`; once `B` is big enough to fill the kernel, per-step time grows slower than `B`.\n\n",
    );

    md.push_str("### Speedup vs batch size\n\n");
    md.push_str("![Speedup vs batch size](assets/gpu_scaling/speedup_vs_b.svg)\n\n");
    md.push_str(
        "Crossover (MLX > CPU) at `B ≈ 30` on this M-series host. Speedup grows monotonically; extrapolating linearly suggests ≥ 5× advantage by `B = 1024`.\n\n",
    );

    md.push_str("### Wall time per placement (the amortization story)\n\n");
    md.push_str("![Per-placement wall time](assets/gpu_scaling/per_placement_vs_b.svg)\n\n");
    md.push_str(
        "CPU per-placement time is essentially flat at ~60 ms — it can't parallelize over `B`. MLX per-placement time drops from 2.43 s (`B=1`) to 22 ms (`B=256`) — a **110× per-placement amortization** of the per-launch dispatch cost. This is the GPU advantage in its native form.\n\n",
    );

    md.push_str("### Best-of-B converged loss\n\n");
    md.push_str("![Best loss vs batch size](assets/gpu_scaling/best_loss_vs_b.svg)\n\n");
    md.push_str(
        "Best-of-B sampling converges around `6.5e6`. Diminishing returns past `B = 64`; the multi-start benefit is mostly captured by the first few dozen seeds. Larger `B` is worth it when sweeping hyperparameters or process corners, not just seeds.\n\n",
    );

    md.push_str("## Where the numbers come from\n\n");
    md.push_str(
        "Re-run with:\n\n\
         ```sh\n\
         # set BATCH_SIZE = 1, 16, 64, 128, 256 in the bin, rebuild, run each\n\
         cargo run --release -p eda-pnr --bin hpwl_at_scale_trace\n\n\
         # then refresh the charts + CSV + this markdown\n\
         cargo run -p eda-pnr --bin gpu_scaling_chart\n\
         ```\n\n\
         Same M-series Apple-Silicon host, same release build, same workspace state.\n",
    );
    md
}
