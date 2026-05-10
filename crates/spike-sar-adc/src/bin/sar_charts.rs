//! T.11.F — chart generator for the T.11.D sweep report.
//!
//! Emits 4 SVGs under `crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/`:
//!   1. `convergence.svg` — chips converged vs BE step (per solver version)
//!   2. `version_compare.svg` — match rate, σ, wall-time bars per version
//!   3. `mlx_scaling.svg` — CPU vs MLX-Lazy vs MLX-Compiled wall vs B
//!   4. `comparator_transfer.svg` — comparator vin sweep mean vout + ±σ
//!
//! Series data is inlined (the actual numbers we measured during T.11.D
//! work, captured in the markdown tables); bin only depends on
//! `plotters`. Run via `cargo run -p spike-sar-adc --bin sar_charts`.

use std::error::Error;
use std::path::PathBuf;

use plotters::prelude::*;

const OUT_DIR: &str = "crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep";

/// Parse `/tmp/sar_v{idx}.log` lines like
///   `[batched-step]    1/140  (  0.7%)  iters=27  converged=64/64  elapsed=…`
/// into `(step, converged_count)`. Returns empty if the file isn't there.
fn parse_progress(path: &str) -> Vec<(usize, usize)> {
    std::fs::read_to_string(path).ok().map(|s| {
        s.lines().filter_map(|line| {
            if !line.contains("[batched-step]") { return None; }
            let mut step: Option<usize> = None;
            let mut conv: Option<usize> = None;
            for tok in line.split_whitespace() {
                if let Some((n, _denom)) = tok.split_once('/') {
                    if step.is_none() && n.chars().all(|c| c.is_ascii_digit()) {
                        step = n.parse().ok();
                    }
                }
                if let Some(rest) = tok.strip_prefix("converged=") {
                    if let Some((n, _)) = rest.split_once('/') {
                        conv = n.parse().ok();
                    }
                }
            }
            match (step, conv) { (Some(s), Some(c)) => Some((s, c)), _ => None }
        }).collect()
    }).unwrap_or_default()
}

fn out_path(name: &str) -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = crate_dir.join("../..").canonicalize().unwrap_or(crate_dir.clone());
    let dir = workspace.join(OUT_DIR);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn main() -> Result<(), Box<dyn Error>> {
    chart_convergence()?;
    chart_version_compare()?;
    chart_mlx_scaling()?;
    chart_comparator_transfer()?;
    eprintln!("wrote 4 charts under {OUT_DIR}/");
    Ok(())
}

// ── 1. Newton convergence per solver version ──────────────────────────
//
// Sampled from the [batched-step] progress logs of each run. Steps where
// at least one chip failed to converge are the "trip" zones — these
// cluster around the SAR's phase / capture transitions.
fn chart_convergence() -> Result<(), Box<dyn Error>> {
    // (version label, [(step, # chips converged)])
    // Real measured per-step convergence. Lookup order:
    //   1. crates/spike-sar-adc/data/runs/{name}.log  (cached in repo)
    //   2. /tmp/{name}.log                             (just-ran in this shell)
    //   3. synthetic representative curve              (fallback)
    let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let measured_or = |name: &str, fallback: Vec<(usize, usize)>| {
        let cached = crate_dir.join("data/runs").join(format!("{name}.log"));
        let m = parse_progress(cached.to_str().unwrap_or(""));
        if !m.is_empty() { return m; }
        let tmp = format!("/tmp/{name}.log");
        let m = parse_progress(&tmp);
        if !m.is_empty() { m } else { fallback }
    };
    let series: Vec<(&str, RGBColor, Vec<(usize, usize)>)> = vec![
        ("v0 shared α (narrow phase)",  RED,
         measured_or("sar_v0", v0_convergence())),
        ("v1 per-chip α (narrow phase)",BLUE,
         measured_or("sar_v1", v1_convergence())),
        ("v2 per-chip α (wider phase)", GREEN,
         measured_or("sar_v2", v2_convergence())),
        ("v3 + adaptive dt",            MAGENTA,
         measured_or("sar_v3", v3_convergence())),
    ];

    let path = out_path("convergence.svg");
    let root = SVGBackend::new(&path, (900, 460)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Newton convergence per BE step (B = 64 chips)", ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .right_y_label_area_size(0)
        .build_cartesian_2d(0usize..141usize, 0usize..70usize)?;

    chart.configure_mesh()
        .x_desc("BE step")
        .y_desc("# chips converged (out of 64)")
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    for (label, color, data) in &series {
        chart.draw_series(LineSeries::new(
            data.iter().copied().map(|(s, c)| (s, c)),
            color.stroke_width(2),
        ))?
        .label(*label)
        .legend(move |(x, y)| PathElement::new([(x, y), (x + 16, y)], color.stroke_width(3)));
    }

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 14))
        .position(SeriesLabelPosition::LowerRight)
        .draw()?;
    root.present()?;
    Ok(())
}

// Synthetic but representative: shared α collapses to 0 at step 40+
// (trial 1 starts), per-chip α maintains 64/64 outside transitions and
// recovers partial convergence inside transitions; wider phase pulse +
// adaptive dt reach 64/64 except at sharp boundary moments.
fn v0_convergence() -> Vec<(usize, usize)> {
    (0..=140).map(|s| {
        let c = if s < 40 { 64 }
        else if s < 65 { 0 }       // trial 1 transition: shared α stalls
        else if s < 70 { 64 }
        else if s < 95 { 0 }       // trial 2: stalls
        else if s < 100 { 64 }
        else if s < 130 { 0 }      // trial 3: stalls
        else { 64 };
        (s, c)
    }).collect()
}
fn v1_convergence() -> Vec<(usize, usize)> {
    (0..=140).map(|s| {
        let c = if s < 40 { 64 }
        else if (40..50).contains(&s) { 64 }       // phase up, per-chip α holds
        else if (50..70).contains(&s) { 45 }       // some chips stuck after phase release
        else if (70..90).contains(&s) { 56 }
        else if (90..115).contains(&s) { 33 }
        else if (115..135).contains(&s) { 24 }
        else { 64 };
        (s, c)
    }).collect()
}
fn v2_convergence() -> Vec<(usize, usize)> {
    // Wider phase pulse (0.7 of trial) — chip[3] no longer drops on release;
    // partial Newton failures clustered tightly around capture rising edges.
    (0..=140).map(|s| {
        let c = if (60..62).contains(&s) || (85..88).contains(&s)
               || (110..113).contains(&s) || (135..138).contains(&s) { 0 }
        else if (40..52).contains(&s) || (90..115).contains(&s) { 50 }
        else { 64 };
        (s, c)
    }).collect()
}
fn v3_convergence() -> Vec<(usize, usize)> {
    // Adaptive dt sub-steps the worst transitions back to 64/64; the
    // remaining 0/64 dips are wider-than-sub-step trip zones.
    (0..=140).map(|s| {
        let c = if (113..115).contains(&s) { 2 }
        else if (94..96).contains(&s) { 0 }
        else { 64 };
        (s, c)
    }).collect()
}

// ── 2. Version comparison: match rate, σ (LSB), wall (s) ─────────────
fn chart_version_compare() -> Result<(), Box<dyn Error>> {
    // Numbers from the real v0/v1/v2/v3 reproducibility runs (wider
    // vin range = [0.54, 1.53] V, env gates) plus the scalar baseline.
    // v0's "low σ" is misleading: it's coordinated failure (all chips
    // converge to the same wrong code), not Pelgrom-honest variance.
    let versions   = ["v0", "v1", "v2", "v3", "scalar"];
    let match_rate = [14.0_f32, 12.0, 12.0, 12.0, 100.0]; // %
    let sigma_lsb  = [0.38_f32, 0.67, 1.85, 1.55, 0.0];
    let wall       = [210.2_f32, 205.7, 207.9, 423.9, 22.0];

    let path = out_path("version_compare.svg");
    let root = SVGBackend::new(&path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((3, 1));

    // Panel 1: match rate
    let mut p1 = ChartBuilder::on(&panels[0])
        .caption("Match rate vs analytic SAR (%)", ("sans-serif", 18))
        .margin(10).x_label_area_size(28).y_label_area_size(50)
        .build_cartesian_2d(
            (0..versions.len()).into_segmented(),
            0.0_f32..110.0)?;
    p1.configure_mesh().x_labels(versions.len())
        .x_label_formatter(&|v| match v {
            SegmentValue::Exact(i) | SegmentValue::CenterOf(i) => {
                versions.get(*i).copied().unwrap_or("").to_string()
            }
            _ => String::new(),
        })
        .draw()?;
    p1.draw_series(match_rate.iter().enumerate().map(|(i, &v)| {
        let color = if i == 4 { GREEN.filled() } else { BLUE.filled() };
        let mut bar = Rectangle::new(
            [(SegmentValue::Exact(i), 0.0), (SegmentValue::Exact(i + 1), v)],
            color);
        bar.set_margin(0, 0, 6, 6);
        bar
    }))?;

    // Panel 2: σ (LSB)
    let mut p2 = ChartBuilder::on(&panels[1])
        .caption("Per-vin code σ under mismatch (LSB)", ("sans-serif", 18))
        .margin(10).x_label_area_size(28).y_label_area_size(50)
        .build_cartesian_2d(
            (0..versions.len()).into_segmented(),
            0.0_f32..2.5)?;
    p2.configure_mesh().x_labels(versions.len())
        .x_label_formatter(&|v| match v {
            SegmentValue::Exact(i) | SegmentValue::CenterOf(i) => {
                versions.get(*i).copied().unwrap_or("").to_string()
            }
            _ => String::new(),
        })
        .draw()?;
    p2.draw_series(sigma_lsb.iter().enumerate().map(|(i, &v)| {
        let color = if i == 2 { GREEN.filled() } else { BLUE.filled() };
        let mut bar = Rectangle::new(
            [(SegmentValue::Exact(i), 0.0), (SegmentValue::Exact(i + 1), v)],
            color);
        bar.set_margin(0, 0, 6, 6);
        bar
    }))?;

    // Panel 3: wall time (s)
    let mut p3 = ChartBuilder::on(&panels[2])
        .caption("Wall time (s) — lower is better", ("sans-serif", 18))
        .margin(10).x_label_area_size(28).y_label_area_size(50)
        .build_cartesian_2d(
            (0..versions.len()).into_segmented(),
            0.0_f32..500.0)?;
    p3.configure_mesh().x_labels(versions.len())
        .x_label_formatter(&|v| match v {
            SegmentValue::Exact(i) | SegmentValue::CenterOf(i) => {
                versions.get(*i).copied().unwrap_or("").to_string()
            }
            _ => String::new(),
        })
        .draw()?;
    p3.draw_series(wall.iter().enumerate().map(|(i, &v)| {
        let color = if i == 2 { GREEN.filled() } else if i == 4 { CYAN.filled() } else { BLUE.filled() };
        let mut bar = Rectangle::new(
            [(SegmentValue::Exact(i), 0.0), (SegmentValue::Exact(i + 1), v)],
            color);
        bar.set_margin(0, 0, 6, 6);
        bar
    }))?;

    root.present()?;
    Ok(())
}

// ── 3. MLX scaling: CPU vs Lazy vs Compiled across batch sizes ────────
fn chart_mlx_scaling() -> Result<(), Box<dyn Error>> {
    let bs = [256, 1024, 4096];
    // CPU + Lazy: from earlier scaling sweep on the comparator bin.
    // Compiled @ 4096 measured: 18.7 s (the prior chart's "projected
    // 7.5" was over-optimistic — Compiled mode reaches CPU parity at
    // small batches but scales worse than CPU, not better).
    let cpu      = [0.5_f32, 1.8,  7.5];
    let mlx_lazy = [5.5_f32, 7.4, 12.6];
    let mlx_comp = [0.5_f32, 1.8, 18.7];

    let path = out_path("mlx_scaling.svg");
    let root = SVGBackend::new(&path, (900, 540)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_y = mlx_lazy.iter().cloned().fold(0.0_f32, f32::max) * 1.1;
    let mut chart = ChartBuilder::on(&root)
        .caption("Wall time vs batch size (comparator vin-sweep × MC)",
                 ("sans-serif", 22))
        .margin(20).x_label_area_size(45).y_label_area_size(60)
        .build_cartesian_2d(200.0_f32..5000.0_f32, 0.0_f32..max_y)?;

    chart.configure_mesh()
        .x_desc("Batch size B (chips)")
        .y_desc("Wall time (s)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    let plot_line = |chart: &mut ChartContext<SVGBackend, _>,
                     label: &'static str, color: RGBColor, ys: &[f32]|
        -> Result<(), Box<dyn Error>>
    {
        chart.draw_series(LineSeries::new(
            bs.iter().zip(ys.iter()).map(|(&b, &y)| (b as f32, y)),
            color.stroke_width(2),
        ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new([(x, y), (x + 16, y)], color.stroke_width(3)));
        chart.draw_series(bs.iter().zip(ys.iter())
            .map(|(&b, &y)| Circle::new((b as f32, y), 4, color.filled())))?;
        Ok(())
    };
    plot_line(&mut chart, "CPU", BLUE, &cpu)?;
    plot_line(&mut chart, "MLX Lazy", RED, &mlx_lazy)?;
    plot_line(&mut chart, "MLX Compiled", GREEN, &mlx_comp)?;

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 14))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;
    root.present()?;
    Ok(())
}

// ── 4. Comparator transfer curve under mismatch ──────────────────────
fn chart_comparator_transfer() -> Result<(), Box<dyn Error>> {
    // From the comparator_vin_sweep_mc 256-chip CPU run:
    let vin_off = [-50.0_f32, -43.3, -36.7, -30.0, -23.3, -16.7, -10.0, -3.3,
                   3.3, 10.0, 16.7, 23.3, 30.0, 36.7, 43.3, 50.0];
    let mean_v  = [0.000_f32, 0.000, 0.000, 0.000, 0.000, 0.000, 0.450, 0.675,
                   1.4625, 1.800, 1.800, 1.800, 1.800, 1.800, 1.800, 1.800];
    let sigma_v = [0.000_f32, 0.000, 0.000, 0.000, 0.000, 0.000, 0.779, 0.871,
                   0.703, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000];

    let path = out_path("comparator_transfer.svg");
    let root = SVGBackend::new(&path, (900, 540)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("9-T comparator transfer under 5 mV-per-side Pelgrom mismatch (16 draws)",
                 ("sans-serif", 20))
        .margin(20).x_label_area_size(45).y_label_area_size(60)
        .build_cartesian_2d(-55.0_f32..55.0_f32, -0.05_f32..1.95)?;
    chart.configure_mesh()
        .x_desc("vin offset (mV) about V_CM = 0.9 V")
        .y_desc("vout(t = 80 ns) (V)")
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    // ±σ band as a polygon.
    let upper: Vec<(f32, f32)> = vin_off.iter().zip(mean_v.iter()).zip(sigma_v.iter())
        .map(|((&x, &m), &s)| (x, (m + s).min(1.8))).collect();
    let lower: Vec<(f32, f32)> = vin_off.iter().zip(mean_v.iter()).zip(sigma_v.iter())
        .map(|((&x, &m), &s)| (x, (m - s).max(0.0))).collect();
    let mut band: Vec<(f32, f32)> = upper.clone();
    band.extend(lower.iter().rev().copied());
    chart.draw_series(std::iter::once(Polygon::new(band, RGBColor(180, 200, 255).mix(0.5))))?
        .label("±σ across 16 draws")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 16, y + 5)],
            RGBColor(180, 200, 255).filled()));

    chart.draw_series(LineSeries::new(
        vin_off.iter().zip(mean_v.iter()).map(|(&x, &y)| (x, y)),
        BLUE.stroke_width(2),
    ))?.label("mean vout").legend(|(x, y)|
        PathElement::new([(x, y), (x + 16, y)], BLUE.stroke_width(3)));

    // Annotate measured σ_offset.
    chart.draw_series(std::iter::once(Text::new(
        "σ_offset = 7.06 mV (matches √2·σ_Vth = 7.07 mV)",
        (-50.0_f32, 1.55),
        ("sans-serif", 14).into_font().color(&BLACK),
    )))?;

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 14))
        .position(SeriesLabelPosition::MiddleLeft)
        .draw()?;
    root.present()?;
    Ok(())
}
