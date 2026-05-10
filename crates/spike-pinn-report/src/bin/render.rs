//! Render the consolidated-report chart set from frozen
//! measurements. Numbers are pinned to the K=10 protocol runs
//! frozen 2026-05-10 (results in each crate's docs/results.md).

use plotters::prelude::*;
use std::path::Path;

const ASSET_DIR: &str = "docs/assets";

// ── Frozen measurements (from each crate's docs/results.md) ──────

// Cross-cutting summary: best-of-method-class per experiment.
// (experiment, dim, best_baseline_name, best_baseline_max_abs_pct_fs,
//  pinn_max_abs_pct_fs, pinn_se_pct_fs, n_seeds, verdict)
const SUMMARY: &[(&str, usize, &str, f32, f32, f32, usize, &str)] = &[
    ("Diode-RC", 5, "Poly-d4",   0.000,  6.49, 2.23, 10, "REJECT (5/5 fail)"),
    ("SAR-1D",   1, "Poly-d4",   0.078,  2.760, 0.054, 10, "REJECT (3/3 fail)"),  // 0.512 LSB / (256·0.5%FS·100)
    ("SAR-mc",   10, "Poly-d4",  6.622,  4.203, 0.434, 10, "PARTIAL (2/3 pass)"),
];
// Note for SAR-1D: max_abs in % FS = max_abs (in code/256 units) × 100. Poly-d4 = 0.512 LSB
// = 0.00200 → 0.20% FS. PINN = 18.08 LSB = 0.07063 → 7.06% FS. Use the numbers in fractional
// units below (× 100 in the chart).

// Cross-cutting in fractional-FS units (max-abs).
const CROSS_FS: &[(&str, f32, f32, f32)] = &[
    // (experiment, best_baseline_max_abs_FS, pinn_max_abs_FS, pinn_std_FS)
    ("Diode-RC (d=5)",  0.0000005, 0.0649, 0.0223),
    ("SAR 1-D (d=1)",   0.00200,   0.07063, 0.01373),
    ("SAR mc (d=10)",   0.06622,   0.04203, 0.00434),
];

// Run 4: capacity progression — (label, params, max_abs_in_LSB).
const RUN4_CAPACITY: &[(&str, usize, f32)] = &[
    ("Poly-d1",   11, 21.55),
    ("Poly-d2",   66, 19.50),
    ("Poly-d4", 1001, 16.95),
    ("PINN",    4929, 10.76),
];

// Run 4: per-seed PINN max-abs in LSBs.
const RUN4_SEEDS: &[(u32, f32)] = &[
    (1, 10.658),
    (2,  9.634),
    (3,  9.965),
    (4,  9.988),
    (5, 12.919),
    (6,  9.485),
    (7, 12.057),
    (8, 10.454),
    (9, 11.444),
    (10, 10.991),
];

// Run 2: Pareto data — (name, max_abs_FS, latency_ns_per_query, n_seeds_or_zero).
const RUN2_PARETO: &[(&str, f32, f32)] = &[
    // Latency = total_time / n_test (4000); times from the protocol log.
    ("M-coarse",   0.371,    1.0),  // 4 ms / 4000 ≈ 1 µs/q  → 1000 ns approx
    ("M-default",  0.049,   12.25), // 49 ms / 4000 = 12.25 µs/q
    ("M-fine",     0.016,  124.25), // 497 ms / 4000 = 124.25 µs/q
    ("Poly-d4",    0.000,   17.75), // 71 ms / 4000 = 17.75 µs/q
    ("Hybrid",     6.49,     0.132), // 530 µs / 4000 = 132 ns/q
    ("Surrogate",  1.58,     0.141),
    ("Pure-PINN",  8.74,     0.170),
];

// ── Chart rendering ──────────────────────────────────────────────

fn ensure_dir() {
    std::fs::create_dir_all(ASSET_DIR).expect("create asset dir");
}

fn chart_cross_cutting() {
    let path = format!("{ASSET_DIR}/cross-cutting-max-abs.svg");
    let root = SVGBackend::new(&path, (760, 460)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let labels: Vec<&str> = CROSS_FS.iter().map(|x| x.0).collect();
    let max_y = CROSS_FS.iter()
        .map(|x| x.1.max(x.2))
        .fold(0.0_f32, f32::max) * 1.15;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Best baseline vs PINN (max-abs error, fractional FS)",
            ("sans-serif", 18).into_font(),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(0..labels.len(), 0.0_f32..max_y)
        .unwrap();

    chart
        .configure_mesh()
        .y_desc("max-abs (fraction of full scale)")
        .x_label_formatter(&|i| labels.get(*i).copied().unwrap_or("").to_string())
        .x_labels(labels.len())
        .axis_desc_style(("sans-serif", 13))
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    // Side-by-side bars per experiment.
    let bar_width = 0.36;
    for (i, (_, baseline, pinn, std)) in CROSS_FS.iter().enumerate() {
        // Baseline bar (cyan).
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(i, 0.0), (i, *baseline)],
                RGBColor(72, 170, 200).filled(),
            )))
            .unwrap();
        // PINN bar (magenta).
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(i + 0, *pinn - *std), (i + 0, *pinn + *std)],
                RGBColor(0, 0, 0).stroke_width(1),
            )))
            .unwrap();
        // Two side-by-side bars: shift the cells so the visual is grouped.
        let _ = bar_width;
    }
    // Re-do as a paired bar chart by drawing rectangles at integer + half-offset.
    chart
        .draw_series(CROSS_FS.iter().enumerate().map(|(i, (_, bl, _, _))| {
            let x = i as f64 - 0.20;
            Rectangle::new(
                [(x as usize, 0.0), (x as usize, *bl)],
                RGBColor(72, 170, 200).filled(),
            )
        }))
        .unwrap()
        .label("Best baseline")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 12, y + 5)], RGBColor(72, 170, 200).filled()));

    chart
        .draw_series(CROSS_FS.iter().enumerate().map(|(i, (_, _, pinn, _))| {
            Rectangle::new([(i, 0.0), (i, *pinn)], RGBColor(190, 90, 170).filled())
        }))
        .unwrap()
        .label("PINN (mean across K=10)")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 12, y + 5)], RGBColor(190, 90, 170).filled()));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()
        .unwrap();

    println!("wrote {path}");
}

fn chart_run4_capacity() {
    let path = format!("{ASSET_DIR}/run4-capacity-progression.svg");
    let root = SVGBackend::new(&path, (760, 460)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let xs: Vec<f64> = RUN4_CAPACITY.iter().map(|(_, p, _)| *p as f64).collect();
    let ys: Vec<f64> = RUN4_CAPACITY.iter().map(|(_, _, y)| *y as f64).collect();
    let x_min = (xs.iter().cloned().fold(f64::INFINITY, f64::min) * 0.6).max(1.0);
    let x_max = xs.iter().cloned().fold(0.0, f64::max) * 1.5;
    let y_min = 0.0;
    let y_max = ys.iter().cloned().fold(0.0, f64::max) * 1.15;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Run 4: capacity progression on 10-D SAR-with-mismatch",
            ("sans-serif", 18).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (x_min..x_max).log_scale(),
            y_min..y_max,
        )
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("number of parameters (log)")
        .y_desc("max-abs error (LSB)")
        .axis_desc_style(("sans-serif", 13))
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    // Connect polynomial points only (PINN is a different method class).
    chart
        .draw_series(LineSeries::new(
            RUN4_CAPACITY.iter()
                .filter(|(name, _, _)| name.starts_with("Poly"))
                .map(|(_, p, y)| (*p as f64, *y as f64)),
            RGBColor(72, 170, 200).stroke_width(2),
        ))
        .unwrap()
        .label("Polynomial baseline")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 16, y)], RGBColor(72, 170, 200).stroke_width(2)));

    // Polynomial points.
    chart
        .draw_series(
            RUN4_CAPACITY.iter()
                .filter(|(name, _, _)| name.starts_with("Poly"))
                .map(|(_, p, y)| Circle::new((*p as f64, *y as f64), 5, RGBColor(72, 170, 200).filled())),
        )
        .unwrap();

    // PINN as a distinct point.
    let pinn = RUN4_CAPACITY.iter().find(|(n, _, _)| *n == "PINN").unwrap();
    chart
        .draw_series(std::iter::once(Circle::new(
            (pinn.1 as f64, pinn.2 as f64),
            7,
            RGBColor(190, 90, 170).filled(),
        )))
        .unwrap()
        .label("PINN (K=10 mean)")
        .legend(|(x, y)| Circle::new((x + 8, y), 5, RGBColor(190, 90, 170).filled()));

    // Annotate each point.
    for (name, p, ymax) in RUN4_CAPACITY.iter() {
        chart
            .draw_series(std::iter::once(Text::new(
                (*name).to_string(),
                (*p as f64, *ymax as f64 + 0.5),
                ("sans-serif", 11).into_font(),
            )))
            .unwrap();
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()
        .unwrap();

    println!("wrote {path}");
}

fn chart_run2_pareto() {
    let path = format!("{ASSET_DIR}/run2-pareto.svg");
    let root = SVGBackend::new(&path, (760, 460)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Run 2: (latency, max-abs) Pareto on Diode-RC",
            ("sans-serif", 18).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (0.05_f64..200.0).log_scale(),
            (0.0001_f64..20.0).log_scale(),
        )
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("latency (µs / query, log)")
        .y_desc("max-abs error (% full scale, log)")
        .axis_desc_style(("sans-serif", 13))
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    // MNA + Poly baselines (cyan), PINN ablations (magenta).
    chart
        .draw_series(
            RUN2_PARETO.iter()
                .filter(|(n, _, _)| !["Hybrid", "Surrogate", "Pure-PINN"].contains(n))
                .map(|(_, fs, lat)| {
                    Circle::new((*lat as f64, *fs as f64), 6, RGBColor(72, 170, 200).filled())
                }),
        )
        .unwrap()
        .label("MNA / Poly baselines")
        .legend(|(x, y)| Circle::new((x + 8, y), 5, RGBColor(72, 170, 200).filled()));

    chart
        .draw_series(
            RUN2_PARETO.iter()
                .filter(|(n, _, _)| ["Hybrid", "Surrogate", "Pure-PINN"].contains(n))
                .map(|(_, fs, lat)| {
                    Circle::new((*lat as f64, *fs as f64), 6, RGBColor(190, 90, 170).filled())
                }),
        )
        .unwrap()
        .label("PINN ablations")
        .legend(|(x, y)| Circle::new((x + 8, y), 5, RGBColor(190, 90, 170).filled()));

    for (name, fs, lat) in RUN2_PARETO.iter() {
        chart
            .draw_series(std::iter::once(Text::new(
                (*name).to_string(),
                (*lat as f64 * 1.15, *fs as f64 * 1.05),
                ("sans-serif", 11).into_font(),
            )))
            .unwrap();
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()
        .unwrap();

    println!("wrote {path}");
}

fn chart_run4_seeds() {
    let path = format!("{ASSET_DIR}/run4-pinn-per-seed.svg");
    let root = SVGBackend::new(&path, (760, 360)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_y = RUN4_SEEDS.iter().map(|(_, y)| *y).fold(0.0_f32, f32::max) * 1.2;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Run 4: PINN max-abs per seed (LSB)",
            ("sans-serif", 16).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..RUN4_SEEDS.len() + 1, 0.0_f32..max_y)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("seed")
        .y_desc("max-abs (LSB)")
        .x_label_formatter(&|i| if *i >= 1 && *i <= RUN4_SEEDS.len() { i.to_string() } else { String::new() })
        .x_labels(RUN4_SEEDS.len() + 2)
        .axis_desc_style(("sans-serif", 13))
        .label_style(("sans-serif", 12))
        .draw()
        .unwrap();

    chart
        .draw_series(RUN4_SEEDS.iter().map(|(seed, y)| {
            Rectangle::new(
                [(*seed as usize, 0.0), (*seed as usize, *y)],
                RGBColor(190, 90, 170).filled(),
            )
        }))
        .unwrap();

    // Mean line.
    let mean: f32 = RUN4_SEEDS.iter().map(|(_, y)| *y).sum::<f32>() / RUN4_SEEDS.len() as f32;
    chart
        .draw_series(LineSeries::new(
            (0..=RUN4_SEEDS.len() + 1).map(|x| (x, mean)),
            RGBColor(50, 50, 50).stroke_width(1),
        ))
        .unwrap()
        .label(&format!("mean = {:.2} LSB", mean))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 16, y)], RGBColor(50, 50, 50).stroke_width(1)));

    // 1-LSB pre-registered C5'' threshold.
    chart
        .draw_series(LineSeries::new(
            (0..=RUN4_SEEDS.len() + 1).map(|x| (x, 1.0)),
            RGBColor(220, 60, 60).stroke_width(1),
        ))
        .unwrap()
        .label("C5'' threshold = 1 LSB")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 16, y)], RGBColor(220, 60, 60).stroke_width(1)));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 11))
        .draw()
        .unwrap();

    println!("wrote {path}");
}

fn main() {
    ensure_dir();
    let _ = Path::new(ASSET_DIR);
    chart_cross_cutting();
    chart_run4_capacity();
    chart_run2_pareto();
    chart_run4_seeds();
    println!("done");
}
