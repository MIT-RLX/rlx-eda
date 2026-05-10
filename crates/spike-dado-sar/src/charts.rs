//! Trajectory plots for DADO vs EDA. Mirrors the chart helpers from
//! `spike-dado-r2r/src/charts.rs` (1200×720 PNGs via `eda_waveform::plot`)
//! but adapted to the SAR experiment's smaller seed budget.

use std::collections::BTreeMap;
use std::path::Path;

use eda_waveform::plot::{png_to_path, Marker, PlotConfig, PlotError};
use eda_waveform::Waveform;

use crate::RunTrace;

const W: u32 = 1200;
const H: u32 = 720;

fn mean_trajectory<F>(traces: &[RunTrace], pick: F) -> Vec<f64>
where F: Fn(&RunTrace) -> &Vec<f64> {
    let n = pick(&traces[0]).len();
    let mut out = vec![0.0_f64; n];
    for tr in traces {
        for (i, &x) in pick(tr).iter().enumerate() { out[i] += x; }
    }
    let n_t = traces.len() as f64;
    for x in out.iter_mut() { *x /= n_t; }
    out
}

/// DADO vs EDA, best+mean per iter (mean across seeds), with optional
/// `optimum` reference line.
pub fn write_trajectory_png(
    title: &str,
    dado: &[RunTrace],
    eda: &[RunTrace],
    optimum: Option<f64>,
    path: impl AsRef<Path>,
) -> Result<(), PlotError> {
    let n = dado[0].best.len();
    let axis: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let mut signals = BTreeMap::new();
    signals.insert("DADO best".to_string(), mean_trajectory(dado, |t| &t.best));
    signals.insert("DADO mean".to_string(), mean_trajectory(dado, |t| &t.mean));
    signals.insert("EDA best".to_string(),  mean_trajectory(eda,  |t| &t.best));
    signals.insert("EDA mean".to_string(),  mean_trajectory(eda,  |t| &t.mean));

    let wave = Waveform::Real { axis_name: "iter".into(), axis, signals };
    let mut cfg = PlotConfig::new().with_title(title).with_size(W, H);
    if let Some(opt) = optimum {
        cfg = cfg.add_marker(Marker::Horizontal {
            y: opt, label: Some("optimum".into()),
        });
    }
    png_to_path(&wave, path, &cfg)
}
