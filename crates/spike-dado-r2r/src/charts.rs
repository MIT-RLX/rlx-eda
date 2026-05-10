//! Plotting helpers — wrappers around `eda-waveform::plot` that turn
//! optimization traces, distribution snapshots, and per-design DAC
//! sweeps into PNG charts.
//!
//! Everything goes through `Waveform::Real` + `plot::png_to_path` so
//! the rendering style matches the rest of the workspace's gallery.

use std::collections::BTreeMap;
use std::path::Path;

use eda_waveform::plot::{png_to_path, Marker, PlotConfig, PlotError};
use eda_waveform::Waveform;
use spike_dac_r2r::ideal_vout;

use crate::{
    r_in_idx, r_sp_idx, r_term_idx, solve_r2r, Design, DistSnapshot, RunTrace, DEVIATIONS,
    N_BITS, N_CODES, N_NODES,
};

// Larger default chart size — the previous 900×540 looked compressed
// when markdown viewers fit images to column width.
const W: u32 = 1200;
const H: u32 = 720;

/// Mean trajectory across seeds for a single algorithm.
fn mean_trajectory<F>(traces: &[RunTrace], pick: F) -> Vec<f64>
where
    F: Fn(&RunTrace) -> &Vec<f64>,
{
    let n = pick(&traces[0]).len();
    let mut out = vec![0.0_f64; n];
    for tr in traces {
        let v = pick(tr);
        for (i, &x) in v.iter().enumerate() { out[i] += x; }
    }
    let n_t = traces.len() as f64;
    for x in out.iter_mut() { *x /= n_t; }
    out
}

/// Optimization-trajectory plot: best-so-far + mean of total score for
/// DADO and EDA on one objective. X axis is iteration. Snapshot iters
/// are drawn as vertical markers so readers can locate the per-iter
/// images relative to the curves.
pub fn write_trajectory_png(
    title: &str,
    dado: &[RunTrace],
    eda: &[RunTrace],
    snapshot_iters: &[usize],
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

    let wave = Waveform::Real {
        axis_name: "iter".into(),
        axis,
        signals,
    };
    let mut cfg = PlotConfig::new().with_title(title).with_size(W, H);
    for &it in snapshot_iters {
        cfg = cfg.add_marker(Marker::Vertical {
            x: it as f64,
            label: Some(format!("iter {it}")),
        });
    }
    if let Some(opt) = optimum {
        cfg = cfg.add_marker(Marker::Horizontal {
            y: opt,
            label: Some("optimum".into()),
        });
    }
    png_to_path(&wave, path, &cfg)
}

/// Resistor-name labels in the legend.
fn resistor_label(v: usize) -> String {
    if v == 0 { "r_term".into() }
    else if v <= N_BITS { format!("r_in[{}]", v - 1) }
    else { format!("r_sp[{}]", v - N_BITS - 1) }
}

/// Resistor groupings — used to split per-resistor plots into more
/// readable subsets. `feeders` covers the 8 input feeders + termination
/// (vertical resistors in the schematic); `spine` covers the 7 spine
/// resistors (horizontal in the schematic, the ones that form the
/// junction-tree separators).
fn feeder_vars() -> Vec<usize> {
    let mut v = Vec::with_capacity(N_BITS + 1);
    v.push(r_term_idx());
    for b in 0..N_BITS { v.push(r_in_idx(b)); }
    v
}
fn spine_vars() -> Vec<usize> {
    (0..(N_NODES - 1)).map(r_sp_idx).collect()
}

fn write_evolution_subset(
    title: &str,
    snapshots: &[DistSnapshot],
    vars: &[usize],
    path: impl AsRef<Path>,
) -> Result<(), PlotError> {
    let axis: Vec<f64> = snapshots.iter().map(|s| s.iter as f64).collect();
    let mut signals = BTreeMap::new();
    for &v in vars {
        let series: Vec<f64> = snapshots.iter().map(|s| s.expected_dev_idx[v]).collect();
        signals.insert(resistor_label(v), series);
    }
    let wave = Waveform::Real { axis_name: "iter".into(), axis, signals };
    let cfg = PlotConfig::new()
        .with_title(title)
        .with_size(W, H)
        .add_marker(Marker::Horizontal {
            y: ((DEVIATIONS.len() - 1) as f64) / 2.0,   // dev idx 2 = nominal
            label: Some("nominal".into()),
        });
    png_to_path(&wave, path, &cfg)
}

/// Per-resistor expected-deviation-index evolution across snapshots,
/// emitted as **two** PNGs at `path_root_{spine,feeder}.png` to keep
/// each plot to 7–9 lines instead of 16.
pub fn write_param_evolution_png(
    title_prefix: &str,
    snapshots: &[DistSnapshot],
    path_root: &Path,
) -> Result<(), PlotError> {
    let stem = path_root.with_extension("");
    let stem = stem.to_string_lossy();
    write_evolution_subset(
        &format!("{title_prefix} — spine resistors (separators)"),
        snapshots,
        &spine_vars(),
        format!("{stem}_spine.png"),
    )?;
    write_evolution_subset(
        &format!("{title_prefix} — feeders + termination"),
        snapshots,
        &feeder_vars(),
        format!("{stem}_feeder.png"),
    )?;
    Ok(())
}

fn write_marginals_subset(
    title: &str,
    snap: &DistSnapshot,
    vars: &[usize],
    path: impl AsRef<Path>,
) -> Result<(), PlotError> {
    let axis: Vec<f64> = DEVIATIONS.iter().map(|d| d * 100.0).collect();
    let mut signals = BTreeMap::new();
    for &v in vars {
        let series: Vec<f64> = snap.marginals[v].to_vec();
        signals.insert(resistor_label(v), series);
    }
    let wave = Waveform::Real {
        axis_name: "deviation_pct".into(),
        axis,
        signals,
    };
    let cfg = PlotConfig::new()
        .with_title(title)
        .with_size(W, H)
        .add_marker(Marker::Horizontal {
            y: 1.0 / DEVIATIONS.len() as f64,   // 0.2 = uniform prior
            label: Some("uniform".into()),
        });
    png_to_path(&wave, path, &cfg)
}

/// Per-resistor marginal at one snapshot, split spine / feeder for
/// legibility. Writes `<path_root>_{spine,feeder}.png`.
pub fn write_marginals_png(
    title_prefix: &str,
    snap: &DistSnapshot,
    path_root: &Path,
) -> Result<(), PlotError> {
    let stem = path_root.with_extension("");
    let stem = stem.to_string_lossy();
    write_marginals_subset(
        &format!("{title_prefix} — spine"),
        snap,
        &spine_vars(),
        format!("{stem}_spine.png"),
    )?;
    write_marginals_subset(
        &format!("{title_prefix} — feeders + term"),
        snap,
        &feeder_vars(),
        format!("{stem}_feeder.png"),
    )?;
    Ok(())
}

/// DAC staircase from the analytical evaluator: ideal vs the design's
/// vout for every code. Useful for snapshot packages and for the final
/// best-design comparison.
pub fn write_staircase_png(
    title: &str,
    designs: &[(String, Design)],
    path: impl AsRef<Path>,
) -> Result<(), PlotError> {
    let axis: Vec<f64> = (0..N_CODES).map(|c| c as f64).collect();
    let mut signals = BTreeMap::new();

    let ideal: Vec<f64> = (0..N_CODES as u32)
        .map(|c| ideal_vout(c, N_BITS as u32, 1.0, 0.0))
        .collect();
    signals.insert("ideal".to_string(), ideal);
    for (name, d) in designs {
        let v: Vec<f64> = (0..N_CODES as u32).map(|c| solve_r2r(d, c, 1.0, 0.0)).collect();
        signals.insert(name.clone(), v);
    }

    let wave = Waveform::Real { axis_name: "code".into(), axis, signals };
    let cfg = PlotConfig::new().with_title(title).with_size(W, H);
    png_to_path(&wave, path, &cfg)
}

/// INL plot: vout - ideal_vout, per code, for one or more designs.
/// Adds `y=0` as the ideal reference.
pub fn write_inl_png(
    title: &str,
    designs: &[(String, Design)],
    path: impl AsRef<Path>,
) -> Result<(), PlotError> {
    let axis: Vec<f64> = (0..N_CODES).map(|c| c as f64).collect();
    let mut signals = BTreeMap::new();
    for (name, d) in designs {
        let v: Vec<f64> = (0..N_CODES as u32).map(|c| {
            solve_r2r(d, c, 1.0, 0.0) - ideal_vout(c, N_BITS as u32, 1.0, 0.0)
        }).collect();
        signals.insert(name.clone(), v);
    }
    let wave = Waveform::Real { axis_name: "code".into(), axis, signals };
    let cfg = PlotConfig::new()
        .with_title(title)
        .with_size(W, H)
        .add_marker(Marker::Horizontal { y: 0.0, label: Some("ideal".into()) });
    png_to_path(&wave, path, &cfg)
}
