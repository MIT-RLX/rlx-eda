//! Monte Carlo / corner aggregation harness.
//!
//! Production verification runs the same testbench across many corners
//! (PVT, mismatch seeds, layout-extracted vs. schematic) and asks:
//!
//! - What's the worst-case ENOB? Which seed produced it?
//! - What's the σ of phase margin across mismatch?
//! - What fraction of corners meet the spec?
//!
//! Single-run primitives (in `spectrum`, `ac`, `linearity`, …) answer
//! those questions for *one* `Waveform`. This module folds many
//! per-run results into the aggregate numbers a release gate cares
//! about.
//!
//! ## Shape
//!
//! - [`Run`] — a labeled scalar metric value (`label = "tt_27c"`,
//!   `metric = 11.8`).
//! - [`map_metric`] / [`try_map_metric`] — apply a `Waveform → f64`
//!   extractor across a labeled set of waveforms, producing runs.
//! - [`collect_stats`] — fold runs into mean / std / min / max / median /
//!   worst-case.
//! - [`check_spec`] — yield against a predicate.
//!
//! ## Worst-case direction
//!
//! [`Worst`] picks the polarity. Use `Min` for "bigger is better"
//! metrics (ENOB, phase margin, gain margin), `Max` for "smaller is
//! better" (THD magnitude, settling time, INL). The worst case is
//! reported with its label so the release notes can finger the corner.
//!
//! ## NaN / failed runs
//!
//! [`map_metric`] takes an infallible extractor; if your metric can
//! fail (e.g. UGBW doesn't exist for an over-compensated loop), use
//! [`try_map_metric`] which separates successful runs from errored
//! ones into two vecs. The error vec is part of the audit trail.

use crate::stats;
use crate::Waveform;

/// One labeled run result.
#[derive(Debug, Clone)]
pub struct Run<M> {
    pub label: String,
    pub metric: M,
}

/// Polarity of "worst" — which direction is bad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Worst {
    /// Smallest value is worst (e.g. ENOB, phase margin).
    Min,
    /// Largest value is worst (e.g. THD, INL, settling time).
    Max,
}

/// Aggregate stats over `Run<f64>`. Means / std / min / max / median
/// are over the full `runs` slice; `worst` is the run identified by
/// the chosen polarity.
#[derive(Debug, Clone)]
pub struct Stats {
    pub n: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    /// `(label, metric)` of the worst run, if any.
    pub worst: Option<(String, f64)>,
}

/// Fold a slice of runs into aggregate statistics.
///
/// Returns `None` if `runs` is empty. Single-run input gives `std = 0`.
pub fn collect_stats(runs: &[Run<f64>], worst: Worst) -> Option<Stats> {
    if runs.is_empty() {
        return None;
    }
    let values: Vec<f64> = runs.iter().map(|r| r.metric).collect();
    let mean = stats::mean(&values)?;
    let std = stats::std(&values).unwrap_or(0.0);
    let (min, max) = stats::min_max(&values)?;
    let median = stats::percentile(&values, 50.0)?;

    let worst_run = runs.iter().fold(None, |acc: Option<&Run<f64>>, r| {
        match acc {
            None => Some(r),
            Some(cur) => {
                let take = match worst {
                    Worst::Min => r.metric < cur.metric,
                    Worst::Max => r.metric > cur.metric,
                };
                if take { Some(r) } else { Some(cur) }
            }
        }
    });
    let worst = worst_run.map(|r| (r.label.clone(), r.metric));

    Some(Stats {
        n: runs.len(),
        mean,
        std,
        min,
        max,
        median,
        worst,
    })
}

/// Apply an infallible `Waveform → f64` extractor to each labeled
/// waveform. NaN-producing extractors are passed through; downstream
/// `collect_stats` will flag them via `min`/`max`.
pub fn map_metric<F>(
    waves: &[(String, Waveform)],
    mut extract: F,
) -> Vec<Run<f64>>
where
    F: FnMut(&Waveform) -> f64,
{
    waves
        .iter()
        .map(|(label, w)| Run {
            label: label.clone(),
            metric: extract(w),
        })
        .collect()
}

/// Apply a fallible extractor; partition into successful runs and
/// errors. Use this when the extractor can legitimately fail (UGBW
/// doesn't always exist, AC analysis can return `None`, etc.).
pub fn try_map_metric<F, E>(
    waves: &[(String, Waveform)],
    mut extract: F,
) -> (Vec<Run<f64>>, Vec<(String, E)>)
where
    F: FnMut(&Waveform) -> Result<f64, E>,
{
    let mut ok = Vec::new();
    let mut err = Vec::new();
    for (label, w) in waves {
        match extract(w) {
            Ok(v) => ok.push(Run {
                label: label.clone(),
                metric: v,
            }),
            Err(e) => err.push((label.clone(), e)),
        }
    }
    (ok, err)
}

/// Yield statistics against a pass/fail predicate.
#[derive(Debug, Clone)]
pub struct SpecResult {
    pub n_total: usize,
    pub n_pass: usize,
    pub yield_frac: f64,
    /// Runs that failed the predicate, in input order.
    pub failures: Vec<Run<f64>>,
}

/// Apply `pred` to each run's metric. Reports yield + the failing runs.
pub fn check_spec<P>(runs: &[Run<f64>], mut pred: P) -> SpecResult
where
    P: FnMut(f64) -> bool,
{
    let mut failures = Vec::new();
    let mut n_pass = 0usize;
    for r in runs {
        if pred(r.metric) {
            n_pass += 1;
        } else {
            failures.push(r.clone());
        }
    }
    let n_total = runs.len();
    let yield_frac = if n_total == 0 {
        0.0
    } else {
        n_pass as f64 / n_total as f64
    };
    SpecResult {
        n_total,
        n_pass,
        yield_frac,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn run(label: &str, m: f64) -> Run<f64> {
        Run {
            label: label.into(),
            metric: m,
        }
    }

    fn dummy_wave(name: &str, v: f64) -> Waveform {
        let mut signals = BTreeMap::new();
        signals.insert(name.to_string(), vec![v, v, v]);
        Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1.0, 2.0],
            signals,
        }
    }

    #[test]
    fn stats_aggregate_correctly() {
        let runs = vec![
            run("a", 10.0),
            run("b", 12.0),
            run("c", 11.0),
            run("d", 14.0),
            run("e", 13.0),
        ];
        let s = collect_stats(&runs, Worst::Min).unwrap();
        assert_eq!(s.n, 5);
        assert_eq!(s.min, 10.0);
        assert_eq!(s.max, 14.0);
        assert_eq!(s.median, 12.0);
        assert_eq!(s.mean, 12.0);
        // Worst (Min) → label "a", metric 10.0
        assert_eq!(s.worst.as_ref().unwrap().0, "a");
        assert_eq!(s.worst.as_ref().unwrap().1, 10.0);
    }

    #[test]
    fn worst_max_picks_largest() {
        let runs = vec![run("low", 1.0), run("high", 99.0), run("mid", 50.0)];
        let s = collect_stats(&runs, Worst::Max).unwrap();
        assert_eq!(s.worst.as_ref().unwrap().0, "high");
        assert_eq!(s.worst.as_ref().unwrap().1, 99.0);
    }

    #[test]
    fn empty_runs_returns_none() {
        let runs: Vec<Run<f64>> = Vec::new();
        assert!(collect_stats(&runs, Worst::Min).is_none());
    }

    #[test]
    fn map_metric_extracts_per_run() {
        // Three "corners" with different DC values; metric = the constant.
        let waves = vec![
            ("tt".to_string(), dummy_wave("v", 1.0)),
            ("ff".to_string(), dummy_wave("v", 1.1)),
            ("ss".to_string(), dummy_wave("v", 0.9)),
        ];
        let runs = map_metric(&waves, |w| w.real("v").unwrap()[0]);
        assert_eq!(runs.len(), 3);
        let s = collect_stats(&runs, Worst::Min).unwrap();
        assert!((s.mean - 1.0).abs() < 1e-12);
        assert_eq!(s.worst.as_ref().unwrap().0, "ss");
    }

    #[test]
    fn try_map_metric_partitions_ok_and_err() {
        let waves = vec![
            ("good".to_string(), dummy_wave("v", 1.0)),
            ("bad".to_string(), dummy_wave("v", 1.0)),
        ];
        let (ok, err) = try_map_metric(&waves, |w| {
            let v = w.real("v").unwrap()[0];
            if v < 1.5 {
                Err("too small")
            } else {
                Ok(v)
            }
        });
        assert!(ok.is_empty());
        assert_eq!(err.len(), 2);
        assert_eq!(err[0].0, "good");
        assert_eq!(err[1].0, "bad");
    }

    #[test]
    fn check_spec_reports_yield_and_failures() {
        // Spec: ENOB ≥ 11.5. Runs: 11.8, 11.4, 11.6, 11.2, 12.0.
        let runs = vec![
            run("c1", 11.8),
            run("c2", 11.4),
            run("c3", 11.6),
            run("c4", 11.2),
            run("c5", 12.0),
        ];
        let r = check_spec(&runs, |m| m >= 11.5);
        assert_eq!(r.n_total, 5);
        assert_eq!(r.n_pass, 3);
        assert!((r.yield_frac - 0.6).abs() < 1e-12);
        assert_eq!(r.failures.len(), 2);
        assert_eq!(r.failures[0].label, "c2");
        assert_eq!(r.failures[1].label, "c4");
    }

    #[test]
    fn check_spec_empty_yields_zero() {
        let r = check_spec::<fn(f64) -> bool>(&[], |_| true);
        assert_eq!(r.n_total, 0);
        assert_eq!(r.yield_frac, 0.0);
    }

    #[test]
    fn single_run_has_zero_std() {
        let runs = vec![run("only", 7.0)];
        let s = collect_stats(&runs, Worst::Min).unwrap();
        assert_eq!(s.std, 0.0);
        assert_eq!(s.median, 7.0);
        assert_eq!(s.mean, 7.0);
    }

    #[test]
    fn end_to_end_corner_sweep_with_pm_metric() {
        // Synthetic "phase margin" values across 4 corners; spec is PM ≥ 45°.
        let runs = vec![
            run("tt_27c", 60.0),
            run("ff_85c", 52.0),
            run("ss_-40c", 38.0), // failing corner
            run("fs_27c", 47.0),
        ];
        let s = collect_stats(&runs, Worst::Min).unwrap();
        assert_eq!(s.worst.as_ref().unwrap().0, "ss_-40c");
        let yield_ = check_spec(&runs, |pm| pm >= 45.0);
        assert_eq!(yield_.n_pass, 3);
        assert_eq!(yield_.failures.len(), 1);
        assert_eq!(yield_.failures[0].label, "ss_-40c");
    }
}
