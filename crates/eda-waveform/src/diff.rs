//! Per-signal `Waveform` comparison with rtol/atol envelopes.
//!
//! The validation pyramid wants a single primitive that answers "do
//! these two waveforms agree?" — golden vs. candidate, ngspice vs.
//! LTspice, analytic vs. FD vs. ngspice. This is that primitive.
//!
//! Builds on [`eda_validate::is_close_f64`] / [`eda_validate::lerp`]
//! so the `|a - b| <= atol + rtol * |b|` envelope is the same one used
//! everywhere else in the stack. Time grids may differ — the candidate
//! is interpolated onto the golden's axis over the overlap interval,
//! matching `eda_validate::compare_transient_traces`.
//!
//! Compared to that helper, this layer:
//! - Operates on [`Waveform::Real`] directly so callers don't unpack maps.
//! - Reports both **stats** (max-abs, RMS) and the **first divergence**
//!   per signal, so a CI gate can both fail loud and pinpoint where.
//! - Allows a per-signal tolerance override — e.g. tighten the clock
//!   node, loosen a noisy sense node.
//!
//! Signals present in only one side are surfaced via `missing_in_*`,
//! not silently dropped.
//!
//! ## Why this lives in eda-waveform
//!
//! `eda-validate` deliberately stays Waveform-agnostic (it's used by
//! crates that don't depend on the waveform IR). This module is the
//! Waveform-shaped convenience layer on top.

use std::collections::{BTreeMap, BTreeSet};

use eda_validate::{is_close_f64, lerp};
use thiserror::Error;

use crate::Waveform;

#[derive(Debug, Error)]
pub enum DiffError {
    #[error("waveforms must be real-valued for diff")]
    NotReal,
    #[error("waveform axis is empty")]
    Empty,
    #[error("waveform axes do not overlap (golden: [{ga}, {gb}], candidate: [{ca}, {cb}])")]
    NoOverlap { ga: f64, gb: f64, ca: f64, cb: f64 },
}

/// Tolerance envelope. Same shape as the one in `eda-validate`.
#[derive(Debug, Clone, Copy)]
pub struct Tol {
    pub rtol: f64,
    pub atol: f64,
}

impl Tol {
    pub const fn new(rtol: f64, atol: f64) -> Self {
        Self { rtol, atol }
    }
}

/// Per-signal diff stats over the time-axis overlap.
#[derive(Debug, Clone)]
pub struct SignalDiff {
    pub max_abs: f64,
    pub rms: f64,
    /// `(idx_in_golden, t, golden_value, candidate_at_t)` for the first
    /// sample where `is_close_f64` rejected, if any.
    pub first_divergence: Option<(usize, f64, f64, f64)>,
    /// Tolerance applied to this signal (after per-signal override).
    pub tol: Tol,
}

/// Aggregate report.
#[derive(Debug, Clone)]
pub struct DiffReport {
    pub per_signal: BTreeMap<String, SignalDiff>,
    /// `(signal, max_abs)` for the worst signal by max-abs diff.
    pub worst: Option<(String, f64)>,
    /// Signals present in candidate but missing from golden.
    pub missing_in_golden: Vec<String>,
    /// Signals present in golden but missing from candidate.
    pub missing_in_candidate: Vec<String>,
    /// Time-axis overlap actually used: `[t_lo, t_hi]`.
    pub overlap: (f64, f64),
}

impl DiffReport {
    pub fn is_ok(&self) -> bool {
        self.per_signal.values().all(|d| d.first_divergence.is_none())
            && self.missing_in_golden.is_empty()
            && self.missing_in_candidate.is_empty()
    }

    /// Names of signals whose envelope was breached.
    pub fn divergent(&self) -> Vec<&str> {
        self.per_signal
            .iter()
            .filter_map(|(n, d)| d.first_divergence.as_ref().map(|_| n.as_str()))
            .collect()
    }

    /// Panic with a useful message if anything diverged or signals are missing.
    #[track_caller]
    pub fn assert_ok(&self, label: &str) {
        if !self.missing_in_candidate.is_empty() {
            panic!(
                "[{label}] candidate is missing signals: {:?}",
                self.missing_in_candidate
            );
        }
        if !self.missing_in_golden.is_empty() {
            panic!(
                "[{label}] candidate has extra signals not in golden: {:?}",
                self.missing_in_golden
            );
        }
        for (name, d) in &self.per_signal {
            if let Some((idx, t, a, b)) = d.first_divergence {
                let env = d.tol.atol + d.tol.rtol * b.abs();
                panic!(
                    "[{label}] signal {name:?} diverged at idx {idx}, t={t:+.6e}:\n  golden    = {a:+.9e}\n  candidate = {b:+.9e}\n  |a-b| = {diff:.3e}  (envelope = {env:.3e})\n  max_abs over overlap = {max_abs:.3e}, rms = {rms:.3e}",
                    diff = (a - b).abs(),
                    max_abs = d.max_abs,
                    rms = d.rms,
                );
            }
        }
    }
}

/// Compare two real waveforms signal-by-signal.
///
/// `golden`'s time axis is the comparison grid; `candidate` is linearly
/// interpolated onto it over the overlap interval. Tolerance defaults
/// from `default`, with optional per-signal overrides keyed by signal
/// name (matching exact `Waveform` keys).
pub fn diff(
    golden: &Waveform,
    candidate: &Waveform,
    default: Tol,
    per_signal: &BTreeMap<String, Tol>,
) -> Result<DiffReport, DiffError> {
    let (g_axis, g_signals) = real_view(golden)?;
    let (c_axis, c_signals) = real_view(candidate)?;
    if g_axis.is_empty() || c_axis.is_empty() {
        return Err(DiffError::Empty);
    }

    let t_lo = g_axis[0].max(c_axis[0]);
    let t_hi = g_axis[g_axis.len() - 1].min(c_axis[c_axis.len() - 1]);
    if t_hi < t_lo {
        return Err(DiffError::NoOverlap {
            ga: g_axis[0],
            gb: g_axis[g_axis.len() - 1],
            ca: c_axis[0],
            cb: c_axis[c_axis.len() - 1],
        });
    }

    let g_keys: BTreeSet<&str> = g_signals.keys().map(|s| s.as_str()).collect();
    let c_keys: BTreeSet<&str> = c_signals.keys().map(|s| s.as_str()).collect();
    let missing_in_candidate: Vec<String> = g_keys
        .difference(&c_keys)
        .map(|s| (*s).to_string())
        .collect();
    let missing_in_golden: Vec<String> = c_keys
        .difference(&g_keys)
        .map(|s| (*s).to_string())
        .collect();

    let mut per_signal_out = BTreeMap::new();
    let mut worst: Option<(String, f64)> = None;
    for name in g_keys.intersection(&c_keys) {
        let ya = &g_signals[*name];
        let yb = &c_signals[*name];
        let tol = per_signal.get(*name).copied().unwrap_or(default);

        let mut max_abs = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut count = 0usize;
        let mut first_div: Option<(usize, f64, f64, f64)> = None;
        for (i, (&t, &a)) in g_axis.iter().zip(ya.iter()).enumerate() {
            if t < t_lo || t > t_hi {
                continue;
            }
            let b = lerp(c_axis, yb, t);
            // NaN-vs-NaN: treat as match (same unknown). NaN-vs-number: divergence.
            let agree = if a.is_nan() && b.is_nan() {
                true
            } else if a.is_nan() || b.is_nan() {
                false
            } else {
                is_close_f64(a, b, tol.rtol, tol.atol)
            };
            let d = if a.is_nan() || b.is_nan() {
                f64::INFINITY
            } else {
                (a - b).abs()
            };
            if d.is_finite() {
                if d > max_abs {
                    max_abs = d;
                }
                sum_sq += d * d;
                count += 1;
            }
            if first_div.is_none() && !agree {
                first_div = Some((i, t, a, b));
            }
        }
        let rms = if count > 0 {
            (sum_sq / count as f64).sqrt()
        } else {
            0.0
        };
        if worst.as_ref().is_none_or(|(_, w)| max_abs > *w) {
            worst = Some(((*name).to_string(), max_abs));
        }
        per_signal_out.insert(
            (*name).to_string(),
            SignalDiff {
                max_abs,
                rms,
                first_divergence: first_div,
                tol,
            },
        );
    }

    Ok(DiffReport {
        per_signal: per_signal_out,
        worst,
        missing_in_golden,
        missing_in_candidate,
        overlap: (t_lo, t_hi),
    })
}

fn real_view(w: &Waveform) -> Result<(&Vec<f64>, &BTreeMap<String, Vec<f64>>), DiffError> {
    match w {
        Waveform::Real { axis, signals, .. } => Ok((axis, signals)),
        Waveform::Complex { .. } => Err(DiffError::NotReal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(name: &str, slope: f64, n: usize, dt: f64) -> Waveform {
        let axis: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let y: Vec<f64> = axis.iter().map(|t| slope * t).collect();
        let mut signals = BTreeMap::new();
        signals.insert(name.to_string(), y);
        Waveform::Real {
            axis_name: "time".into(),
            axis,
            signals,
        }
    }

    #[test]
    fn matches_on_different_grids() {
        // y = 2t, sampled finely vs coarsely. Within float epsilon.
        let a = ramp("y", 2.0, 11, 0.1);
        let b = ramp("y", 2.0, 21, 0.05);
        let r = diff(&a, &b, Tol::new(1e-9, 1e-12), &BTreeMap::new()).unwrap();
        r.assert_ok("ramp");
        assert!(r.is_ok());
        assert_eq!(r.divergent(), Vec::<&str>::new());
    }

    #[test]
    fn detects_first_divergence() {
        let a = ramp("y", 2.0, 5, 1.0); // 0, 2, 4, 6, 8
        let mut b = a.clone();
        if let Waveform::Real { signals, .. } = &mut b {
            signals.get_mut("y").unwrap()[3] = 6.5; // off by 0.5 at t=3
        }
        let r = diff(&a, &b, Tol::new(1e-6, 1e-6), &BTreeMap::new()).unwrap();
        let d = &r.per_signal["y"];
        let (idx, t, _, _) = d.first_divergence.expect("expected divergence");
        assert_eq!(idx, 3);
        assert_eq!(t, 3.0);
        assert!((d.max_abs - 0.5).abs() < 1e-12);
    }

    #[test]
    fn per_signal_override_loosens_one_signal() {
        let mut a_signals = BTreeMap::new();
        a_signals.insert("clk".into(), vec![0.0, 1.0, 0.0]);
        a_signals.insert("noisy".into(), vec![0.0, 0.5, 1.0]);
        let a = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1.0, 2.0],
            signals: a_signals,
        };
        let mut b_signals = BTreeMap::new();
        b_signals.insert("clk".into(), vec![0.0, 1.0, 0.0]);
        b_signals.insert("noisy".into(), vec![0.0, 0.6, 1.0]); // 0.1 off
        let b = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1.0, 2.0],
            signals: b_signals,
        };

        // Tight default fails on noisy …
        let strict = diff(&a, &b, Tol::new(1e-6, 1e-6), &BTreeMap::new()).unwrap();
        assert_eq!(strict.divergent(), vec!["noisy"]);

        // … per-signal override loosens just the noisy node.
        let mut overrides = BTreeMap::new();
        overrides.insert("noisy".to_string(), Tol::new(0.0, 0.2));
        let r = diff(&a, &b, Tol::new(1e-6, 1e-6), &overrides).unwrap();
        r.assert_ok("with override");
    }

    #[test]
    fn reports_missing_signals() {
        let mut a_signals = BTreeMap::new();
        a_signals.insert("x".into(), vec![0.0, 1.0]);
        a_signals.insert("y".into(), vec![0.0, 2.0]);
        let a = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1.0],
            signals: a_signals,
        };
        let mut b_signals = BTreeMap::new();
        b_signals.insert("x".into(), vec![0.0, 1.0]);
        b_signals.insert("z".into(), vec![0.0, 3.0]); // extra
        let b = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1.0],
            signals: b_signals,
        };
        let r = diff(&a, &b, Tol::new(1e-6, 1e-6), &BTreeMap::new()).unwrap();
        assert_eq!(r.missing_in_candidate, vec!["y".to_string()]);
        assert_eq!(r.missing_in_golden, vec!["z".to_string()]);
        assert!(!r.is_ok());
    }

    #[test]
    #[should_panic(expected = "diverged at idx 3")]
    fn assert_ok_panics_with_first_divergence() {
        let a = ramp("y", 1.0, 5, 1.0);
        let mut b = a.clone();
        if let Waveform::Real { signals, .. } = &mut b {
            signals.get_mut("y").unwrap()[3] = 99.0;
        }
        let r = diff(&a, &b, Tol::new(1e-6, 1e-6), &BTreeMap::new()).unwrap();
        r.assert_ok("test");
    }

    #[test]
    fn rejects_complex() {
        let a = Waveform::Complex {
            axis_name: "frequency".into(),
            axis: vec![1e3],
            signals: BTreeMap::new(),
        };
        let b = a.clone();
        assert!(matches!(
            diff(&a, &b, Tol::new(1e-6, 1e-6), &BTreeMap::new()),
            Err(DiffError::NotReal)
        ));
    }

    #[test]
    fn nan_vs_nan_matches() {
        let mut a_signals = BTreeMap::new();
        a_signals.insert("u".into(), vec![f64::NAN, 1.0]);
        let a = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1.0],
            signals: a_signals,
        };
        let b = a.clone();
        let r = diff(&a, &b, Tol::new(1e-6, 1e-6), &BTreeMap::new()).unwrap();
        r.assert_ok("nan-nan");
    }
}
