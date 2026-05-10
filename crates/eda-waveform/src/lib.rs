//! Common waveform IR + parsers + writers.
//!
//! Phase 1's "validation pipe" hinges on every external simulator
//! (ngspice, LTspice, Xyce, …) and every rlx-eda native solver speaking
//! the same in-memory waveform shape. This crate owns that shape and the
//! parsers / writers that bridge it to the outside world.
//!
//! ## Layers
//!
//! - [`Waveform`] — the shared IR. Real-valued time/frequency-domain
//!   trace bundle: one independent axis + named dependent series.
//!   `Waveform::Real` covers transient and DC sweeps; `Waveform::Complex`
//!   covers AC.
//! - [`nutmeg`] — parser for ngspice and LTspice "raw" / Nutmeg files
//!   (ASCII + binary). Used by `eda-extern-ngspice` and
//!   `eda-extern-ltspice`. The same format is what the cicwave viewer
//!   consumes, so test fixtures stay openable in cicwave for visual
//!   debug.
//! - [`csv`] — minimal CSV writer / reader. Round-trips a `Waveform`
//!   through pandas-friendly tabular form. cicwave can also read this.
//! - [`vcd`] — Value Change Dump writer + reader for digital waveforms.
//!   Useful for ADC bit traces, clock signals, SAR logic outputs, and
//!   for ingesting external dumps from cocotb / Verilator / Surfer.
//!
//! ## Why split this out of `eda-extern-ngspice`
//!
//! The Nutmeg format is shared between ngspice and LTspice. Phase 1 of
//! the rlx-eda roadmap adds an LTspice driver as a second external
//! validator; both drivers now depend on this crate instead of
//! re-implementing the parser. New simulator drivers (Xyce, Spectre via
//! intermediate, …) drop into the same shape.

pub mod nutmeg;
pub mod csv;
pub mod vcd;
pub mod plot;
pub mod diff;
pub mod timing;
pub mod spectrum;
pub mod assertions;
pub mod ac;
pub mod golden;
pub mod stats;
pub mod linearity;
pub mod align;
pub mod mc;
pub mod power;

use std::collections::BTreeMap;

/// In-memory waveform: an independent axis (time, frequency, sweep
/// variable) plus a name-keyed bundle of dependent series.
///
/// `BTreeMap` keeps deterministic iteration for snapshot tests and CSV
/// column ordering. Series share the same length as `axis`.
#[derive(Debug, Clone)]
pub enum Waveform {
    /// Real-valued: transient, DC sweep, operating-point fan-out.
    Real {
        /// Axis name — `"time"`, `"v-sweep"`, etc. cicwave keys plot axes
        /// off this string.
        axis_name: String,
        axis: Vec<f64>,
        /// `signal_name → samples` aligned 1:1 with `axis`.
        signals: BTreeMap<String, Vec<f64>>,
    },
    /// Complex-valued: AC small-signal response.
    Complex {
        axis_name: String,
        /// Frequencies. Independent variable in AC plots is also written
        /// with a zero-imaginary part by ngspice/LTspice; we strip it.
        axis: Vec<f64>,
        /// `signal_name → (re, im)` samples aligned with `axis`.
        signals: BTreeMap<String, Vec<(f64, f64)>>,
    },
}

impl Waveform {
    pub fn axis_name(&self) -> &str {
        match self {
            Waveform::Real { axis_name, .. } | Waveform::Complex { axis_name, .. } => axis_name,
        }
    }

    pub fn axis(&self) -> &[f64] {
        match self {
            Waveform::Real { axis, .. } | Waveform::Complex { axis, .. } => axis,
        }
    }

    /// Look up a real-valued signal by name. Case-insensitive — Nutmeg
    /// headers are mixed-case across simulators.
    pub fn real(&self, name: &str) -> Option<&[f64]> {
        match self {
            Waveform::Real { signals, .. } => {
                let needle = name.to_lowercase();
                signals
                    .iter()
                    .find(|(k, _)| k.to_lowercase() == needle)
                    .map(|(_, v)| v.as_slice())
            }
            _ => None,
        }
    }

    /// Look up a complex signal by name. Case-insensitive.
    pub fn complex(&self, name: &str) -> Option<&[(f64, f64)]> {
        match self {
            Waveform::Complex { signals, .. } => {
                let needle = name.to_lowercase();
                signals
                    .iter()
                    .find(|(k, _)| k.to_lowercase() == needle)
                    .map(|(_, v)| v.as_slice())
            }
            _ => None,
        }
    }

    pub fn signal_names(&self) -> Vec<&str> {
        match self {
            Waveform::Real { signals, .. } => signals.keys().map(|s| s.as_str()).collect(),
            Waveform::Complex { signals, .. } => signals.keys().map(|s| s.as_str()).collect(),
        }
    }

    /// Return a clone of this waveform with one extra signal:
    /// `signals[p] − signals[n]`, named `out_name`.
    ///
    /// The two existing signals stay untouched. Useful for production
    /// analog where the headline output is differential
    /// (`v(out_p) - v(out_n)`) but the simulator records each leg
    /// separately. Pair with [`Self::with_common_mode`] when you also
    /// want the common-mode trace for CMRR work.
    ///
    /// Returns `None` if either `p` or `n` is missing.
    pub fn with_differential(&self, p: &str, n: &str, out_name: &str) -> Option<Waveform> {
        match self {
            Waveform::Real {
                axis_name,
                axis,
                signals,
            } => {
                let yp = signals.get(p)?;
                let yn = signals.get(n)?;
                let diff: Vec<f64> = yp.iter().zip(yn.iter()).map(|(a, b)| a - b).collect();
                let mut new_signals = signals.clone();
                new_signals.insert(out_name.to_string(), diff);
                Some(Waveform::Real {
                    axis_name: axis_name.clone(),
                    axis: axis.clone(),
                    signals: new_signals,
                })
            }
            Waveform::Complex {
                axis_name,
                axis,
                signals,
            } => {
                let yp = signals.get(p)?;
                let yn = signals.get(n)?;
                let diff: Vec<(f64, f64)> = yp
                    .iter()
                    .zip(yn.iter())
                    .map(|(&(rp, ip), &(rn, in_))| (rp - rn, ip - in_))
                    .collect();
                let mut new_signals = signals.clone();
                new_signals.insert(out_name.to_string(), diff);
                Some(Waveform::Complex {
                    axis_name: axis_name.clone(),
                    axis: axis.clone(),
                    signals: new_signals,
                })
            }
        }
    }

    /// Like [`Self::with_differential`] but stores the common-mode
    /// trace `(signals[p] + signals[n]) / 2` under `out_name`.
    pub fn with_common_mode(&self, p: &str, n: &str, out_name: &str) -> Option<Waveform> {
        match self {
            Waveform::Real {
                axis_name,
                axis,
                signals,
            } => {
                let yp = signals.get(p)?;
                let yn = signals.get(n)?;
                let cm: Vec<f64> = yp
                    .iter()
                    .zip(yn.iter())
                    .map(|(a, b)| 0.5 * (a + b))
                    .collect();
                let mut new_signals = signals.clone();
                new_signals.insert(out_name.to_string(), cm);
                Some(Waveform::Real {
                    axis_name: axis_name.clone(),
                    axis: axis.clone(),
                    signals: new_signals,
                })
            }
            Waveform::Complex {
                axis_name,
                axis,
                signals,
            } => {
                let yp = signals.get(p)?;
                let yn = signals.get(n)?;
                let cm: Vec<(f64, f64)> = yp
                    .iter()
                    .zip(yn.iter())
                    .map(|(&(rp, ip), &(rn, in_))| (0.5 * (rp + rn), 0.5 * (ip + in_)))
                    .collect();
                let mut new_signals = signals.clone();
                new_signals.insert(out_name.to_string(), cm);
                Some(Waveform::Complex {
                    axis_name: axis_name.clone(),
                    axis: axis.clone(),
                    signals: new_signals,
                })
            }
        }
    }

    /// Return a new waveform restricted to samples whose axis value
    /// falls in `[t0, t1]` (inclusive).
    ///
    /// Useful when a transient run is hundreds of thousands of samples
    /// long and you want to focus a plot or analysis on a single
    /// conversion cycle / clock period / event of interest. Out-of-window
    /// samples are dropped from every signal in lockstep with the axis.
    ///
    /// Returns an empty waveform (axis + signals all length 0) if no
    /// samples land inside the window.
    pub fn slice_window(&self, t0: f64, t1: f64) -> Waveform {
        assert!(t1 >= t0, "slice_window: t1 ({t1}) must be >= t0 ({t0})");
        match self {
            Waveform::Real {
                axis_name,
                axis,
                signals,
            } => {
                let keep: Vec<usize> = axis
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &t)| if t >= t0 && t <= t1 { Some(i) } else { None })
                    .collect();
                let new_axis: Vec<f64> = keep.iter().map(|&i| axis[i]).collect();
                let new_signals: BTreeMap<String, Vec<f64>> = signals
                    .iter()
                    .map(|(k, v)| (k.clone(), keep.iter().map(|&i| v[i]).collect()))
                    .collect();
                Waveform::Real {
                    axis_name: axis_name.clone(),
                    axis: new_axis,
                    signals: new_signals,
                }
            }
            Waveform::Complex {
                axis_name,
                axis,
                signals,
            } => {
                let keep: Vec<usize> = axis
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &t)| if t >= t0 && t <= t1 { Some(i) } else { None })
                    .collect();
                let new_axis: Vec<f64> = keep.iter().map(|&i| axis[i]).collect();
                let new_signals: BTreeMap<String, Vec<(f64, f64)>> = signals
                    .iter()
                    .map(|(k, v)| (k.clone(), keep.iter().map(|&i| v[i]).collect()))
                    .collect();
                Waveform::Complex {
                    axis_name: axis_name.clone(),
                    axis: new_axis,
                    signals: new_signals,
                }
            }
        }
    }
}

/// Build a [`Waveform::Real`] from a Nutmeg plot, dropping the
/// independent axis from the signals map (it lives in `axis` instead).
///
/// Returns `None` if `plot` is complex-flavored — call [`from_nutmeg_complex`]
/// in that case.
pub fn from_nutmeg_real(plot: &nutmeg::NutmegPlot) -> Option<Waveform> {
    if plot.flavor != nutmeg::NutmegFlavor::Real {
        return None;
    }
    let (axis_name, axis_idx) = independent_axis(&plot.var_names)?;
    let axis = plot.values.get(axis_idx)?.clone();
    let mut signals = BTreeMap::new();
    for (i, name) in plot.var_names.iter().enumerate() {
        if i == axis_idx {
            continue;
        }
        if let Some(v) = plot.values.get(i) {
            signals.insert(name.clone(), v.clone());
        }
    }
    Some(Waveform::Real { axis_name, axis, signals })
}

/// Build a [`Waveform::Complex`] from a Nutmeg plot. Strips the zero
/// imaginary part on the frequency axis.
pub fn from_nutmeg_complex(plot: &nutmeg::NutmegPlot) -> Option<Waveform> {
    if plot.flavor != nutmeg::NutmegFlavor::Complex {
        return None;
    }
    let (axis_name, axis_idx) = independent_axis(&plot.var_names)?;
    let axis = plot
        .complex_values
        .get(axis_idx)?
        .iter()
        .map(|(re, _)| *re)
        .collect();
    let mut signals = BTreeMap::new();
    for (i, name) in plot.var_names.iter().enumerate() {
        if i == axis_idx {
            continue;
        }
        if let Some(v) = plot.complex_values.get(i) {
            signals.insert(name.clone(), v.clone());
        }
    }
    Some(Waveform::Complex { axis_name, axis, signals })
}

/// First var that looks like an independent axis. ngspice / LTspice
/// always put it first, but we match by name so a hand-built plot with
/// `time` in slot 1 still works.
fn independent_axis(var_names: &[String]) -> Option<(String, usize)> {
    for (i, n) in var_names.iter().enumerate() {
        let l = n.to_lowercase();
        if l == "time" || l == "frequency" || l.ends_with("-sweep") || l == "v-sweep" {
            return Some((n.clone(), i));
        }
    }
    // Fall back to the first variable.
    var_names.first().map(|n| (n.clone(), 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_real() -> nutmeg::NutmegPlot {
        nutmeg::NutmegPlot {
            plotname: "T".into(),
            flavor: nutmeg::NutmegFlavor::Real,
            var_names: vec!["time".into(), "v(out)".into()],
            values: vec![vec![0.0, 1e-9, 2e-9], vec![0.0, 0.5, 0.75]],
            complex_values: vec![],
        }
    }

    #[test]
    fn lift_real_drops_axis_from_signals() {
        let w = from_nutmeg_real(&fixture_real()).unwrap();
        assert_eq!(w.axis_name(), "time");
        assert_eq!(w.axis(), &[0.0, 1e-9, 2e-9]);
        assert_eq!(w.real("v(out)").unwrap(), &[0.0, 0.5, 0.75]);
        assert!(w.real("time").is_none()); // axis is not in signals
    }

    #[test]
    fn case_insensitive_lookup() {
        let w = from_nutmeg_real(&fixture_real()).unwrap();
        assert!(w.real("V(OUT)").is_some());
    }

    #[test]
    fn slice_window_keeps_inclusive_endpoints() {
        let mut signals = BTreeMap::new();
        signals.insert("v(out)".into(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9, 2e-9, 3e-9, 4e-9],
            signals,
        };
        let s = w.slice_window(1e-9, 3e-9);
        assert_eq!(s.axis(), &[1e-9, 2e-9, 3e-9]);
        assert_eq!(s.real("v(out)").unwrap(), &[0.25, 0.5, 0.75]);
    }

    #[test]
    fn slice_window_empty_when_no_samples_inside() {
        let mut signals = BTreeMap::new();
        signals.insert("v".into(), vec![0.0, 1.0]);
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1.0],
            signals,
        };
        let s = w.slice_window(5.0, 10.0);
        assert!(s.axis().is_empty());
        assert_eq!(s.real("v").unwrap().len(), 0);
    }

    #[test]
    fn with_differential_real_subtracts_n_from_p() {
        // Use exactly representable values (halves) so we can use
        // exact equality without f64 representation drift.
        let mut signals = BTreeMap::new();
        signals.insert("v(outp)".into(), vec![0.5, 1.0, 1.5]);
        signals.insert("v(outn)".into(), vec![0.25, 0.5, 0.75]);
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9, 2e-9],
            signals,
        };
        let d = w.with_differential("v(outp)", "v(outn)", "v(out_diff)").unwrap();
        assert_eq!(d.real("v(out_diff)").unwrap(), &[0.25, 0.5, 0.75]);
        // Original signals preserved.
        assert_eq!(d.real("v(outp)").unwrap(), &[0.5, 1.0, 1.5]);
    }

    #[test]
    fn with_common_mode_real_averages() {
        let mut signals = BTreeMap::new();
        signals.insert("v(outp)".into(), vec![0.6, 0.7]);
        signals.insert("v(outn)".into(), vec![0.4, 0.3]);
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9],
            signals,
        };
        let c = w.with_common_mode("v(outp)", "v(outn)", "v(cm)").unwrap();
        assert_eq!(c.real("v(cm)").unwrap(), &[0.5, 0.5]);
    }

    #[test]
    fn with_differential_returns_none_for_missing_signal() {
        let mut signals = BTreeMap::new();
        signals.insert("v(only)".into(), vec![1.0]);
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0],
            signals,
        };
        assert!(w.with_differential("v(only)", "missing", "diff").is_none());
        assert!(w.with_differential("missing", "v(only)", "diff").is_none());
    }

    #[test]
    fn with_differential_works_on_complex() {
        // AC sweep with two named complex signals: vp = 1 + 0j, vn = 0 - j.
        // Differential: (1 - 0, 0 - (-1)) = (1, 1).
        let mut signals = BTreeMap::new();
        signals.insert("vp".into(), vec![(1.0, 0.0)]);
        signals.insert("vn".into(), vec![(0.0, -1.0)]);
        let w = Waveform::Complex {
            axis_name: "frequency".into(),
            axis: vec![1e3],
            signals,
        };
        let d = w.with_differential("vp", "vn", "vdiff").unwrap();
        assert_eq!(d.complex("vdiff").unwrap(), &[(1.0, 1.0)]);
    }

    #[test]
    fn slice_window_works_on_complex() {
        let mut signals = BTreeMap::new();
        signals.insert("h".into(), vec![(1.0, 0.0), (0.5, -0.5), (0.1, -0.1)]);
        let w = Waveform::Complex {
            axis_name: "frequency".into(),
            axis: vec![1.0, 10.0, 100.0],
            signals,
        };
        let s = w.slice_window(5.0, 50.0);
        assert_eq!(s.axis(), &[10.0]);
        assert_eq!(s.complex("h").unwrap(), &[(0.5, -0.5)]);
    }
}
