//! Time-align two waveforms by cross-correlation.
//!
//! ngspice and LTspice (and even two ngspice runs with different `.tran`
//! arguments) often start the meaningful part of the trace at slightly
//! different `t`. A `diff` between them then sees a "shape match,
//! shifted" — every sample diverges by an amount that's actually just
//! the time offset.
//!
//! This module finds the best integer-then-sub-sample lag between two
//! uniformly-sampled signals and returns the shift in seconds.
//! Subtract that shift from one waveform's time axis (or add to the
//! other's) and the diff collapses to the real numerical disagreement.
//!
//! ## Sign convention
//!
//! [`best_lag`] returns the lag `k` (in samples) that maximizes
//! `Σᵢ a[i] · b[i + k]`. Equivalently:
//!
//! - `a[i]` corresponds to the same physical instant as `b[i + k]`.
//! - `b(t + k·dt) ≈ a(t)` in the overlap.
//! - If `k > 0`, `b` lags `a` (b's events happened later in the
//!   sample-index sense; you'd shift `b` *earlier* by `k·dt` to align).
//!
//! [`align_waveforms`] returns `shift_seconds = k · dt` after sub-sample
//! refinement via parabolic interpolation around the peak.
//!
//! ## When this is the right tool
//!
//! Use it for *small* shifts — fractions of a percent of the trace
//! length up to a few percent. For larger offsets, the signals
//! probably differ in more than just timing and the cross-correlation
//! peak isn't meaningful. The default `max_shift_seconds` parameter
//! exists precisely to bound the search.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::Waveform;

#[derive(Debug, Error)]
pub enum AlignError {
    #[error("waveforms must be real-valued")]
    NotReal,
    #[error("signal {0:?} not found in one of the waveforms")]
    MissingSignal(String),
    #[error("waveforms must have at least 4 samples each")]
    TooShort,
    #[error("waveforms have non-overlapping time ranges")]
    NoOverlap,
}

/// Best integer lag (in samples) and the cross-correlation value at
/// that lag. The search is constrained to `|lag| <= max_lag`.
///
/// `a` and `b` must be equal-length and uniformly sampled. NaN samples
/// are treated as zero (so a missing sample contributes nothing rather
/// than poisoning the sum).
pub fn best_lag(a: &[f64], b: &[f64], max_lag: usize) -> (i64, f64) {
    assert_eq!(a.len(), b.len(), "a/b length mismatch");
    let n = a.len() as i64;
    let max_lag = (max_lag as i64).min(n - 1);

    let prod = |k: i64| -> f64 {
        let (lo, hi) = if k >= 0 {
            (0i64, n - k)
        } else {
            (-k, n)
        };
        let mut s = 0.0;
        for i in lo..hi {
            let av = a[i as usize];
            let bv = b[(i + k) as usize];
            if av.is_finite() && bv.is_finite() {
                s += av * bv;
            }
        }
        s
    };

    let mut best_k = 0i64;
    let mut best_v = f64::NEG_INFINITY;
    for k in -max_lag..=max_lag {
        let v = prod(k);
        if v > best_v {
            best_v = v;
            best_k = k;
        }
    }
    (best_k, best_v)
}

/// Find the time shift (seconds) that aligns `b`'s named signal with
/// `a`'s. Both waveforms are first resampled onto a uniform grid spanning
/// their time-axis overlap; cross-correlation is bounded to
/// `±max_shift_seconds`. Returns the sub-sample-refined shift.
///
/// Apply the result by subtracting it from `b`'s time axis (or adding
/// it to `a`'s) before calling [`crate::diff::diff`].
pub fn align_waveforms(
    a: &Waveform,
    b: &Waveform,
    signal: &str,
    max_shift_seconds: f64,
) -> Result<f64, AlignError> {
    let (a_axis, a_signals) = real_view(a)?;
    let (b_axis, b_signals) = real_view(b)?;
    let ya = a_signals
        .get(signal)
        .ok_or_else(|| AlignError::MissingSignal(signal.to_string()))?;
    let yb = b_signals
        .get(signal)
        .ok_or_else(|| AlignError::MissingSignal(signal.to_string()))?;
    if a_axis.len() < 4 || b_axis.len() < 4 {
        return Err(AlignError::TooShort);
    }
    let t_lo = a_axis[0].max(b_axis[0]);
    let t_hi = a_axis[a_axis.len() - 1].min(b_axis[b_axis.len() - 1]);
    if t_hi <= t_lo {
        return Err(AlignError::NoOverlap);
    }

    // Resample to a uniform grid. Pick the finer of the two natural
    // step sizes so the cross-correlation has the resolution to find
    // small shifts.
    let dt_a = (a_axis[a_axis.len() - 1] - a_axis[0]) / (a_axis.len() - 1).max(1) as f64;
    let dt_b = (b_axis[b_axis.len() - 1] - b_axis[0]) / (b_axis.len() - 1).max(1) as f64;
    let dt = dt_a.min(dt_b);
    let n = (((t_hi - t_lo) / dt).floor() as usize + 1).max(8);
    let resampled_a: Vec<f64> = (0..n)
        .map(|i| eda_validate::lerp(a_axis, ya, t_lo + i as f64 * dt))
        .collect();
    let resampled_b: Vec<f64> = (0..n)
        .map(|i| eda_validate::lerp(b_axis, yb, t_lo + i as f64 * dt))
        .collect();

    // De-mean both — cross-correlation of sinusoids around a non-zero
    // mean is dominated by the DC term, which doesn't carry timing info.
    let mean_a = resampled_a.iter().sum::<f64>() / n as f64;
    let mean_b = resampled_b.iter().sum::<f64>() / n as f64;
    let zm_a: Vec<f64> = resampled_a.iter().map(|v| v - mean_a).collect();
    let zm_b: Vec<f64> = resampled_b.iter().map(|v| v - mean_b).collect();

    let max_lag = ((max_shift_seconds / dt).abs().ceil() as usize).min(n - 2);
    let (k, _) = best_lag(&zm_a, &zm_b, max_lag);

    // Parabolic refinement around the peak.
    let refined = parabolic_refine(&zm_a, &zm_b, k);
    Ok(refined * dt)
}

/// Sub-sample peak position via parabolic interpolation through the
/// three-point neighborhood around `k`.
fn parabolic_refine(a: &[f64], b: &[f64], k: i64) -> f64 {
    let kk = k as i64;
    let n = a.len() as i64;
    if kk.abs() >= n - 1 {
        return kk as f64;
    }
    let prod = |kp: i64| -> f64 {
        let (lo, hi) = if kp >= 0 { (0, n - kp) } else { (-kp, n) };
        let mut s = 0.0;
        for i in lo..hi {
            let av = a[i as usize];
            let bv = b[(i + kp) as usize];
            if av.is_finite() && bv.is_finite() {
                s += av * bv;
            }
        }
        s
    };
    let y0 = prod(kk - 1);
    let y1 = prod(kk);
    let y2 = prod(kk + 1);
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() < 1e-30 {
        return kk as f64;
    }
    let delta = 0.5 * (y0 - y2) / denom;
    // Guard against runaway when the parabola is nearly flat.
    let delta = delta.clamp(-1.0, 1.0);
    kk as f64 + delta
}

fn real_view(w: &Waveform) -> Result<(&Vec<f64>, &BTreeMap<String, Vec<f64>>), AlignError> {
    match w {
        Waveform::Real { axis, signals, .. } => Ok((axis, signals)),
        Waveform::Complex { .. } => Err(AlignError::NotReal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::f64::consts::PI;

    fn shifted_sine(axis: &[f64], shift: f64) -> Vec<f64> {
        axis.iter()
            .map(|&t| (2.0 * PI * 1e6 * (t - shift)).sin())
            .collect()
    }

    fn make_wave(name: &str, axis: Vec<f64>, samples: Vec<f64>) -> Waveform {
        let mut signals = BTreeMap::new();
        signals.insert(name.to_string(), samples);
        Waveform::Real {
            axis_name: "time".into(),
            axis,
            signals,
        }
    }

    #[test]
    fn best_lag_recovers_known_shift() {
        // Sine at 1 MHz sampled at 100 MHz, b shifted by +5 samples.
        let n = 1024;
        let dt = 1e-8;
        let axis: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let a = shifted_sine(&axis, 0.0);
        let b = shifted_sine(&axis, 5.0 * dt); // delayed by 5 samples
        // a(t) = b(t + 5·dt), so peak lag should be +5.
        let (k, _) = best_lag(&a, &b, 50);
        assert_eq!(k, 5);
    }

    #[test]
    fn align_waveforms_finds_subsample_shift() {
        // True shift of 3.7 samples → parabolic refinement on a
        // non-coherent sine recovers it within ~0.3 samples (parabolic
        // interp is exact only for an exact parabola; the autocorr of a
        // sinusoid is a cosine, which deviates from parabolic away
        // from the peak).
        let n = 1024;
        let dt = 1e-8;
        let axis: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let a_samples = shifted_sine(&axis, 0.0);
        let b_samples = shifted_sine(&axis, 3.7 * dt);
        let a = make_wave("v", axis.clone(), a_samples);
        let b = make_wave("v", axis, b_samples);
        let shift = align_waveforms(&a, &b, "v", 100.0 * dt).unwrap();
        let expected = 3.7 * dt;
        assert!(
            (shift - expected).abs() < 0.3 * dt,
            "got shift = {} (expected ~{})",
            shift,
            expected
        );
    }

    #[test]
    fn align_waveforms_zero_shift_for_identical() {
        let n = 256;
        let dt = 1e-9;
        let axis: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let s = shifted_sine(&axis, 0.0);
        let a = make_wave("v", axis.clone(), s.clone());
        let b = make_wave("v", axis, s);
        let shift = align_waveforms(&a, &b, "v", 50.0 * dt).unwrap();
        assert!(shift.abs() < 0.05 * dt, "shift = {}", shift);
    }

    #[test]
    fn align_rejects_complex() {
        let mut signals = BTreeMap::new();
        signals.insert("v".into(), vec![(1.0, 0.0)]);
        let a = Waveform::Complex {
            axis_name: "frequency".into(),
            axis: vec![1e3],
            signals: signals.clone(),
        };
        let b = a.clone();
        assert!(matches!(
            align_waveforms(&a, &b, "v", 1.0),
            Err(AlignError::NotReal)
        ));
    }

    #[test]
    fn align_errors_on_missing_signal() {
        let axis = vec![0.0, 1.0, 2.0, 3.0];
        let a = make_wave("x", axis.clone(), vec![0.0; 4]);
        let b = make_wave("y", axis, vec![0.0; 4]);
        assert!(matches!(
            align_waveforms(&a, &b, "x", 1.0),
            Err(AlignError::MissingSignal(_))
        ));
    }
}
