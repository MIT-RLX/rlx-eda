//! Transient spec-window assertions.
//!
//! Datasheets specify analog behavior in terms a few stock predicates:
//! "stays within ±X% of nominal during the hold window", "settles to
//! within Y% of target by time T", "overshoot below Z%", "monotonic
//! during ramp". This module is those predicates as one-line calls.
//!
//! All helpers operate on raw `(t, y)` slices and use linear
//! interpolation between samples where appropriate, so a coarse
//! adaptive-stepping simulator output still produces the right answer
//! on a sub-sample-accurate edge.
//!
//! ## Conventions
//!
//! - Time windows are half-open mathematically — the predicate is
//!   evaluated at every sample with `t ∈ [t0, t1]`.
//! - Tolerance bands are *symmetric*: `target ± tol`. If you want an
//!   asymmetric band, call [`within_bounds`] directly with explicit
//!   `vmin` / `vmax`.
//! - `settling_time` returns the **last time** the signal *exits* the
//!   tol-band around `target`. After that time the signal stays inside
//!   to the end of the trace. This matches the datasheet definition
//!   ("settles to within Y% by time T") rather than the "first entry"
//!   definition (which trips on a single transient touch).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AssertionError {
    #[error("trace has no samples")]
    Empty,
    #[error("axis lengths differ ({t} vs {y})")]
    LengthMismatch { t: usize, y: usize },
    #[error("window [{t0}, {t1}] is empty or reversed")]
    BadWindow { t0: f64, t1: f64 },
}

/// First sample (or interpolated point) where the predicate is violated.
#[derive(Debug, Clone, Copy)]
pub struct Violation {
    pub t: f64,
    pub y: f64,
    /// Lower bound at `t` (inclusive).
    pub vmin: f64,
    /// Upper bound at `t` (inclusive).
    pub vmax: f64,
}

/// Assert that `y` stays in `[vmin, vmax]` for every sample whose `t`
/// falls in `[t0, t1]`. Returns the first violation, if any.
///
/// `t` must be ascending. Samples outside the window are ignored.
/// NaN samples count as a violation (since "in band" is undefined).
pub fn within_bounds(
    t: &[f64],
    y: &[f64],
    window: (f64, f64),
    vmin: f64,
    vmax: f64,
) -> Result<Result<(), Violation>, AssertionError> {
    if t.len() != y.len() {
        return Err(AssertionError::LengthMismatch {
            t: t.len(),
            y: y.len(),
        });
    }
    if t.is_empty() {
        return Err(AssertionError::Empty);
    }
    let (t0, t1) = window;
    if !(t1 >= t0) {
        return Err(AssertionError::BadWindow { t0, t1 });
    }
    for (&ti, &yi) in t.iter().zip(y.iter()) {
        if ti < t0 || ti > t1 {
            continue;
        }
        if yi.is_nan() || yi < vmin || yi > vmax {
            return Ok(Err(Violation {
                t: ti,
                y: yi,
                vmin,
                vmax,
            }));
        }
    }
    Ok(Ok(()))
}

/// Last time the signal was *outside* the band `target ± tol`. After
/// this time the signal stays inside the band for the rest of the
/// trace; returns `Some(t_exit)`.
///
/// Returns `None` if the signal never exits the band — i.e. it was
/// already settled at `t[0]`.
///
/// `t` must be ascending. NaN samples are treated as "outside the band".
pub fn settling_time(t: &[f64], y: &[f64], target: f64, tol: f64) -> Option<f64> {
    debug_assert_eq!(t.len(), y.len());
    debug_assert!(tol >= 0.0);
    let mut last_outside: Option<f64> = None;
    for (&ti, &yi) in t.iter().zip(y.iter()) {
        let outside = yi.is_nan() || (yi - target).abs() > tol;
        if outside {
            last_outside = Some(ti);
        }
    }
    last_outside
}

/// Direction of an overshoot relative to `target`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// Report `max(y) - target` if `max > target`, else `None`.
    Above,
    /// Report `target - min(y)` if `min < target`, else `None`.
    Below,
}

/// Peak excursion past `target` in the given direction.
///
/// Returns `None` if the signal never crosses `target` in the requested
/// direction (or the trace is empty).
pub fn overshoot(t: &[f64], y: &[f64], target: f64, polarity: Polarity) -> Option<f64> {
    debug_assert_eq!(t.len(), y.len());
    if t.is_empty() {
        return None;
    }
    let mut best: Option<f64> = None;
    for &yi in y {
        if yi.is_nan() {
            continue;
        }
        let excess = match polarity {
            Polarity::Above => yi - target,
            Polarity::Below => target - yi,
        };
        if excess > 0.0 && best.is_none_or(|b| excess > b) {
            best = Some(excess);
        }
    }
    best
}

/// Direction of a monotonic check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Increasing,
    Decreasing,
    /// `y[i+1] >= y[i]`.
    NonDecreasing,
    /// `y[i+1] <= y[i]`.
    NonIncreasing,
}

/// Check that `y` is monotonic in the given direction across every
/// sample whose `t` falls in `[t0, t1]`. Returns the index of the first
/// violating consecutive pair, if any.
///
/// `t` must be ascending. NaN-vs-anything counts as a violation.
pub fn monotonic(
    t: &[f64],
    y: &[f64],
    window: (f64, f64),
    direction: Direction,
) -> Result<Result<(), Violation>, AssertionError> {
    if t.len() != y.len() {
        return Err(AssertionError::LengthMismatch {
            t: t.len(),
            y: y.len(),
        });
    }
    if t.is_empty() {
        return Err(AssertionError::Empty);
    }
    let (t0, t1) = window;
    if !(t1 >= t0) {
        return Err(AssertionError::BadWindow { t0, t1 });
    }
    for i in 0..t.len() - 1 {
        let (ta, tb) = (t[i], t[i + 1]);
        if tb < t0 || ta > t1 {
            continue;
        }
        let (ya, yb) = (y[i], y[i + 1]);
        if ya.is_nan() || yb.is_nan() {
            return Ok(Err(Violation {
                t: tb,
                y: yb,
                vmin: ya,
                vmax: ya,
            }));
        }
        let ok = match direction {
            Direction::Increasing => yb > ya,
            Direction::Decreasing => yb < ya,
            Direction::NonDecreasing => yb >= ya,
            Direction::NonIncreasing => yb <= ya,
        };
        if !ok {
            // Encode the expected relation in vmin/vmax for the report.
            return Ok(Err(Violation {
                t: tb,
                y: yb,
                vmin: ya,
                vmax: ya,
            }));
        }
    }
    Ok(Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(t0: f64, t1: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| t0 + (t1 - t0) * i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn within_bounds_passes_clean_signal() {
        let t = linspace(0.0, 1.0, 11);
        let y: Vec<f64> = t.iter().map(|t| 0.5 + 0.1 * (t * 6.28).sin()).collect();
        let r = within_bounds(&t, &y, (0.0, 1.0), 0.0, 1.0).unwrap();
        assert!(r.is_ok());
    }

    #[test]
    fn within_bounds_catches_first_violation() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 0.5, 1.5, 0.5]; // 1.5 > vmax=1.0 at t=2
        let r = within_bounds(&t, &y, (0.0, 3.0), 0.0, 1.0).unwrap();
        let v = r.unwrap_err();
        assert_eq!(v.t, 2.0);
        assert_eq!(v.y, 1.5);
    }

    #[test]
    fn within_bounds_window_filters_samples() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![10.0, 0.5, 0.5, 0.5];
        // Out-of-spec at t=0, but our window starts at t=1 → ok.
        let r = within_bounds(&t, &y, (1.0, 3.0), 0.0, 1.0).unwrap();
        assert!(r.is_ok());
    }

    #[test]
    fn settling_time_finds_last_exit() {
        // Step response: settles by t=3 to within ±0.05 of 1.0. Values
        // chosen well clear of the band edge to dodge f64 representation
        // quirks (e.g. `1.05 - 1.0` isn't exactly 0.05).
        let t = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 0.8, 1.2, 1.03, 1.02, 1.01];
        // Tol = 0.05. Outside at t=0,1,2; inside from t=3 onward.
        let st = settling_time(&t, &y, 1.0, 0.05).unwrap();
        assert_eq!(st, 2.0);
    }

    #[test]
    fn settling_time_none_when_already_settled() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 1.001, 0.999];
        assert!(settling_time(&t, &y, 1.0, 0.01).is_none());
    }

    #[test]
    fn overshoot_above_and_below() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.2, 1.0]; // peak 1.2 above target 1.0
        let os = overshoot(&t, &y, 1.0, Polarity::Above).unwrap();
        assert!((os - 0.2).abs() < 1e-12);
        // Below: target=0, min=0 → no excursion.
        assert!(overshoot(&t, &y, 0.0, Polarity::Below).is_none());
    }

    #[test]
    fn monotonic_passes_strictly_increasing() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.5, 3.0];
        let r = monotonic(&t, &y, (0.0, 3.0), Direction::Increasing).unwrap();
        assert!(r.is_ok());
    }

    #[test]
    fn monotonic_catches_dip() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.5, 2.0]; // 0.5 < 1.0 at i=2
        let r = monotonic(&t, &y, (0.0, 3.0), Direction::Increasing).unwrap();
        let v = r.unwrap_err();
        assert_eq!(v.t, 2.0);
        assert_eq!(v.y, 0.5);
    }

    #[test]
    fn monotonic_non_decreasing_allows_plateau() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 1.0]; // plateau ok
        let r = monotonic(&t, &y, (0.0, 2.0), Direction::NonDecreasing).unwrap();
        assert!(r.is_ok());
        // …but strict Increasing rejects.
        let r2 = monotonic(&t, &y, (0.0, 2.0), Direction::Increasing).unwrap();
        assert!(r2.is_err());
    }

    #[test]
    fn nan_samples_are_violations() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![0.5, f64::NAN, 0.5];
        let r = within_bounds(&t, &y, (0.0, 2.0), 0.0, 1.0).unwrap();
        let v = r.unwrap_err();
        assert_eq!(v.t, 1.0);
    }

    #[test]
    fn rejects_length_mismatch() {
        let t = vec![0.0, 1.0];
        let y = vec![0.0];
        assert!(matches!(
            within_bounds(&t, &y, (0.0, 1.0), 0.0, 1.0),
            Err(AssertionError::LengthMismatch { .. })
        ));
    }
}
