//! Numerical statistics on waveform sample slices.
//!
//! Trivial primitives, but they're used everywhere — from
//! [`crate::linearity`] (histogram normalization) through noise
//! characterization to one-line "what's the RMS ripple at the output"
//! style queries. Pulled out so each consumer doesn't reinvent them.
//!
//! ## NaN handling
//!
//! Every primitive **skips NaN samples** (matching numpy's `nan*`
//! family), and returns `None` when there are no non-NaN samples to
//! aggregate. This avoids accidentally poisoning a stat with a single
//! `x`/`z` bit that leaked through the VCD reader.

/// Arithmetic mean. `None` if all samples are NaN or the slice is empty.
pub fn mean(y: &[f64]) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;
    for &v in y {
        if v.is_finite() {
            sum += v;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some(sum / n as f64)
    }
}

/// Sample standard deviation (Bessel-corrected, divides by `n - 1`).
/// `None` if fewer than 2 finite samples.
pub fn std(y: &[f64]) -> Option<f64> {
    let m = mean(y)?;
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for &v in y {
        if v.is_finite() {
            let d = v - m;
            sum_sq += d * d;
            n += 1;
        }
    }
    if n < 2 {
        None
    } else {
        Some((sum_sq / (n - 1) as f64).sqrt())
    }
}

/// Root mean square. `None` if all samples are NaN.
pub fn rms(y: &[f64]) -> Option<f64> {
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for &v in y {
        if v.is_finite() {
            sum_sq += v * v;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some((sum_sq / n as f64).sqrt())
    }
}

/// `(min, max)` over finite samples. `None` if all NaN.
pub fn min_max(y: &[f64]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut n = 0usize;
    for &v in y {
        if v.is_finite() {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some((lo, hi))
    }
}

/// `max - min` over finite samples.
pub fn peak_to_peak(y: &[f64]) -> Option<f64> {
    min_max(y).map(|(lo, hi)| hi - lo)
}

/// Percentile (`p ∈ [0.0, 100.0]`) using linear interpolation between
/// ranks. `None` if all NaN. Mirrors numpy's `nanpercentile` with the
/// default linear method.
pub fn percentile(y: &[f64], p: f64) -> Option<f64> {
    assert!((0.0..=100.0).contains(&p), "percentile must be in [0, 100]");
    let mut sorted: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    // Numpy "linear" rank: rank = p/100 * (n - 1).
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = rank - lo as f64;
    Some(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn mean_skips_nan() {
        assert!(approx(mean(&[1.0, 2.0, 3.0]).unwrap(), 2.0, 1e-12));
        assert!(approx(
            mean(&[1.0, f64::NAN, 3.0]).unwrap(),
            2.0,
            1e-12
        ));
        assert!(mean(&[]).is_none());
        assert!(mean(&[f64::NAN, f64::NAN]).is_none());
    }

    #[test]
    fn std_matches_unbiased() {
        // [1, 2, 3, 4, 5], unbiased std = sqrt(2.5) ≈ 1.58113883…
        let s = std(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(approx(s, (2.5_f64).sqrt(), 1e-12));
    }

    #[test]
    fn std_requires_two_samples() {
        assert!(std(&[1.0]).is_none());
        assert!(std(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn rms_of_unit_sine_is_sqrt2_inv() {
        let n = 1024;
        let y: Vec<f64> =
            (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()).collect();
        let r = rms(&y).unwrap();
        assert!(approx(r, 1.0 / 2f64.sqrt(), 1e-3));
    }

    #[test]
    fn peak_to_peak_works() {
        assert_eq!(peak_to_peak(&[1.0, -2.0, 3.0, 0.5]).unwrap(), 5.0);
    }

    #[test]
    fn percentile_endpoints_and_median() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&y, 0.0).unwrap(), 1.0);
        assert_eq!(percentile(&y, 100.0).unwrap(), 5.0);
        assert_eq!(percentile(&y, 50.0).unwrap(), 3.0);
        assert!(approx(percentile(&y, 25.0).unwrap(), 2.0, 1e-12));
    }

    #[test]
    fn percentile_skips_nan() {
        let y = vec![1.0, f64::NAN, 3.0, 5.0];
        assert_eq!(percentile(&y, 50.0).unwrap(), 3.0);
    }
}
