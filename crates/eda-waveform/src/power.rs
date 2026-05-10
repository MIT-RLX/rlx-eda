//! Power and energy integration.
//!
//! Two stock numbers analog datasheets quote — average power and total
//! energy over a window — both fall out of trapezoidal integration of
//! `v · i`. This module is those two integrals as one-line calls.
//!
//! ## Conventions
//!
//! - `v` and `i` must be sampled on the same time axis as `t`. If
//!   they aren't (e.g. ngspice transient with internal subgrids),
//!   resample to a common axis first via `eda_validate::lerp` or
//!   `spectrum::resample_uniform`.
//! - Sign convention: positive `i` flowing into a positive-`v` node is
//!   power *delivered* to the node. Pass current with the sign you
//!   want; this module doesn't second-guess.
//! - NaN samples poison the segment they appear in (the whole pair is
//!   skipped). For analog waveforms this should never happen — surface
//!   it as a non-zero `n_skipped` if you want diagnostics.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PowerError {
    #[error("axis lengths differ (t={t}, v={v}, i={i})")]
    LengthMismatch { t: usize, v: usize, i: usize },
    #[error("at least 2 samples required")]
    TooShort,
    #[error("time axis must be ascending (idx {idx}: {a} >= {b})")]
    NonMonotonic { idx: usize, a: f64, b: f64 },
}

/// Total energy `∫ v(t)·i(t) dt` over the trace, by trapezoidal rule.
/// Result is in joules when `v` is volts and `i` is amps.
pub fn total_energy(t: &[f64], v: &[f64], i: &[f64]) -> Result<f64, PowerError> {
    if t.len() != v.len() || t.len() != i.len() {
        return Err(PowerError::LengthMismatch {
            t: t.len(),
            v: v.len(),
            i: i.len(),
        });
    }
    if t.len() < 2 {
        return Err(PowerError::TooShort);
    }
    let mut total = 0.0_f64;
    for k in 0..t.len() - 1 {
        let dt = t[k + 1] - t[k];
        if dt <= 0.0 {
            return Err(PowerError::NonMonotonic {
                idx: k + 1,
                a: t[k],
                b: t[k + 1],
            });
        }
        let p0 = v[k] * i[k];
        let p1 = v[k + 1] * i[k + 1];
        if !p0.is_finite() || !p1.is_finite() {
            continue;
        }
        total += 0.5 * (p0 + p1) * dt;
    }
    Ok(total)
}

/// Average power = `total_energy / (t_last - t_first)`. Watts when
/// `v` is volts and `i` is amps.
pub fn average_power(t: &[f64], v: &[f64], i: &[f64]) -> Result<f64, PowerError> {
    let energy = total_energy(t, v, i)?;
    let span = t[t.len() - 1] - t[0];
    if span <= 0.0 {
        return Err(PowerError::TooShort);
    }
    Ok(energy / span)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn constant_v_and_i_match_closed_form() {
        // V = 5V, I = 0.1A over 1ms → P = 0.5W, E = 0.5mJ.
        let n = 100;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 1e-5).collect(); // 0..1ms
        let v = vec![5.0; n];
        let i = vec![0.1; n];
        let energy = total_energy(&t, &v, &i).unwrap();
        let power = average_power(&t, &v, &i).unwrap();
        let span = t[n - 1] - t[0];
        assert!(approx(energy, 0.5 * span, 1e-12));
        assert!(approx(power, 0.5, 1e-9));
    }

    #[test]
    fn linearly_ramped_load_matches_analytical() {
        // V = 1V constant. I = 2t over [0, 1s]. ∫ V·I dt = ∫ 2t dt = 1.
        // average power = 1 / 1 = 1.
        let n = 1001;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let v = vec![1.0; n];
        let i: Vec<f64> = t.iter().map(|t| 2.0 * t).collect();
        let energy = total_energy(&t, &v, &i).unwrap();
        let power = average_power(&t, &v, &i).unwrap();
        assert!(approx(energy, 1.0, 1e-6));
        assert!(approx(power, 1.0, 1e-6));
    }

    #[test]
    fn signed_current_changes_sign() {
        // Current reverses sign halfway through; net energy = 0.
        let t: Vec<f64> = (0..=100).map(|i| i as f64 * 0.01).collect();
        let v = vec![1.0; 101];
        let i: Vec<f64> = t.iter().map(|t| if *t < 0.5 { 1.0 } else { -1.0 }).collect();
        let energy = total_energy(&t, &v, &i).unwrap();
        // Allow some tolerance from the trapezoidal rule's handling of
        // the discontinuity.
        assert!(energy.abs() < 0.02, "energy = {energy}");
    }

    #[test]
    fn rejects_length_mismatch() {
        let t = vec![0.0, 1.0];
        let v = vec![1.0, 1.0];
        let i = vec![1.0];
        assert!(matches!(
            total_energy(&t, &v, &i),
            Err(PowerError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_non_monotonic_time() {
        let t = vec![0.0, 1.0, 0.5];
        let v = vec![1.0; 3];
        let i = vec![1.0; 3];
        assert!(matches!(
            total_energy(&t, &v, &i),
            Err(PowerError::NonMonotonic { .. })
        ));
    }

    #[test]
    fn rejects_too_short() {
        let t = vec![0.0];
        let v = vec![1.0];
        let i = vec![1.0];
        assert!(matches!(
            total_energy(&t, &v, &i),
            Err(PowerError::TooShort)
        ));
    }
}
