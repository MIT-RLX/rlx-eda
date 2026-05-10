//! Static ADC linearity via the sine-histogram method.
//!
//! Pair this with [`crate::spectrum`] to get the standard "static +
//! dynamic" picture of an ADC:
//!
//! - **Spectrum** answers "how clean is the conversion?" — SNDR, ENOB,
//!   SFDR, THD from a single-tone capture.
//! - **Linearity** answers "is the transfer characteristic actually
//!   monotonic and evenly-spaced?" — DNL and INL per output code.
//!
//! Both feed off the same simulation: a long full-scale-sine run. The
//! captured ADC output codes get histogrammed here; the same sample
//! sequence (after one resample) goes into `spectrum::adc_metrics`.
//!
//! ## Theory
//!
//! For input `x = A·sin(θ)` with `θ` uniform on `[0, 2π)`, the PDF is
//! `f(x) = 1 / (π·√(A² - x²))`. The probability that `x` falls in
//! the voltage band assigned to code `k` integrates to a closed-form
//! arcsine difference. We compare the actual histogram against this
//! ideal:
//!
//! - `DNL[k] = h_actual[k] / h_ideal[k] - 1` (units: LSB)
//! - `INL[k] = Σ DNL[0..=k]` (running sum; "endpoint" INL is the
//!   caller's job to compute by subtracting a linear fit if desired).
//!
//! ## Edge codes
//!
//! Real captures with a full-scale-or-larger sine pile *all* the
//! out-of-range samples into codes `0` and `n_codes - 1`. Including
//! those codes in the histogram normalization wrecks the inner DNL.
//! We exclude them from analysis: their `dnl_lsb` and `inl_lsb` are
//! reported as `0.0`, and the headline `max_dnl` / `max_inl` only
//! consider inner codes.
//!
//! ## When the result lies
//!
//! - Capture too short → DNL has shot noise. Rule of thumb: at least
//!   2¹⁶ samples per code on average.
//! - Sine amplitude well below full-scale → end codes get few hits;
//!   normalization variance is high. Use slightly *over* full-scale
//!   so saturation pads the edge codes (which we ignore).
//! - Coherent sampling actually hurts the histogram — for sine-
//!   histogram you want *non*-coherent sampling so the input visits
//!   every voltage uniformly. The opposite of what `spectrum::adc_metrics`
//!   wants. Run two separate captures.

use std::f64::consts::PI;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LinearityError {
    #[error("n_codes must be at least 4 (got {0})")]
    TooFewCodes(usize),
    #[error("no samples provided")]
    Empty,
    #[error("all inner codes received zero hits — capture too short or wrong range")]
    AllInnerEmpty,
}

/// Result bundle for an INL/DNL run.
#[derive(Debug, Clone)]
pub struct LinearityResult {
    /// DNL per code, in LSB. Edge codes (`0` and `n_codes - 1`) are `0.0`.
    pub dnl_lsb: Vec<f64>,
    /// Cumulative INL per code, in LSB. Edge codes are `0.0`.
    pub inl_lsb: Vec<f64>,
    /// Worst-case |DNL| over inner codes.
    pub max_dnl: f64,
    /// Worst-case |INL| over inner codes.
    pub max_inl: f64,
    /// Sample count that landed in inner codes (used for normalization).
    pub n_inner: usize,
    /// Inner codes that received zero hits — strong sign of a missing
    /// code in the converter.
    pub missing_codes: Vec<usize>,
}

/// Bin a stream of fractional ADC outputs into integer-code histogram
/// bins. Values are rounded then clamped to `[0, n_codes - 1]`. NaN
/// samples are skipped.
pub fn sine_histogram(codes: &[f64], n_codes: usize) -> Vec<u64> {
    let mut h = vec![0u64; n_codes];
    let max = (n_codes - 1) as f64;
    for &c in codes {
        if c.is_nan() {
            continue;
        }
        let k = c.round().clamp(0.0, max) as usize;
        h[k] += 1;
    }
    h
}

/// DNL/INL from a captured-code stream, sine-histogram method.
///
/// `codes` is the sequence of integer codes the ADC produced for a
/// (long, full-scale-or-larger) sine input. Values are interpreted as
/// f64 for convenience (this is what falls out of a `Waveform` bus
/// signal); they're rounded before binning.
pub fn linearity_sine(codes: &[f64], n_codes: usize) -> Result<LinearityResult, LinearityError> {
    if n_codes < 4 {
        return Err(LinearityError::TooFewCodes(n_codes));
    }
    if codes.is_empty() {
        return Err(LinearityError::Empty);
    }
    let hist = sine_histogram(codes, n_codes);

    // Ideal arcsine probability per code. v_edge(k) = 2k/N - 1, scaled
    // so the input range is exactly [-1, 1] (i.e. a unit-amplitude sine
    // riding the full-scale rails). This matches *normalized* ADC
    // behavior; absolute scale cancels in the DNL ratio.
    let n = n_codes as f64;
    let p_ideal: Vec<f64> = (0..n_codes)
        .map(|k| {
            let v_lo = (2.0 * k as f64 / n - 1.0).clamp(-1.0, 1.0);
            let v_hi = (2.0 * (k + 1) as f64 / n - 1.0).clamp(-1.0, 1.0);
            (v_hi.asin() - v_lo.asin()) / PI
        })
        .collect();

    // Inner-code renormalization: drop k=0 and k=n_codes-1.
    let n_inner: u64 = hist[1..n_codes - 1].iter().sum();
    if n_inner == 0 {
        return Err(LinearityError::AllInnerEmpty);
    }
    let p_inner_total: f64 = p_ideal[1..n_codes - 1].iter().sum();

    let mut dnl = vec![0.0_f64; n_codes];
    let mut missing = Vec::new();
    for k in 1..n_codes - 1 {
        let p_ideal_norm = p_ideal[k] / p_inner_total;
        let h_norm = hist[k] as f64 / n_inner as f64;
        dnl[k] = h_norm / p_ideal_norm - 1.0;
        if hist[k] == 0 {
            missing.push(k);
        }
    }

    // INL via running sum, anchored at code 1 = 0.
    let mut inl = vec![0.0_f64; n_codes];
    let mut acc = 0.0;
    for k in 1..n_codes - 1 {
        acc += dnl[k];
        inl[k] = acc;
    }

    let max_dnl = dnl[1..n_codes - 1]
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);
    let max_inl = inl[1..n_codes - 1]
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);

    Ok(LinearityResult {
        dnl_lsb: dnl,
        inl_lsb: inl,
        max_dnl,
        max_inl,
        n_inner: n_inner as usize,
        missing_codes: missing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate a perfect N-bit ADC fed a sine of amplitude `amp` (in
    /// LSB units; full-scale = N/2) over `n_samples` non-coherent points.
    fn simulate_perfect_adc(n_codes: usize, amp_frac: f64, n_samples: usize) -> Vec<f64> {
        // Use an irrational frequency ratio so we don't fall into a
        // coherent loop. φ = 1/(π·1.234567) — anything irrational works.
        let step = 1.0 / (PI * 1.234567);
        let mut codes = Vec::with_capacity(n_samples);
        let n = n_codes as f64;
        for i in 0..n_samples {
            let theta = 2.0 * PI * (i as f64 * step);
            // Sine in [-1, 1] times amp_frac.
            let x = amp_frac * theta.sin();
            // Map x ∈ [-1, 1] to code ∈ [0, n_codes-1]: code = (x+1)/2 · n_codes.
            let code = ((x + 1.0) * 0.5 * n).floor().clamp(0.0, n - 1.0);
            codes.push(code);
        }
        codes
    }

    #[test]
    fn perfect_adc_has_near_zero_dnl_and_inl() {
        // 6-bit ADC, ~1M samples → enough to keep histogram noise small.
        let n_codes = 64;
        let codes = simulate_perfect_adc(n_codes, 1.0, 200_000);
        let r = linearity_sine(&codes, n_codes).unwrap();
        // With 200k samples / 64 codes ≈ 3k per code, statistical std
        // on DNL is ~1/√3000 ≈ 0.018 LSB; allow a generous envelope.
        assert!(
            r.max_dnl < 0.10,
            "max DNL = {:.3} LSB (expected < 0.10)",
            r.max_dnl
        );
        assert!(
            r.max_inl < 0.6,
            "max INL = {:.3} LSB (expected < 0.6)",
            r.max_inl
        );
        assert!(r.missing_codes.is_empty());
    }

    #[test]
    fn missing_code_shows_up_as_negative_dnl() {
        // Take the perfect sequence and remap every code 17 to code 16
        // → code 17 is missing, code 16 has double hits.
        let n_codes = 32;
        let mut codes = simulate_perfect_adc(n_codes, 1.0, 100_000);
        for c in codes.iter_mut() {
            if (*c - 17.0).abs() < 0.5 {
                *c = 16.0;
            }
        }
        let r = linearity_sine(&codes, n_codes).unwrap();
        assert!(r.missing_codes.contains(&17));
        // DNL[17] should be ≈ -1.0 (missing entirely).
        assert!(
            (r.dnl_lsb[17] + 1.0).abs() < 0.05,
            "DNL[17] = {:.3} (expected ~-1.0)",
            r.dnl_lsb[17]
        );
        // DNL[16] should be ≈ +1.0 (received its share + 17's share).
        assert!(
            (r.dnl_lsb[16] - 1.0).abs() < 0.1,
            "DNL[16] = {:.3} (expected ~+1.0)",
            r.dnl_lsb[16]
        );
    }

    #[test]
    fn sine_histogram_clamps_and_skips_nan() {
        let codes = vec![-2.0, -0.5, 0.0, 1.4, 5.0, f64::NAN, 7.0];
        let h = sine_histogram(&codes, 4);
        // -2.0 and -0.5 → 0; 0.0 → 0; 1.4 → 1; 5.0 → 3; 7.0 → 3; NaN → skipped.
        // So h[0] = 3, h[1] = 1, h[2] = 0, h[3] = 2.
        assert_eq!(h, vec![3, 1, 0, 2]);
    }

    #[test]
    fn rejects_tiny_n_codes() {
        let codes = vec![0.0, 1.0];
        assert!(matches!(
            linearity_sine(&codes, 2),
            Err(LinearityError::TooFewCodes(2))
        ));
    }

    #[test]
    fn rejects_empty_codes() {
        assert!(matches!(
            linearity_sine(&[], 16),
            Err(LinearityError::Empty)
        ));
    }
}
