//! Spectral analysis for ADC / DAC verification.
//!
//! For a data converter, the headline number is **ENOB** — the
//! effective number of bits, derived from the signal-to-noise-and-
//! distortion ratio (SNDR) of a single-tone FFT. Adjacent figures of
//! merit are SFDR (largest spur) and THD (sum of harmonic powers).
//!
//! ## Pipeline
//!
//! 1. SPICE transient produces an analog or code stream on an
//!    LTE-controlled adaptive grid → call [`resample_uniform`] to land
//!    on a fixed Δt.
//! 2. Choose a power-of-two `N` and a fundamental bin `k` with
//!    `gcd(k, N) = 1` → coherent sampling, no spectral leakage.
//! 3. Call [`adc_metrics`] with `Window::Rectangular`. For
//!    non-coherent sampling, pick `BlackmanHarris` instead and accept
//!    the wider skirt.
//!
//! ## Conventions
//!
//! - One-sided power spectrum, normalized so a unit-amplitude sine at
//!   bin `k` yields total signal power `0.5` (i.e. RMS²).
//! - Powers are linear (V²); ratios are reported in dB.
//! - SFDR includes harmonic bins as candidate spurs — a fat 3rd
//!   harmonic *is* the SFDR floor for many converters.
//! - DC (bin 0) is excluded from signal/noise/spur accounting.
//!
//! ## Why hand-rolled FFT
//!
//! The verification flow is dominated by the SPICE run; FFT cost is
//! negligible. A 100-line radix-2 keeps the dep tree small. If we ever
//! need non-power-of-two lengths, swap in Bluestein or pull `rustfft`.

use std::f64::consts::PI;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpectrumError {
    #[error("FFT length must be a power of 2 (got {0})")]
    NotPow2(usize),
    #[error("at least 2 samples required (got {0})")]
    TooShort(usize),
    #[error("re/im length mismatch ({re} vs {im})")]
    LengthMismatch { re: usize, im: usize },
    #[error("signal bin {bin} out of one-sided range [1, {max}]")]
    BinOutOfRange { bin: usize, max: usize },
    #[error("overlap fraction must be in [0, 1) (got {0})")]
    OverlapOutOfRange(f64),
}

/// Tapering window applied before the FFT.
///
/// Use `Rectangular` when the input is coherently sampled (the
/// fundamental sits exactly on a bin); use `BlackmanHarris` for
/// general inputs and accept ~3 bins of leakage.
#[derive(Debug, Clone, Copy)]
pub enum Window {
    Rectangular,
    Hann,
    /// 4-term Blackman-Harris.
    BlackmanHarris,
    /// 5-term flat-top — for amplitude-accurate (non-coherent) measurements.
    FlatTop,
}

impl Window {
    /// Number of bins on either side of a tone to count as "in-signal"
    /// when totaling power. Wider windows leak more.
    fn skirt(self) -> usize {
        match self {
            Window::Rectangular => 0,
            Window::Hann => 2,
            Window::BlackmanHarris => 3,
            Window::FlatTop => 4,
        }
    }

    fn coef(self, n: usize, len: usize) -> f64 {
        let m = (len - 1) as f64;
        let x = 2.0 * PI * n as f64 / m;
        match self {
            Window::Rectangular => 1.0,
            Window::Hann => 0.5 - 0.5 * x.cos(),
            Window::BlackmanHarris => {
                0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos()
                    - 0.01168 * (3.0 * x).cos()
            }
            Window::FlatTop => {
                0.21557895 - 0.41663158 * x.cos() + 0.277263158 * (2.0 * x).cos()
                    - 0.083578947 * (3.0 * x).cos()
                    + 0.006947368 * (4.0 * x).cos()
            }
        }
    }
}

/// Multiply each sample by the window taper.
pub fn apply_window(samples: &[f64], window: Window) -> Vec<f64> {
    let n = samples.len();
    samples
        .iter()
        .enumerate()
        .map(|(i, &s)| s * window.coef(i, n))
        .collect()
}

/// In-place radix-2 forward FFT. `re.len()` must equal `im.len()` and
/// be a power of 2.
pub fn fft_pow2(re: &mut [f64], im: &mut [f64]) -> Result<(), SpectrumError> {
    if re.len() != im.len() {
        return Err(SpectrumError::LengthMismatch {
            re: re.len(),
            im: im.len(),
        });
    }
    let n = re.len();
    if n < 2 {
        return Err(SpectrumError::TooShort(n));
    }
    if !n.is_power_of_two() {
        return Err(SpectrumError::NotPow2(n));
    }

    // Bit-reverse permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Cooley-Tukey butterflies.
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let theta = -2.0 * PI / size as f64;
        let w_step_r = theta.cos();
        let w_step_i = theta.sin();
        let mut block = 0;
        while block < n {
            let mut wr = 1.0_f64;
            let mut wi = 0.0_f64;
            for k in 0..half {
                let a = block + k;
                let b = a + half;
                let xr = re[b] * wr - im[b] * wi;
                let xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
                let new_wr = wr * w_step_r - wi * w_step_i;
                let new_wi = wr * w_step_i + wi * w_step_r;
                wr = new_wr;
                wi = new_wi;
            }
            block += size;
        }
        size <<= 1;
    }
    Ok(())
}

/// Resample an irregularly-sampled `(t, y)` trace onto a uniform grid
/// of `n` points spanning `[t[0], t[last]]`, via linear interpolation.
///
/// Returns `(dt, samples)` where `dt = (t_last - t_0) / (n - 1)` and
/// `samples.len() == n`. The sampling rate `fs = 1.0 / dt`.
pub fn resample_uniform(t: &[f64], y: &[f64], n: usize) -> (f64, Vec<f64>) {
    assert_eq!(t.len(), y.len(), "t and y length mismatch");
    assert!(t.len() >= 2, "need at least 2 samples to resample");
    assert!(n >= 2, "need at least 2 output samples");
    let t0 = t[0];
    let t1 = t[t.len() - 1];
    let dt = (t1 - t0) / (n - 1) as f64;
    let samples: Vec<f64> = (0..n)
        .map(|i| eda_validate::lerp(t, y, t0 + i as f64 * dt))
        .collect();
    (dt, samples)
}

/// One-sided power spectrum (linear V²), length `n/2 + 1`.
///
/// Normalization: a unit-amplitude sine at bin `k` (coherent, rectangular
/// window) puts power `0.5` in bin `k` — matching the time-domain RMS².
pub fn power_spectrum(samples: &[f64], window: Window) -> Result<Vec<f64>, SpectrumError> {
    let n = samples.len();
    if !n.is_power_of_two() {
        return Err(SpectrumError::NotPow2(n));
    }
    if n < 2 {
        return Err(SpectrumError::TooShort(n));
    }

    // Window normalization: divide by the coherent gain so the bin
    // amplitude reflects the original signal amplitude rather than the
    // attenuated windowed amplitude.
    let cg = (0..n).map(|i| window.coef(i, n)).sum::<f64>() / n as f64;
    let cg = if cg.abs() < 1e-30 { 1.0 } else { cg };

    let mut re: Vec<f64> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| s * window.coef(i, n) / cg)
        .collect();
    let mut im = vec![0.0_f64; n];
    fft_pow2(&mut re, &mut im)?;

    let inv_n2 = 1.0 / (n as f64).powi(2);
    let mut out = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n / 2 {
        let mag2 = (re[k] * re[k] + im[k] * im[k]) * inv_n2;
        // Double interior bins to fold negative frequencies onto the
        // one-sided spectrum.
        let factor = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
        out.push(mag2 * factor);
    }
    Ok(out)
}

/// Options for [`welch_psd`]. Built fluent-style so callers don't have
/// to spell out every knob.
#[derive(Debug, Clone, Copy)]
pub struct WelchOptions {
    /// Segment length (must be a power of 2). Defaults to the largest
    /// power of two ≤ `samples.len() / 8` — eight non-overlapping
    /// segments at default 50% overlap give roughly 15 averages, which
    /// is the scipy.signal.welch default flavor.
    pub segment_len: Option<usize>,
    /// Overlap fraction in `[0, 1)`. Default `0.5`.
    pub overlap_frac: f64,
    /// Window. Default `Hann`.
    pub window: Window,
}

impl Default for WelchOptions {
    fn default() -> Self {
        Self {
            segment_len: None,
            overlap_frac: 0.5,
            window: Window::Hann,
        }
    }
}

impl WelchOptions {
    pub fn with_segment_len(mut self, n: usize) -> Self {
        self.segment_len = Some(n);
        self
    }
    pub fn with_overlap(mut self, f: f64) -> Self {
        self.overlap_frac = f;
        self
    }
    pub fn with_window(mut self, w: Window) -> Self {
        self.window = w;
        self
    }
}

/// Welch's averaged-periodogram power spectral density.
///
/// Returns `Vec<(freq_hz, psd_v2_per_hz)>` of length `L/2 + 1`, where
/// `L` is the segment length. The PSD is normalized so that
/// `∫₀^{fs/2} PSD(f) df ≈ Var(samples)` for stationary input — i.e. it's
/// a one-sided density in V²/Hz.
///
/// For a coherent single tone you generally want [`adc_metrics`] instead;
/// Welch trades a stable noise-floor estimate for spectral leakage on
/// the tone (Hann smears it across ~3 bins).
pub fn welch_psd(
    samples: &[f64],
    fs: f64,
    opts: WelchOptions,
) -> Result<Vec<(f64, f64)>, SpectrumError> {
    let n = samples.len();
    let l = opts.segment_len.unwrap_or_else(|| {
        let target = (n / 8).max(8);
        // Largest power of two ≤ target.
        1usize << (usize::BITS - 1 - target.leading_zeros())
    });
    if !l.is_power_of_two() {
        return Err(SpectrumError::NotPow2(l));
    }
    if l < 4 {
        return Err(SpectrumError::TooShort(l));
    }
    if l > n {
        return Err(SpectrumError::TooShort(n));
    }
    if !(0.0..1.0).contains(&opts.overlap_frac) {
        return Err(SpectrumError::OverlapOutOfRange(opts.overlap_frac));
    }

    let step = (((1.0 - opts.overlap_frac) * l as f64).round() as usize).max(1);
    let n_segs = if n >= l { (n - l) / step + 1 } else { 0 };
    if n_segs == 0 {
        return Err(SpectrumError::TooShort(n));
    }

    // PSD normalization: 1 / (fs · Σ w²). The window-energy term replaces
    // the FFT's `1/N` with the noise-equivalent bandwidth correction.
    let w_sq_sum: f64 = (0..l)
        .map(|i| {
            let v = opts.window.coef(i, l);
            v * v
        })
        .sum();
    let norm = 1.0 / (fs * w_sq_sum);

    let n_bins = l / 2 + 1;
    let mut accum = vec![0.0_f64; n_bins];

    for seg_idx in 0..n_segs {
        let start = seg_idx * step;
        let mut re: Vec<f64> = (0..l)
            .map(|i| samples[start + i] * opts.window.coef(i, l))
            .collect();
        let mut im = vec![0.0_f64; l];
        fft_pow2(&mut re, &mut im)?;
        for (k, slot) in accum.iter_mut().enumerate() {
            let mag2 = re[k] * re[k] + im[k] * im[k];
            let factor = if k == 0 || k == l / 2 { 1.0 } else { 2.0 };
            *slot += mag2 * factor;
        }
    }

    let inv_segs = 1.0 / n_segs as f64;
    let df = fs / l as f64;
    Ok((0..n_bins)
        .map(|k| (k as f64 * df, accum[k] * norm * inv_segs))
        .collect())
}

/// Headline ADC figures of merit derived from the FFT of a single-tone
/// capture.
#[derive(Debug, Clone, Copy)]
pub struct AdcMetrics {
    /// Bin of the fundamental (echoed back for convenience).
    pub signal_bin: usize,
    /// Total power in the fundamental skirt (linear V²).
    pub signal_power: f64,
    /// Signal-to-noise-and-distortion ratio in dB.
    pub sndr_db: f64,
    /// Effective number of bits = (SNDR - 1.76) / 6.02.
    pub enob: f64,
    /// Spurious-free dynamic range in dB: ratio of fundamental peak
    /// to largest other bin (harmonics included).
    pub sfdr_db: f64,
    /// Total harmonic distortion in dB: ratio of summed harmonic
    /// power to fundamental power. Negative — closer to `-∞` is better.
    pub thd_db: f64,
}

/// Compute ENOB / SNDR / SFDR / THD from a uniformly-sampled, single-tone
/// capture.
///
/// - `signal_bin` is the bin of the fundamental in the one-sided spectrum.
///   For coherent sampling pick `signal_bin` such that `gcd(signal_bin, N)
///   = 1` — that's how you avoid leakage.
/// - `harmonics` is how many harmonic orders to include in THD (typically
///   `5` to `7`). Harmonic bins are alias-folded into the Nyquist range.
/// - `window` should be `Rectangular` for coherent inputs, `BlackmanHarris`
///   otherwise.
pub fn adc_metrics(
    samples: &[f64],
    signal_bin: usize,
    harmonics: usize,
    window: Window,
) -> Result<AdcMetrics, SpectrumError> {
    let ps = power_spectrum(samples, window)?;
    let n_bins = ps.len();
    let nyq = n_bins - 1;
    if signal_bin == 0 || signal_bin >= n_bins {
        return Err(SpectrumError::BinOutOfRange {
            bin: signal_bin,
            max: nyq,
        });
    }
    let skirt = window.skirt();

    // Helper: indices in [signal_bin - skirt, signal_bin + skirt] ∩ [1, nyq].
    let signal_band: Vec<usize> = ((signal_bin.saturating_sub(skirt)).max(1)
        ..=(signal_bin + skirt).min(nyq))
        .collect();
    let signal_power: f64 = signal_band.iter().map(|&k| ps[k]).sum();

    // Harmonic bins — fold each multiple back into the Nyquist range.
    let mut harmonic_bins: Vec<usize> = Vec::with_capacity(harmonics);
    for h in 2..=(harmonics + 1) {
        let folded = fold_bin(signal_bin * h, nyq);
        if folded != 0 && folded != nyq && !signal_band.contains(&folded) {
            harmonic_bins.push(folded);
        }
    }
    let mut harmonic_power = 0.0_f64;
    let mut harmonic_band: Vec<usize> = Vec::new();
    for &hb in &harmonic_bins {
        let lo = hb.saturating_sub(skirt).max(1);
        let hi = (hb + skirt).min(nyq);
        for k in lo..=hi {
            if !signal_band.contains(&k) && !harmonic_band.contains(&k) {
                harmonic_power += ps[k];
                harmonic_band.push(k);
            }
        }
    }

    // Noise: everything that isn't DC, signal, or harmonic.
    let total_above_dc: f64 = ps[1..].iter().sum();
    let noise_power = (total_above_dc - signal_power - harmonic_power).max(0.0);

    let denom = noise_power + harmonic_power;
    let sndr_db = if denom > 0.0 && signal_power > 0.0 {
        10.0 * (signal_power / denom).log10()
    } else {
        f64::INFINITY
    };
    let enob = (sndr_db - 1.76) / 6.02;

    let thd_db = if harmonic_power > 0.0 && signal_power > 0.0 {
        10.0 * (harmonic_power / signal_power).log10()
    } else {
        f64::NEG_INFINITY
    };

    // SFDR: largest non-signal bin (harmonics included) vs signal peak.
    let signal_peak: f64 = signal_band.iter().map(|&k| ps[k]).fold(0.0, f64::max);
    let mut max_spur = 0.0_f64;
    for (k, &p) in ps.iter().enumerate() {
        if k == 0 || signal_band.contains(&k) {
            continue;
        }
        if p > max_spur {
            max_spur = p;
        }
    }
    let sfdr_db = if max_spur > 0.0 && signal_peak > 0.0 {
        10.0 * (signal_peak / max_spur).log10()
    } else {
        f64::INFINITY
    };

    Ok(AdcMetrics {
        signal_bin,
        signal_power,
        sndr_db,
        enob,
        sfdr_db,
        thd_db,
    })
}

/// Pick an FFT bin near `target` that's **coprime** with `fft_len` —
/// the condition for coherent (no-leakage) sampling of a single tone.
///
/// For `fft_len = 2^m`, coprime ⇔ odd, so this always lands within ±1
/// of the target. For composite lengths it may step further. Search is
/// constrained to the open Nyquist range `(0, fft_len/2)`.
///
/// Returns `None` if `fft_len < 4` or the search exhausted without
/// finding a coprime bin (only possible for pathologically small
/// lengths).
pub fn pick_coherent_bin(fft_len: usize, target: usize) -> Option<usize> {
    if fft_len < 4 {
        return None;
    }
    let nyq = fft_len / 2;
    if target == 0 || target >= nyq {
        return None;
    }
    // Step outward from target: 0, +1, -1, +2, -2, …
    for offset in 0..nyq {
        for &sign in &[1i64, -1] {
            if offset == 0 && sign == -1 {
                continue;
            }
            let candidate = target as i64 + sign * offset as i64;
            if candidate <= 0 || candidate >= nyq as i64 {
                continue;
            }
            let c = candidate as usize;
            if gcd(c, fft_len) == 1 {
                return Some(c);
            }
        }
    }
    None
}

/// Pick a coherent input-tone frequency near `target_fin` for a sweep
/// of length `fft_len` at sampling rate `fs`.
///
/// Returns `(bin, fin_actual)` where `fin_actual = bin * fs / fft_len`.
/// Use `bin` directly with [`adc_metrics`].
pub fn pick_coherent_freq(
    fs: f64,
    fft_len: usize,
    target_fin: f64,
) -> Option<(usize, f64)> {
    let target_bin = (target_fin * fft_len as f64 / fs).round() as i64;
    if target_bin <= 0 {
        return None;
    }
    let bin = pick_coherent_bin(fft_len, target_bin as usize)?;
    Some((bin, bin as f64 * fs / fft_len as f64))
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Integrate a one-sided PSD curve over `[f_lo, f_hi]` and return the
/// RMS voltage that integrated power corresponds to: `√∫ PSD df`.
///
/// Use this with a [`welch_psd`] result to get the headline datasheet
/// number — "X µV RMS noise from 1 Hz to 10 kHz" — without
/// hand-rolling the trapezoidal sum.
///
/// `psd` is `Vec<(freq_hz, v2_per_hz)>` with strictly ascending
/// frequencies (which is what `welch_psd` returns). Bins straddling
/// the band edges are linearly interpolated. Returns `0.0` if the band
/// doesn't intersect the PSD's frequency range.
pub fn integrate_psd(psd: &[(f64, f64)], f_lo: f64, f_hi: f64) -> f64 {
    assert!(f_hi >= f_lo, "integrate_psd: f_hi must be >= f_lo");
    if psd.len() < 2 || f_hi == f_lo {
        return 0.0;
    }
    let mut power = 0.0_f64;
    for w in psd.windows(2) {
        let (f0, p0) = w[0];
        let (f1, p1) = w[1];
        // Skip segments entirely outside the band.
        if f1 < f_lo || f0 > f_hi {
            continue;
        }
        // Clip the segment to the band.
        let a = f0.max(f_lo);
        let b = f1.min(f_hi);
        if b <= a {
            continue;
        }
        // Linear interpolation of PSD at the clipped endpoints.
        let denom = f1 - f0;
        let pa = if denom > 0.0 {
            p0 + (p1 - p0) * (a - f0) / denom
        } else {
            p0
        };
        let pb = if denom > 0.0 {
            p0 + (p1 - p0) * (b - f0) / denom
        } else {
            p1
        };
        power += 0.5 * (pa + pb) * (b - a);
    }
    power.max(0.0).sqrt()
}

/// Map a (possibly above-Nyquist) bin index into the one-sided spectrum
/// via the `f → fs - f` aliasing rule. `nyq = N/2`.
fn fold_bin(raw: usize, nyq: usize) -> usize {
    let n = nyq * 2; // FFT length
    let folded = raw % n;
    if folded > nyq {
        n - folded
    } else {
        folded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate `N` samples of `A * sin(2π k n / N)`.
    fn coherent_sine(n: usize, k: usize, amp: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * k as f64 * i as f64 / n as f64).sin())
            .collect()
    }

    /// Quantize to `bits` (mid-tread, [-1, 1] full scale).
    fn quantize(x: &[f64], bits: u32) -> Vec<f64> {
        let levels = (1u64 << bits) as f64;
        let lsb = 2.0 / levels;
        x.iter()
            .map(|&v| (v / lsb).round() * lsb)
            .collect()
    }

    #[test]
    fn fft_recovers_sine_at_bin() {
        // Coherent sine at bin 7 in N=64 → power_spectrum has all energy
        // in bin 7; rectangular window means no leakage.
        let n = 64;
        let k = 7;
        let x = coherent_sine(n, k, 1.0);
        let ps = power_spectrum(&x, Window::Rectangular).unwrap();
        // Total power should be ≈ 0.5 (= A²/2 for unit-amplitude sine).
        let total: f64 = ps.iter().sum();
        assert!((total - 0.5).abs() < 1e-9);
        // All of it concentrated at bin 7.
        assert!((ps[k] - 0.5).abs() < 1e-9);
        for (i, &p) in ps.iter().enumerate() {
            if i != k {
                assert!(p < 1e-12, "leakage at bin {i}: {p:.3e}");
            }
        }
    }

    #[test]
    fn fft_round_trip_via_inverse_is_identity_in_magnitude() {
        // FFT on real input → check Parseval: |X|² sum equals |x|² sum.
        let n = 32;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin() + 0.5).collect();
        let mut re = x.clone();
        let mut im = vec![0.0; n];
        fft_pow2(&mut re, &mut im).unwrap();
        let energy_freq: f64 = re.iter().zip(&im).map(|(r, i)| r * r + i * i).sum::<f64>()
            / n as f64;
        let energy_time: f64 = x.iter().map(|v| v * v).sum();
        assert!(
            (energy_freq - energy_time).abs() < 1e-9,
            "Parseval violation: freq={energy_freq:.6} time={energy_time:.6}"
        );
    }

    #[test]
    fn enob_matches_textbook_formula_for_quantized_sine() {
        // 12-bit quantized coherent sine → ENOB should be very close to 12.
        // Theoretical SNDR = 6.02·B + 1.76 dB → ENOB = B exactly.
        // Finite N + bin-coherent quantization noise gives a slight
        // deviation; allow ±0.5 bits.
        let n = 4096;
        let k = 137; // prime to N=4096 ⇒ coherent
        let bits = 12;
        let x = coherent_sine(n, k, 0.99); // just under full-scale to avoid clipping
        let xq = quantize(&x, bits);
        let m = adc_metrics(&xq, k, 5, Window::Rectangular).unwrap();
        assert!(
            (m.enob - bits as f64).abs() < 0.5,
            "ENOB = {:.3}, expected ~{}, SNDR = {:.2} dB",
            m.enob,
            bits,
            m.sndr_db
        );
    }

    #[test]
    fn ideal_sine_has_extreme_sndr() {
        // Unquantized coherent sine: SNDR should be enormous (limited
        // by f64 round-off in the FFT, not real noise).
        let n = 1024;
        let k = 31;
        let x = coherent_sine(n, k, 0.5);
        let m = adc_metrics(&x, k, 5, Window::Rectangular).unwrap();
        assert!(m.sndr_db > 200.0, "got SNDR = {} dB", m.sndr_db);
        assert!(m.enob > 30.0, "got ENOB = {}", m.enob);
    }

    #[test]
    fn third_harmonic_dominates_sfdr() {
        // Sine + a deliberately injected 3rd harmonic at -40 dBc.
        let n = 1024;
        let k = 11;
        let mut x = coherent_sine(n, k, 1.0);
        let h3 = coherent_sine(n, 3 * k, 0.01); // 0.01 / 1.0 → -40 dB amplitude → -40 dBc
        for (xi, hi) in x.iter_mut().zip(h3.iter()) {
            *xi += hi;
        }
        let m = adc_metrics(&x, k, 5, Window::Rectangular).unwrap();
        // SFDR should be ~40 dB, dominated by the 3rd harmonic.
        assert!(
            (m.sfdr_db - 40.0).abs() < 1.0,
            "SFDR = {:.3} dB (expected ~40)",
            m.sfdr_db
        );
        // THD should also be ~-40 dB.
        assert!(
            (m.thd_db + 40.0).abs() < 1.0,
            "THD = {:.3} dB (expected ~-40)",
            m.thd_db
        );
    }

    #[test]
    fn fold_bin_aliases_correctly() {
        // N=16, Nyquist=8. Bin 9 folds to 7, bin 17 folds to 1.
        assert_eq!(fold_bin(9, 8), 7);
        assert_eq!(fold_bin(17, 8), 1);
        assert_eq!(fold_bin(8, 8), 8);
        assert_eq!(fold_bin(0, 8), 0);
    }

    #[test]
    fn fft_rejects_non_pow2() {
        let mut re = vec![0.0; 6];
        let mut im = vec![0.0; 6];
        assert!(matches!(
            fft_pow2(&mut re, &mut im),
            Err(SpectrumError::NotPow2(6))
        ));
    }

    #[test]
    fn pick_coherent_bin_finds_odd_neighbor_for_pow2() {
        // FFT length 1024 — coprime ⇔ odd. Target 100 → adjacent odd, 101.
        let bin = pick_coherent_bin(1024, 100).unwrap();
        assert_eq!(bin, 101);
        // Already coprime → unchanged.
        assert_eq!(pick_coherent_bin(1024, 137), Some(137));
        // Edge case: target on Nyquist boundary → None.
        assert_eq!(pick_coherent_bin(1024, 512), None);
        assert_eq!(pick_coherent_bin(1024, 0), None);
    }

    #[test]
    fn pick_coherent_bin_handles_composite_length() {
        // FFT length 12, target 4 (gcd 4). 4±1 → 5 or 3, both coprime.
        // Algorithm tries +1 first → 5.
        assert_eq!(pick_coherent_bin(12, 4), Some(5));
    }

    #[test]
    fn pick_coherent_freq_round_trips_to_bin() {
        let fs = 1e9; // 1 GS/s
        let n = 4096;
        // Target 10 MHz: target bin = round(10e6 · 4096 / 1e9) = 41 (coprime).
        let (bin, fin) = pick_coherent_freq(fs, n, 10e6).unwrap();
        assert_eq!(bin, 41);
        assert!((fin - 41.0 * fs / n as f64).abs() < 1e-6);
    }

    #[test]
    fn pick_coherent_pipeline_yields_pure_tone() {
        // Use the picker to set up a sine, then verify spectrum::adc_metrics
        // reports near-infinite SNDR (pure float-arithmetic noise floor).
        let n = 1024;
        let target = 73_usize; // any non-coprime would leak
        let bin = pick_coherent_bin(n, target).unwrap();
        let x = coherent_sine(n, bin, 0.5);
        let m = adc_metrics(&x, bin, 5, Window::Rectangular).unwrap();
        assert!(m.sndr_db > 200.0, "got SNDR = {} dB", m.sndr_db);
    }

    /// Deterministic pseudo-random generator (xorshift) so the white-noise
    /// PSD test stays reproducible without pulling in `rand`.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> f64 {
            // xorshift64 → uniform → standard-normal via Box-Muller.
            let mut s = self.0;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            self.0 = s;
            // Normalize to (0, 1].
            ((s as u32) as f64 + 1.0) / (u32::MAX as f64 + 2.0)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.next();
            let u2 = self.next();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }

    #[test]
    fn welch_psd_of_white_noise_is_flat_at_2_over_fs() {
        // Unit-variance gaussian → one-sided PSD = 2/fs (constant over freq).
        let fs = 1.0;
        let n = 16384;
        let mut rng = XorShift(0xdead_beef);
        let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let psd = welch_psd(&x, fs, WelchOptions::default()).unwrap();
        // Average over interior bins (drop DC and Nyquist where edge
        // effects + DC removal hit hardest).
        let interior: Vec<f64> = psd[1..psd.len() - 1].iter().map(|(_, p)| *p).collect();
        let mean = interior.iter().sum::<f64>() / interior.len() as f64;
        // Expected mean ≈ 2/fs = 2.0. Welch on Hann + 50% overlap leaves
        // a few % bias in the limit; the std-of-mean dominates the
        // tolerance for reasonable n. ±15 % envelope.
        assert!(
            (mean - 2.0).abs() < 0.3,
            "got mean PSD = {:.4} (expected ~2.0)",
            mean
        );
    }

    #[test]
    fn welch_psd_resolves_tone_above_floor() {
        // White noise + small sine. Welch should show a peak in the
        // sine's bin well above the surrounding floor.
        let fs = 1000.0;
        let n = 8192;
        let l = 1024;
        let f_tone = 100.0; // Hz
        let mut rng = XorShift(0xabad_cafe);
        let x: Vec<f64> = (0..n)
            .map(|i| {
                0.05 * rng.normal()
                    + 1.0 * (2.0 * PI * f_tone * i as f64 / fs).sin()
            })
            .collect();
        let psd = welch_psd(
            &x,
            fs,
            WelchOptions::default().with_segment_len(l),
        )
        .unwrap();
        // Bin closest to f_tone:
        let target = (f_tone * l as f64 / fs).round() as usize;
        let peak = psd[target].1;
        // Compare against a "background" sample at 3·f_tone to dodge
        // window leakage from the tone itself.
        let background = psd[3 * target].1;
        assert!(
            peak / background > 100.0,
            "tone/background = {:.2} (expected >> 1)",
            peak / background
        );
    }

    #[test]
    fn welch_psd_rejects_bad_overlap() {
        let x: Vec<f64> = (0..1024).map(|i| (i as f64).sin()).collect();
        let r = welch_psd(
            &x,
            1.0,
            WelchOptions::default().with_overlap(1.5),
        );
        assert!(matches!(r, Err(SpectrumError::OverlapOutOfRange(_))));
    }

    #[test]
    fn integrate_psd_recovers_white_noise_variance() {
        // Unit-variance white noise → one-sided PSD is flat at 2/fs.
        // Integrating from 0 to fs/2 should give back unit power → unit RMS.
        let fs = 1.0;
        let n = 16384;
        let mut rng = XorShift(0x1234_5678);
        let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let psd = welch_psd(&x, fs, WelchOptions::default()).unwrap();
        let rms = integrate_psd(&psd, 0.0, fs / 2.0);
        // Expect ≈ 1.0; allow ±15 % envelope for finite-sample variance
        // and Welch bias.
        assert!(
            (rms - 1.0).abs() < 0.15,
            "integrated RMS = {:.4} (expected ~1.0)",
            rms
        );
    }

    #[test]
    fn integrate_psd_band_excludes_dc_tone() {
        // Build a PSD curve manually: tone at 100 Hz only.
        let psd: Vec<(f64, f64)> = (0..=1000).map(|i| {
            let f = i as f64;
            let p = if (f - 100.0).abs() < 0.5 { 1e-6 } else { 0.0 };
            (f, p)
        }).collect();
        let in_band = integrate_psd(&psd, 50.0, 200.0);
        let out_of_band = integrate_psd(&psd, 200.0, 500.0);
        assert!(in_band > out_of_band);
        assert!(out_of_band < 1e-12, "out_of_band = {out_of_band:.3e}");
    }

    #[test]
    fn integrate_psd_zero_when_band_outside_range() {
        let psd: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 1.0)).collect();
        // Band entirely above the PSD's max frequency.
        assert_eq!(integrate_psd(&psd, 100.0, 200.0), 0.0);
        // Band entirely below the PSD's min frequency.
        assert_eq!(integrate_psd(&psd, -10.0, -1.0), 0.0);
    }

    #[test]
    fn welch_psd_default_segment_len_is_pow2() {
        // n = 16384 → default segment = 16384/8 = 2048 (already pow2).
        let x = vec![0.0; 16384];
        let psd = welch_psd(&x, 1.0, WelchOptions::default()).unwrap();
        // segment_len 2048 ⇒ output length 1025.
        assert_eq!(psd.len(), 2048 / 2 + 1);
    }

    #[test]
    fn resample_uniform_preserves_endpoints() {
        let t = vec![0.0, 0.3, 1.0];
        let y = vec![10.0, 13.0, 20.0];
        let (dt, s) = resample_uniform(&t, &y, 5);
        assert!((dt - 0.25).abs() < 1e-12);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 10.0).abs() < 1e-12);
        assert!((s[4] - 20.0).abs() < 1e-12);
    }
}
