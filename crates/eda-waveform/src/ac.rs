//! Loop-stability figures of merit on `Waveform::Complex`.
//!
//! For a feedback loop, verification engineers want four numbers from
//! the open-loop AC response:
//!
//! - **DC gain** — `|H|` at the lowest swept frequency, in dB. Sets the
//!   loop's static accuracy.
//! - **Unity-gain bandwidth (UGBW)** — frequency where `|H|` crosses
//!   `0 dB` going down. The closed-loop bandwidth lands here for a
//!   well-behaved single-pole-dominant loop.
//! - **Phase margin** — phase + 180° at UGBW. >45° is the usual
//!   healthy threshold; < 0° means the loop is unstable.
//! - **Gain margin** — `-20·log10(|H|)` at the frequency where phase
//!   crosses `-180°`. `+∞` if phase never reaches `-180°` (which is
//!   what you want).
//!
//! Plus a fifth that's nice to surface:
//!
//! - **Peaking** — `max|H| - dc_gain` in dB, measuring how much the
//!   response humps before rolling off. Zero for a clean
//!   single-pole-dominant loop; large peaking foreshadows ringing.
//!
//! ## Conventions
//!
//! - Frequency axis must be ascending. AC sweeps are usually log-spaced;
//!   crossings are interpolated in `log10(f)` space, which matches how
//!   the human reading the Bode plot would eyeball it and is much more
//!   accurate than linear-frequency interp on a log grid.
//! - Phase is unwrapped before margin calculation so a -180° crossing
//!   isn't masked by the ±π branch cut.
//! - The signal looked up by name is treated as `H(jω)` — open-loop
//!   transfer in a Bode-plot sense. If your sweep records `vout` and
//!   you injected `vin = 1`, that's already `H`.

use std::f64::consts::PI;

use thiserror::Error;

use crate::Waveform;

#[derive(Debug, Error)]
pub enum AcError {
    #[error("waveform must be complex-valued (frequency-domain) for AC analysis")]
    NotComplex,
    #[error("signal {0:?} not found in waveform")]
    MissingSignal(String),
    #[error("frequency axis is empty or has fewer than 2 points")]
    TooShort,
    #[error("frequency axis must be strictly positive (got f={0})")]
    NonPositiveFreq(f64),
    #[error("frequency axis must be ascending (idx {idx}: {a} >= {b})")]
    NonMonotonic { idx: usize, a: f64, b: f64 },
}

/// Loop-stability summary.
#[derive(Debug, Clone, Copy)]
pub struct AcMetrics {
    pub dc_gain_db: f64,
    /// Frequency at which `|H|` first crosses 1.0 (= 0 dB) going down.
    /// `None` if `|H|` never crosses 0 dB in the swept range.
    pub unity_gain_bandwidth: Option<f64>,
    /// Phase + 180° at UGBW, in degrees. `None` if UGBW is not in range.
    pub phase_margin_deg: Option<f64>,
    /// `-20·log10(|H|)` at the frequency where phase crosses `-180°`.
    /// `None` if phase never reaches `-180°` (gain margin = `+∞`,
    /// meaning the loop is unconditionally stable in phase).
    pub gain_margin_db: Option<f64>,
    /// `max|H|` over the sweep, in dB, minus `dc_gain_db`. Always ≥ 0.
    pub peaking_db: f64,
}

/// Analyze the open-loop response named `signal` in `wave`.
pub fn analyze(wave: &Waveform, signal: &str) -> Result<AcMetrics, AcError> {
    let (axis, samples) = match wave {
        Waveform::Complex { axis, signals, .. } => {
            let s = signals
                .get(signal)
                .ok_or_else(|| AcError::MissingSignal(signal.to_string()))?;
            (axis, s)
        }
        Waveform::Real { .. } => return Err(AcError::NotComplex),
    };
    if axis.len() < 2 || samples.len() != axis.len() {
        return Err(AcError::TooShort);
    }
    for (i, &f) in axis.iter().enumerate() {
        if !(f > 0.0) {
            return Err(AcError::NonPositiveFreq(f));
        }
        if i > 0 && axis[i - 1] >= f {
            return Err(AcError::NonMonotonic {
                idx: i,
                a: axis[i - 1],
                b: f,
            });
        }
    }

    let mag_db: Vec<f64> = samples
        .iter()
        .map(|&(re, im)| 20.0 * (re * re + im * im).sqrt().log10())
        .collect();
    let phase_deg = unwrap_phase_deg(samples);

    let dc_gain_db = mag_db[0];
    let peaking_db = mag_db
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - dc_gain_db;

    // UGBW: first 0-dB crossing going down. We look for a sign change in
    // `mag_db` from + to - (or exact zero followed by negative).
    let ugbw = first_down_crossing(axis, &mag_db, 0.0);

    let phase_margin_deg = ugbw.map(|f| {
        let phase_at_ugbw = lerp_log_x(axis, &phase_deg, f);
        180.0 + phase_at_ugbw
    });

    // Gain margin: where unwrapped phase first crosses -180°.
    let f_180 = first_phase_crossing(axis, &phase_deg, -180.0);
    let gain_margin_db = f_180.map(|f| {
        let mag_at_180 = lerp_log_x(axis, &mag_db, f);
        -mag_at_180
    });

    Ok(AcMetrics {
        dc_gain_db,
        unity_gain_bandwidth: ugbw,
        phase_margin_deg,
        gain_margin_db,
        peaking_db,
    })
}

/// Phase in degrees, unwrapped to a continuous curve (no ±π jumps).
fn unwrap_phase_deg(samples: &[(f64, f64)]) -> Vec<f64> {
    let mut out = Vec::with_capacity(samples.len());
    if samples.is_empty() {
        return out;
    }
    let raw0 = samples[0].1.atan2(samples[0].0);
    out.push(raw0.to_degrees());
    let mut prev = raw0;
    let mut acc = raw0;
    for &(re, im) in &samples[1..] {
        let raw = im.atan2(re);
        let mut d = raw - prev;
        while d > PI {
            d -= 2.0 * PI;
        }
        while d < -PI {
            d += 2.0 * PI;
        }
        acc += d;
        out.push(acc.to_degrees());
        prev = raw;
    }
    out
}

/// First frequency at which `y` crosses `level` going down (positive
/// → negative). Linear interp in `log10(f)` for accuracy on log-spaced
/// sweeps.
fn first_down_crossing(f: &[f64], y: &[f64], level: f64) -> Option<f64> {
    for i in 0..f.len() - 1 {
        let y0 = y[i] - level;
        let y1 = y[i + 1] - level;
        if y0 >= 0.0 && y1 < 0.0 {
            return Some(interp_x_at_y(f[i], f[i + 1], y[i], y[i + 1], level));
        }
    }
    None
}

/// First frequency at which `y` crosses `level`, either direction.
fn first_phase_crossing(f: &[f64], y: &[f64], level: f64) -> Option<f64> {
    for i in 0..f.len() - 1 {
        let d0 = y[i] - level;
        let d1 = y[i + 1] - level;
        let crossed = (d0 <= 0.0 && d1 > 0.0) || (d0 >= 0.0 && d1 < 0.0);
        if crossed {
            return Some(interp_x_at_y(f[i], f[i + 1], y[i], y[i + 1], level));
        }
    }
    None
}

/// Interpolate the frequency at which y crosses `target` between
/// (f0, y0) and (f1, y1), with the x-axis treated as log10(f).
fn interp_x_at_y(f0: f64, f1: f64, y0: f64, y1: f64, target: f64) -> f64 {
    if (y1 - y0).abs() < f64::EPSILON {
        return f0;
    }
    let frac = (target - y0) / (y1 - y0);
    let lf0 = f0.log10();
    let lf1 = f1.log10();
    10f64.powf(lf0 + frac * (lf1 - lf0))
}

/// Linear interpolation of `y` at frequency `fq`, in log10(f) space.
fn lerp_log_x(f: &[f64], y: &[f64], fq: f64) -> f64 {
    if fq <= f[0] {
        return y[0];
    }
    if fq >= f[f.len() - 1] {
        return y[y.len() - 1];
    }
    let i = match f.binary_search_by(|x| {
        x.partial_cmp(&fq).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(j) => return y[j],
        Err(j) => j - 1,
    };
    let lf = fq.log10();
    let lf0 = f[i].log10();
    let lf1 = f[i + 1].log10();
    let t = (lf - lf0) / (lf1 - lf0);
    y[i] + t * (y[i + 1] - y[i])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// One-pole low-pass: H(jω) = A0 / (1 + jω/ωp), ωp = 2π·fp.
    fn one_pole(a0: f64, fp: f64, freqs: &[f64]) -> Waveform {
        let samples: Vec<(f64, f64)> = freqs
            .iter()
            .map(|&f| {
                let w = 2.0 * PI * f;
                let wp = 2.0 * PI * fp;
                // H = A0 / (1 + j w/wp); rationalize: A0 (1 - j w/wp) / (1 + (w/wp)^2)
                let r = w / wp;
                let denom = 1.0 + r * r;
                (a0 / denom, -a0 * r / denom)
            })
            .collect();
        let mut signals = BTreeMap::new();
        signals.insert("vout".into(), samples);
        Waveform::Complex {
            axis_name: "frequency".into(),
            axis: freqs.to_vec(),
            signals,
        }
    }

    fn log_sweep(decades: std::ops::RangeInclusive<i32>, per_decade: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for d in decades.clone() {
            for i in 0..per_decade {
                out.push(10f64.powf(d as f64 + i as f64 / per_decade as f64));
            }
        }
        // include endpoint
        out.push(10f64.powf(*decades.end() as f64 + 1.0));
        out
    }

    #[test]
    fn one_pole_dc_gain_and_ugbw() {
        // A0 = 1000 (60 dB), fp = 100 Hz → UGBW ≈ A0 · fp = 100 kHz.
        let freqs = log_sweep(0..=8, 20);
        let w = one_pole(1000.0, 100.0, &freqs);
        let m = analyze(&w, "vout").unwrap();
        // DC gain ≈ 60 dB.
        assert!((m.dc_gain_db - 60.0).abs() < 0.5);
        // UGBW ≈ 100 kHz (within 1 % since we're at 20 pts/dec).
        let ugbw = m.unity_gain_bandwidth.unwrap();
        assert!(
            (ugbw / 1e5 - 1.0).abs() < 0.01,
            "got UGBW = {} Hz",
            ugbw
        );
    }

    #[test]
    fn one_pole_has_90deg_phase_margin() {
        // A single-pole loop maxes out at -90° phase → PM = 90°.
        let freqs = log_sweep(0..=8, 20);
        let w = one_pole(100.0, 1e3, &freqs);
        let m = analyze(&w, "vout").unwrap();
        let pm = m.phase_margin_deg.unwrap();
        assert!((pm - 90.0).abs() < 1.0, "got PM = {} deg", pm);
        // Phase never reaches -180° → no gain margin.
        assert!(m.gain_margin_db.is_none());
    }

    #[test]
    fn double_pole_gives_low_phase_margin() {
        // H = A0 / (1 + jω/ω1)^2 with ω1 = 2π·1 kHz, A0 = 100.
        // |H| = 1 ⇒ r = √(A0-1) = √99 ≈ 9.95
        // arg(H) = -2·atan(r) ≈ -168.6° ⇒ PM ≈ 11.4°.
        // Asymptotically (A0 → ∞) PM → 0; finite A0 leaves a small margin.
        let freqs = log_sweep(0..=8, 20);
        let a0 = 100.0_f64;
        let f1 = 1e3_f64;
        let samples: Vec<(f64, f64)> = freqs
            .iter()
            .map(|&f| {
                let w = 2.0 * PI * f;
                let wp = 2.0 * PI * f1;
                let r = w / wp;
                // H = A0 / ((1 + jr)^2)
                // (1 + jr)^2 = 1 - r^2 + 2jr
                let denom_re = 1.0 - r * r;
                let denom_im = 2.0 * r;
                let denom_mag2 = denom_re * denom_re + denom_im * denom_im;
                let re = a0 * denom_re / denom_mag2;
                let im = -a0 * denom_im / denom_mag2;
                (re, im)
            })
            .collect();
        let mut signals = BTreeMap::new();
        signals.insert("vout".into(), samples);
        let w = Waveform::Complex {
            axis_name: "frequency".into(),
            axis: freqs.clone(),
            signals,
        };
        let m = analyze(&w, "vout").unwrap();
        let pm = m.phase_margin_deg.unwrap();
        let expected_pm = 180.0 - 2.0 * (99.0_f64.sqrt()).atan().to_degrees();
        assert!(
            (pm - expected_pm).abs() < 1.0,
            "got PM = {} deg (expected {})",
            pm,
            expected_pm
        );
        // Phase asymptotes at -180° (never quite reaches it), so the
        // first-crossing detector returns None → gain margin is +∞ in
        // an analytic sense.
        assert!(m.gain_margin_db.is_none());
    }

    #[test]
    fn one_pole_has_no_peaking() {
        let freqs = log_sweep(0..=6, 20);
        let w = one_pole(10.0, 100.0, &freqs);
        let m = analyze(&w, "vout").unwrap();
        assert!(m.peaking_db.abs() < 1e-6, "peaking = {}", m.peaking_db);
    }

    #[test]
    fn rejects_real_waveform() {
        let mut signals = BTreeMap::new();
        signals.insert("v".into(), vec![1.0]);
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0],
            signals,
        };
        assert!(matches!(analyze(&w, "v"), Err(AcError::NotComplex)));
    }

    #[test]
    fn missing_signal_errors() {
        let freqs = log_sweep(0..=2, 5);
        let w = one_pole(1.0, 10.0, &freqs);
        assert!(matches!(
            analyze(&w, "nope"),
            Err(AcError::MissingSignal(_))
        ));
    }
}
