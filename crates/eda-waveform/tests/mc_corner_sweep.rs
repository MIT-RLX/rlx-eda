//! End-to-end test of the MC harness: synthetic ADC corner sweep
//! → ENOB extraction via `spectrum::adc_metrics` → aggregation →
//! spec check. Exercises the full mc.rs API on real Waveforms and
//! a real metric extractor (not a mock).

use std::collections::BTreeMap;
use std::f64::consts::PI;

use eda_waveform::{
    mc::{self, Worst},
    spectrum::{self, Window},
    Waveform,
};

/// Synthesize one ADC capture: B-bit quantized coherent sine plus
/// gaussian noise of a given std (in LSB units). Returns a Waveform
/// whose "code" signal holds the integer codes as f64.
fn synth_adc_capture(
    n: usize,
    bin: usize,
    bits: u32,
    noise_lsb_std: f64,
    seed: u64,
) -> Waveform {
    let levels = (1u64 << bits) as f64;
    let lsb = 2.0 / levels;
    // Tiny seeded LCG so each "corner" has reproducible but distinct noise.
    let mut s = seed;
    let mut next_uniform = || -> f64 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (s >> 33) as u32;
        bits as f64 / u32::MAX as f64
    };
    let mut next_normal = || -> f64 {
        // Box-Muller.
        let u1 = next_uniform().max(1e-12);
        let u2 = next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    };
    let codes: Vec<f64> = (0..n)
        .map(|i| {
            let x = 0.99 * (2.0 * PI * bin as f64 * i as f64 / n as f64).sin();
            let noisy = x + noise_lsb_std * lsb * next_normal();
            // Quantize to nearest LSB, clamp to range.
            (noisy / lsb).round() * lsb
        })
        .collect();
    let axis: Vec<f64> = (0..n).map(|i| i as f64 * 1e-9).collect();
    let mut signals = BTreeMap::new();
    signals.insert("code".to_string(), codes);
    Waveform::Real {
        axis_name: "time".into(),
        axis,
        signals,
    }
}

/// Extract ENOB from a Waveform whose "code" signal holds the ADC samples.
fn enob(w: &Waveform) -> Result<f64, spectrum::SpectrumError> {
    let samples = w.real("code").expect("missing 'code' signal");
    let n = samples.len();
    let bin = spectrum::pick_coherent_bin(n, n / 8).expect("coherent bin");
    spectrum::adc_metrics(samples, bin, 5, Window::Rectangular).map(|m| m.enob)
}

#[test]
fn corner_sweep_finds_worst_enob_and_yield() {
    let n = 4096;
    // Pick a coherent bin in the same way the metric does, so the synthetic
    // data and the analyzer agree on which bin holds the fundamental.
    let bin = spectrum::pick_coherent_bin(n, n / 8).unwrap();

    // Five "corners": the typical case is clean (low noise), the bad
    // corner has 4× the noise std. ENOB scales as B - log2(noise_lsb).
    let corners = vec![
        ("tt_27c", 0.25, 0xa5_5a_a5_5a),
        ("ff_85c", 0.30, 0xdead_beef),
        ("ss_-40c", 1.0, 0xfeed_face),
        ("fs_27c", 0.40, 0xbabe_face),
        ("sf_27c", 0.35, 0xc0de_cafe),
    ];
    let waves: Vec<(String, Waveform)> = corners
        .iter()
        .map(|(label, sigma, seed)| {
            (
                label.to_string(),
                synth_adc_capture(n, bin, 12, *sigma, *seed),
            )
        })
        .collect();

    // Map ENOB across corners. `enob` is fallible; we want to surface
    // failures rather than silently dropping them.
    let (runs, errors) = mc::try_map_metric(&waves, enob);
    assert!(errors.is_empty(), "extractor errors: {:?}", errors);
    assert_eq!(runs.len(), 5);

    // Aggregate. ENOB — bigger is better → Worst::Min.
    let stats = mc::collect_stats(&runs, Worst::Min).unwrap();
    assert_eq!(stats.n, 5);
    // The high-noise corner should land at the worst ENOB.
    assert_eq!(
        stats.worst.as_ref().unwrap().0,
        "ss_-40c",
        "worst-case ENOB should be the high-noise corner; got {:?}",
        stats.worst
    );
    // 12-bit ADC with σ ≤ 0.4 LSB hits ENOB ≥ 10 comfortably.
    let tt_enob = runs
        .iter()
        .find(|r| r.label == "tt_27c")
        .map(|r| r.metric)
        .unwrap();
    assert!(tt_enob > 10.0, "tt ENOB = {tt_enob:.2}");
    // The high-noise corner should fall meaningfully below tt.
    let ss_enob = stats.worst.as_ref().unwrap().1;
    assert!(
        ss_enob < tt_enob - 1.0,
        "expected the high-noise corner to be > 1 ENOB worse; tt={tt_enob:.2}, ss={ss_enob:.2}"
    );

    // Spec gate: ENOB ≥ 10. Yield should be 4/5 (everyone except the
    // high-noise corner).
    let spec = mc::check_spec(&runs, |e| e >= 10.0);
    assert_eq!(spec.n_total, 5);
    assert_eq!(spec.n_pass, 4);
    assert!((spec.yield_frac - 0.8).abs() < 1e-12);
    assert_eq!(spec.failures.len(), 1);
    assert_eq!(spec.failures[0].label, "ss_-40c");
}

#[test]
fn corner_sweep_with_infallible_metric() {
    // Demonstrate map_metric (no Result) on a simple per-run scalar:
    // peak amplitude of the "code" signal. This is the path most users
    // will take when their metric can't fail.
    let waves = vec![
        ("a".to_string(), simple_wave(&[0.0, 0.5, 1.0, 0.5])),
        ("b".to_string(), simple_wave(&[0.0, 0.7, 1.4, 0.7])),
        ("c".to_string(), simple_wave(&[0.0, 0.2, 0.4, 0.2])),
    ];
    let runs = mc::map_metric(&waves, |w| {
        let samples = w.real("v").unwrap();
        samples.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    });
    let stats = mc::collect_stats(&runs, Worst::Max).unwrap();
    // Worst (Max) is "highest peak", which is corner b at 1.4.
    assert_eq!(stats.worst.as_ref().unwrap().0, "b");
    assert!((stats.worst.as_ref().unwrap().1 - 1.4).abs() < 1e-12);
}

fn simple_wave(samples: &[f64]) -> Waveform {
    let mut signals = BTreeMap::new();
    signals.insert("v".to_string(), samples.to_vec());
    Waveform::Real {
        axis_name: "time".into(),
        axis: (0..samples.len()).map(|i| i as f64 * 1e-9).collect(),
        signals,
    }
}
