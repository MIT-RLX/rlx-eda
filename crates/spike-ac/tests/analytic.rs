//! Tier 1: rlx AC response vs closed-form Bode reference.
//!
//! For a unit-driven RC LP, `H(jω) = 1/(1 + jωRC)`. We sweep across
//! 6 decades of frequency and compare both the complex value and the
//! magnitude/phase derived from it.

use eda_validate::assert_traces_close;
use spike_ac::{analytic_h, analytic_mag, analytic_phase, run_ac_point, run_ac_sweep};

const R: f64 = 1_000.0;
const C: f64 = 1e-9;

#[test]
fn rlx_complex_matches_analytic_complex_per_point() {
    let (freq, re, im) = run_ac_sweep(1e3, 1e9, 8, R, C);
    for i in 0..freq.len() {
        let omega = 2.0 * std::f64::consts::PI * freq[i];
        let (re_a, im_a) = analytic_h(omega, R, C);
        assert!((re[i] - re_a).abs() < 1e-9 + 1e-9 * re_a.abs(),
            "f={:.2e}: re rlx={} analytic={}", freq[i], re[i], re_a);
        assert!((im[i] - im_a).abs() < 1e-9 + 1e-9 * im_a.abs(),
            "f={:.2e}: im rlx={} analytic={}", freq[i], im[i], im_a);
    }
}

#[test]
fn rlx_magnitude_trace_matches_analytic_trace() {
    let (freq, re, im) = run_ac_sweep(1e3, 1e9, 8, R, C);
    let mag: Vec<f64> = re.iter().zip(&im).map(|(r, i)| (r * r + i * i).sqrt()).collect();
    let mag_a: Vec<f64> = freq.iter()
        .map(|f| analytic_mag(2.0 * std::f64::consts::PI * f, R, C))
        .collect();
    assert_traces_close(&freq, &mag, &freq, &mag_a, 1e-9, 1e-12,
        "rlx |H| vs analytic Bode magnitude");
}

#[test]
fn rlx_phase_at_corner_is_minus_45_degrees() {
    // Exact corner frequency f_c = 1/(2πRC); at ω=ω_c, ωRC = 1 so the
    // ideal Bode reading is |H|=1/√2 and ∠H = -π/4.
    let fc = 1.0 / (2.0 * std::f64::consts::PI * R * C);
    let omega = 2.0 * std::f64::consts::PI * fc;
    let (re, im) = run_ac_point(omega, R, C);
    let phase = im.atan2(re);
    let phase_analytic = analytic_phase(omega, R, C);
    assert!((phase - phase_analytic).abs() < 1e-12);
    assert!((phase - (-std::f64::consts::FRAC_PI_4)).abs() < 1e-12,
        "phase at corner = {phase} rad (expected -π/4)");
    let mag = (re * re + im * im).sqrt();
    assert!((mag - 1.0 / std::f64::consts::SQRT_2).abs() < 1e-12,
        "|H| at corner = {mag} (expected 1/√2)");
}
