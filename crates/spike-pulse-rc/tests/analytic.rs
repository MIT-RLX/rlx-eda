//! Tier 1: rlx outer-loop trace vs piecewise-analytic continuum reference.
//!
//! BE is exact for a linear RC (recurrence with `α = RC/(h+RC)`), so as
//! `h → 0` the BE solution converges to the analytic. We pick `h` small
//! enough relative to `RC` that the BE/analytic gap is below `5e-4`
//! relative — sufficient for the rlx forward to be validated against the
//! step / charge / fall / relax structure of the closed form.

use eda_hir::SourceWaveform;
use spike_pulse_rc::{analytic_pulse_at, run_transient_trace};

const R: f64 = 1_000.0;
const C: f64 = 1e-9;
const TAU: f64 = R * C; // 1 µs

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] not close: a={a:+.6e} b={b:+.6e} |a-b|={:.3e} env={env:.3e}",
            (a - b).abs());
    }
}

#[test]
fn rlx_matches_analytic_across_pulse_regions() {
    // 0V→1V pulse at td=200ns, pw=2µs (twice tau, so high plateau gets
    // ~86% of the way). h chosen as tau/100 = 10ns; t_stop = 4 tau
    // covers fall + relax with margin.
    let v1 = 0.0;
    let v2 = 1.0;
    let td = 200e-9;
    let pw = 2.0 * TAU;
    let w = SourceWaveform::pulse(v1, v2, td, 0.0, 0.0, pw, 0.0);

    let t_stop = 4.0 * TAU;
    // BE is first-order: per-step error ~ h/(2τ). At h=τ/1000 the
    // cumulative drift after a few τ is ~5e-4, well inside our envelope.
    let h = TAU / 1_000.0;
    let n_steps = (t_stop / h).round() as usize;
    let (t, v) = run_transient_trace(n_steps, h, R, C, v1, &w);

    // Sample several characteristic times — one in each region. Tolerance
    // is dominated by BE truncation, hence the ~2e-3 rtol envelope.
    let probes = [
        // (t,                    rtol, atol, label)
        (50e-9,                   1e-6, 1e-6, "before delay"),
        (td + 0.5 * TAU,          2e-3, 1e-4, "rising charge (one-half tau)"),
        (td + TAU,                2e-3, 1e-4, "rising charge (one tau ~ 63%)"),
        (td + pw - 1e-12,         2e-3, 1e-4, "end of plateau"),
        (td + pw + 0.5 * TAU,     2e-3, 1e-4, "relaxation (half tau)"),
        (td + pw + 2.0 * TAU,     2e-3, 1e-4, "relaxation (two tau)"),
    ];
    for (tp, rtol, atol, label) in probes {
        let i = (tp / h).round() as usize;
        let i = i.min(t.len() - 1);
        let rlx_v = v[i];
        let ana_v = analytic_pulse_at(t[i], v1, v2, td, pw, R, C);
        assert_close(rlx_v, ana_v, rtol, atol, label);
    }
}

#[test]
fn rlx_starts_at_initial_condition() {
    let w = SourceWaveform::pulse(0.0, 1.0, 1e-9, 0.0, 0.0, 1.0, 0.0);
    let (_t, v) = run_transient_trace(10, 1e-10, R, C, 0.0, &w);
    assert_eq!(v[0], 0.0, "initial condition not preserved");
}

#[test]
fn periodic_pulse_runs_without_panic() {
    // We don't assert against the analytic here (which only covers the
    // first pulse), but exercise the periodic path to catch NaNs / step-
    // graph runtime errors. Period is 5τ with 50% duty so the cap rides
    // ~0.91/0.09 high/low; we check it crossed both halves.
    let w = SourceWaveform::pulse(0.0, 1.0, 0.0, 0.0, 0.0, 2.5 * TAU, 5.0 * TAU);
    let h = TAU / 100.0;
    let n_steps = (10.0 * TAU / h).round() as usize;
    let (_t, v) = run_transient_trace(n_steps, h, R, C, 0.0, &w);
    assert!(v.iter().all(|x| x.is_finite()), "non-finite output");
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(max > 0.7, "pulse never charged the cap above 0.7 (got max={max:.3})");
    assert!(min < 0.3, "cap never relaxed below 0.3 (got min={min:.3})");
}
