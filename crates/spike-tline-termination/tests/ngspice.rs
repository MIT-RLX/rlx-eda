//! Tier 3: ngspice `T`-element trace vs the FDTD reference, sampled
//! strictly inside the constant plateaus of the bounce diagram.
//!
//! Why plateau-sampling rather than full-trace comparison: at the bounce
//! transitions both engines show a one-sample-wide edge artifact, but for
//! different reasons — FDTD samples the source waveform at integer-h and
//! a `tr = 1 ps` ramp ends up pre-edge at sample `k` and post-edge at
//! sample `k+1`, whereas ngspice resolves the ramp with sub-h substeps
//! and interpolates the output back to the requested grid. Both engines
//! agree to within mV inside a plateau; the edge mismatch is a sampling
//! artifact, not a physics disagreement, so we don't ask it of the test.
//!
//! Backend selection follows the workspace convention used by
//! `spike-dado-sar::invoker_from_env`: `NGSPICE_BACKEND=docker` switches
//! to a pinned image (good for CI / reproducibility), default uses the
//! local `ngspice` on PATH (good for tight dev loops).

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{
    DockerInvoker, Invoker, LocalBinary, NgspiceError, OutputRequest, TransientAnalysis,
};
use eda_hir::SourceWaveform;
use spike_tline_termination::{fdtd_trace, spice_deck, Topology};

fn invoker_from_env() -> Result<Box<dyn Invoker>, NgspiceError> {
    match std::env::var("NGSPICE_BACKEND").as_deref() {
        Ok("docker") => {
            let d = DockerInvoker::from_env()?;
            d.ensure_image()?;
            Ok(Box::new(d))
        }
        _ => Ok(Box::new(LocalBinary::from_env()?)),
    }
}

/// Linearly interpolate `y` defined on `t` (assumed sorted) at `t_q`.
fn interp(t: &[f64], y: &[f64], t_q: f64) -> f64 {
    if t_q <= t[0] { return y[0]; }
    if t_q >= *t.last().unwrap() { return *y.last().unwrap(); }
    // Binary search would be tidier; linear scan is fine for test sizes.
    let i = t.iter().position(|&ti| ti >= t_q).unwrap();
    let (t0, t1) = (t[i - 1], t[i]);
    let (y0, y1) = (y[i - 1], y[i]);
    y0 + (y1 - y0) * (t_q - t0) / (t1 - t0)
}

/// Sample the FDTD and ngspice traces at the midpoints of each bounce
/// plateau — `edge + (2k+2)·TD` — and assert they agree there. Plateaus
/// are wide (2·TD = 2 ns at our settings) so midpoints are far from any
/// edge artifact.
fn assert_plateau_agreement(
    label: &str,
    edge: f64, td: f64,
    t_fd: &[f64], v_fd: &[f64],
    t_ng: &[f64], v_ng: &[f64],
    n_plateaus: usize,
    rtol: f64, atol: f64,
) {
    for k in 0..n_plateaus {
        let t_mid = edge + (2.0 * k as f64 + 2.0) * td;
        if t_mid > *t_fd.last().unwrap() || t_mid > *t_ng.last().unwrap() {
            break;
        }
        let v_fd_q = interp(t_fd, v_fd, t_mid);
        let v_ng_q = interp(t_ng, v_ng, t_mid);
        let env = atol + rtol * v_fd_q.abs();
        if (v_fd_q - v_ng_q).abs() > env {
            panic!(
                "[{label}] plateau {k} mismatch at t={:.3} ns: \
                 FDTD={:+.4} V, ngspice={:+.4} V, |Δ|={:.3e} > env={:.3e}",
                t_mid * 1e9, v_fd_q, v_ng_q, (v_fd_q - v_ng_q).abs(), env,
            );
        }
    }
}

#[test]
fn ngspice_unterminated_matches_fdtd() {
    let ng = match invoker_from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    let td = 1e-9;
    let topo = Topology::unterminated(td);

    let h = td / 50.0;
    let t_stop = 12.0 * td;
    let n_steps = (t_stop / h).round() as usize;

    let edge = 0.5 * td;
    let tr = 1e-12;
    let w = SourceWaveform::pulse(0.0, 3.3, edge, tr, tr, t_stop * 2.0, 0.0);

    let (t_fd, v_fd) = fdtd_trace(topo, h, n_steps, &w);

    let trace = ng.run_transient_trace(
        &spice_deck(topo, &w),
        &TransientAnalysis::new(h, t_stop).with_t_max(h),
        &[OutputRequest::NodeVoltage("vrx".into())],
    ).expect("ngspice .tran trace failed");
    let v_ng = &trace.node_voltages["vrx"];

    // Inside-plateau agreement. With the geometric ringing |Γ_S|=2/3, by
    // the 5th plateau the FDTD-vs-ngspice mismatch should be ~5 mV; the
    // envelope here is conservative.
    assert_plateau_agreement(
        "ngspice T-element vs FDTD (unterminated, ringing)",
        edge, td,
        &t_fd, &v_fd,
        &trace.time, v_ng,
        5, 1e-2, 1e-2,
    );

    // Quiz number: ngspice peak should reach ~5.5 V (167% of 3.3 V).
    let v_ng_peak = v_ng.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((v_ng_peak - 5.5).abs() < 0.05,
            "expected ~5.5 V overshoot, ngspice peak = {v_ng_peak:.3} V");
}

#[test]
fn ngspice_matched_settles_at_vs() {
    let ng = match invoker_from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    let td = 1e-9;
    let topo = Topology::series_matched(td);
    let h = td / 50.0;
    let t_stop = 8.0 * td;
    let n_steps = (t_stop / h).round() as usize;

    let edge = 0.5 * td;
    let tr = 1e-12;
    let vs = 3.3;
    let w = SourceWaveform::pulse(0.0, vs, edge, tr, tr, t_stop * 2.0, 0.0);

    let (t_fd, v_fd) = fdtd_trace(topo, h, n_steps, &w);

    let trace = ng.run_transient_trace(
        &spice_deck(topo, &w),
        &TransientAnalysis::new(h, t_stop).with_t_max(h),
        &[OutputRequest::NodeVoltage("vrx".into())],
    ).expect("ngspice .tran trace failed");
    let v_ng = &trace.node_voltages["vrx"];

    assert_plateau_agreement(
        "ngspice T-element vs FDTD (series matched)",
        edge, td,
        &t_fd, &v_fd,
        &trace.time, v_ng,
        3, 5e-3, 5e-3,
    );

    // Late-time samples should sit at vs to within mV.
    let late = trace.time.iter().enumerate()
        .filter(|(_, &t)| t > 2.0 * td)
        .map(|(i, _)| v_ng[i]);
    for (i, v) in late.enumerate() {
        assert!((v - vs).abs() < 1e-2,
                "matched ngspice late sample {i} not at vs: {v:.4}");
    }
}
