//! Tier 3: rlx outer-loop trace against ngspice `.tran` trace, with a
//! `SourceWaveform::Pulse`-driven RC LP. Both engines see the same pulse
//! definition (`Netlist::add_waveform_source`); both run BDF1 with the
//! same `h`. Trace-level comparison via `eda_validate::assert_traces_close`.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_hir::SourceWaveform;
use eda_validate::assert_traces_close;
use spike_pulse_rc::*;

#[test]
fn rlx_pulse_trace_matches_ngspice_trace() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let tau = r * c;

    // Use a non-zero tr/tf (1 ps) — far below h — so the two simulators
    // can't disagree about the source value at the discontinuity. With
    // tr=tf=0, rlx evaluates V(td) = v2 while ngspice's first sample at
    // t=td still reports v1. A 1 ps ramp is invisible at our 10 ns
    // timestep but eliminates the ambiguity.
    let w = SourceWaveform::pulse(0.0, 1.0, 200e-9, 1e-12, 1e-12, 2.0 * tau, 0.0);
    let t_stop = 4.0 * tau;
    // h = τ/200 matches the constant-DC trace test in `spike-rc-transient`
    // and keeps cumulative BE-stepping error inside the 5e-3 envelope.
    let h = tau / 200.0;
    let n_steps = (t_stop / h).round() as usize;

    let (rlx_t, rlx_v) = run_transient_trace(n_steps, h, r, c, 0.0, &w);

    let trace = ng
        .run_transient_trace(
            &spice_deck(r, c, &w),
            // Pin ngspice's adaptive substep to our uniform h so the two
            // BDF1 solvers walk the same time grid and don't drift O(h)
            // around the pulse edges.
            &TransientAnalysis::new(h, t_stop).with_t_max(h),
            &[OutputRequest::NodeVoltage("vout".into())],
        )
        .expect("ngspice .tran trace failed");
    let ng_v = &trace.node_voltages["vout"];

    // Tolerance is honest BE-vs-BDF1 with mismatched grids: ngspice picks
    // its step size adaptively (tmax > h between edges) while rlx is
    // uniform-h. The two BDF1 solvers can drift ~3-5% near step edges
    // even though they're nominally the same numerical method. The
    // constant-DC trace test runs cleaner (5e-3) because it has no edge
    // for the LTE controller to respond to. Future tightening: add a
    // `tmax` field to `TransientAnalysis` and set `tmax = t_step` on the
    // ngspice .tran line, forcing uniform substeps.
    // With matched stimulus (the per=0 → 1e30 substitution in
    // `add_waveform_source`) and `tmax=h` pinning, the two BDF1 solvers
    // walk the same trajectory. Tolerance matches the constant-DC trace
    // test (5e-3) plus a small atol cushion for ngspice's edge-resolution
    // sub-substeps inside the 1ps fall.
    assert_traces_close(
        &rlx_t, &rlx_v,
        &trace.time, ng_v,
        5e-3, 1e-4,
        "rlx vs ngspice RC LP pulse-driven trace",
    );
}
