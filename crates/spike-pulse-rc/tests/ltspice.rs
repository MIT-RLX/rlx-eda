//! Tier 3 (LTspice flavor): same shape as `tests/ngspice.rs`, validating
//! the rlx outer-loop trace against an LTspice `.tran` trace driven by
//! the same `SourceWaveform::Pulse`.
//!
//! Requires the `ltspice` Cargo feature **and** an LTspice install.
//! Soft-skips when LTspice isn't on the machine, so `cargo test
//! --features ltspice` works on CI runners without LTspice provisioned.

#![cfg(feature = "ltspice")]

use eda_extern_ltspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_hir::SourceWaveform;
use eda_validate::assert_traces_close;
use spike_pulse_rc::*;

#[test]
fn rlx_pulse_trace_matches_ltspice_trace() {
    let Some(lt) = LocalBinary::from_env_optional() else {
        eprintln!("LTspice not installed; skipping rlx_pulse_trace_matches_ltspice_trace");
        return;
    };

    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let tau = r * c;
    let w = SourceWaveform::pulse(0.0, 1.0, 200e-9, 1e-12, 1e-12, 2.0 * tau, 0.0);
    let t_stop = 4.0 * tau;
    let h = tau / 200.0;
    let n_steps = (t_stop / h).round() as usize;

    let (rlx_t, rlx_v) = run_transient_trace(n_steps, h, r, c, 0.0, &w);

    let trace = lt
        .run_transient_trace(
            &spice_deck(r, c, &w),
            &TransientAnalysis::new(h, t_stop),
            &[OutputRequest::NodeVoltage("vout".into())],
        )
        .expect("LTspice .tran trace failed");
    let lt_v = &trace.node_voltages["vout"];

    // Same envelope as the ngspice version. LTspice's adaptive stepper
    // is independent from rlx's uniform-h grid; trace_close handles the
    // grid mismatch by linear interpolation of the candidate (LTspice)
    // onto the reference (rlx) grid before comparing samples.
    assert_traces_close(&rlx_t, &rlx_v, &trace.time, lt_v, 5e-2, 5e-3, "rlx vs LTspice");
}
