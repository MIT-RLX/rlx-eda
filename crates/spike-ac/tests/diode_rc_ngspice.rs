//! Tier 3: rlx Bode response of the linearised diode-RC matches
//! ngspice `.ac` over a multi-decade sweep. ngspice does its own
//! pre-DC + small-signal linearisation internally; we do it explicitly
//! via `dc_op_f64` + `small_signal_conductance`. Both approaches
//! should land at the same `g_d`, so the per-frequency complex
//! response should agree to ngspice's solver tolerance.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{AcAnalysis, Invoker, LocalBinary, OutputRequest};
use eda_validate::assert_traces_close;
use spike_ac::*;

const VT: f64 = 0.025_852;

#[test]
fn diode_rc_ac_matches_ngspice_ac() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let is_ = 1e-12_f64;
    let v_dc = 1.0_f64;
    let f_start = 1e3;
    let f_stop = 1e9;
    let pts_per_dec = 8;

    let (rlx_freq, rlx_re, rlx_im) = run_diode_rc_ac_sweep(
        f_start, f_stop, pts_per_dec, r, c, is_, VT, v_dc, 30);
    let rlx_mag: Vec<f64> = rlx_re.iter().zip(&rlx_im)
        .map(|(re, im)| (re * re + im * im).sqrt()).collect();
    let rlx_phase: Vec<f64> = rlx_re.iter().zip(&rlx_im)
        .map(|(re, im)| im.atan2(*re)).collect();

    let ac = ng.run_ac(
        &diode_rc_spice_deck(v_dc, r, c, is_),
        &AcAnalysis::dec(pts_per_dec, f_start, f_stop),
        &[OutputRequest::NodeVoltage("vmid".into())],
    ).expect("ngspice .ac failed");
    let ng_v = &ac.node_voltages["vmid"];
    let ng_mag: Vec<f64> = ng_v.iter()
        .map(|(re, im)| (re * re + im * im).sqrt()).collect();
    let ng_phase: Vec<f64> = ng_v.iter()
        .map(|(re, im)| im.atan2(*re)).collect();

    // 1e-3 relative envelope is honest: rlx uses Vt=25.852 mV exactly
    // while ngspice's TNOM=300.15 K gives ~25.865 mV; the resulting
    // g_d differs by ~0.05% which propagates to the AC response
    // multiplicatively.
    assert_traces_close(
        &rlx_freq, &rlx_mag,
        &ac.frequency, &ng_mag,
        1e-3, 1e-9,
        "rlx vs ngspice |H(jω)| diode-RC",
    );
    assert_traces_close(
        &rlx_freq, &rlx_phase,
        &ac.frequency, &ng_phase,
        1e-2, 1e-7,
        "rlx vs ngspice ∠H(jω) diode-RC",
    );
}
