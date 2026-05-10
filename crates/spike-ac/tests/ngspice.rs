//! Tier 3: rlx AC sweep vs ngspice `.ac` complex response. Both engines
//! see the same V1 with `AC 1`. Trace comparison uses
//! `eda_validate::assert_traces_close` on |H| and ∠H separately.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{AcAnalysis, Invoker, LocalBinary, OutputRequest};
use eda_validate::assert_traces_close;
use spike_ac::*;

#[test]
fn rlx_ac_matches_ngspice_ac() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let f_start = 1e3;
    let f_stop = 1e9;
    let pts_per_dec = 8;

    let (rlx_freq, rlx_re, rlx_im) = run_ac_sweep(f_start, f_stop, pts_per_dec, r, c);
    let rlx_mag: Vec<f64> = rlx_re.iter().zip(&rlx_im)
        .map(|(re, im)| (re * re + im * im).sqrt()).collect();
    let rlx_phase: Vec<f64> = rlx_re.iter().zip(&rlx_im)
        .map(|(re, im)| im.atan2(*re)).collect();

    let ac = ng
        .run_ac(
            &spice_deck(r, c),
            &AcAnalysis::dec(pts_per_dec, f_start, f_stop),
            &[OutputRequest::NodeVoltage("vout".into())],
        )
        .expect("ngspice .ac failed");
    let ng_v = &ac.node_voltages["vout"];
    let ng_mag: Vec<f64> = ng_v.iter().map(|(re, im)| (re * re + im * im).sqrt()).collect();
    let ng_phase: Vec<f64> = ng_v.iter().map(|(re, im)| im.atan2(*re)).collect();

    // |H| should match to f64 precision modulo ngspice's solver
    // tolerance. 1e-6 envelope is honest for a linear circuit.
    assert_traces_close(
        &rlx_freq, &rlx_mag,
        &ac.frequency, &ng_mag,
        1e-6, 1e-9,
        "rlx vs ngspice |H(jω)|",
    );

    assert_traces_close(
        &rlx_freq, &rlx_phase,
        &ac.frequency, &ng_phase,
        1e-6, 1e-9,
        "rlx vs ngspice ∠H(jω)",
    );
}
