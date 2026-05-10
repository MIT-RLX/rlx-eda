//! Tier 3 (per the validation pyramid): rlx diode-RC transient against
//! ngspice `.tran`. Both run BDF1 (= Backward Euler) with the same
//! uniform timestep `h`. With a constant V drive and no `uic`, both
//! simulators start at the same DC operating point and stay there —
//! the test confirms the rlx outer-loop reproduces ngspice's settled
//! state to f32 precision.
//!
//! Skipped (rather than failed) when no ngspice binary is on PATH; CI
//! that wants strict witness coverage should set `NGSPICE` to the
//! binary path.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use spike_diode::{run_transient_forward, spice_deck, VT};

const N_NEWTON_DC:   usize = 30;
const N_NEWTON_STEP: usize = 5;

fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    if !close(a, b, rtol, atol) {
        panic!(
            "[{label}] not close:\n  a    = {a:+.15e}\n  b    = {b:+.15e}\n  |a-b|= {diff:.3e}",
            diff = (a - b).abs()
        );
    }
}

#[test]
fn diode_rc_transient_matches_ngspice() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    // (v_dc, R, Is, C, h, n_steps)
    // Each case settles many τ's of constant drive; the BE outer loop
    // should land on the same DC OP ngspice computes via its pre-tran
    // analysis. Larger Is => smaller forward drop, so the cases sweep
    // both the operating point and the time constant.
    let cases = &[
        (1.0_f64, 1_000.0_f64, 1e-12_f64, 1e-9_f64, 1e-7_f64, 60),
        (3.3,     2_200.0,     1e-14,     2.2e-9,   1e-7,    100),
        (0.7,       470.0,     5e-12,     1e-9,     1e-7,     80),
    ];

    for &(v_dc, r, is_, c, h, n_steps) in cases {
        let v_per_step: Vec<f32> = vec![v_dc as f32; n_steps];
        let rlx = run_transient_forward(
            v_dc as f32, &v_per_step, VT, h as f32,
            r as f32, is_ as f32, c as f32,
            N_NEWTON_DC, N_NEWTON_STEP,
        ) as f64;

        let t_stop = h * n_steps as f64;
        let res = ng.run_transient_final(
            &spice_deck(v_dc, r, c, is_),
            &TransientAnalysis::new(h, t_stop),
            &[OutputRequest::NodeVoltage("vmid".into())],
        ).expect("ngspice .tran failed");
        let ng_v = res.node_voltages["vmid"];

        // f32 rlx vs f64 ngspice with `meas/print` ~6 sig figs: 1e-3
        // relative is the honest envelope after the diode exp's
        // sensitivity to tnom (rlx uses Vt=25.852 mV exactly, ngspice's
        // default T=300.15 K gives ~25.865 mV).
        assert_close(rlx, ng_v, 1e-3, 1e-5,
            &format!("ngspice @ V={v_dc}, R={r}, Is={is_:.0e}, C={c:.2e}"));
    }
}
