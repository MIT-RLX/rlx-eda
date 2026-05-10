//! Tier 3: rlx outer-loop transient against ngspice `.tran`.
//!
//! Both use Backward Euler with the same uniform timestep `h`, both start
//! from `vout = 0` (we set `IC=0` and `uic`). For a linear RC LP the BE
//! result is `V·(1 − α^N)` with `α = RC/(h+RC)`, exact in either
//! simulator — so they should agree to f64 precision modulo ngspice's
//! print-format truncation. ngspice's default LTE-controlled stepper may
//! still take internal substeps, so we compare at common t_stop values
//! that lie far enough from the transient-event boundary to be tolerant.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use spike_rc_transient::*;

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
fn rlx_transient_matches_ngspice() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    let cases = &[
        // (v_dc, R,        C,       n_steps, h_factor·RC, t_stop)
        // We use h = t_stop / n_steps; t_stop is N·RC time constants.
        (1.0_f64, 1_000.0_f64, 1e-9_f64, 100, 1.0_f64),     // T = 1·RC, ~63% charged
        (1.0,     2_200.0,     2.2e-9,    50, 0.5),         // T = 0.5·RC
        (3.3,     10_000.0,    100e-12,  200, 2.0),         // T = 2·RC
    ];

    for &(v_dc, r, c, n_steps, t_in_rc) in cases {
        let rc = r * c;
        let t_stop = t_in_rc * rc;
        let h = t_stop / n_steps as f64;

        let rlx = run_transient(n_steps, h, r, c, 0.0, |_| v_dc);

        let res = ng.run_transient_final(
            &spice_deck(v_dc, r, c),
            &TransientAnalysis::new(h, t_stop),
            &[OutputRequest::NodeVoltage("vout".into())],
        ).expect("ngspice .tran failed");
        let ng_v = res.node_voltages["vout"];

        // We force BDF1 (= BE) on the ngspice side via `.options
        // method=gear maxord=1` in the deck, so both solvers do the same
        // discretization. Residual disagreement comes from (a) ngspice's
        // initial-step internal adjustments under maxord=1 and (b) the
        // ~6-sig-fig precision of `meas/print` output. Honest envelope
        // is ~5e-4 relative.
        assert_close(rlx, ng_v, 5e-4, 1e-7,
            &format!("ngspice @ V={v_dc}, R={r}, C={c:.2e}, T={t_in_rc}·RC"));
    }
}
