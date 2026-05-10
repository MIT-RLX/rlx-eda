//! Tier 3: rlx MNA forward against ngspice DC (the same MNA system, two
//! independent implementations). Linear, well-conditioned, both at ~f64
//! precision — agreement should be very tight.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};
use spike_divider_mna::*;

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
fn rlx_mna_matches_ngspice_dc() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    for &(v, r1, r2) in &[
        (1.0_f64, 1_000.0_f64, 1_000.0_f64),
        (5.0,     1_000.0,     2_000.0),
        (3.3,     10_000.0,    330.0),
    ] {
        let rlx = run_forward_mna(v, r1, r2);
        let res = ng.run_dc(
            &spice_deck(v, r1, r2),
            &[OutputRequest::NodeVoltage("vout".into())],
        ).expect("ngspice run failed");
        let ng_v = res.node_voltages["vout"];

        // Two MNA solvers, same circuit; ngspice prints with ~6 sig figs by
        // default in `print` form, so 1e-6 rtol is the practical limit set
        // by the printer, not the solver.
        assert_close(rlx, ng_v, 1e-6, 1e-9, &format!("ngspice @ V={v}, R1={r1}, R2={r2}"));
    }
}
