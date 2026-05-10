//! Tier 3: rlx forward against an external simulator (ngspice).
//!
//! Gated by the `ngspice` feature so the build doesn't require ngspice on
//! every developer's machine. CI installs ngspice and turns it on.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};
use eda_validate::assert_close;
use spike_divider::*;

#[test]
fn rlx_forward_matches_ngspice_dc() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: {e}");
            return; // Treat missing binary as a soft skip even with feature on.
        }
    };

    for &(v, r1, r2) in &[
        (1.0_f32, 1_000.0_f32, 1_000.0_f32),
        (5.0,     1_000.0,     2_000.0),
        (3.3,     10_000.0,    330.0),
    ] {
        let rlx = run_forward(v, r1, r2);

        let deck = spice_deck(v, r1, r2);
        let res = ng.run_dc(&deck, &[OutputRequest::NodeVoltage("vout".into())])
            .expect("ngspice run failed");
        let ng_vout = res.node_voltages["vout"] as f32;

        // Linear circuit, no Newton, both sides at f32-ish precision: tight.
        assert_close(rlx, ng_vout, 1e-5, 1e-7, &format!("ngspice @ V={v}, R1={r1}, R2={r2}"));
    }
}
