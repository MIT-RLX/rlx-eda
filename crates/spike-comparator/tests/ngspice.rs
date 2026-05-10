//! Tier 3: rlx forward `vout` vs ngspice `.op` reading from a B-source
//! computing the same closed-form expression.
//!
//! Both engines evaluate `vol + (voh-vol)·½·(1+tanh(k·(v+ − v−)))` —
//! ngspice via its arbitrary-source (`B`) syntax, rlx via the graph
//! built in `vout_subgraph`. They should agree to f64 precision modulo
//! ngspice's internal solver tolerance.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};
use spike_comparator::*;

#[test]
fn rlx_vout_matches_ngspice_op() {
    let Ok(ng) = LocalBinary::from_env() else {
        eprintln!("ngspice missing"); return;
    };

    let comp = Comparator::default();
    let cases = &[
        // (v+, v−, label)
        (-5e-3,    0.0,    "deep negative (vol rail)"),
        (-1e-3,    0.0,    "active negative"),
        (0.0,      0.0,    "threshold midrail"),
        (1e-3,     0.0,    "active positive"),
        (5e-3,     0.0,    "deep positive (voh rail)"),
        (0.901,    0.9,    "common-mode @ vdd/2, Δv=+1mV"),
        (0.899,    0.9,    "common-mode @ vdd/2, Δv=-1mV"),
    ];

    for &(v_plus, v_minus, label) in cases {
        let v_rlx = run_vout(v_plus, v_minus, comp.k, comp.voh, comp.vol);
        let res = ng.run_dc(
            &spice_deck(v_plus, v_minus, &comp),
            &[OutputRequest::NodeVoltage("out".into())],
        ).expect("ngspice .op");
        let v_ng = res.node_voltages["out"];

        // ngspice's .op converges to ~1e-6 absolute by default. f64
        // rlx should match well within that.
        assert!(
            (v_rlx - v_ng).abs() < 1e-6,
            "[{label}] rlx={v_rlx:+.6e} ng={v_ng:+.6e} |Δ|={:.3e}",
            (v_rlx - v_ng).abs(),
        );
    }
}
