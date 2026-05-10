//! Tier 3-bis: rlx outer-loop transient against ngspice **full trace**.
//!
//! Where `ngspice.rs` validated only the final-time scalar `vout(t_stop)`,
//! this test compares the entire waveform: every interior timestep, not
//! just the endpoint. The ngspice grid is adaptive (LTE-controlled, even
//! with `method=gear maxord=1`) while the rlx grid is uniform — so we use
//! `eda_validate::assert_traces_close`, which interpolates one onto the
//! other before tolerance-checking.
//!
//! This is the first test that exercises the new
//! `Invoker::run_transient_trace` + Nutmeg-binary parser path.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_validate::assert_traces_close;
use spike_rc_transient::*;

#[test]
fn rlx_trace_matches_ngspice_trace() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: {e}"); return; }
    };

    // One representative case: 1 RC time constant, 200 BE steps. Plenty
    // of interior points to catch any per-step drift between the two
    // solvers. Two ngspice and rlx waveforms should match to ~5e-3
    // relative — looser than the endpoint check because interpolation
    // onto ngspice's near-singular early steps amplifies noise.
    let v_dc = 1.0_f64;
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let n_steps = 200_usize;
    let rc = r * c;
    let t_stop = 1.0 * rc;
    let h = t_stop / n_steps as f64;

    // rlx side: uniform-grid BE.
    let (rlx_t, rlx_v) = run_transient_trace(n_steps, h, r, c, 0.0, |_| v_dc);

    // ngspice side: adaptive .tran with BDF1 forced on.
    let trace = ng
        .run_transient_trace(
            &spice_deck(v_dc, r, c),
            &TransientAnalysis::new(h, t_stop),
            &[OutputRequest::NodeVoltage("vout".into())],
        )
        .expect("ngspice .tran trace failed");
    let ng_v = &trace.node_voltages["vout"];

    assert_traces_close(
        &rlx_t, &rlx_v,
        &trace.time, ng_v,
        5e-3, 1e-6,
        "rlx vs ngspice RC LP transient trace",
    );

    // Sanity: trace endpoints agree to roughly the same envelope as the
    // existing `run_transient_final` test, confirming we're measuring the
    // same circuit through a different code path.
    let rlx_final = *rlx_v.last().unwrap();
    let ng_final = *ng_v.last().unwrap();
    let rel_err = (rlx_final - ng_final).abs() / ng_final.abs();
    assert!(rel_err < 5e-3, "endpoint divergence: rlx={rlx_final} ng={ng_final}");
}
