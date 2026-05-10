//! Three-way triangulation: rlx outer-loop BE vs ngspice transient vs
//! LTspice transient, all driven by the same `SourceWaveform::Pulse`
//! through the same `eda_spice_emit::Netlist::add_waveform_source`.
//!
//! ## What this proves
//!
//! All three solvers agree on a PULSE-driven RC transient to within the
//! published tolerance. Same deck text fed to ngspice and LTspice; same
//! `SourceWaveform::value_at(t)` sampled by the rlx outer loop. The
//! validation is the Phase-1 harness (`compare_transient_traces`)
//! handling the inevitable adaptive-vs-uniform grid mismatch via
//! interpolation.
//!
//! ## Visual artifact
//!
//! When run with both features, the test also dumps a PNG to
//! `crates/spike-pulse-rc/docs/triangulate.png` overlaying the three
//! traces. Useful as the canonical "look, they really do agree"
//! screenshot.
//!
//! ## Soft-skip behavior
//!
//! Each backend's presence is checked at runtime via
//! `LocalBinary::from_env_optional` — the test passes (with eprintln
//! notes) when one or both simulators are missing, so CI without
//! LTspice still validates the rlx-vs-ngspice arc.

#![cfg(all(feature = "ngspice", feature = "ltspice"))]

use std::collections::BTreeMap;
use std::path::PathBuf;

use eda_extern_ltspice::{self as lt, Invoker as _};
use eda_extern_ngspice::{self as ng, Invoker as _};
use eda_hir::SourceWaveform;
use eda_validate::compare_transient_traces;
use eda_waveform::{plot, Waveform};
use spike_pulse_rc::*;

#[test]
fn rlx_ngspice_ltspice_agree_on_pulse_rc() {
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let tau = r * c;
    let w = SourceWaveform::pulse(0.0, 1.0, 200e-9, 1e-12, 1e-12, 2.0 * tau, 0.0);
    let t_stop = 4.0 * tau;
    let h = tau / 200.0;
    let n_steps = (t_stop / h).round() as usize;

    // — rlx outer-loop BE
    let (rlx_t, rlx_v) = run_transient_trace(n_steps, h, r, c, 0.0, &w);

    // — ngspice .tran (pinned tmax = h to match the uniform grid)
    let (ng_t, ng_v) = match ng::LocalBinary::from_env() {
        Ok(invoker) => {
            let trace = invoker
                .run_transient_trace(
                    &spice_deck(r, c, &w),
                    &ng::TransientAnalysis::new(h, t_stop).with_t_max(h),
                    &[ng::OutputRequest::NodeVoltage("vout".into())],
                )
                .expect("ngspice tran");
            (trace.time, trace.node_voltages["vout"].clone())
        }
        Err(e) => {
            eprintln!("ngspice missing ({e}); test soft-skips here");
            return;
        }
    };

    // — LTspice .tran
    let (lt_t, lt_v) = match lt::LocalBinary::from_env_optional() {
        Some(invoker) => {
            let trace = invoker
                .run_transient_trace(
                    &spice_deck(r, c, &w),
                    &lt::TransientAnalysis::new(h, t_stop),
                    &[lt::OutputRequest::NodeVoltage("vout".into())],
                )
                .expect("LTspice tran");
            (trace.time, trace.node_voltages["vout"].clone())
        }
        None => {
            eprintln!("LTspice missing; finishing as rlx-vs-ngspice only");
            // Render with the two we have and exit successfully.
            render_overlay(&[
                ("rlx (BE)", &rlx_t, &rlx_v),
                ("ngspice", &ng_t, &ng_v),
            ]);
            return;
        }
    };

    // Triangulation diff. Reference grid is the rlx uniform-h trace;
    // ngspice and LTspice get interpolated onto it.
    let mut rlx_map = BTreeMap::new();
    rlx_map.insert("vout".to_string(), rlx_v.clone());
    let mut ng_map = BTreeMap::new();
    ng_map.insert("vout".to_string(), ng_v.clone());
    let mut lt_map = BTreeMap::new();
    lt_map.insert("vout".to_string(), lt_v.clone());

    // BTreeMap → HashMap for the diff helper. Per-pair tolerance: 5%
    // peak relative + 5 mV absolute (BDF1-vs-BDF1 plateau), same as
    // the per-backend tests.
    use std::collections::HashMap;
    let to_hash = |m: &BTreeMap<String, Vec<f64>>| -> HashMap<String, Vec<f64>> {
        m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    };

    let r_ng = compare_transient_traces(&rlx_t, &to_hash(&rlx_map), &ng_t, &to_hash(&ng_map));
    let r_lt = compare_transient_traces(&rlx_t, &to_hash(&rlx_map), &lt_t, &to_hash(&lt_map));
    eprintln!("rlx vs ngspice: {:?}", r_ng.worst);
    eprintln!("rlx vs LTspice: {:?}", r_lt.worst);
    // Plateau peak ≈ 1V; tolerance 5% peak ⇒ envelope ≈ 50 mV.
    r_ng.assert_within(0.0, 0.05, /*peak*/ 1.0, "rlx vs ngspice");
    r_lt.assert_within(0.0, 0.05, /*peak*/ 1.0, "rlx vs LTspice");

    render_overlay(&[
        ("rlx (BE)", &rlx_t, &rlx_v),
        ("ngspice", &ng_t, &ng_v),
        ("LTspice", &lt_t, &lt_v),
    ]);
}

/// Build a `Waveform::Real` with one signal per backend and dump it as a
/// stacked PNG into the crate's `docs/` directory. The stacked layout
/// makes it easy to spot a divergence — if all three traces look
/// identical visually, the test PASSED for the right reason.
fn render_overlay(traces: &[(&str, &[f64], &[f64])]) {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();

    // For an overlay, each trace gets its own series — but they sit on
    // *different* time grids. Resample each onto a common dense grid
    // (the rlx grid is fine — it's the densest uniform one).
    let (_, ref_t, _) = traces[0];
    let t_axis: Vec<f64> = ref_t.to_vec();

    let mut signals = BTreeMap::new();
    for (name, t, v) in traces {
        let resampled: Vec<f64> = t_axis
            .iter()
            .map(|&tq| eda_validate::lerp(t, v, tq))
            .collect();
        signals.insert((*name).to_string(), resampled);
    }
    let wave = Waveform::Real {
        axis_name: "time (s)".into(),
        axis: t_axis,
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title("PULSE → RC: rlx vs ngspice vs LTspice (overlay)")
        .with_size(900, 500);
    let png = out_dir.join("triangulate.png");
    let svg = out_dir.join("triangulate.svg");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, &svg, &cfg).expect("svg");
    eprintln!("triangulation overlay written to {}", png.display());
}
