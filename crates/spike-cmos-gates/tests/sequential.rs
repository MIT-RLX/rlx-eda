//! DLatch + Dff validation through ngspice (and LTspice when present).
//!
//! ## DLatch (gated D latch, level-sensitive)
//!
//! Validated via `.op` enumeration. The classical truth-table approach
//! breaks down for sequential cells because the latch has *memory* —
//! the output depends on previous state. Workaround: hold `en = 1` so
//! the latch is transparent, then the steady-state Vout = D·Vdd. That
//! exercises the input gating + the SR latch when forced into the
//! "track" mode. Hold mode (en=0) is then validated by a transient
//! (clock falls, then D changes — Q must stay).
//!
//! ## Dff (positive-edge-triggered)
//!
//! Validated via transient: PULSE clock at 1 MHz, PWL data that toggles
//! a few times *between* clock edges. After each rising edge of `clk`,
//! `q` must equal whatever `d` was *just before* that edge. We sample
//! `q` 200 ns after each rising edge (well into the slave's
//! transparency window) and assert the match.
//!
//! ## Soft-skip
//!
//! Both backends gated by Cargo features and runtime presence checks.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, SpiceEmit};
use spike_cmos_gates::{deck_for_levels, DLatch, Dff};

const VDD: f64 = 1.8;
const RAIL_TOL: f64 = 0.05;

fn vout(ng: &LocalBinary, deck: &str, node: &str) -> f64 {
    let res = ng
        .run_dc(deck, &[OutputRequest::NodeVoltage(node.into())])
        .expect("ngspice .op");
    res.node_voltages[node]
}

fn assert_rail(actual: f64, expected_bit: u8, label: &str) {
    let target = if expected_bit == 1 { VDD } else { 0.0 };
    assert!(
        (actual - target).abs() < RAIL_TOL,
        "{label}: got {actual:.4} V, expected ≈ {target:.1} V (env {RAIL_TOL})",
    );
}

#[test]
fn dlatch_transparent_when_enabled_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let l = DLatch::default();
    // Hold en=1 (transparent). Vary D. Steady-state Q must follow D.
    for d_bit in [0u8, 1] {
        let mut net = Netlist::new("DLatch transparent probe");
        net.add_dc_source("dd", "vdd", "0", VDD);
        net.add_dc_source("ind", "d", "0", if d_bit == 1 { VDD } else { 0.0 });
        net.add_dc_source("ine", "en", "0", VDD); // en = 1
        l.emit_spice(&mut net, &["d", "en", "q", "qb", "vdd", "0"], "l1").unwrap();
        let q = vout(&ng, &net.deck(), "q");
        assert_rail(q, d_bit, &format!("DLatch transparent d={d_bit}"));
    }
}

/// Avoid relying on .op picking the right SR-latch state when both
/// active-low inputs are forced to 1 (the "memory" case). Use transient
/// instead: bring the latch out of reset by enabling, set D, then drop
/// EN and toggle D — Q must stay.
#[test]
fn dlatch_holds_when_disabled_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let l = DLatch::default();
    let mut net = Netlist::new("DLatch hold probe");
    net.add_dc_source("dd", "vdd", "0", VDD);
    // EN: 0V for 0..1µs, 1V for 1..2µs, 0V after — captures whatever
    // D is at the falling edge.
    net.add_pulse_source(
        "ine",
        "en",
        "0",
        &eda_spice_emit::Pulse {
            v_initial: 0.0,
            v_pulsed: VDD,
            t_delay: 1e-6,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 1e-6 - 2e-9,
            period: 1e30,
        },
    );
    // D: 1V from t=0..3µs, then 0V — toggles AFTER en falls back to 0.
    // Q (sampled at t=4µs) must still be 1V because the latch held the
    // value captured during the EN window.
    net.add_pulse_source(
        "ind",
        "d",
        "0",
        &eda_spice_emit::Pulse {
            v_initial: VDD,
            v_pulsed: 0.0,
            t_delay: 3e-6,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 100.0,
            period: 1e30,
        },
    );
    l.emit_spice(&mut net, &["d", "en", "q", "qb", "vdd", "0"], "l1").unwrap();

    // Disable uic: let ngspice solve a DC operating point first so the
    // cross-coupled SR latch has a defined initial state. With uic and
    // Q=qb=0V, the symmetric initial condition leaves the latch in
    // numerical metastability and it never resolves to a clean rail.
    let analysis = TransientAnalysis {
        t_step: 10e-9,
        t_stop: 4e-6,
        use_initial_conditions: false,
        t_max: Some(10e-9),
    };
    let trace = ng
        .run_transient_trace(
            &net.deck(),
            &analysis,
            &[
                OutputRequest::NodeVoltage("q".into()),
                OutputRequest::NodeVoltage("qb".into()),
                OutputRequest::NodeVoltage("d".into()),
                OutputRequest::NodeVoltage("en".into()),
                OutputRequest::NodeVoltage("l1_a".into()),
                OutputRequest::NodeVoltage("l1_b".into()),
            ],
        )
        .expect("ngspice tran");
    // Render for debug visibility regardless of pass/fail.
    render_dlatch_debug(&trace);

    let q = &trace.node_voltages["q"];
    let t = &trace.time;
    let idx = t.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - 3.9e-6).abs().partial_cmp(&(*b - 3.9e-6).abs()).unwrap())
        .unwrap()
        .0;
    assert!(
        (q[idx] - VDD).abs() < RAIL_TOL,
        "DLatch hold failed: at t={:.2e} q={:.4} V, expected ~{VDD}",
        t[idx], q[idx],
    );
}

fn render_dlatch_debug(trace: &eda_extern_ngspice::TransientTrace) {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use eda_waveform::{plot, Waveform};
    let mut signals = BTreeMap::new();
    for k in ["en", "d", "l1_a", "l1_b", "q", "qb"] {
        if let Some(v) = trace.node_voltages.get(k) {
            signals.insert(k.into(), v.clone());
        }
    }
    let wave = Waveform::Real {
        axis_name: "time (s)".into(),
        axis: trace.time.clone(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title("DLatch hold-mode debug trace")
        .with_layout(plot::Layout::Stacked)
        .with_size(900, 900);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    plot::png_to_path(&wave, out_dir.join("dlatch_hold_debug.png"), &cfg).expect("png");
    eprintln!("DLatch debug trace at {}", out_dir.join("dlatch_hold_debug.png").display());
}

/// Positive-edge-triggered DFF: drive `d` with a slow PWL that flips
/// between clock edges, sample `q` shortly *after* each rising edge,
/// assert it matches the `d` value held *during* that edge.
#[test]
fn dff_captures_d_on_rising_edge_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let f = Dff::default();
    let mut net = Netlist::new("DFF rising-edge capture");
    net.add_dc_source("dd", "vdd", "0", VDD);
    // Clock: 1 MHz, 50% duty. Period 1µs, width 0.5µs, rise/fall 1ns.
    net.add_pulse_source(
        "clk",
        "clk",
        "0",
        &eda_spice_emit::Pulse {
            v_initial: 0.0,
            v_pulsed: VDD,
            t_delay: 0.5e-6, // first rising edge at t=0.5µs
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 0.5e-6 - 2e-9,
            period: 1e-6,
        },
    );
    // Data: PWL chosen so that at each rising edge, d has a known value.
    //   t=0      : d=1   ⇒ at clk@0.5µs, capture 1
    //   t=0.7µs  : d=0   ⇒ at clk@1.5µs, capture 0
    //   t=1.7µs  : d=1   ⇒ at clk@2.5µs, capture 1
    //   t=2.7µs  : d=1   ⇒ at clk@3.5µs, capture 1 (no change)
    use eda_spice_emit::Pwl;
    net.add_pwl_source(
        "ind",
        "d",
        "0",
        &Pwl {
            points: vec![
                (0.0,        VDD),
                (0.7e-6,     VDD),
                (0.7e-6 + 1e-9, 0.0),
                (1.7e-6,     0.0),
                (1.7e-6 + 1e-9, VDD),
                (3.5e-6,     VDD),
            ],
        },
    );
    f.emit_spice(&mut net, &["d", "clk", "q", "qb", "vdd", "0"], "f1").unwrap();

    let trace = ng
        .run_transient_trace(
            &net.deck(),
            &TransientAnalysis::new(10e-9, 4e-6).with_t_max(10e-9),
            &[
                OutputRequest::NodeVoltage("clk".into()),
                OutputRequest::NodeVoltage("d".into()),
                OutputRequest::NodeVoltage("q".into()),
            ],
        )
        .expect("ngspice tran");
    let t = &trace.time;
    let q = &trace.node_voltages["q"];

    // Sample q 200ns after each rising edge — well past the slave
    // transparency-window settling time (LEVEL=1 inverter at this size
    // settles in <50ns for a 1.8V swing).
    let sample_after = |t_edge: f64, expected_bit: u8, label: &str| {
        let target_t = t_edge + 200e-9;
        let idx = t.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - target_t).abs().partial_cmp(&(*b - target_t).abs()).unwrap())
            .unwrap()
            .0;
        assert_rail(q[idx], expected_bit, &format!("{label} (q@t={:.2e})", t[idx]));
    };
    sample_after(0.5e-6, 1, "DFF capture clk@0.5µs (d=1)");
    sample_after(1.5e-6, 0, "DFF capture clk@1.5µs (d=0)");
    sample_after(2.5e-6, 1, "DFF capture clk@2.5µs (d=1)");
    sample_after(3.5e-6, 1, "DFF capture clk@3.5µs (d=1)");

    // Render the transient as PNG so we can eyeball the staircase.
    render_transient(&trace);
}

fn render_transient(trace: &eda_extern_ngspice::TransientTrace) {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use eda_waveform::{plot, Waveform};

    let mut signals = BTreeMap::new();
    signals.insert("clk".into(), trace.node_voltages["clk"].clone());
    signals.insert("d".into(),   trace.node_voltages["d"].clone());
    signals.insert("q".into(),   trace.node_voltages["q"].clone());
    let wave = Waveform::Real {
        axis_name: "time (s)".into(),
        axis: trace.time.clone(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title("D flip-flop: q captures d on rising clk edges")
        .with_layout(plot::Layout::Stacked)
        .with_size(900, 700);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("dff_transient.png");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, out_dir.join("dff_transient.svg"), &cfg).expect("svg");
    eprintln!("DFF transient written to {}", png.display());
}
