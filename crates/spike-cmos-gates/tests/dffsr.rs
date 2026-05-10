//! `DffSR` validation through ngspice (and LTspice when present).
//!
//! Three properties to validate:
//!   1. **Set override**: with `reset_b = 1`, asserting `set_b = 0`
//!      forces `q = 1` regardless of `clk` and `d`.
//!   2. **Reset override**: with `set_b = 1`, asserting `reset_b = 0`
//!      forces `q = 0` regardless of `clk` and `d`.
//!   3. **Normal mode**: with both `set_b = reset_b = 1`, behaves as a
//!      plain positive-edge-triggered DFF — exercised by reusing the
//!      same shape of test as the [`Dff`] suite.
//!
//! The set/reset overrides are tested via transient: hold `clk` low
//! (so the slave is opaque) and `d` opposite to the override target,
//! then pulse the override low and sample `q` immediately.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, SpiceEmit};
use spike_cmos_gates::DffSR;

const VDD: f64 = 1.8;
const RAIL_TOL: f64 = 0.05;

fn dc_high() -> f64 { VDD }
fn dc_low() -> f64 { 0.0 }

fn assert_rail(actual: f64, expected_bit: u8, label: &str) {
    let target = if expected_bit == 1 { VDD } else { 0.0 };
    assert!(
        (actual - target).abs() < RAIL_TOL,
        "{label}: got {actual:.4} V, expected ≈ {target:.1} V (env {RAIL_TOL})",
    );
}

/// Sample `q` at `t_query` from a transient trace.
fn sample_at(t: &[f64], y: &[f64], t_query: f64) -> f64 {
    let idx = t
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - t_query).abs().partial_cmp(&(*b - t_query).abs()).unwrap()
        })
        .unwrap()
        .0;
    y[idx]
}

#[test]
fn dffsr_set_b_forces_q_high_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let f = DffSR::default();
    let mut net = Netlist::new("DffSR set override");
    net.add_dc_source("dd", "vdd", "0", VDD);
    // d held LOW (would normally drive q→0 on rising edge).
    net.add_dc_source("ind", "d", "0", dc_low());
    // clk held LOW the entire run — no rising edge ever fires.
    net.add_dc_source("inc", "clk", "0", dc_low());
    // reset_b held HIGH (inactive).
    net.add_dc_source("inrb", "rb", "0", dc_high());
    // set_b: HIGH for first 200 ns, then LOW (asserts set), then HIGH again.
    use eda_spice_emit::Pulse;
    net.add_pulse_source(
        "insb",
        "sb",
        "0",
        &Pulse {
            v_initial: VDD,
            v_pulsed: 0.0,
            t_delay: 200e-9,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 200e-9,
            period: 1e30,
        },
    );
    f.emit_spice(&mut net, &["d", "clk", "sb", "rb", "q", "qb", "vdd", "0"], "f1")
        .unwrap();

    // Use DC operating point first so the latches don't start
    // metastable.
    let analysis = TransientAnalysis {
        t_step: 1e-9,
        t_stop: 600e-9,
        use_initial_conditions: false,
        t_max: Some(1e-9),
    };
    let trace = ng
        .run_transient_trace(
            &net.deck(),
            &analysis,
            &[OutputRequest::NodeVoltage("q".into())],
        )
        .expect("ngspice tran");
    let q = &trace.node_voltages["q"];
    let t = &trace.time;

    // While set_b is asserted (250 ns to 400 ns), q must be ~VDD.
    assert_rail(sample_at(t, q, 350e-9), 1, "DffSR set_b asserted, q@350ns");
    // Set_b releases at t=400ns; q should HOLD high (no clock edge to
    // overwrite, no reset).
    assert_rail(sample_at(t, q, 550e-9), 1, "DffSR set_b released, q@550ns");
}

#[test]
fn dffsr_reset_b_forces_q_low_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let f = DffSR::default();
    let mut net = Netlist::new("DffSR reset override");
    net.add_dc_source("dd", "vdd", "0", VDD);
    // d held HIGH (would normally drive q→1 on rising edge).
    net.add_dc_source("ind", "d", "0", dc_high());
    // clk held LOW.
    net.add_dc_source("inc", "clk", "0", dc_low());
    // set_b held HIGH (inactive).
    net.add_dc_source("insb", "sb", "0", dc_high());
    // reset_b: HIGH then LOW (assert) then HIGH.
    use eda_spice_emit::Pulse;
    net.add_pulse_source(
        "inrb",
        "rb",
        "0",
        &Pulse {
            v_initial: VDD,
            v_pulsed: 0.0,
            t_delay: 200e-9,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 200e-9,
            period: 1e30,
        },
    );
    f.emit_spice(&mut net, &["d", "clk", "sb", "rb", "q", "qb", "vdd", "0"], "f1")
        .unwrap();

    let analysis = TransientAnalysis {
        t_step: 1e-9,
        t_stop: 600e-9,
        use_initial_conditions: false,
        t_max: Some(1e-9),
    };
    let trace = ng
        .run_transient_trace(
            &net.deck(),
            &analysis,
            &[OutputRequest::NodeVoltage("q".into())],
        )
        .expect("ngspice tran");
    let q = &trace.node_voltages["q"];
    let t = &trace.time;

    assert_rail(sample_at(t, q, 350e-9), 0, "DffSR reset_b asserted, q@350ns");
    // After reset releases, q must HOLD low (no clock edge fires).
    assert_rail(sample_at(t, q, 550e-9), 0, "DffSR reset_b released, q@550ns");
}

/// Mirror of `dff_captures_d_on_rising_edge_ngspice` from the basic Dff
/// suite: with both overrides inactive, DffSR behaves like a plain DFF.
#[test]
fn dffsr_normal_mode_matches_plain_dff_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let f = DffSR::default();
    let mut net = Netlist::new("DffSR normal mode");
    net.add_dc_source("dd", "vdd", "0", VDD);
    // Both overrides inactive.
    net.add_dc_source("insb", "sb", "0", dc_high());
    net.add_dc_source("inrb", "rb", "0", dc_high());
    // 1 MHz clock, first rising edge at t=0.5µs.
    use eda_spice_emit::{Pulse, Pwl};
    net.add_pulse_source(
        "clk",
        "clk",
        "0",
        &Pulse {
            v_initial: 0.0,
            v_pulsed: VDD,
            t_delay: 0.5e-6,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 0.5e-6 - 2e-9,
            period: 1e-6,
        },
    );
    // Same PWL data shape as the basic Dff test.
    net.add_pwl_source(
        "ind",
        "d",
        "0",
        &Pwl {
            points: vec![
                (0.0,            VDD),
                (0.7e-6,         VDD),
                (0.7e-6 + 1e-9,  0.0),
                (1.7e-6,         0.0),
                (1.7e-6 + 1e-9,  VDD),
                (3.5e-6,         VDD),
            ],
        },
    );
    f.emit_spice(&mut net, &["d", "clk", "sb", "rb", "q", "qb", "vdd", "0"], "f1")
        .unwrap();

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
            &[OutputRequest::NodeVoltage("q".into())],
        )
        .expect("ngspice tran");
    let q = &trace.node_voltages["q"];
    let t = &trace.time;
    let sample_after_edge = |t_edge: f64, expected: u8, label: &str| {
        assert_rail(sample_at(t, q, t_edge + 200e-9), expected, label);
    };
    sample_after_edge(0.5e-6, 1, "DffSR normal capture clk@0.5µs (d=1)");
    sample_after_edge(1.5e-6, 0, "DffSR normal capture clk@1.5µs (d=0)");
    sample_after_edge(2.5e-6, 1, "DffSR normal capture clk@2.5µs (d=1)");
    sample_after_edge(3.5e-6, 1, "DffSR normal capture clk@3.5µs (d=1)");
}

/// Combined transient mirroring Fig 5.1 of the LTspice paper: reset
/// pulse first, then a few normal clocked captures, then a set pulse
/// at the end. Renders a stacked PNG showing the three operating
/// modes in one trace.
#[test]
fn dffsr_combined_reset_normal_set_renders_png() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let f = DffSR::default();
    let mut net = Netlist::new("DffSR Fig 5.1 reproduction");
    net.add_dc_source("dd", "vdd", "0", VDD);
    use eda_spice_emit::{Pulse, Pwl};
    // 1 MHz clock for the whole run.
    net.add_pulse_source(
        "clk",
        "clk",
        "0",
        &Pulse {
            v_initial: 0.0,
            v_pulsed: VDD,
            t_delay: 0.5e-6,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 0.5e-6 - 2e-9,
            period: 1e-6,
        },
    );
    // d toggles between rising edges so each capture sees a different value.
    net.add_pwl_source(
        "ind",
        "d",
        "0",
        &Pwl {
            points: vec![
                (0.0,            VDD),
                (1.2e-6,         VDD),
                (1.2e-6 + 1e-9, 0.0),
                (2.2e-6,         0.0),
                (2.2e-6 + 1e-9, VDD),
                (8.0e-6,         VDD),
            ],
        },
    );
    // reset_b: low for first 0.5 µs (initial reset).
    net.add_pulse_source(
        "inrb",
        "rb",
        "0",
        &Pulse {
            v_initial: 0.0,
            v_pulsed: VDD,
            t_delay: 0.5e-6,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 100.0,
            period: 1e30,
        },
    );
    // set_b: low for 5.5 µs..6.5 µs (forces q=1 mid-stream).
    net.add_pulse_source(
        "insb",
        "sb",
        "0",
        &Pulse {
            v_initial: VDD,
            v_pulsed: 0.0,
            t_delay: 5.5e-6,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 1.0e-6,
            period: 1e30,
        },
    );
    f.emit_spice(&mut net, &["d", "clk", "sb", "rb", "q", "qb", "vdd", "0"], "f1")
        .unwrap();

    let analysis = TransientAnalysis {
        t_step: 10e-9,
        t_stop: 8e-6,
        use_initial_conditions: false,
        t_max: Some(10e-9),
    };
    let trace = ng
        .run_transient_trace(
            &net.deck(),
            &analysis,
            &[
                OutputRequest::NodeVoltage("clk".into()),
                OutputRequest::NodeVoltage("d".into()),
                OutputRequest::NodeVoltage("sb".into()),
                OutputRequest::NodeVoltage("rb".into()),
                OutputRequest::NodeVoltage("q".into()),
            ],
        )
        .expect("ngspice tran");
    let q = &trace.node_voltages["q"];
    let t = &trace.time;

    // Asserts at three checkpoints for the three operating modes.
    // PWL drives d=1 until t=1.2µs, d=0 from 1.2 to 2.2µs, d=1 thereafter.
    // Clock rising edges at 0.5, 1.5, 2.5, 3.5, 4.5 µs ⇒ q-tracks-d as
    // 1, 0, 1, 1, 1 (subject to reset/set overrides at the edges).
    assert_rail(sample_at(t, q, 0.4e-6), 0, "reset window: q low");
    assert_rail(sample_at(t, q, 2.7e-6), 1, "post-reset clk@2.5µs (d=1) → q=1");
    assert_rail(sample_at(t, q, 6.0e-6), 1, "set window: q high");

    render_combined(&trace);
}

fn render_combined(trace: &eda_extern_ngspice::TransientTrace) {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use eda_waveform::{plot, Waveform};

    let mut signals = BTreeMap::new();
    for k in ["clk", "d", "sb", "rb", "q"] {
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
        .with_title("DffSR: reset → normal clocked → set (Fig 5.1 reproduction)")
        .with_layout(plot::Layout::Stacked)
        .with_size(900, 900);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("dffsr_combined.png");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, out_dir.join("dffsr_combined.svg"), &cfg).expect("svg");
    eprintln!("DffSR combined trace at {}", png.display());
}
