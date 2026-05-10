//! 4-bit ripple counter at 4 MHz. Each stage's q must oscillate at
//! `4 MHz / 2^(stage+1)`. Validate by counting zero-crossings in the
//! transient trace and comparing to the expected count for the
//! observation window.
//!
//! Why zero-crossings instead of an FFT? At 4 MHz with a 10 µs window
//! we have only 40 input cycles — too few for an FFT to give clean
//! frequency bins. Zero-crossings give an exact integer count and
//! work even when the transient hasn't reached steady state.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pulse, SpiceEmit};
use spike_ripple_counter::RippleCounter;

const VDD: f64 = 1.8;
const F_IN: f64 = 4e6; // 4 MHz input clock
const T_STOP: f64 = 10e-6; // 10 µs window = 40 input cycles
const T_RESET_RELEASE: f64 = 0.2e-6; // Hold reset for 200 ns at startup

/// Count rising zero-crossings (low→high transitions across vdd/2) in
/// `y`, restricted to `t ≥ start_t`. Skipping the first 200 ns avoids
/// the reset window.
fn count_rising_edges(t: &[f64], y: &[f64], start_t: f64) -> usize {
    let thr = VDD / 2.0;
    let mut count = 0usize;
    let mut prev_above = false;
    for (i, &ts) in t.iter().enumerate() {
        if ts < start_t { prev_above = y[i] >= thr; continue; }
        let above = y[i] >= thr;
        if above && !prev_above { count += 1; }
        prev_above = above;
    }
    count
}

#[test]
fn ripple_counter_4_bit_at_4_mhz_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let counter: RippleCounter<4> = RippleCounter::default();

    let mut net = Netlist::new("4-bit ripple counter @ 4 MHz");
    net.add_dc_source("dd", "vdd", "0", VDD);
    // 4 MHz square clock (50% duty), starting at t = 0.
    let period = 1.0 / F_IN;
    net.add_pulse_source(
        "in",
        "clk_in",
        "0",
        &Pulse {
            v_initial: 0.0,
            v_pulsed: VDD,
            t_delay: 0.0,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: period * 0.5 - 2e-9,
            period,
        },
    );
    // reset_b: low for first 200 ns, then high.
    net.add_pulse_source(
        "rb",
        "rb",
        "0",
        &Pulse {
            v_initial: 0.0,
            v_pulsed: VDD,
            t_delay: T_RESET_RELEASE,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 100.0,
            period: 1e30,
        },
    );

    let qs: Vec<String> = (0..4).map(|i| format!("q{i}")).collect();
    let mut nets: Vec<&str> = vec!["clk_in", "rb"];
    nets.extend(qs.iter().map(String::as_str));
    nets.push("vdd");
    nets.push("0");
    counter.emit_spice(&mut net, &nets, "rc").unwrap();

    let analysis = TransientAnalysis {
        t_step: 5e-9,
        t_stop: T_STOP,
        use_initial_conditions: false,
        t_max: Some(5e-9),
    };
    let mut requests = vec![OutputRequest::NodeVoltage("clk_in".into())];
    for q in &qs {
        requests.push(OutputRequest::NodeVoltage(q.clone()));
    }
    let trace = ng
        .run_transient_trace(&net.deck(), &analysis, &requests)
        .expect("ngspice tran");

    let t = &trace.time;
    // After reset releases (t ≥ 0.5 µs to give the counter a clock or two
    // to start ticking), count rising edges over the remaining window.
    let observe_start = 0.5e-6;
    let observe_window = T_STOP - observe_start;
    let observe_cycles_in = (observe_window * F_IN).round() as usize;
    eprintln!("observation window: {observe_window:.2e} s, {observe_cycles_in} input cycles");

    // Expected rising-edge counts:
    //   q0: F_IN / 2 → input_cycles / 2
    //   q1: F_IN / 4 → input_cycles / 4
    //   q2: F_IN / 8 → input_cycles / 8
    //   q3: F_IN / 16 → input_cycles / 16
    // Allow ±1 tolerance for window-boundary effects.
    let expected = [
        observe_cycles_in / 2,
        observe_cycles_in / 4,
        observe_cycles_in / 8,
        observe_cycles_in / 16,
    ];
    for (i, q_name) in qs.iter().enumerate() {
        let q = &trace.node_voltages[q_name];
        let edges = count_rising_edges(t, q, observe_start);
        let exp = expected[i];
        eprintln!("q{i}: {edges} rising edges (expected ~{exp})");
        assert!(
            edges.abs_diff(exp) <= 1,
            "q{i}: counted {edges} edges, expected {exp} ± 1",
        );
    }

    render(&trace);
}

fn render(trace: &eda_extern_ngspice::TransientTrace) {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use eda_waveform::{plot, Waveform};

    let mut signals = BTreeMap::new();
    signals.insert("clk_in (4 MHz)".into(), trace.node_voltages["clk_in"].clone());
    signals.insert("q0 (2 MHz)".into(), trace.node_voltages["q0"].clone());
    signals.insert("q1 (1 MHz)".into(), trace.node_voltages["q1"].clone());
    signals.insert("q2 (500 kHz)".into(), trace.node_voltages["q2"].clone());
    signals.insert("q3 (250 kHz)".into(), trace.node_voltages["q3"].clone());
    let wave = Waveform::Real {
        axis_name: "time (s)".into(),
        axis: trace.time.clone(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title("4-bit ripple counter: divide-by-2 chain")
        .with_layout(plot::Layout::Stacked)
        .with_size(900, 800);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("ripple_counter.png");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, out_dir.join("ripple_counter.svg"), &cfg).expect("svg");
    eprintln!("ripple counter trace at {}", png.display());
}
