//! Drive the OutputDoor's 8 inputs to a fixed pattern, pulse `clk`,
//! and verify every output bit equals its input bit. Renders all 17
//! signals (clk + 8 in + 8 out) as a stacked PNG to make the parallel
//! capture visually obvious.
//!
//! Pattern under test: `0xA5 = 10100101`. Picked because its bits are
//! a mix of 0s and 1s (catches "all bits stuck high/low" failures) and
//! it's not symmetric (catches MSB↔LSB swaps).

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, SpiceEmit};
use spike_output_door::OutputDoor;

const VDD: f64 = 1.8;
const RAIL_TOL: f64 = 0.05;
const PATTERN: u8 = 0xA5; // 10100101

fn assert_rail(actual: f64, expected_bit: u8, label: &str) {
    let target = if expected_bit == 1 { VDD } else { 0.0 };
    assert!(
        (actual - target).abs() < RAIL_TOL,
        "{label}: got {actual:.4} V, expected ≈ {target:.1} V (env {RAIL_TOL})",
    );
}

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
fn parallel_capture_8bit_pattern_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let door: OutputDoor<8> = OutputDoor::default();

    let mut net = Netlist::new("OutputDoor 8-bit parallel capture");
    net.add_dc_source("dd", "vdd", "0", VDD);
    // Drive each input to its bit's level via DC source. Stable inputs
    // before, during, and after the clock edge — exactly the SAR's
    // intended use of this block.
    for i in 0..8usize {
        let bit = (PATTERN >> i) & 1;
        let v = if bit == 1 { VDD } else { 0.0 };
        net.add_dc_source(&format!("b{i}"), &format!("in{i}"), "0", v);
    }
    // Single clock pulse: stays low until 200ns, then rises and stays
    // high. Captures all inputs on the rising edge.
    use eda_spice_emit::Pulse;
    net.add_pulse_source(
        "clk",
        "clk",
        "0",
        &Pulse {
            v_initial: 0.0,
            v_pulsed: VDD,
            t_delay: 200e-9,
            t_rise: 1e-9,
            t_fall: 1e-9,
            pulse_width: 100.0,
            period: 1e30,
        },
    );

    // Compose the OutputDoor's net list.
    let in_names: Vec<String>  = (0..8).map(|i| format!("in{i}")).collect();
    let out_names: Vec<String> = (0..8).map(|i| format!("out{i}")).collect();
    let mut nets: Vec<&str> = in_names.iter().map(String::as_str).collect();
    nets.push("clk");
    nets.extend(out_names.iter().map(String::as_str));
    nets.push("vdd");
    nets.push("0");
    door.emit_spice(&mut net, &nets, "od").unwrap();

    // DC operating point first so the latches don't start metastable.
    let analysis = TransientAnalysis {
        t_step: 5e-9,
        t_stop: 600e-9,
        use_initial_conditions: false,
        t_max: Some(5e-9),
    };
    let mut requests = vec![OutputRequest::NodeVoltage("clk".into())];
    for i in 0..8usize {
        requests.push(OutputRequest::NodeVoltage(format!("in{i}")));
        requests.push(OutputRequest::NodeVoltage(format!("out{i}")));
    }
    let trace = ng
        .run_transient_trace(&net.deck(), &analysis, &requests)
        .expect("ngspice tran");
    let t = &trace.time;

    // Sample 200ns past the rising edge (~400ns absolute) — well past
    // the slave's settling time.
    for i in 0..8usize {
        let bit = (PATTERN >> i) & 1;
        let out = &trace.node_voltages[&format!("out{i}")];
        let v = sample_at(t, out, 400e-9);
        assert_rail(v, bit, &format!("OutputDoor out{i} (expected bit {bit})"));
    }

    render_capture(&trace);
}

fn render_capture(trace: &eda_extern_ngspice::TransientTrace) {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use eda_waveform::{plot, Waveform};

    let mut signals = BTreeMap::new();
    signals.insert("clk".into(), trace.node_voltages["clk"].clone());
    // Outputs first (so they read top-down 0..7 in stacked layout).
    for i in 0..8usize {
        let key = format!("out{i}");
        if let Some(v) = trace.node_voltages.get(&key) {
            signals.insert(key, v.clone());
        }
    }
    let wave = Waveform::Real {
        axis_name: "time (s)".into(),
        axis: trace.time.clone(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title(format!(
            "OutputDoor: parallel capture of 0x{PATTERN:02X} = {PATTERN:08b} on clk rising edge",
        ))
        .with_layout(plot::Layout::Stacked)
        .with_size(900, 1100)
        .add_marker(plot::Marker::Vertical {
            x: 200e-9,
            label: Some("clk↑".into()),
        });
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("output_door_capture.png");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, out_dir.join("output_door_capture.svg"), &cfg).expect("svg");
    eprintln!("OutputDoor capture trace at {}", png.display());
}
