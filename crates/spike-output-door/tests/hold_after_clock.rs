//! Latch hold property: drive pattern A, pulse clock to capture, then
//! change inputs to pattern B without another clock edge — outputs
//! must continue to read pattern A. Witness against the
//! `behavioral_capture` truth-table reference.
//!
//! This is the property that distinguishes a latch from a buffer: if
//! the OutputDoor were missing the storage element, outputs would
//! follow inputs through and the test would fail.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pulse, Pwl, SpiceEmit};
use spike_output_door::{behavioral_capture, OutputDoor};

const VDD: f64 = 1.8;
const RAIL_TOL: f64 = 0.05;

/// Pattern A captured at the clock edge; pattern B applied later.
/// Bits are independent so an MSB↔LSB routing bug or a bit-leak
/// fault on any single bit fails its assertion.
const PATTERN_A: u8 = 0xA5; // 10100101
const PATTERN_B: u8 = 0x5A; // 01011010 — bitwise inverse of A

const T_CLK_RISE:    f64 = 200e-9;
const T_INPUTS_FLIP: f64 = 400e-9;     // inputs change here, no clock edge
const T_SAMPLE:      f64 = 600e-9;     // outputs sampled here, well after the flip

fn assert_rail(actual: f64, expected_bit: u8, label: &str) {
    let target = if expected_bit == 1 { VDD } else { 0.0 };
    assert!(
        (actual - target).abs() < RAIL_TOL,
        "{label}: got {actual:.4} V, expected ≈ {target:.1} V (env {RAIL_TOL})",
    );
}

fn sample_at(t: &[f64], y: &[f64], t_query: f64) -> f64 {
    let idx = t.iter().enumerate()
        .min_by(|(_, a), (_, b)| (*a - t_query).abs().partial_cmp(&(*b - t_query).abs()).unwrap())
        .unwrap().0;
    y[idx]
}

#[test]
fn outputs_hold_when_inputs_change_without_clock_edge() {
    let Ok(ng) = LocalBinary::from_env() else {
        eprintln!("ngspice missing"); return;
    };

    // Behavioral truth at T_SAMPLE: clock rose at T_CLK_RISE so
    // outputs latched A; the post-flip change at T_INPUTS_FLIP isn't
    // accompanied by another clock edge, so outputs should still be A.
    let expected = behavioral_capture(PATTERN_A as u64, 0, true);
    assert_eq!(expected, PATTERN_A as u64,
        "behavioral reference disagrees with itself");

    let door: OutputDoor<8> = OutputDoor::default();
    let mut net = Netlist::new("OutputDoor hold-after-clock");
    net.add_dc_source("dd", "vdd", "0", VDD);

    // PWL inputs: stay at A until 50ns after the clock edge, then
    // ramp to B over 1ns. The 50ns post-edge dwell guarantees the
    // master/slave latches have absorbed A before B arrives.
    for i in 0..8usize {
        let bit_a = (PATTERN_A >> i) & 1;
        let bit_b = (PATTERN_B >> i) & 1;
        let v_a = if bit_a == 1 { VDD } else { 0.0 };
        let v_b = if bit_b == 1 { VDD } else { 0.0 };
        net.add_pwl_source(
            &format!("in{i}"),
            &format!("in{i}"),
            "0",
            &Pwl {
                points: vec![
                    (0.0,                v_a),
                    (T_INPUTS_FLIP,      v_a),
                    (T_INPUTS_FLIP+1e-9, v_b),
                    (1e-3,               v_b),
                ],
            },
        );
    }
    // Single rising edge at T_CLK_RISE; never falls.
    net.add_pulse_source(
        "clk", "clk", "0",
        &Pulse {
            v_initial: 0.0, v_pulsed: VDD,
            t_delay: T_CLK_RISE, t_rise: 1e-9, t_fall: 1e-9,
            pulse_width: 100.0, period: 1e30,
        },
    );

    let in_names: Vec<String>  = (0..8).map(|i| format!("in{i}")).collect();
    let out_names: Vec<String> = (0..8).map(|i| format!("out{i}")).collect();
    let mut nets: Vec<&str> = in_names.iter().map(String::as_str).collect();
    nets.push("clk");
    nets.extend(out_names.iter().map(String::as_str));
    nets.push("vdd");
    nets.push("0");
    door.emit_spice(&mut net, &nets, "od").unwrap();

    let analysis = TransientAnalysis {
        t_step: 5e-9, t_stop: 700e-9,
        use_initial_conditions: false, t_max: Some(5e-9),
    };
    let mut requests = vec![OutputRequest::NodeVoltage("clk".into())];
    for i in 0..8usize {
        requests.push(OutputRequest::NodeVoltage(format!("in{i}")));
        requests.push(OutputRequest::NodeVoltage(format!("out{i}")));
    }
    let trace = ng.run_transient_trace(&net.deck(), &analysis, &requests)
        .expect("ngspice tran");
    let t = &trace.time;

    // Sanity: at T_INPUTS_FLIP+50ns (still pre-flip dwell), outputs are A.
    for i in 0..8usize {
        let bit = ((expected >> i) & 1) as u8;
        let out = &trace.node_voltages[&format!("out{i}")];
        let v = sample_at(t, out, T_INPUTS_FLIP - 1e-9);
        assert_rail(v, bit, &format!("post-capture out{i}"));
    }

    // Hold property: at T_SAMPLE (well past the input flip), outputs
    // are still A — even though inputs are now B. If the latch leaks,
    // out_i would have moved toward bit B_i and tripped this assert.
    for i in 0..8usize {
        let want = ((expected >> i) & 1) as u8;
        let post_flip_in = (PATTERN_B >> i) & 1;
        // The test is only meaningful for bits where A and B differ —
        // a stuck-at-A bit on a same-bit position would falsely pass.
        // PATTERN_B = ~PATTERN_A so every bit differs by construction;
        // assert that here so a future PATTERN edit can't quietly
        // drop the coverage.
        assert_ne!(want, post_flip_in,
            "PATTERN_A and PATTERN_B must differ at every bit; bit {i}");
        let out = &trace.node_voltages[&format!("out{i}")];
        let v = sample_at(t, out, T_SAMPLE);
        assert_rail(v, want,
            &format!("HOLD out{i}: input flipped to {post_flip_in}, output should still read {want}"));
    }
}
