//! Drive a 4-bit ripple counter + ClockDecoder with a clock and verify
//! each strobe fires exactly when the counter is in its target state.
//!
//! ## Approach
//!
//! 1. Build a deck: clock generator (PULSE), 4-bit RippleCounter,
//!    ClockDecoder<4>. Reset asserted briefly at startup so all q's
//!    start at 0.
//! 2. Run a transient long enough for the counter to roll over twice
//!    (~32 input clock cycles).
//! 3. For each strobe trace, find the times where it's high. At each
//!    such "fire" time, sample the counter state q[3:0] and assert it
//!    matches the strobe's target.
//!
//! Counter+decoder propagation delay smears the strobe edges, so we
//! sample the counter state at the **midpoint** of each strobe-high
//! interval. That dodges the ringing on rising and falling edges.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pulse, SpiceEmit};
use spike_clock_decoder::{ClockDecoder, DecoderStates};
use spike_ripple_counter::RippleCounter;

const VDD: f64 = 1.8;
const F_IN: f64 = 1e6; // 1 MHz input clock
const T_RESET_RELEASE: f64 = 0.5e-6;

fn build_deck() -> String {
    let mut net = Netlist::new("ClockDecoder + RippleCounter integration");
    net.add_dc_source("dd", "vdd", "0", VDD);

    // Clock: 1 MHz, 50% duty.
    let period = 1.0 / F_IN;
    net.add_pulse_source("in", "clk_in", "0", &Pulse {
        v_initial: 0.0,
        v_pulsed: VDD,
        t_delay: 0.0,
        t_rise: 1e-9,
        t_fall: 1e-9,
        pulse_width: period * 0.5 - 2e-9,
        period,
    });

    // Reset: held low (active) for 500 ns at startup, then high
    // forever — single PULSE with the inverse polarity does it.
    net.add_pulse_source("rst", "reset_b", "0", &Pulse {
        v_initial: 0.0,
        v_pulsed: VDD,
        t_delay: T_RESET_RELEASE,
        t_rise: 1e-9,
        t_fall: 1e-9,
        pulse_width: 100e-3, // effectively forever for our 35 µs sim
        period: 1e30,
    });

    let counter: RippleCounter<4> = RippleCounter::default();
    counter.emit_spice(
        &mut net,
        &["clk_in", "reset_b", "q0", "q1", "q2", "q3", "vdd", "0"],
        "rc",
    ).unwrap();

    let decoder: ClockDecoder<4> = ClockDecoder::default();  // s_sh=0, s_door=9, s_reset=10
    decoder.emit_spice(
        &mut net,
        &["q0", "q1", "q2", "q3", "sh", "door", "rst", "vdd", "0"],
        "cd",
    ).unwrap();

    net.deck()
}

/// Sample-points where each strobe trace is "high" (above vdd/2). Returns
/// `(t_lo, t_hi)` for each maximal high interval. Skips intervals
/// shorter than `min_width` (filters glitches from gate-delay ringing
/// on the counter output transitions).
fn high_intervals(t: &[f64], y: &[f64], min_width: f64) -> Vec<(f64, f64)> {
    let thr = VDD / 2.0;
    let mut intervals = Vec::new();
    let mut start: Option<f64> = None;
    for (i, &ts) in t.iter().enumerate() {
        let above = y[i] >= thr;
        match (above, start) {
            (true,  None)        => start = Some(ts),
            (false, Some(s_lo))  => {
                if ts - s_lo >= min_width { intervals.push((s_lo, ts)); }
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s_lo) = start {
        let last = *t.last().unwrap();
        if last - s_lo >= min_width { intervals.push((s_lo, last)); }
    }
    intervals
}

/// Read the counter state q[3:0] at time `t` (linear interp, threshold
/// at vdd/2). LSB first.
fn counter_state_at(
    t_arr: &[f64], q0: &[f64], q1: &[f64], q2: &[f64], q3: &[f64], t: f64,
) -> u32 {
    let qs = [q0, q1, q2, q3];
    let mut state = 0u32;
    for (i, q) in qs.iter().enumerate() {
        let v = lerp(t_arr, q, t);
        if v >= VDD / 2.0 { state |= 1 << i; }
    }
    state
}

/// Linear-interpolated trace lookup at time `t`. Out-of-range queries
/// clamp to the nearest endpoint.
fn lerp(xs: &[f64], ys: &[f64], xq: f64) -> f64 {
    if xq <= xs[0] { return ys[0]; }
    if xq >= xs[xs.len() - 1] { return ys[ys.len() - 1]; }
    let i = match xs.binary_search_by(|x| x.partial_cmp(&xq).unwrap()) {
        Ok(j) => return ys[j],
        Err(j) => j - 1,
    };
    let t = (xq - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + t * (ys[i + 1] - ys[i])
}

#[test]
fn each_strobe_fires_at_its_target_state() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };

    // Run long enough for the counter to roll over at least twice.
    // Ripple counter at 1 MHz divides by 16 → full cycle = 16 µs.
    // 35 µs gives ~2 full rollovers + reset settling.
    let h = 5e-9;       // 5 ns timestep — fine vs 250 ns shortest q[0] half-period
    let t_stop = 35e-6;
    let analysis = TransientAnalysis::new(h, t_stop).with_t_max(h);

    let trace = ng.run_transient_trace(
        &build_deck(),
        &analysis,
        &[
            OutputRequest::NodeVoltage("q0".into()),
            OutputRequest::NodeVoltage("q1".into()),
            OutputRequest::NodeVoltage("q2".into()),
            OutputRequest::NodeVoltage("q3".into()),
            OutputRequest::NodeVoltage("sh".into()),
            OutputRequest::NodeVoltage("door".into()),
            OutputRequest::NodeVoltage("rst".into()),
        ],
    ).expect("ngspice transient");

    let t = &trace.time;
    let q0 = &trace.node_voltages["q0"];
    let q1 = &trace.node_voltages["q1"];
    let q2 = &trace.node_voltages["q2"];
    let q3 = &trace.node_voltages["q3"];

    // Any strobe-high interval shorter than ~half a state-period is a
    // glitch (gate ringing as q transitions). State period at the
    // input clock is 1 µs (each q[0] state lasts ½ input period =
    // 0.5 µs); a 0.3 µs threshold cleanly filters glitches.
    let min_width = 300e-9;

    let states = DecoderStates::default();
    for (name, target) in [
        ("sh",   states.s_sh),
        ("door", states.s_door),
        ("rst",  states.s_reset),
    ] {
        let trace_y = &trace.node_voltages[name];
        let intervals = high_intervals(t, trace_y, min_width);
        // Skip the startup window — reset-release ringing can produce
        // a stray sh=1 at t<1 µs before the counter has settled.
        let intervals: Vec<(f64, f64)> = intervals
            .into_iter()
            .filter(|(lo, _)| *lo > T_RESET_RELEASE + 1e-6)
            .collect();
        assert!(
            !intervals.is_empty(),
            "strobe {name}: no firing intervals found in {t_stop}s sim",
        );
        for (lo, hi) in &intervals {
            let mid = 0.5 * (lo + hi);
            let state = counter_state_at(t, q0, q1, q2, q3, mid);
            assert_eq!(
                state, target,
                "strobe {name} fired during state {state:04b} but target is {target:04b} (t={mid:.3e}s)",
            );
        }
    }
}
