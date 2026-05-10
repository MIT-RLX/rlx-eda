//! Drive a 4-bit `SarRegister<4>` with PWL phase signals and a
//! synthetic comparator pattern, then verify the captured bits match
//! the ideal SAR algorithm.
//!
//! ## Why we don't close the loop here
//!
//! A real SAR ADC closes the loop: comparator output depends on the
//! DAC output, which depends on the SAR register's previous bits. This
//! test deliberately *opens* the loop — comparator is driven by a
//! pre-canned PWL — so we validate the register's per-bit set/latch
//! behavior in isolation. End-to-end SAR ADC validation (with a real
//! DAC + comparator + register loop) is a future spike.
//!
//! ## Expected behavior
//!
//! For each of three test cases we pick a target output code, derive
//! the comparator pattern that produces it (1 during bit-i's phase
//! iff bit i should be 1, 0 otherwise — matching the ideal SAR
//! decision rule), and assert the register's bit outputs match the
//! target after `phase_done`.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pwl, Pulse, SpiceEmit};
use spike_sar_register::SarRegister;

const VDD: f64 = 1.8;
const PHASE_DURATION: f64 = 1e-6; // 1 µs per phase
const EDGE: f64 = 5e-9;           // 5 ns rising/falling edges
const RESET_HOLD: f64 = 0.5e-6;   // reset_b low for 500 ns at startup

/// Build a deck for an N=4 SarRegister test driven by external PWL
/// signals on phase[0..3], phase_done, and cmp. `target_code` selects
/// the comparator pattern.
fn build_deck(target_code: u32) -> String {
    let mut net = Netlist::new("SarRegister<4> open-loop conversion");
    net.add_dc_source("dd", "vdd", "0", VDD);

    // Reset: low until RESET_HOLD, then high forever.
    net.add_pulse_source("rb", "reset_b", "0", &Pulse {
        v_initial: 0.0, v_pulsed: VDD,
        t_delay: RESET_HOLD,
        t_rise: EDGE, t_fall: EDGE,
        pulse_width: 1.0,  // forever vs our short sim
        period: 1e30,
    });

    // Phases: mutually-exclusive 1 µs windows. Captures land in the
    // gap between adjacent phases (so the next phase's set hasn't yet
    // perturbed the comparator).
    //
    //   phase[3] hi:  [t0,        t0 + T]
    //   cap[3]   hi:  [t0 + T + ε,  t0 + T + ε + CAP_W]
    //   phase[2] hi:  [t0 + T + GAP,  ...]
    //   ...
    //
    // GAP > CAP_W ensures the capture pulse fully resolves before the
    // next phase rises.
    let t0 = RESET_HOLD + 50e-9;
    let cap_w = 100e-9;       // 100 ns capture pulse width
    let cap_off = 50e-9;      // 50 ns delay after phase falls
    let gap = cap_off + cap_w + 50e-9;  // total inter-phase gap
    let phase_window = |idx: u32| -> Pwl {
        let start = t0 + idx as f64 * (PHASE_DURATION + gap);
        let end   = start + PHASE_DURATION;
        Pwl { points: vec![
            (0.0,         0.0),
            (start - EDGE, 0.0),
            (start,        VDD),
            (end - EDGE,   VDD),
            (end,          0.0),
            (1.0,          0.0),
        ]}
    };
    let capture_window = |idx: u32| -> Pwl {
        let phase_start = t0 + idx as f64 * (PHASE_DURATION + gap);
        let phase_end   = phase_start + PHASE_DURATION;
        let cap_start   = phase_end + cap_off;
        let cap_end     = cap_start + cap_w;
        Pwl { points: vec![
            (0.0,             0.0),
            (cap_start - EDGE, 0.0),
            (cap_start,        VDD),
            (cap_end - EDGE,   VDD),
            (cap_end,          0.0),
            (1.0,              0.0),
        ]}
    };
    // Bit 3 fires at idx=0 (MSB), bit 0 at idx=3 (LSB).
    net.add_pwl_source("p3", "p3", "0", &phase_window(0));
    net.add_pwl_source("p2", "p2", "0", &phase_window(1));
    net.add_pwl_source("p1", "p1", "0", &phase_window(2));
    net.add_pwl_source("p0", "p0", "0", &phase_window(3));
    net.add_pwl_source("c3", "c3", "0", &capture_window(0));
    net.add_pwl_source("c2", "c2", "0", &capture_window(1));
    net.add_pwl_source("c1", "c1", "0", &capture_window(2));
    net.add_pwl_source("c0", "c0", "0", &capture_window(3));

    // Comparator pattern: during phase[i] AND through bit-i's capture
    // pulse, cmp = bit i of target_code.
    let bit_at = |i: usize| -> f64 {
        if (target_code >> i) & 1 == 1 { VDD } else { 0.0 }
    };
    let setup = 20e-9;
    let mut cmp_pts: Vec<(f64, f64)> = vec![(0.0, 0.0)];
    for idx in 0..4 {
        let bit = 3 - idx;  // idx 0 → bit 3 (MSB), idx 3 → bit 0 (LSB)
        let phase_start = t0 + idx as f64 * (PHASE_DURATION + gap);
        let cap_end = phase_start + PHASE_DURATION + cap_off + cap_w;
        let v = bit_at(bit);
        cmp_pts.push((phase_start - setup, v));
        cmp_pts.push((cap_end + setup, v));
    }
    cmp_pts.push((1.0, *cmp_pts.last().map(|(_, v)| v).unwrap_or(&0.0)));
    net.add_pwl_source("cmp", "cmp", "0", &Pwl { points: cmp_pts });

    // SarRegister<4>: net order
    //   [p0..p3, c0..c3, cmp, reset_b, b0..b3, vdd, gnd]
    let r: SarRegister<4> = SarRegister::default();
    r.emit_spice(
        &mut net,
        &[
            "p0", "p1", "p2", "p3",
            "c0", "c1", "c2", "c3",
            "cmp", "reset_b",
            "b0", "b1", "b2", "b3",
            "vdd", "0",
        ],
        "u1",
    ).unwrap();

    net.deck()
}

/// Sample bit_i at t and threshold at vdd/2.
fn read_bit(t_arr: &[f64], y: &[f64], t: f64) -> u32 {
    let mut v = y[0];
    for i in 1..t_arr.len() {
        if t_arr[i] >= t {
            // Linear interp
            let frac = (t - t_arr[i - 1]) / (t_arr[i] - t_arr[i - 1]);
            v = y[i - 1] + frac * (y[i] - y[i - 1]);
            break;
        }
    }
    if v >= VDD / 2.0 { 1 } else { 0 }
}

fn run_case(target_code: u32) -> u32 {
    let ng = LocalBinary::from_env().expect("ngspice missing");
    let h = 5e-9;
    // 4 phases × (1 µs phase + ~200 ns gap) = ~4.8 µs of active conversion
    // after RESET_HOLD release; pad to 6 µs for sampling tail.
    let cycle = PHASE_DURATION + 200e-9;
    let t_stop = RESET_HOLD + 50e-9 + 4.0 * cycle + 1e-6;
    let analysis = TransientAnalysis::new(h, t_stop).with_t_max(h);

    let trace = ng.run_transient_trace(
        &build_deck(target_code),
        &analysis,
        &[
            OutputRequest::NodeVoltage("b0".into()),
            OutputRequest::NodeVoltage("b1".into()),
            OutputRequest::NodeVoltage("b2".into()),
            OutputRequest::NodeVoltage("b3".into()),
        ],
    ).expect("ngspice transient");

    // Sample after phase_done has captured the LSB.
    let t_sample = RESET_HOLD + 50e-9 + 5.5 * PHASE_DURATION;
    let t = &trace.time;
    let b0 = read_bit(t, &trace.node_voltages["b0"], t_sample);
    let b1 = read_bit(t, &trace.node_voltages["b1"], t_sample);
    let b2 = read_bit(t, &trace.node_voltages["b2"], t_sample);
    let b3 = read_bit(t, &trace.node_voltages["b3"], t_sample);
    b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)
}

#[test]
fn captures_target_code_1000_msb_set() {
    if LocalBinary::from_env().is_err() { eprintln!("ngspice missing"); return; }
    let got = run_case(0b1000);
    assert_eq!(got, 0b1000, "MSB-only: SAR captured {got:04b}, expected 1000");
}

#[test]
fn captures_target_code_1010_msb_and_b1() {
    if LocalBinary::from_env().is_err() { eprintln!("ngspice missing"); return; }
    let got = run_case(0b1010);
    assert_eq!(got, 0b1010, "MSB+b1: SAR captured {got:04b}, expected 1010");
}

#[test]
fn captures_target_code_0110_no_msb() {
    if LocalBinary::from_env().is_err() { eprintln!("ngspice missing"); return; }
    let got = run_case(0b0110);
    assert_eq!(got, 0b0110, "mid: SAR captured {got:04b}, expected 0110");
}

#[test]
fn captures_target_code_0000_all_clear() {
    if LocalBinary::from_env().is_err() { eprintln!("ngspice missing"); return; }
    let got = run_case(0b0000);
    assert_eq!(got, 0b0000, "all-zero: SAR captured {got:04b}, expected 0000");
}

#[test]
fn captures_target_code_1111_all_set() {
    if LocalBinary::from_env().is_err() { eprintln!("ngspice missing"); return; }
    let got = run_case(0b1111);
    assert_eq!(got, 0b1111, "all-one: SAR captured {got:04b}, expected 1111");
}
