//! End-to-end closed-loop SAR ADC validation.
//!
//! Drives `vin` at fixed DC values across a 4-bit conversion cycle.
//! Phase signals are PWL-generated (clock generation is validated
//! independently in `spike-ripple-counter` and `spike-clock-decoder`).
//! After `phase_done`, samples each `bit[i]` and reconstructs the
//! output code; asserts it matches `ideal_sar_code(vin, vref, 4)`.
//!
//! ## Cycle layout
//!
//! Each bit's eval is followed by a brief capture pulse during the gap
//! between phases — that's the rising edge that latches the comparator
//! decision into the bit. Without the gap, the next phase's set would
//! perturb the comparator before the master DFF locks.
//!
//! ```text
//!   reset    sample      bit3              bit2              bit1              bit0          done
//!   ┌──┐    ┌────┐    ┌─────────┐ │c3│ ┌─────────┐ │c2│ ┌─────────┐ │c1│ ┌─────────┐ │c0│
//!   ┘  └────┘    └────┘         └─────┘         └─────┘         └─────┘         └─────┘
//! ```

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pwl, SpiceEmit};
use spike_sar_adc::{ideal_sar_code, SarAdc};

const VDD: f64 = 1.8;
const VREF: f64 = VDD;

fn build_deck(vin: f64) -> String {
    let mut net = Netlist::new("Closed-loop 4-bit SAR ADC");
    net.add_dc_source("dd", "vdd", "0", VDD);
    net.add_dc_source("in", "vin", "0", vin);

    // reset_b: low for 400 ns, then high forever.
    net.add_pwl_source("rb", "reset_b", "0", &Pwl { points: vec![
        (0.0,             0.0),
        (0.4e-6 - 5e-9,   0.0),
        (0.4e-6,          VDD),
        (10.0,            VDD),
    ]});

    // clk_sh: high during 0.5 → 1.0 µs (sample phase).
    net.add_pwl_source("clk", "clk_sh", "0", &Pwl { points: vec![
        (0.0,             0.0),
        (0.5e-6 - 5e-9,   0.0),
        (0.5e-6,          VDD),
        (1.0e-6 - 5e-9,   VDD),
        (1.0e-6,          0.0),
        (10.0,            0.0),
    ]});

    // Phase + capture pairs. Each phase is 800 ns active with a 200 ns
    // gap between adjacent phases. The capture pulse falls in the
    // middle of the gap — comparator has settled, next phase hasn't
    // started disturbing it yet.
    //
    // bit i: phase active in [start_i, start_i + 0.8 µs]
    //        capture active in [start_i + 0.85, start_i + 0.95]
    //
    //   bit 3 (MSB): start = 1.1 µs  (after sample)
    //   bit 2:       start = 2.2 µs
    //   bit 1:       start = 3.3 µs
    //   bit 0 (LSB): start = 4.4 µs
    let phase_pwl = |start: f64| -> Pwl {
        let end = start + 0.8e-6;
        Pwl { points: vec![
            (0.0,           0.0),
            (start - 5e-9,  0.0),
            (start,         VDD),
            (end - 5e-9,    VDD),
            (end,           0.0),
            (10.0,          0.0),
        ]}
    };
    let cap_pwl = |start: f64| -> Pwl {
        let cap_start = start + 0.85e-6;
        let cap_end   = start + 0.95e-6;
        Pwl { points: vec![
            (0.0,                0.0),
            (cap_start - 5e-9,   0.0),
            (cap_start,          VDD),
            (cap_end - 5e-9,     VDD),
            (cap_end,            0.0),
            (10.0,               0.0),
        ]}
    };
    let bit_starts = [1.1e-6, 2.2e-6, 3.3e-6, 4.4e-6];
    // Phases (LSB-first net order): p0..p3 with bit-3 starting first.
    net.add_pwl_source("p3", "p3", "0", &phase_pwl(bit_starts[0]));
    net.add_pwl_source("p2", "p2", "0", &phase_pwl(bit_starts[1]));
    net.add_pwl_source("p1", "p1", "0", &phase_pwl(bit_starts[2]));
    net.add_pwl_source("p0", "p0", "0", &phase_pwl(bit_starts[3]));
    net.add_pwl_source("c3", "c3", "0", &cap_pwl(bit_starts[0]));
    net.add_pwl_source("c2", "c2", "0", &cap_pwl(bit_starts[1]));
    net.add_pwl_source("c1", "c1", "0", &cap_pwl(bit_starts[2]));
    net.add_pwl_source("c0", "c0", "0", &cap_pwl(bit_starts[3]));

    let adc: SarAdc<4> = SarAdc::default();
    adc.emit_spice(
        &mut net,
        &[
            "vin",
            "p0", "p1", "p2", "p3",
            "c0", "c1", "c2", "c3",
            "clk_sh", "reset_b",
            "b0", "b1", "b2", "b3",
            "vdd", "0",
        ],
        "u1",
    ).unwrap();

    net.deck()
}

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

fn read_bit(t_arr: &[f64], y: &[f64], t: f64) -> u32 {
    if lerp(t_arr, y, t) >= VDD / 2.0 { 1 } else { 0 }
}

fn convert_one(vin: f64) -> u32 {
    let ng = LocalBinary::from_env().expect("ngspice missing");
    let h = 5e-9;
    let t_stop = 7e-6;
    let analysis = TransientAnalysis::new(h, t_stop).with_t_max(h);

    let trace = ng.run_transient_trace(
        &build_deck(vin),
        &analysis,
        &[
            OutputRequest::NodeVoltage("b0".into()),
            OutputRequest::NodeVoltage("b1".into()),
            OutputRequest::NodeVoltage("b2".into()),
            OutputRequest::NodeVoltage("b3".into()),
        ],
    ).expect("ngspice transient");

    // Sample after the LSB capture pulse. bit-0 phase ends at
    // 4.4 + 0.8 = 5.2 µs; capture pulse fires at 5.25–5.35 µs;
    // sample at 5.5 µs.
    let t_sample = 5.5e-6;
    let t = &trace.time;
    let b0 = read_bit(t, &trace.node_voltages["b0"], t_sample);
    let b1 = read_bit(t, &trace.node_voltages["b1"], t_sample);
    let b2 = read_bit(t, &trace.node_voltages["b2"], t_sample);
    let b3 = read_bit(t, &trace.node_voltages["b3"], t_sample);
    b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)
}

fn case(vin: f64, expected: u32) {
    if LocalBinary::from_env().is_err() { eprintln!("ngspice missing"); return; }
    let got = convert_one(vin);
    let ideal = ideal_sar_code(vin, VREF, 4);
    assert_eq!(ideal, expected, "test setup error: ideal_sar_code mismatch");
    // Allow ±1 LSB tolerance: the smoothed comparator + S/H droop +
    // DAC settling can each shift the boundary by a few mV.
    let diff = (got as i32 - expected as i32).abs();
    assert!(diff <= 1,
        "vin = {vin:.3} V: got code {got:04b}, expected {expected:04b} (diff = {diff} LSB)");
}

#[test] fn vin_0v0_gives_code_0()    { case(0.0,   0);  }
#[test] fn vin_0v5_gives_code_4()    { case(0.5,   4);  }
#[test] fn vin_0v95_gives_code_8()   { case(0.95,  8);  }
#[test] fn vin_1v4_gives_code_12()   { case(1.4,  12);  }
#[test] fn vin_1v7_gives_code_15()   { case(1.7,  15);  }
