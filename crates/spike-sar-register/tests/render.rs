//! Render a 4-bit SAR conversion as a stacked PNG so the binary-search
//! progression is visible: phases march left-to-right (MSB → LSB),
//! each result bit goes high during its trial then either holds or
//! drops to 0 based on the comparator's decision.
//!
//! Reuses `tests/conversion.rs`'s deck/PWL machinery (kept private to
//! that test for now — small amount of duplication is OK to keep the
//! render purely visual without coupling test outcomes).

#![cfg(feature = "ngspice")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pulse, Pwl, SpiceEmit};
use eda_waveform::{plot, Waveform};
use spike_sar_register::SarRegister;

const VDD: f64 = 1.8;
const PHASE_DURATION: f64 = 1e-6;
const EDGE: f64 = 5e-9;
const RESET_HOLD: f64 = 0.5e-6;

#[test]
fn render_sar_conversion_for_target_1010() {
    let Ok(ng) = LocalBinary::from_env() else {
        eprintln!("ngspice missing; skipping render");
        return;
    };

    let target = 0b1010_u32;
    let mut net = Netlist::new("SarRegister<4> conversion render");
    net.add_dc_source("dd", "vdd", "0", VDD);
    net.add_pulse_source("rb", "reset_b", "0", &Pulse {
        v_initial: 0.0, v_pulsed: VDD,
        t_delay: RESET_HOLD,
        t_rise: EDGE, t_fall: EDGE,
        pulse_width: 1.0,
        period: 1e30,
    });

    // Phase + capture timing (matches `tests/conversion.rs`):
    //   eval phase: 1 µs active per bit
    //   capture pulse: 100 ns wide, 50 ns after phase falls
    //   gap between phases: 200 ns total
    let t0 = RESET_HOLD + 50e-9;
    let cap_w = 100e-9;
    let cap_off = 50e-9;
    let gap = cap_off + cap_w + 50e-9;
    let phase_window = |idx: u32| -> Pwl {
        let start = t0 + idx as f64 * (PHASE_DURATION + gap);
        let end = start + PHASE_DURATION;
        Pwl { points: vec![
            (0.0, 0.0), (start - EDGE, 0.0), (start, VDD),
            (end - EDGE, VDD), (end, 0.0), (1.0, 0.0),
        ]}
    };
    let capture_window = |idx: u32| -> Pwl {
        let phase_end = t0 + idx as f64 * (PHASE_DURATION + gap) + PHASE_DURATION;
        let cs = phase_end + cap_off;
        let ce = cs + cap_w;
        Pwl { points: vec![
            (0.0, 0.0), (cs - EDGE, 0.0), (cs, VDD),
            (ce - EDGE, VDD), (ce, 0.0), (1.0, 0.0),
        ]}
    };
    net.add_pwl_source("p3", "p3", "0", &phase_window(0));
    net.add_pwl_source("p2", "p2", "0", &phase_window(1));
    net.add_pwl_source("p1", "p1", "0", &phase_window(2));
    net.add_pwl_source("p0", "p0", "0", &phase_window(3));
    net.add_pwl_source("c3", "c3", "0", &capture_window(0));
    net.add_pwl_source("c2", "c2", "0", &capture_window(1));
    net.add_pwl_source("c1", "c1", "0", &capture_window(2));
    net.add_pwl_source("c0", "c0", "0", &capture_window(3));

    let bit_at = |i: usize| -> f64 {
        if (target >> i) & 1 == 1 { VDD } else { 0.0 }
    };
    let setup = 20e-9;
    let mut cmp_pts: Vec<(f64, f64)> = vec![(0.0, 0.0)];
    for idx in 0..4 {
        let bit = 3 - idx;
        let phase_start = t0 + idx as f64 * (PHASE_DURATION + gap);
        let cap_end = phase_start + PHASE_DURATION + cap_off + cap_w;
        let v = bit_at(bit);
        cmp_pts.push((phase_start - setup, v));
        cmp_pts.push((cap_end + setup, v));
    }
    cmp_pts.push((1.0, *cmp_pts.last().map(|(_, v)| v).unwrap_or(&0.0)));
    net.add_pwl_source("cmp", "cmp", "0", &Pwl { points: cmp_pts });

    let r: SarRegister<4> = SarRegister::default();
    r.emit_spice(
        &mut net,
        &["p0", "p1", "p2", "p3",
          "c0", "c1", "c2", "c3",
          "cmp", "reset_b",
          "b0", "b1", "b2", "b3", "vdd", "0"],
        "u1",
    ).unwrap();

    let h = 5e-9;
    let cycle = PHASE_DURATION + gap;
    let t_stop = RESET_HOLD + 50e-9 + 4.0 * cycle + 1e-6;
    let analysis = TransientAnalysis::new(h, t_stop).with_t_max(h);

    let trace = ng.run_transient_trace(
        &net.deck(),
        &analysis,
        &[
            OutputRequest::NodeVoltage("p3".into()),
            OutputRequest::NodeVoltage("p2".into()),
            OutputRequest::NodeVoltage("p1".into()),
            OutputRequest::NodeVoltage("p0".into()),
            OutputRequest::NodeVoltage("c3".into()),
            OutputRequest::NodeVoltage("c2".into()),
            OutputRequest::NodeVoltage("c1".into()),
            OutputRequest::NodeVoltage("c0".into()),
            OutputRequest::NodeVoltage("cmp".into()),
            OutputRequest::NodeVoltage("b3".into()),
            OutputRequest::NodeVoltage("b2".into()),
            OutputRequest::NodeVoltage("b1".into()),
            OutputRequest::NodeVoltage("b0".into()),
        ],
    ).expect("ngspice transient");

    // Stack signals in narrative order: phases (MSB down), captures
    // (MSB down), cmp, then result bits (MSB down).
    let mut signals = BTreeMap::new();
    for (label, key) in [
        ("01_phase[3]_MSB",   "p3"),
        ("02_phase[2]",       "p2"),
        ("03_phase[1]",       "p1"),
        ("04_phase[0]_LSB",   "p0"),
        ("05_capture[3]",     "c3"),
        ("06_capture[2]",     "c2"),
        ("07_capture[1]",     "c1"),
        ("08_capture[0]",     "c0"),
        ("09_comparator",     "cmp"),
        ("10_bit[3]_MSB",     "b3"),
        ("11_bit[2]",         "b2"),
        ("12_bit[1]",         "b1"),
        ("13_bit[0]_LSB",     "b0"),
    ] {
        signals.insert(label.into(), trace.node_voltages[key].clone());
    }
    let wave = Waveform::Real {
        axis_name: "time (s)".into(),
        axis: trace.time.clone(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title(format!("4-bit SAR conversion → target code 0x{target:X} = {target:04b}"))
        .with_layout(plot::Layout::Stacked)
        .with_size(1100, 1400);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("sar_conversion.png");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, out_dir.join("sar_conversion.svg"), &cfg).expect("svg");
    eprintln!("SAR conversion render at {}", png.display());
}
