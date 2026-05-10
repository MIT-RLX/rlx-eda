//! Diagnostic — runs a single 4-bit conversion at vin=0.95V and dumps
//! intermediate waveforms (vhold, v_dac, cmp, b0..b3) at key times.
//!
//! Not a unit test — only runs when the `diagnose` feature gate is on,
//! invoked manually with `cargo test -p spike-sar-adc --features 'ngspice diagnose' diagnose -- --nocapture`.

#![cfg(all(feature = "ngspice", feature = "diagnose"))]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pwl, SpiceEmit};
use spike_sar_adc::SarAdc;

const VDD: f64 = 1.8;

#[test]
fn dump_signals_at_vin_0v95() {
    let Ok(ng) = LocalBinary::from_env() else { return; };

    let mut net = Netlist::new("diagnose");
    net.add_dc_source("dd", "vdd", "0", VDD);
    let probe_vin = std::env::var("PROBE_VIN").ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.95);
    net.add_dc_source("in", "vin", "0", probe_vin);
    println!("=== vin = {probe_vin} V ===");
    net.add_pwl_source("rb", "reset_b", "0", &Pwl { points: vec![(0.0, 0.0), (0.4e-6, VDD), (10.0, VDD)]});
    net.add_pwl_source("clk", "clk_sh", "0", &Pwl { points: vec![(0.0, 0.0), (0.5e-6, VDD), (1.0e-6, 0.0), (10.0, 0.0)]});
    let phase_pwl = |start: f64| Pwl { points: vec![
        (0.0, 0.0), (start - 5e-9, 0.0), (start, VDD),
        (start + 0.8e-6 - 5e-9, VDD), (start + 0.8e-6, 0.0), (10.0, 0.0)
    ]};
    let cap_pwl = |start: f64| Pwl { points: vec![
        (0.0, 0.0),
        (start + 0.85e-6 - 5e-9, 0.0), (start + 0.85e-6, VDD),
        (start + 0.95e-6 - 5e-9, VDD), (start + 0.95e-6, 0.0),
        (10.0, 0.0)
    ]};
    let bit_starts = [1.1e-6, 2.2e-6, 3.3e-6, 4.4e-6];
    net.add_pwl_source("p3", "p3", "0", &phase_pwl(bit_starts[0]));
    net.add_pwl_source("p2", "p2", "0", &phase_pwl(bit_starts[1]));
    net.add_pwl_source("p1", "p1", "0", &phase_pwl(bit_starts[2]));
    net.add_pwl_source("p0", "p0", "0", &phase_pwl(bit_starts[3]));
    net.add_pwl_source("c3", "c3", "0", &cap_pwl(bit_starts[0]));
    net.add_pwl_source("c2", "c2", "0", &cap_pwl(bit_starts[1]));
    net.add_pwl_source("c1", "c1", "0", &cap_pwl(bit_starts[2]));
    net.add_pwl_source("c0", "c0", "0", &cap_pwl(bit_starts[3]));

    let adc: SarAdc<4> = SarAdc::default();
    adc.emit_spice(&mut net, &[
        "vin",
        "p0", "p1", "p2", "p3",
        "c0", "c1", "c2", "c3",
        "clk_sh", "reset_b",
        "b0", "b1", "b2", "b3",
        "vdd", "0",
    ], "u1").unwrap();

    let trace = ng.run_transient_trace(
        &net.deck(),
        &TransientAnalysis::new(5e-9, 7e-6).with_t_max(5e-9),
        &[
            OutputRequest::NodeVoltage("u1_vhold".into()),
            OutputRequest::NodeVoltage("u1_vdac".into()),
            OutputRequest::NodeVoltage("u1_cmp".into()),
            OutputRequest::NodeVoltage("b0".into()),
            OutputRequest::NodeVoltage("b1".into()),
            OutputRequest::NodeVoltage("b2".into()),
            OutputRequest::NodeVoltage("b3".into()),
            OutputRequest::NodeVoltage("p3".into()),
            OutputRequest::NodeVoltage("p2".into()),
            OutputRequest::NodeVoltage("c3".into()),
            OutputRequest::NodeVoltage("c2".into()),
        ],
    ).expect("ngspice");

    let lerp = |xs: &[f64], ys: &[f64], xq: f64| -> f64 {
        if xq <= xs[0] { return ys[0]; }
        if xq >= xs[xs.len()-1] { return ys[ys.len()-1]; }
        let i = match xs.binary_search_by(|x| x.partial_cmp(&xq).unwrap()) {
            Ok(j) => return ys[j], Err(j) => j-1,
        };
        let t = (xq - xs[i]) / (xs[i+1] - xs[i]);
        ys[i] + t * (ys[i+1] - ys[i])
    };

    let t = &trace.time;
    println!("\nvin = 0.95V, expected code = 1000 (= 8)");
    println!("{:>8}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}",
        "t [µs]", "vhold", "v_dac", "cmp", "b3", "b2", "b1", "b0", "");
    let mut t_probe_us = 0.5;
    while t_probe_us < 6.0 {
        let t_probe = t_probe_us * 1e-6;
        let vhold = lerp(t, &trace.node_voltages["u1_vhold"], t_probe);
        let v_dac = lerp(t, &trace.node_voltages["u1_vdac"], t_probe);
        let cmp = lerp(t, &trace.node_voltages["u1_cmp"], t_probe);
        let b0 = lerp(t, &trace.node_voltages["b0"], t_probe);
        let b1 = lerp(t, &trace.node_voltages["b1"], t_probe);
        let b2 = lerp(t, &trace.node_voltages["b2"], t_probe);
        let b3 = lerp(t, &trace.node_voltages["b3"], t_probe);
        let bit = |v: f64| if v >= 0.9 { '1' } else { '0' };
        println!("{:>8.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>6.3}  {}{}{}{}",
            t_probe_us, vhold, v_dac, cmp, b3, b2, b1, b0,
            bit(b3), bit(b2), bit(b1), bit(b0));
        t_probe_us += 0.1;
    }
}
