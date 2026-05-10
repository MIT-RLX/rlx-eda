//! VTC + transient validation of `CmosComparator`.
//!
//! Two flavors of evidence:
//!
//! 1. **VTC sweep**: hold `vm = vdd/2`, sweep `vp` from 0 to vdd via
//!    multiple `.op` runs, plot `vout(vp)`. A working comparator
//!    snaps from low to high near `vp = vm` with a sharp transition.
//! 2. **Transient with sine**: drive `vp` with a slow sine centered
//!    on `vm = vdd/2`, observe `vout` produces a clean rail-to-rail
//!    square wave matching the input zero-crossings.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pulse, Sine, SpiceEmit};
use spike_comparator_cmos::CmosComparator;

const VDD: f64 = 1.8;
const VBIAS: f64 = 0.8; // tail current source gate, above NMOS Vto
const VM_DEFAULT: f64 = 0.9; // vdd/2

fn deck_for_op(vp: f64) -> String {
    let mut net = Netlist::new("CmosComparator VTC point");
    net.add_dc_source("dd", "vdd", "0", VDD);
    net.add_dc_source("bs", "vbias", "0", VBIAS);
    net.add_dc_source("vm", "vm", "0", VM_DEFAULT);
    net.add_dc_source("vp", "vp", "0", vp);
    let cmp = CmosComparator::default();
    cmp.emit_spice(&mut net, &["vp", "vm", "vout", "vbias", "vdd", "0"], "u1").unwrap();
    net.deck()
}

fn ngspice_vout_op(ng: &LocalBinary, deck: &str) -> f64 {
    let res = ng
        .run_dc(deck, &[OutputRequest::NodeVoltage("vout".into())])
        .expect("ngspice .op");
    res.node_voltages["vout"]
}

#[test]
fn vtc_sweep_shows_switching_threshold_near_vm_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };

    // Sweep vp from 0 to VDD in 41 points.
    const N: usize = 41;
    let mut vps = Vec::with_capacity(N);
    let mut vouts = Vec::with_capacity(N);
    for i in 0..N {
        let vp = VDD * i as f64 / (N - 1) as f64;
        let v = ngspice_vout_op(&ng, &deck_for_op(vp));
        vps.push(vp);
        vouts.push(v);
    }

    // Find the switching threshold: the vp where vout crosses vdd/2.
    let mut sw_idx = None;
    for i in 1..N {
        if (vouts[i - 1] - VDD / 2.0).signum() != (vouts[i] - VDD / 2.0).signum() {
            sw_idx = Some(i);
            break;
        }
    }
    let sw_idx = sw_idx.expect("vout never crossed vdd/2");
    let sw_vp = {
        // Linear interpolation for sub-step precision.
        let (v0, v1) = (vouts[sw_idx - 1], vouts[sw_idx]);
        let t = (VDD / 2.0 - v0) / (v1 - v0);
        vps[sw_idx - 1] + t * (vps[sw_idx] - vps[sw_idx - 1])
    };
    eprintln!("comparator switches at vp = {sw_vp:.4} V (vm = {VM_DEFAULT})");

    // The simple diff-pair comparator's offset is sensitive to the
    // exact LEVEL=1 sizing and Vbias choice — in practice we end up
    // somewhere in [vm - 0.3, vm + 0.3]. A real design would trim
    // this with cascode or regenerative-latch topologies. Generous
    // tolerance here:
    assert!(
        (sw_vp - VM_DEFAULT).abs() < 0.4,
        "switching threshold {sw_vp:.4} V too far from vm = {VM_DEFAULT} V",
    );

    // Also assert the output reaches both rails over the sweep.
    let max_vout = vouts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_vout = vouts.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(max_vout > VDD - 0.1, "vout never reached high rail; max = {max_vout:.4}");
    assert!(min_vout < 0.1, "vout never reached low rail; min = {min_vout:.4}");

    render_vtc(&vps, &vouts, sw_vp);
}

#[test]
fn transient_sine_produces_square_wave_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };

    let mut net = Netlist::new("CmosComparator sine transient");
    net.add_dc_source("dd", "vdd", "0", VDD);
    net.add_dc_source("bs", "vbias", "0", VBIAS);
    net.add_dc_source("vm", "vm", "0", VM_DEFAULT);

    // 100 kHz sine, 0.5 V peak, centered on vm. With this swing the
    // comparator decisions are clear (300 mV either side of vm).
    net.add_sine_source("vp", "vp", "0", &Sine {
        v_offset: VM_DEFAULT,
        v_amplitude: 0.5,
        frequency: 100e3,
        t_delay: 0.0,
        damping: 0.0,
    });
    let cmp = CmosComparator::default();
    cmp.emit_spice(&mut net, &["vp", "vm", "vout", "vbias", "vdd", "0"], "u1").unwrap();

    let analysis = TransientAnalysis {
        t_step: 50e-9,
        t_stop: 30e-6, // 3 sine periods
        use_initial_conditions: false,
        t_max: Some(50e-9),
    };
    let trace = ng
        .run_transient_trace(
            &net.deck(),
            &analysis,
            &[
                OutputRequest::NodeVoltage("vp".into()),
                OutputRequest::NodeVoltage("vout".into()),
            ],
        )
        .expect("ngspice tran");

    // vout should have one full square-wave period per sine period.
    // Count rising edges in the steady-state portion (skip first 5 µs
    // for startup).
    let t = &trace.time;
    let v = &trace.node_voltages["vout"];
    let thr = VDD / 2.0;
    let mut count = 0usize;
    let mut prev_above = false;
    for (i, &ts) in t.iter().enumerate() {
        if ts < 5e-6 { prev_above = v[i] >= thr; continue; }
        let above = v[i] >= thr;
        if above && !prev_above { count += 1; }
        prev_above = above;
    }
    // Observation window: 5 µs to 30 µs = 25 µs = 2.5 sine periods.
    // Expect 2 or 3 rising edges.
    eprintln!("vout rising-edges in observation window: {count} (expected 2-3)");
    assert!((2..=3).contains(&count), "expected 2-3 rising edges, got {count}");

    render_transient(&trace);
}

fn render_vtc(vp: &[f64], vout: &[f64], sw_vp: f64) {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use eda_waveform::{plot, Waveform};

    let mut signals = BTreeMap::new();
    signals.insert("vout".into(), vout.to_vec());
    let wave = Waveform::Real {
        axis_name: "vp (V)".into(),
        axis: vp.to_vec(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title(format!(
            "CmosComparator VTC: vm = {VM_DEFAULT} V, switching @ vp = {sw_vp:.3} V",
        ))
        .with_size(900, 500)
        .add_marker(plot::Marker::Vertical { x: VM_DEFAULT, label: Some("vm".into()) })
        .add_marker(plot::Marker::Vertical { x: sw_vp, label: Some(format!("sw@{sw_vp:.3}")) })
        .add_marker(plot::Marker::Horizontal { y: VDD / 2.0, label: Some("vdd/2".into()) });
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("comparator_vtc.png");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, out_dir.join("comparator_vtc.svg"), &cfg).expect("svg");
    eprintln!("VTC at {}", png.display());
}

fn render_transient(trace: &eda_extern_ngspice::TransientTrace) {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use eda_waveform::{plot, Waveform};

    let mut signals = BTreeMap::new();
    signals.insert("vp (sine)".into(), trace.node_voltages["vp"].clone());
    signals.insert("vout (square)".into(), trace.node_voltages["vout"].clone());
    let wave = Waveform::Real {
        axis_name: "time (s)".into(),
        axis: trace.time.clone(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title("CmosComparator transient: 100 kHz sine on vp → square on vout")
        .with_layout(plot::Layout::Stacked)
        .with_size(900, 500)
        .add_marker(plot::Marker::Horizontal { y: VM_DEFAULT, label: Some("vm".into()) });
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("comparator_transient.png");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, out_dir.join("comparator_transient.svg"), &cfg).expect("svg");
    eprintln!("transient at {}", png.display());
}

// suppress unused-import warnings if Pulse goes unused
fn _silence() { let _ = Pulse { v_initial: 0.0, v_pulsed: 0.0, t_delay: 0.0, t_rise: 0.0, t_fall: 0.0, pulse_width: 0.0, period: 0.0 }; }
