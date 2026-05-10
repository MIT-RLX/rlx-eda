//! Transient validation of `SampleHold`: ramp `vin`, pulse `clk_sh`,
//! verify `vhold` tracks during the sample windows and stays constant
//! during hold windows.
//!
//! ## What this catches
//!
//! - **Threshold loss bug:** if the PMOS half of the TG is mis-wired
//!   or missing, `vhold` would saturate at `vdd − Vth_n` ≈ 1.3 V on
//!   the high-side ramp instead of tracking up to ~1.5 V.
//! - **Hold-mode decay:** if the bulk-source junctions of the pass
//!   transistors leak through `vhold` to ground, the held value
//!   would drop measurably across the 1.5 µs hold window.
//! - **Cap missing:** without `C_hold`, `vhold` would have nothing to
//!   integrate onto and would float / oscillate during hold.
//!
//! ## Test pattern
//!
//! ```text
//!     vin: 0.3 V ──── ramp 4 µs ────→ 1.5 V
//!  clk_sh: ┌──┐         ┌──┐
//!          │  │         │  │
//!     ─────┘  └─────────┘  └────────
//!         0  0.5  ...  2.0 2.5  ...
//! ```
//!
//! Two sample/hold cycles let us check that the hold value updates
//! between cycles (rules out a one-time latch artifact).

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pulse, Pwl, SpiceEmit};
use spike_sample_hold::SampleHold;

const VDD: f64 = 1.8;

fn build_deck() -> String {
    let mut net = Netlist::new("SampleHold track-and-hold");
    net.add_dc_source("dd", "vdd", "0", VDD);

    // Vin: ramp from 0.3 to 1.5 V over 4 µs.
    net.add_pwl_source("in", "vin", "0", &Pwl { points: vec![
        (0.0,  0.3),
        (4e-6, 1.5),
        (10.0, 1.5),
    ]});

    // clk_sh: two sample windows.
    //   Sample 1: 0   → 0.5 µs   (high)
    //   Hold 1:   0.5 → 2.0 µs   (low)
    //   Sample 2: 2.0 → 2.5 µs   (high)
    //   Hold 2:   2.5 → 4.0 µs   (low)
    //
    // Use a PWL — Pulse only generates one fixed period.
    net.add_pwl_source("clk", "clk_sh", "0", &Pwl { points: vec![
        (0.0,         VDD),
        (0.5e-6 - 5e-9, VDD),
        (0.5e-6,        0.0),
        (2.0e-6 - 5e-9, 0.0),
        (2.0e-6,        VDD),
        (2.5e-6 - 5e-9, VDD),
        (2.5e-6,        0.0),
        (10.0,          0.0),
    ]});
    let _: Pulse;  // suppress unused-import on Pulse — kept for future tests

    SampleHold::default().emit_spice(
        &mut net,
        &["vin", "vhold", "clk_sh", "vdd", "0"],
        "sh",
    ).unwrap();

    net.deck()
}

/// Linear-interp lookup at time `t` (xs sorted).
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

fn vin_ramp_at(t: f64) -> f64 {
    // 0.3 → 1.5 V linearly across 0 → 4 µs.
    let u = (t / 4e-6).clamp(0.0, 1.0);
    0.3 + 1.2 * u
}

#[test]
fn vhold_tracks_in_sample_and_holds_in_hold() {
    let Ok(ng) = LocalBinary::from_env() else {
        eprintln!("ngspice missing"); return;
    };

    let h = 5e-9;
    let t_stop = 4.0e-6;
    let analysis = TransientAnalysis::new(h, t_stop).with_t_max(h);

    let trace = ng.run_transient_trace(
        &build_deck(),
        &analysis,
        &[
            OutputRequest::NodeVoltage("vin".into()),
            OutputRequest::NodeVoltage("vhold".into()),
        ],
    ).expect("ngspice transient");

    let t = &trace.time;
    let vh = &trace.node_voltages["vhold"];

    // ── Tracking during sample 1 ────────────────────────────────────
    // At t = 0.4 µs (well inside the first sample window), vhold
    // should track vin within a few mV.
    let vh_track1 = lerp(t, vh, 0.4e-6);
    let vin_at_track1 = vin_ramp_at(0.4e-6);
    let track_err = (vh_track1 - vin_at_track1).abs();
    assert!(track_err < 5e-3,
        "sample 1: vhold = {vh_track1:.4} V, vin = {vin_at_track1:.4} V, err = {track_err:.2e}");

    // ── Hold 1: vhold should be ~vin at end of sample 1 (0.5 µs) ────
    let vh_hold1_a = lerp(t, vh, 1.0e-6);
    let vh_hold1_b = lerp(t, vh, 1.5e-6);
    // Ideal held value: vin at the falling edge of clk_sh.
    let vin_at_sample_end = vin_ramp_at(0.5e-6);
    // Reasonable envelope: ~10 mV around the ideal (pass-switch
    // settling + capacitive feedthrough on falling clk edge).
    assert!(
        (vh_hold1_a - vin_at_sample_end).abs() < 0.02,
        "hold 1 at 1.0 µs: vhold = {vh_hold1_a:.4} V, expected ~{vin_at_sample_end:.4} V",
    );
    // Across the hold window, vhold should not drift more than 5 mV.
    assert!(
        (vh_hold1_a - vh_hold1_b).abs() < 5e-3,
        "hold 1 drift: {vh_hold1_a:.4} V → {vh_hold1_b:.4} V over 0.5 µs",
    );

    // ── Tracking during sample 2 ────────────────────────────────────
    // Mid-sample-2 (~2.4 µs), vhold should track vin again.
    let vh_track2 = lerp(t, vh, 2.4e-6);
    let vin_at_track2 = vin_ramp_at(2.4e-6);
    assert!(
        (vh_track2 - vin_at_track2).abs() < 5e-3,
        "sample 2: vhold = {vh_track2:.4} V, vin = {vin_at_track2:.4} V",
    );

    // ── Hold 2: vhold should be ~vin at end of sample 2 (2.5 µs) ────
    let vh_hold2_a = lerp(t, vh, 3.0e-6);
    let vh_hold2_b = lerp(t, vh, 3.8e-6);
    let vin_at_sample2_end = vin_ramp_at(2.5e-6);
    assert!(
        (vh_hold2_a - vin_at_sample2_end).abs() < 0.02,
        "hold 2 at 3.0 µs: vhold = {vh_hold2_a:.4} V, expected ~{vin_at_sample2_end:.4} V",
    );
    assert!(
        (vh_hold2_a - vh_hold2_b).abs() < 5e-3,
        "hold 2 drift: {vh_hold2_a:.4} V → {vh_hold2_b:.4} V over 0.8 µs",
    );

    // ── Updated value: hold-2 differs from hold-1 (rules out latch) ─
    assert!(
        (vh_hold2_a - vh_hold1_a).abs() > 0.3,
        "hold-1 and hold-2 nearly equal — vhold not updating between cycles",
    );
}

#[test]
fn vhold_passes_high_side_via_pmos() {
    // Drive vin to 1.5 V (well above vdd − Vth_n = 1.3 V) and confirm
    // the TG conducts both halves of the rail. With only an NMOS pass
    // gate, vhold would clamp at ~1.3 V; the PMOS lets it reach vin.
    let Ok(ng) = LocalBinary::from_env() else { return; };

    let mut net = Netlist::new("SampleHold high-side passing");
    net.add_dc_source("dd", "vdd", "0", VDD);
    net.add_dc_source("in", "vin", "0", 1.5);
    net.add_dc_source("clk", "clk_sh", "0", VDD);  // hold sample on
    SampleHold::default().emit_spice(
        &mut net,
        &["vin", "vhold", "clk_sh", "vdd", "0"],
        "sh",
    ).unwrap();

    let res = ng.run_dc(
        &net.deck(),
        &[OutputRequest::NodeVoltage("vhold".into())],
    ).expect("ngspice .op");
    let vhold = res.node_voltages["vhold"];
    // Should be within ~10 mV of vin at 1.5 V.
    assert!((vhold - 1.5).abs() < 0.01,
        "high-side pass: vhold = {vhold:.4} V, expected ≈ 1.5 V");
}
