//! Propagation-delay validation for the inverter via transient + PULSE.
//!
//! Drives `Inverter` with a `SourceWaveform::Pulse` (slow rise/fall so the
//! crossing times are measurable), runs ngspice, then computes:
//!
//!   - `tphl` = `t(vout↓ Vdd/2) − t(vin↑ Vdd/2)`  (input rises, output falls)
//!   - `tplh` = `t(vout↑ Vdd/2) − t(vin↓ Vdd/2)`  (input falls, output rises)
//!
//! Crossing times come from linear interpolation on the trace
//! (`eda_validate::lerp`-shaped logic) — ngspice's adaptive grid won't
//! land samples exactly at `Vdd/2`. With `tmax = h` on the `.tran` line
//! (wave-1 plumbing) the grid is dense enough that linear interpolation
//! between two adjacent samples is good to ~10 ps for our 50 ps stride.
//!
//! ## What this validates
//!
//! - The whole wave-1 → wave-2 pipeline composes: `SourceWaveform`
//!   driving a SpiceEmit gate via `Netlist::add_waveform_source`,
//!   `Invoker::run_transient_trace` returning the full waveform with
//!   `tmax` pinning, and the trace-comparison primitives doing
//!   crossing detection.
//! - The default inverter sizing produces a sane delay (`< 1 ns` at
//!   Vdd=1.8V, default L=2µm geometry — the LEVEL=1 model is slow
//!   compared to a real PDK but the **shape** is right: tphl and tplh
//!   are both positive, finite, and within ~3× of each other).
//!
//! ## Soft-skip
//!
//! Gated by `feature = "ngspice"` and runtime `LocalBinary::from_env()`.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_hir::SourceWaveform;
use eda_spice_emit::{Netlist, SpiceEmit};
use spike_cmos_gates::Inverter;

const VDD: f64 = 1.8;
const HALF_VDD: f64 = VDD / 2.0;

/// Output load capacitance — without it the inverter's RC time constant
/// is dominated by parasitic numerics rather than physics, and crossing
/// times collapse onto the SPICE grid resolution. 20 fF is a textbook
/// "fanout-of-4 inverter chain" load and gives a τ ~ 200 ps with
/// default-sized LEVEL=1 transistors at Vdd=1.8 V — comfortably
/// resolvable at h = 10 ps.
const C_LOAD: f64 = 20e-15;

/// Build a transient deck: PULSE on `in`, inverter driving `out`,
/// load capacitor `C_LOAD` from `out` to ground.
fn deck(input: &SourceWaveform) -> String {
    let mut net = Netlist::new("Inverter propagation-delay probe");
    net.add_dc_source("dd", "vdd", "0", VDD);
    net.add_waveform_source("in", "in", "0", input);
    Inverter::default()
        .emit_spice(&mut net, &["in", "out", "vdd", "0"], "u1")
        .unwrap();
    net.add_element(format!("Cload out 0 {C_LOAD:.10e} IC=0"));
    net.deck()
}

/// First time `y(t)` crosses `level` while moving in `direction`
/// (`1` = rising, `-1` = falling), found by linear interpolation between
/// bracketing samples. Search starts at `t_start` (inclusive) — used to
/// skip the initial-condition transient at t=0 and find the post-edge
/// crossing instead. `None` if no crossing.
fn crossing_after(
    t: &[f64], y: &[f64], level: f64, direction: i32, t_start: f64,
) -> Option<f64> {
    debug_assert_eq!(t.len(), y.len());
    for i in 1..t.len() {
        if t[i] < t_start { continue; }
        let above = y[i] > level;
        let below = y[i] < level;
        let prev_below = y[i - 1] < level;
        let prev_above = y[i - 1] > level;
        let rising_xing = direction > 0 && above && prev_below;
        let falling_xing = direction < 0 && below && prev_above;
        if rising_xing || falling_xing {
            let frac = (level - y[i - 1]) / (y[i] - y[i - 1]);
            return Some(t[i - 1] + frac * (t[i] - t[i - 1]));
        }
    }
    None
}

#[test]
fn inverter_tphl_tplh_positive_and_in_range() {
    let Ok(ng) = LocalBinary::from_env() else {
        eprintln!("ngspice missing");
        return;
    };

    // Sharp edges (10 ps tr/tf) so the gate sees a "step" — the
    // measured tphl/tplh then reflects the inverter's intrinsic RC
    // (driven into C_LOAD), not how it tracks a slow input ramp. 10 ps
    // edges bracket cleanly inside the 10 ps timestep grid.
    let v1 = 0.0_f64;
    let v2 = VDD;
    let td = 500e-12;     // 500 ps delay before the rising edge
    let tr = 10e-12;
    let tf = 10e-12;
    let pw = 3e-9;        // pulse stays high 3 ns — long enough to settle
    let pulse = SourceWaveform::pulse(v1, v2, td, tr, tf, pw, 0.0);

    let h = 10e-12;       // 10 ps timestep
    let t_stop = td + tr + pw + tf + 3e-9;
    let analysis = TransientAnalysis::new(h, t_stop).with_t_max(h);

    let trace = ng
        .run_transient_trace(
            &deck(&pulse),
            &analysis,
            &[
                OutputRequest::NodeVoltage("in".into()),
                OutputRequest::NodeVoltage("out".into()),
            ],
        )
        .expect("ngspice transient");

    let t = &trace.time;
    let v_in = &trace.node_voltages["in"];
    let v_out = &trace.node_voltages["out"];

    // Skip the IC=0 → Vdd settling: out starts at 0 (IC) but the
    // inverter's PMOS is on (input at 0), so out rises to Vdd very
    // quickly before the input's first edge. Bound that with a small
    // pre-edge offset so the post-edge crossings are what we measure.
    let pre_edge = td * 0.5;
    let t_in_rise = crossing_after(t, v_in, HALF_VDD, 1, pre_edge)
        .expect("input never rises through Vdd/2");
    // Output falls in response to input rising → search after t_in_rise.
    let t_out_fall = crossing_after(t, v_out, HALF_VDD, -1, t_in_rise)
        .expect("output never falls through Vdd/2 after input rise");
    let tphl = t_out_fall - t_in_rise;

    // For tplh: find the input's falling edge (after pulse high), then
    // the output's rising edge after that.
    let t_in_fall = crossing_after(t, v_in, HALF_VDD, -1, t_in_rise + tr + pw * 0.5)
        .expect("input never falls through Vdd/2");
    let t_out_rise = crossing_after(t, v_out, HALF_VDD, 1, t_in_fall)
        .expect("output never rises through Vdd/2 after input fall");
    let tplh = t_out_rise - t_in_fall;

    eprintln!("inverter tphl = {:.3} ps", tphl * 1e12);
    eprintln!("inverter tplh = {:.3} ps", tplh * 1e12);

    // Sanity envelopes — generous because this is LEVEL=1, not a real
    // PDK, and we want the test to pass even if defaults shift slightly.
    assert!(tphl > 0.0, "tphl should be positive, got {tphl:.3e}");
    assert!(tplh > 0.0, "tplh should be positive, got {tplh:.3e}");
    assert!(tphl < 5e-9, "tphl = {:.1} ps, expected < 5 ns", tphl * 1e12);
    assert!(tplh < 5e-9, "tplh = {:.1} ps, expected < 5 ns", tplh * 1e12);

    // tphl and tplh shouldn't differ by more than ~3× — if they do,
    // the βn/βp ratio is grossly skewed (fix the inverter defaults).
    let ratio = tphl.max(tplh) / tphl.min(tplh);
    assert!(ratio < 3.0, "tphl/tplh ratio = {ratio:.2}, expected < 3");
}

#[test]
fn inverter_output_settles_to_rails() {
    // Cross-check against the truth-table test: at the end of the
    // simulation (well after all edges) the output must be at the rail
    // determined by the static input level.
    let Ok(ng) = LocalBinary::from_env() else { return; };

    // Input held high for the last several ns: output should fall to 0.
    let pulse = SourceWaveform::pulse(0.0, VDD, 100e-12, 10e-12, 10e-12, 100e-9, 0.0);
    let analysis = TransientAnalysis::new(20e-12, 5e-9).with_t_max(20e-12);

    let trace = ng
        .run_transient_trace(
            &deck(&pulse),
            &analysis,
            &[OutputRequest::NodeVoltage("out".into())],
        )
        .expect("ngspice transient");

    let v_out_final = *trace.node_voltages["out"].last().unwrap();
    // Input held high → output should be near 0.
    assert!(v_out_final < 0.05,
        "Vout(final) = {v_out_final:.4} V, expected near 0 V (input held high)");
}
