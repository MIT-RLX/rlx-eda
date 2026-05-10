//! RC low-pass driven by a `SourceWaveform::Pulse` — the time-varying-source
//! analog of `spike-rc-transient`.
//!
//! Where `spike-rc-transient` proved that the rlx outer-loop transient runs
//! correctly under a constant DC source, this spike does the same proof
//! against a step-/pulse-shaped source. That requires:
//!
//! 1. A way to feed `V(t)` into the BE step graph at each timestep — already
//!    in place via the `v_at_step` closure on `spike_rc_transient::run_transient`.
//!    Here we sample [`eda_hir::SourceWaveform::value_at`] inside that
//!    closure so a single `SourceWaveform` value drives both rlx and SPICE.
//! 2. A piecewise-analytic reference. For a `PULSE` with `tr=tf=0` and
//!    initial output settled to `v1`, the response is two stitched
//!    first-order RC segments per period — closed form below.
//! 3. A SPICE deck emitter that uses the same `SourceWaveform`. We use
//!    `eda_spice_emit::Netlist::add_waveform_source` so the rlx side and the
//!    ngspice side cannot disagree on stimulus.
//!
//! ## Scope
//!
//! - **Tier-1 / Tier-2 only**: rlx forward, analytic, and finite differences.
//!   ngspice cross-validation lives in `tests/ngspice.rs` behind the
//!   `ngspice` feature flag.
//! - **Single-pulse waveforms only** (`per <= 0` or `per` larger than the
//!   simulation horizon). Periodic pulses also work, but the analytic
//!   reference here doesn't unwrap them; just call `analytic_pulse_at` once
//!   per region.
//! - **`tr = tf = 0`** in the analytic. The rlx side handles arbitrary
//!   rise/fall, but the closed-form witness in this crate assumes step
//!   transitions to keep the algebra one-line per region.
//!
//! ## What this validates
//!
//! - rlx outer-loop BE under a non-constant source matches the analytic
//!   piecewise response across the rising edge, the high plateau, the
//!   falling edge, and the relaxation tail.
//! - The single-step BE gradients (`spike-rc-transient::run_step_and_grad`)
//!   are independent of the source value once `V` is set, so we don't
//!   re-derive them here — that work was already done. The new gradient
//!   surface this spike adds is `∂vout/∂v2` (the pulsed level), which we
//!   check via FD.

use eda_hir::SourceWaveform;
use eda_spice_emit::{Netlist, SpiceEmit, C, R};
use spike_rc_transient as rct;

/// Run the BE step graph in an outer Rust loop with a [`SourceWaveform`]
/// driving the source. Returns `(time, vout)` aligned to step indices
/// `0..=n_steps` (so `time[0] = 0` carries the initial condition).
pub fn run_transient_trace(
    n_steps: usize,
    h: f64,
    r: f64,
    c: f64,
    vout0: f64,
    waveform: &SourceWaveform,
) -> (Vec<f64>, Vec<f64>) {
    rct::run_transient_trace(n_steps, h, r, c, vout0, |n| {
        // Sample the waveform at the END of step n — Backward Euler
        // evaluates the source at the new timestep, not the old one.
        waveform.value_at(n as f64 * h)
    })
}

/// Final-time output via the same outer loop.
pub fn run_transient_final(
    n_steps: usize,
    h: f64,
    r: f64,
    c: f64,
    vout0: f64,
    waveform: &SourceWaveform,
) -> f64 {
    rct::run_transient(n_steps, h, r, c, vout0, |n| {
        waveform.value_at(n as f64 * h)
    })
}

// ── Analytic piecewise reference (continuous-time, h → 0) ──────────────
//
// First-order RC step response: starting from output `y0` at time `t0`
// with target steady-state `v_inf` and time constant `tau = RC`,
//
//     y(t) = v_inf + (y0 - v_inf) · exp(-(t - t0)/tau).
//
// A PULSE with `tr=tf=0` is two stitched applications:
//   t < td:                 y = v1   (assuming pre-settled)
//   td  <= t < td+pw:       step from v1 → v2, integrate above
//   td+pw <= t:             step from y(td+pw) → v1, integrate above
//
// Higher-rep pulses repeat the same pattern; the reference below handles
// only one pulse for clarity.

/// Continuum (h → 0) analytic vout(t) for a single PULSE with `tr=tf=0`.
/// Assumes `vout(0) = v1` (pre-settled). For a periodic pulse, results are
/// only correct in the first period.
pub fn analytic_pulse_at(t: f64, v1: f64, v2: f64, td: f64, pw: f64, r: f64, c: f64) -> f64 {
    let tau = r * c;
    if t < td {
        v1
    } else if t < td + pw {
        // Charging toward v2 from y0 = v1.
        v2 + (v1 - v2) * (-(t - td) / tau).exp()
    } else {
        // y at end of high plateau, then relaxing back to v1.
        let y_at_fall = v2 + (v1 - v2) * (-pw / tau).exp();
        v1 + (y_at_fall - v1) * (-(t - td - pw) / tau).exp()
    }
}

/// Convenience: pull the same numbers off a `SourceWaveform::Pulse`.
/// Panics on non-Pulse — the analytic only knows that one shape.
pub fn analytic_pulse_at_waveform(t: f64, w: &SourceWaveform, r: f64, c: f64) -> f64 {
    match *w {
        SourceWaveform::Pulse { v1, v2, td, tr, tf, pw, per: _ } => {
            assert!(tr == 0.0 && tf == 0.0, "analytic reference assumes tr=tf=0");
            analytic_pulse_at(t, v1, v2, td, pw, r, c)
        }
        _ => panic!("analytic_pulse_at_waveform: not a Pulse"),
    }
}

// ── SPICE deck (cross-simulator) ───────────────────────────────────────

/// Build a SPICE deck for the RC LP with a `SourceWaveform`-driven source,
/// matching the rlx setup: BDF1 (`method=gear maxord=1`) so both engines
/// run the same numerical discretization, `IC=0` on the cap, `.ic` on
/// the output node so `uic` starts cleanly from a discharged state.
pub fn spice_deck(r: f64, c: f64, waveform: &SourceWaveform) -> String {
    let mut n = Netlist::new("RC LP pulse (rlx-eda spike)");
    n.add_preamble(".options method=gear maxord=1");
    n.add_waveform_source("in", "vin", "0", waveform);
    R { ohms: r }.emit_spice(&mut n, &["vin", "vout"], "1").unwrap();
    n.add_element(format!("C1 vout 0 {c:.10e} IC=0"));
    n.add_element(".ic v(vout)=0");
    // Suppress the C primitive; we want the explicit IC=0 attribute, which
    // the standard `C` emit doesn't carry. (Capacitor with IC will earn a
    // proper `C` field later; for now the manual line is fine.)
    let _ = C { farads: c }; // touch the import for crate-doc visibility
    n.deck()
}
