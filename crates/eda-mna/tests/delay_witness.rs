//! Analytic transport-delay witness for `IdealDelay` + the integrator's
//! per-element history buffer.
//!
//! Topology:
//!
//! ```text
//!   V_in (boundary) ── [IdealDelay τ, G] ── v_out ── [R_term] ── gnd
//! ```
//!
//! KCL at `v_out`:  `G · v_in(t − τ) − v_out / R = 0`  ⇒
//! `v_out(t) = (G · R) · v_in(t − τ)`. With `G = 1 S` and `R = 1 Ω`,
//! the output is the input shifted by `τ`.
//!
//! Stimulus: a clean SPICE PULSE on `V_in` (handled by `SourceWaveform`
//! so the boundary edges are framework-canonical and we don't fight
//! `f32` step-edge rounding).

use std::collections::HashMap;
use eda_hir::SourceWaveform;
use eda_mna::{
    solve_dc, transient_from, transient_pwl,
    Circuit, IdealDelay, NetId, NewtonOptions,
};
use spike_divider_block::Resistor;

#[test]
fn ideal_delay_shifts_pulse_by_tau() {
    let mut c = Circuit::new();
    let v_in  = c.alloc_boundary_net();
    let v_out = c.alloc_unknown_net();

    // τ = 5 ps. Sub-step delays are forbidden by the assert in
    // `init_delay_histories`; this satisfies τ ≥ dt comfortably.
    let dly_name = "dly";
    let tau = 5e-12_f64;
    c.add_delay(IdealDelay::new(dly_name, tau), [v_in, v_out]);

    // Terminator R from v_out to gnd, 1 Ω.
    let r_term = Resistor { length: 10_000, id: "RT".into() };
    c.add_device(r_term.clone(), &[v_out, NetId::GND]);

    // Params: G = 1 S, R = 1 Ω → unity steady-state gain.
    let mut params = HashMap::new();
    params.insert(format!("{dly_name}_G"), 1.0_f32);
    params.insert(eda_hir::Block::name(&r_term), 1.0_f32);

    // 0 → 1 → 0 V pulse on V_in: high from 10 ps to 20 ps.
    let stim = SourceWaveform::pulse(
        0.0, 1.0,
        /* td */ 10e-12, /* tr */ 0.0, /* tf */ 0.0,
        /* pw */ 10e-12, /* per */ 0.0,
    );
    let stim_at = move |t: f32| -> f32 { stim.value_at(t as f64) as f32 };
    let boundary_at = |t: f32| {
        let mut m = HashMap::new();
        m.insert(v_in, stim_at(t));
        m
    };

    let mut ic = HashMap::new();
    ic.insert(v_out, 0.0_f32);

    let dt = 1e-12_f32;
    let n_steps = 40;
    let wave = transient_pwl(&c, &params, boundary_at, &ic,
                             dt, n_steps, NewtonOptions::default());

    for (k, s) in wave.iter().enumerate() {
        assert!(s.converged,
            "step {k} t={:.3e} not converged: residual {:.3e}",
            s.t, s.final_residual_max);
    }

    // For every step k ≥ τ/dt = 5, expect v_out(t) ≈ v_in(t − τ). The
    // edges (t ≈ 15 ps and t ≈ 25 ps in the *output* timeline) are one
    // BE step wide; tolerate generously there, tightly elsewhere.
    let tau_f32 = tau as f32;
    let mut max_interior_err = 0.0_f32;
    for s in wave.iter() {
        let expected = stim_at(s.t - tau_f32);
        let got      = s.voltages[&v_out];
        let near_rise = (s.t - 10e-12 - tau_f32).abs() < 1.5 * dt;
        let near_fall = (s.t - 20e-12 - tau_f32).abs() < 1.5 * dt;
        let on_edge   = near_rise || near_fall;
        let tol = if on_edge { 6e-1 } else { 1e-3 };
        assert!((got - expected).abs() < tol,
            "t={:.3e}: v_out={} expected≈{} (tol {}, on_edge={})",
            s.t, got, expected, tol, on_edge);
        if !on_edge {
            max_interior_err = max_interior_err.max((got - expected).abs());
        }
    }
    // Overall sanity: away from edges, the delayed reproduction is
    // essentially perfect (the delay element is linear and exactly
    // captured in the rational stamp; only the input's discretization
    // matters).
    assert!(max_interior_err < 1e-3,
        "max interior error {max_interior_err} too large");
}

#[test]
fn dc_solve_converges_through_a_delay() {
    // At DC the delay collapses to i_out = G·v_in. With G=1, R_term=0.5,
    // and v_in driven to 0.7 V, the steady-state v_out = G·R·v_in = 0.35 V.
    // Pre-fix `solve_dc` ignored delays, so the loop didn't converge to
    // a sensible OP — this is the regression guard.
    let mut c = Circuit::new();
    let v_in  = c.alloc_boundary_net();
    let v_out = c.alloc_unknown_net();
    c.add_delay(IdealDelay::new("dly", 5e-12), [v_in, v_out]);
    let r_term = Resistor { length: 10_000, id: "RT".into() };
    c.add_device(r_term.clone(), &[v_out, NetId::GND]);

    let mut params = HashMap::new();
    params.insert("dly_G".into(), 1.0_f32);
    params.insert(eda_hir::Block::name(&r_term), 0.5_f32);
    let mut bnd = HashMap::new();
    bnd.insert(v_in, 0.7_f32);

    let op = solve_dc(&c, &params, &bnd, NewtonOptions::default());
    assert!(op.converged, "solve_dc with delay loop did not converge: \
                            residual {:.3e}", op.final_residual_max);
    let v = op.voltages[&v_out];
    assert!((v - 0.35).abs() < 1e-5, "v_out={v}, expected 0.35");
}

#[test]
fn substep_delay_reproduces_input_with_in_graph_interp() {
    // τ < dt: the residual graph couples v_in_now into the BE Jacobian
    // via the in-graph blend formula. Drive a smooth ramp on v_in and
    // check v_out tracks v_in(t − τ).
    let mut c = Circuit::new();
    let v_in  = c.alloc_boundary_net();
    let v_out = c.alloc_unknown_net();
    let tau = 0.4_f64;        // arbitrary units; "seconds" but we
    let dt  = 1.0_f32;        // pick clean numbers so the analytic
    let n_steps = 30;         // expectation is hand-checkable.
    c.add_delay(IdealDelay::new("dly", tau), [v_in, v_out]);
    let r_term = Resistor { length: 10_000, id: "RT".into() };
    c.add_device(r_term.clone(), &[v_out, NetId::GND]);

    let mut params = HashMap::new();
    params.insert("dly_G".into(), 1.0_f32);
    params.insert(eda_hir::Block::name(&r_term), 1.0_f32);

    // Smooth ramp v_in(t) = 0.1·t (avoids step-edge BE error).
    let stim_at = |t: f32| -> f32 { 0.1 * t };
    let boundary_at = |t: f32| {
        let mut m = HashMap::new();
        m.insert(v_in, stim_at(t));
        m
    };
    // Initial condition consistent with constant-history assumption:
    // v_in(t<0) = v_in(0) = 0; steady output v_out(0) = v_in(−τ) = 0.
    let mut ic = HashMap::new();
    ic.insert(v_out, 0.0_f32);

    let wave = transient_pwl(&c, &params, boundary_at, &ic,
                             dt, n_steps, NewtonOptions::default());
    for s in &wave {
        assert!(s.converged, "step at t={:.3} not converged: {:.3e}",
                s.t, s.final_residual_max);
    }

    // Once we're past `t = τ` (the ramp's been "rolling" long enough
    // to flush the constant-history zone), v_out(t) = v_in(t − τ) =
    // 0.1·(t − 0.4) with very high accuracy. BE leaves O(dt) error;
    // the linear interpolation is exact for a linear input, so total
    // error stays at ~floating-point noise.
    let tau_f32 = tau as f32;
    for s in wave.iter().skip(2) {       // skip the warm-up samples
        let expected = stim_at(s.t - tau_f32);
        let got      = s.voltages[&v_out];
        assert!((got - expected).abs() < 1e-4,
            "t={:.3}: v_out={} expected≈{}",
            s.t, got, expected);
    }
}

#[test]
fn tau_param_can_be_overridden_via_params_map() {
    // `<name>_tau` in the `params` map overrides the device's static
    // `delay_seconds()` for both the in-graph α formula AND the
    // integrator-side history indexing. Validates the differentiable-τ
    // wiring end-to-end: passing τ_a vs τ_b through `params` shifts the
    // observed delay accordingly.
    let mut c = Circuit::new();
    let v_in  = c.alloc_boundary_net();
    let v_out = c.alloc_unknown_net();
    // Device's static τ is irrelevant — overridden every run below.
    c.add_delay(IdealDelay::new("dly", 1e-12), [v_in, v_out]);
    let r_term = Resistor { length: 10_000, id: "RT".into() };
    c.add_device(r_term.clone(), &[v_out, NetId::GND]);

    let stim = SourceWaveform::pulse(
        0.0, 1.0, 5e-12, 0.0, 0.0, 100e-12, 0.0,
    );
    let stim_at = move |t: f32| stim.value_at(t as f64) as f32;

    let dt = 1e-12_f32;
    let n_steps = 30;
    let mut ic = HashMap::new();
    ic.insert(v_out, 0.0_f32);

    let run_with_tau = |tau_seconds: f32| -> Vec<f32> {
        let mut params = HashMap::new();
        params.insert("dly_G".into(), 1.0_f32);
        params.insert(eda_hir::Block::name(&r_term), 1.0_f32);
        params.insert("dly_tau".into(), tau_seconds);
        let boundary_at = |t: f32| {
            let mut m = HashMap::new();
            m.insert(v_in, stim_at(t));
            m
        };
        let wave = transient_pwl(&c, &params, boundary_at, &ic,
                                 dt, n_steps, NewtonOptions::default());
        wave.iter().map(|s| s.voltages[&v_out]).collect()
    };

    let trace_a = run_with_tau(3e-12);
    let trace_b = run_with_tau(8e-12);

    // Step 9 (t = 9 ps): with τ_a = 3 ps the delayed pulse has been
    // 'on' for 9 − 5 − 3 = 1 ps → v_out ≈ 1.  With τ_b = 8 ps the
    // delayed pulse hasn't risen yet (9 − 5 − 8 = −4 ps) → v_out ≈ 0.
    assert!(trace_a[9] > 0.9,
        "τ=3ps trace at step 9 = {} (expected ≈1)", trace_a[9]);
    assert!(trace_b[9] < 0.1,
        "τ=8ps trace at step 9 = {} (expected ≈0)", trace_b[9]);

    // Step 15 (t = 15 ps): both rises have completed for both τ values.
    assert!(trace_a[15] > 0.9 && trace_b[15] > 0.9,
        "post-rise: τa={} τb={}", trace_a[15], trace_b[15]);
}

#[test]
fn alpha_finite_difference_matches_in_graph_derivative() {
    // The in-graph stamp has  v_delayed = (1−α) · v_lo + α · v_hi  with
    // α = offset − τ/h. So  ∂v_out/∂τ = −(gain · (v_hi − v_lo)) / h
    // for a unit-gain unit-R termination at the operating point. Run
    // two transients with τ_0 ± ε and check the central FD matches
    // this analytic derivative inside a single integer-step window.
    let mut c = Circuit::new();
    let v_in  = c.alloc_boundary_net();
    let v_out = c.alloc_unknown_net();
    c.add_delay(IdealDelay::new("dly", 1e-12), [v_in, v_out]);
    let r_term = Resistor { length: 10_000, id: "RT".into() };
    c.add_device(r_term.clone(), &[v_out, NetId::GND]);

    let dt  = 1e-12_f32;
    let n_steps = 20;
    let stim = SourceWaveform::pulse(
        0.0, 1.0, 3e-12, 4e-12, 0.0, 100e-12, 0.0,
    );
    let stim_at = move |t: f32| stim.value_at(t as f64) as f32;
    let mut ic = HashMap::new();
    ic.insert(v_out, 0.0_f32);

    let run_with_tau = |tau_seconds: f32| -> Vec<f32> {
        let mut params = HashMap::new();
        params.insert("dly_G".into(), 1.0_f32);
        params.insert(eda_hir::Block::name(&r_term), 1.0_f32);
        params.insert("dly_tau".into(), tau_seconds);
        let boundary_at = |t: f32| {
            let mut m = HashMap::new();
            m.insert(v_in, stim_at(t));
            m
        };
        let wave = transient_pwl(&c, &params, boundary_at, &ic,
                                 dt, n_steps, NewtonOptions::default());
        wave.iter().map(|s| s.voltages[&v_out]).collect()
    };

    // Pick τ_0 = 2.5 ps — *strictly inside* (2 ps, 3 ps), so τ_0 ± ε
    // doesn't cross an integer-step boundary (where `offset` jumps and
    // the stamp is non-smooth). Step k = 9 (t = 9 ps) lands on the
    // smooth ramp portion of the input.
    let tau0  = 2.5e-12_f32;
    let eps   = 5e-14_f32;
    let trace_plus  = run_with_tau(tau0 + eps);
    let trace_minus = run_with_tau(tau0 - eps);
    let trace_at0   = run_with_tau(tau0);

    let k = 9;
    let fd = (trace_plus[k] - trace_minus[k]) / (2.0 * eps);

    // Analytic ∂v_out/∂τ at this op-point: stamp gives
    //   v_out = α · v_hi + (1 − α) · v_lo,  α = offset − τ/h
    //   ⇒ ∂v_out/∂τ = −(v_hi − v_lo) / h.
    // v_hi = stim_at((k − floor(τ/dt))·dt),  v_lo = stim_at(prev sample).
    let i_floor = (tau0 / dt).floor() as i64;
    let t_hi = (k as f32 - i_floor     as f32) * dt;
    let t_lo = (k as f32 - (i_floor+1) as f32) * dt;
    let analytic = -(stim_at(t_hi) - stim_at(t_lo)) / dt;

    let v_at_t0 = trace_at0[k];
    assert!(v_at_t0.is_finite(), "baseline v_out invalid at k={k}");
    // FD and analytic agree to float32 precision; check relative error.
    // Magnitudes here are O(1/dt) = O(1e11), so an absolute tolerance
    // would have to be ≳ 1e5 — relative is the honest comparison.
    let rel = (fd - analytic).abs() / analytic.abs().max(1e-30);
    assert!(rel < 5e-5,
        "FD={fd} vs analytic={analytic} at τ=2.5ps, k=9 \
         (relative err {:.3e})", rel);
}

#[test]
fn delay_with_no_history_holds_initial_constant() {
    // Same topology, but V_in stays at a constant 0.7 V from t=0. The
    // history buffer's "constant history" convention — v_in(t<0) = v_in(0)
    // = 0.7 — should make v_out(t) = 0.7 V from the very first step,
    // without any "warm-up" period.
    let mut c = Circuit::new();
    let v_in  = c.alloc_boundary_net();
    let v_out = c.alloc_unknown_net();

    c.add_delay(IdealDelay::new("dly", 4e-12), [v_in, v_out]);
    let r_term = Resistor { length: 10_000, id: "RT".into() };
    c.add_device(r_term.clone(), &[v_out, NetId::GND]);

    let mut params = HashMap::new();
    params.insert("dly_G".into(), 1.0_f32);
    params.insert(eda_hir::Block::name(&r_term), 1.0_f32);

    let boundary_at = |_t: f32| {
        let mut m = HashMap::new();
        m.insert(v_in, 0.7_f32);
        m
    };
    let mut ic = HashMap::new();
    ic.insert(v_out, 0.7_f32);    // matches steady state with constant history

    let wave = transient_pwl(&c, &params, boundary_at, &ic,
                             1e-12, 20, NewtonOptions::default());
    for s in &wave {
        assert!(s.converged);
        let v = s.voltages[&v_out];
        assert!((v - 0.7).abs() < 1e-4,
            "t={:.3e}: v_out={} should hold at 0.7", s.t, v);
    }
}
