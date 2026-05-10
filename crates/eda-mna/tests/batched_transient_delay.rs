// SPDX-License-Identifier: GPL-3.0-only
// RLX-EDA — circuit-level building blocks on top of rlx.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Phase-5C: batched transient with delay element. Same `IdealDelay`
//! + terminator-R topology that `delay_witness.rs` validates the
//! scalar transport-delay path on, batched over `n_draws` distinct
//! constant boundary V_in values.
//!
//! ```text
//!   V_in (per-draw constant) ── [IdealDelay τ, G=1] ── v_out ── [R=1Ω] ── gnd
//! ```
//!
//! Steady state KCL: `G · v_in - v_out / R = 0` with G = R = 1
//!   ⇒ v_out_ss = V_in. So at t > τ + a few BE steps, v_out should
//!   equal V_in for each draw.
//!
//! What this exercises that prior batched tests don't:
//!
//! * `BatchedDelayHistory` — n_draws independent sample buffers,
//!   each tracking one draw's v_in trajectory at this step's time.
//! * `sample_batched_delay_step` — per-draw v_lo / v_hi lookup with
//!   shared blend / offset (τ shared across batch in v1).
//! * `batched_solve_be_step` with a non-empty `delay_inputs` slice —
//!   binds `delay_v_lo_<id>`, `delay_v_hi_<id>`, `delay_blend_<id>`,
//!   `delay_offset_<id>` per element.

use std::collections::HashMap;

use eda_hir::Block;
use eda_mna::{
    batched_transient_from, transient_from, Circuit, IdealDelay, NetId,
    NewtonOptions,
};
use spike_divider_block::Resistor;

#[test]
fn batched_transient_delay_matches_scalar_per_draw() {
    let mut c = Circuit::new();
    let v_in_net  = c.alloc_boundary_net();
    let v_out_net = c.alloc_unknown_net();

    let dly_name = "dly";
    let tau = 5e-12_f64;
    c.add_delay(IdealDelay::new(dly_name, tau), [v_in_net, v_out_net]);

    let r_term = Resistor { length: 10_000, id: "RT".into() };
    c.add_device(r_term.clone(), &[v_out_net, NetId::GND]);

    let mut params: HashMap<String, f32> = HashMap::new();
    params.insert(format!("{dly_name}_G"), 1.0_f32);
    params.insert(Block::name(&r_term), 1.0_f32);

    // 5 draws of distinct constant V_in. Each draw's v_out should
    // settle to V_in[draw] after τ.
    let v_in_draws: Vec<f32> = vec![0.5, 0.7, 1.0, 1.2, 1.5];
    let n_draws = v_in_draws.len();

    let mut boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
    boundary.insert(v_in_net, v_in_draws.clone());
    let mut ic: HashMap<NetId, Vec<f32>> = HashMap::new();
    ic.insert(v_out_net, vec![0.0; n_draws]);
    let mc_params: HashMap<String, Vec<f32>> = HashMap::new();

    let dt: f32 = 1e-12;
    let n_steps = 40;       // 40 ps total — well past τ = 5 ps

    let opt = NewtonOptions::default();
    let waveform = batched_transient_from(
        &c, n_draws, &params, &mc_params, &boundary, &ic,
        dt, n_steps, opt,
    );

    assert_eq!(waveform.len(), n_steps + 1);
    for (k, s) in waveform.iter().enumerate() {
        assert!(s.converged.iter().all(|&c| c),
            "step {k}: converged={:?}", s.converged);
    }

    // After τ + a few steps, v_out tracks V_in per draw.
    let settle_step = (tau as f32 / dt).ceil() as usize + 5;
    for d in 0..n_draws {
        let v_settled = waveform[settle_step].voltages[&v_out_net][d];
        let expected  = v_in_draws[d];
        let drift = (v_settled - expected).abs();
        assert!(drift < 5e-3,
            "draw {d} V_in={} v_out at step {settle_step} = {v_settled}, expected ≈{expected} (Δ {drift:.3e})",
            v_in_draws[d]);
    }

    // Per-draw scalar parity via transient_from.
    for (idx, &v_in) in v_in_draws.iter().enumerate() {
        let mut bnd_scalar: HashMap<NetId, f32> = HashMap::new();
        bnd_scalar.insert(v_in_net, v_in);
        let mut ic_scalar: HashMap<NetId, f32> = HashMap::new();
        ic_scalar.insert(v_out_net, 0.0);
        let scalar_wf = transient_from(
            &c, &params, &bnd_scalar, &ic_scalar, dt, n_steps, opt,
        );
        // Compare every-other-step samples to keep the assertion
        // count manageable; if one is off the next is too.
        for k in (0..=n_steps).step_by(2) {
            let v_scalar  = scalar_wf[k].voltages[&v_out_net];
            let v_batched = waveform[k].voltages[&v_out_net][idx];
            let drift = (v_batched - v_scalar).abs();
            assert!(drift < 1e-5,
                "draw {idx} step {k}: batched v_out={v_batched} \
                 scalar v_out={v_scalar} (Δ {drift:.3e})");
        }
    }
}
