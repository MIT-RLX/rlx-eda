// SPDX-License-Identifier: GPL-3.0-only
// RLX-EDA — circuit-level building blocks on top of rlx.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Phase-5A: end-to-end parity for `batched_solve_be_step`.
//!
//! Topology — a discharging RC:
//!
//! ```text
//!     vmid ──[R]── gnd
//!     vmid ──[C]── gnd      (initial v_mid = per-draw)
//! ```
//!
//! Closed-form Backward-Euler step for v_mid:
//!   `v(t+h) = v_prev / (1 + h/(R·C))`.
//!
//! For h = R·C exactly, v(t+h) = v_prev / 2 — clean check.
//!
//! Per-draw variation: each draw has its own initial v_mid (so the
//! batched BE step needs to thread per-draw `v_prev` correctly).
//! Verifies against scalar `solve_be_step` ran independently per draw.

use std::collections::HashMap;

use eda_hir::Block;
use eda_mna::{
    batched_solve_be_step, solve_be_step, Circuit, NetId, NewtonOptions,
};
use spike_divider_block::{Capacitor, Resistor};

#[test]
fn batched_be_step_matches_scalar_per_draw_on_rc_discharge() {
    let mut c = Circuit::new();
    let vmid = c.alloc_unknown_net();

    // R between vmid and GND, C between vmid and GND.
    let r   = Resistor { length: 10_000, id: "R1".into() };
    let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
    c.add_device(r.clone(),    &[vmid, NetId::GND]);
    c.add_storage(cap.clone(), [vmid, NetId::GND]);
    assert_eq!(c.n_storage(), 1);

    // Pick R, C, h such that h = R·C → v(t+h) = v_prev / 2 (analytic
    // BE step is exact for any h on a linear RC, no integration error).
    let r_ohms   = 1_000.0_f32;
    let c_farads = 1e-9_f32;
    let h        = r_ohms * c_farads;        // 1 µs

    let mut params = HashMap::new();
    params.insert(Block::name(&r), r_ohms);
    params.insert(format!("{}_C", Block::name(&cap)), c_farads);

    // Per-draw initial v_mid. No boundary nets to vary — vmid is the
    // only thing per-draw.
    let v_init: Vec<f32> = vec![1.0, 0.8, 0.5, 0.2, 0.05];
    let n_draws = v_init.len();
    let mut prev_voltages: HashMap<NetId, Vec<f32>> = HashMap::new();
    prev_voltages.insert(vmid, v_init.clone());

    let boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
    let mc_params: HashMap<String, Vec<f32>> = HashMap::new();
    let opt = NewtonOptions::default();

    let mut inner_cache = eda_mna::InnerSolveCache::default();
    let batched = batched_solve_be_step(
        &c, n_draws, &params, &mc_params, &boundary, &prev_voltages,
        /*delay_inputs=*/ &[],
        h, /*t_prev=*/ 0.0, opt, &mut inner_cache,
    );

    assert!(batched.converged.iter().all(|&c| c),
        "every draw should converge in one BE step (linear circuit): \
         converged={:?} after {} iters", batched.converged, batched.iters);

    // Closed-form expectation: v_new = v_prev / 2 (since h = R·C).
    for (idx, &v_prev) in v_init.iter().enumerate() {
        let v_new_batched = batched.voltages[&vmid][idx];
        let v_new_analytic = v_prev / 2.0;
        let drift_a = (v_new_batched - v_new_analytic).abs();
        assert!(drift_a < 1e-5,
            "draw {idx}: v_prev={v_prev} batched v_new={v_new_batched} \
             analytic v_new={v_new_analytic} (Δ {drift_a:.3e})");
    }

    // Per-draw scalar reference via solve_be_step.
    for (idx, &v_prev) in v_init.iter().enumerate() {
        let mut prev_scalar: HashMap<NetId, f32> = HashMap::new();
        prev_scalar.insert(vmid, v_prev);
        let bnd_scalar: HashMap<NetId, f32> = HashMap::new();
        let scalar_step = solve_be_step(
            &c, &params, &bnd_scalar, &prev_scalar, &[], h, opt,
        );
        assert!(scalar_step.converged,
            "scalar draw {idx} (v_prev={v_prev}) didn't converge");

        let v_new_scalar  = scalar_step.voltages[&vmid];
        let v_new_batched = batched.voltages[&vmid][idx];
        let drift_s = (v_new_batched - v_new_scalar).abs();
        assert!(drift_s < 1e-5,
            "draw {idx}: batched v_new={v_new_batched} scalar v_new={v_new_scalar} \
             (Δ {drift_s:.3e})");
    }

    // Step timestamp should be t_prev + h.
    assert!((batched.t - h).abs() < 1e-12,
        "BatchedTransientStep.t = {}, expected {}", batched.t, h);
}
