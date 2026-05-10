// SPDX-License-Identifier: GPL-3.0-only
// RLX-EDA — circuit-level building blocks on top of rlx.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Phase-5B end-to-end: multi-step `batched_transient_from` on an RC
//! discharge. Loops `batched_solve_be_step` for n_steps, threading
//! per-draw `prev_voltages` from each step's solved output.
//!
//! Topology:
//!
//! ```text
//!     vmid ──[R]── gnd
//!     vmid ──[C]── gnd      (initial v_mid = per-draw)
//! ```
//!
//! Per-draw IC: 5 distinct initial v_C values. Integrate over 5τ at
//! dt = τ/50. Compare per-draw final v_C to:
//!   * scalar `transient_from` per draw (cross-path parity)
//!   * analytic v_init·exp(-5)·BE-correction (BE has O(h) drift, ~3% at this dt)

use std::collections::HashMap;

use eda_hir::Block;
use eda_mna::{
    batched_transient_from, transient_from, Circuit, NetId, NewtonOptions,
};
use spike_divider_block::{Capacitor, Resistor};

#[test]
fn batched_transient_matches_scalar_per_draw_on_rc_discharge() {
    let mut c = Circuit::new();
    let vmid = c.alloc_unknown_net();

    let r   = Resistor { length: 10_000, id: "R1".into() };
    let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
    c.add_device(r.clone(),    &[vmid, NetId::GND]);
    c.add_storage(cap.clone(), [vmid, NetId::GND]);

    let r_ohms   = 1_000.0_f32;
    let c_farads = 1e-9_f32;
    let tau      = r_ohms * c_farads;       // 1 µs
    let dt       = tau / 50.0;
    let n_steps  = 250;                     // → t_end = 5τ

    let mut params = HashMap::new();
    params.insert(Block::name(&r), r_ohms);
    params.insert(format!("{}_C", Block::name(&cap)), c_farads);

    let v_init: Vec<f32> = vec![1.0, 0.8, 0.5, 0.2, 0.05];
    let n_draws = v_init.len();
    let mut ic: HashMap<NetId, Vec<f32>> = HashMap::new();
    ic.insert(vmid, v_init.clone());

    let boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
    let mc_params: HashMap<String, Vec<f32>> = HashMap::new();
    let opt = NewtonOptions::default();

    let waveform = batched_transient_from(
        &c, n_draws, &params, &mc_params, &boundary, &ic,
        dt, n_steps, opt,
    );

    assert_eq!(waveform.len(), n_steps + 1, "n_steps + 1 rows");
    // t=0 echoes initial conditions verbatim.
    for d in 0..n_draws {
        assert_eq!(waveform[0].voltages[&vmid][d], v_init[d]);
    }
    // Every interior + final step converged for every draw (linear).
    for k in 1..=n_steps {
        assert!(waveform[k].converged.iter().all(|&c| c),
            "step {k}: converged={:?}", waveform[k].converged);
    }

    // Per-draw scalar reference. transient_from takes scalar IC so
    // we run it n_draws times.
    for (idx, &v0) in v_init.iter().enumerate() {
        let mut ic_scalar = HashMap::new();
        ic_scalar.insert(vmid, v0);
        let bnd_scalar: HashMap<NetId, f32> = HashMap::new();
        let scalar_wf = transient_from(
            &c, &params, &bnd_scalar, &ic_scalar, dt, n_steps, opt,
        );
        // End-of-window comparison — accumulating BE drift would inflate
        // intermediate-step parity, so check the most-stressed sample.
        let v_end_scalar  = scalar_wf[n_steps].voltages[&vmid];
        let v_end_batched = waveform[n_steps].voltages[&vmid][idx];
        let drift = (v_end_batched - v_end_scalar).abs();
        assert!(
            drift < 1e-5,
            "draw {idx}: batched v(5τ)={v_end_batched} scalar v(5τ)={v_end_scalar} \
             (Δ {drift:.3e})",
        );
    }

    // Analytic spot-check: at t=5τ, v should be ≈ v_init · exp(-5)
    // ≈ 0.00674 · v_init. BE drifts ~3% low at dt=τ/50; the per-draw
    // scalar comparison above is the strict check, this one is a
    // sanity floor.
    let expected_decay = (-5.0_f32).exp();    // ≈ 0.00674
    for (idx, &v0) in v_init.iter().enumerate() {
        let v_end = waveform[n_steps].voltages[&vmid][idx];
        let analytic = v0 * expected_decay;
        let rel = (v_end - analytic).abs() / analytic.max(1e-30);
        assert!(rel < 0.10,
            "draw {idx}: v_end={v_end} analytic={analytic} (rel {rel:.2e})");
    }

    // Timestamps monotone-↑ and last == n_steps · dt.
    assert!((waveform[n_steps].t - n_steps as f32 * dt).abs() < 1e-6);
    for k in 1..=n_steps {
        assert!(waveform[k].t > waveform[k-1].t);
    }
}
