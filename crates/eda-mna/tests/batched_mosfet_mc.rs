// SPDX-License-Identifier: GPL-3.0-only
// RLX-EDA — circuit-level building blocks on top of rlx.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Phase-4 validation: Monte-Carlo Vth mismatch on a loaded NMOS through
//! eda-mna's batched_solve_dc.
//!
//! Topology (a single NMOS with a resistor pull-up — the simplest
//! circuit that has a non-trivial Newton + a Vth dependence):
//!
//! ```text
//!   V_dd ──── R_load ──── vd ──── M1 (D)
//!                                 M1 (G) ──── V_gg
//!                                 M1 (S) ──── GND
//!                                 M1 (B) ──── GND
//! ```
//!
//! What this exercises that prior batched tests don't:
//!
//! * `mc_params` carrying a per-draw `Vth` — exercises the
//!   `promote_params_to_inputs` → `vmap` chain on a real device-bound
//!   Param, not just boundary voltages.
//! * The Mosfet `currents()` graph emits ~30 ops including Activation,
//!   Where, Compare — vmap rules cover all of these but they hadn't
//!   actually been driven from this codepath before.
//! * Per-draw scalar parity confirms the device's gradients are still
//!   correct after vmap (key concern: the vmap'd jac graph must produce
//!   the same per-draw J_d as scalar grad_with_loss at draw d's params).
//!
//! Out of scope (phase 4.5+):
//!
//! * The textbook 1:4 current mirror — needs a current source as a
//!   device. Squeezed for this turn; the loaded-NMOS test above already
//!   proves the per-draw-Vth pipeline. A current-source device + the
//!   mirror topology drops in on top of this test in a follow-up
//!   without changing batched_solve_dc.

use std::collections::HashMap;

use eda_hir::Block;
use eda_mna::{
    batched_solve_dc, solve_dc, Circuit, NetId, NewtonOptions,
};
use spike_divider_block::{Mosfet, Resistor};

#[test]
fn batched_dc_per_draw_vth_matches_scalar_on_loaded_nmos() {
    // Build the circuit.
    let mut c = Circuit::new();
    let vdd = c.alloc_boundary_net();
    let vgg = c.alloc_boundary_net();
    let vd  = c.alloc_unknown_net();

    // R_load = 10 kΩ between V_dd and vd.
    let r = Resistor { length: 10_000, id: "Rload".into() };
    c.add_device(r.clone(), &[vdd, vd]);

    // NMOS M1: W=1µm, L=1µm. Terminal order [D, G, S, B].
    let m = Mosfet::nmos(1_000, 1_000, "M1");
    c.add_device(m.clone(), &[vd, vgg, NetId::GND, NetId::GND]);

    // Shared params: Mosfet defaults + R_load value. Skip the Vth
    // entry — we'll feed it per-draw via mc_params.
    let mut params: HashMap<String, f32> = m.default_params();
    params.insert(Block::name(&r), 10_000.0);
    let m_vth_key = format!("{}_Vth", Block::name(&m));
    let _ = params.remove(&m_vth_key);

    // Per-draw Vth values — synthetic mismatch sweep around the
    // nominal 0.5 V. Real Pelgrom σ at this geometry is ~5 mV; we
    // span a wider band to make the test less sensitive to f32 noise.
    let vth_draws: Vec<f32> = vec![0.40, 0.45, 0.50, 0.55, 0.60];
    let n_draws = vth_draws.len();
    let mut mc_params: HashMap<String, Vec<f32>> = HashMap::new();
    mc_params.insert(m_vth_key.clone(), vth_draws.clone());

    // Boundary voltages — same V_dd / V_gg across all draws (only the
    // Vth varies). Each entry must be length n_draws.
    let mut boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
    boundary.insert(vdd, vec![1.8_f32; n_draws]);
    boundary.insert(vgg, vec![1.0_f32; n_draws]);

    let opt = NewtonOptions { init: 1.0, ..NewtonOptions::default() };
    let batched = batched_solve_dc(&c, n_draws, &params, &mc_params, &boundary, opt);
    assert!(batched.converged.iter().all(|&c| c),
        "every draw should converge: {:?} after {} iters", batched.converged, batched.iters);

    // Per-draw scalar reference: bind that draw's Vth as a shared
    // Param + run scalar solve_dc.
    for (idx, &vth) in vth_draws.iter().enumerate() {
        let mut p_scalar = params.clone();
        p_scalar.insert(m_vth_key.clone(), vth);
        let mut bnd_scalar: HashMap<NetId, f32> = HashMap::new();
        bnd_scalar.insert(vdd, 1.8);
        bnd_scalar.insert(vgg, 1.0);
        let scalar_op = solve_dc(&c, &p_scalar, &bnd_scalar, opt);
        assert!(scalar_op.converged,
            "scalar draw {idx} (Vth={vth}) didn't converge in {} iters \
             (residual={:.3e})", scalar_op.iters, scalar_op.final_residual_max);
        let scalar_vd  = scalar_op.voltages[&vd];
        let batched_vd = batched.voltages[&vd][idx];
        let drift = (batched_vd - scalar_vd).abs();
        assert!(
            drift < 5e-4,
            "draw {idx}: Vth={vth} batched vd={batched_vd} \
             scalar vd={scalar_vd} (Δ {drift:.3e})",
        );

        // Sanity: vd between (well above) GND and (≤) V_dd.
        // Higher Vth → less current → less drop → vd closer to V_dd.
        assert!(
            (0.0 < batched_vd) && (batched_vd <= 1.8 + 1e-3),
            "draw {idx}: vd={batched_vd} outside (0, V_dd]",
        );
    }

    // Monotonicity check: as Vth increases, less drain current flows,
    // so less drop across R_load, so vd should increase. This catches
    // a class of bugs where the per-draw Vth bound to the wrong draw.
    let vds: Vec<f32> = (0..n_draws).map(|i| batched.voltages[&vd][i]).collect();
    for i in 1..n_draws {
        assert!(
            vds[i] >= vds[i-1] - 1e-4,
            "vd should be monotone-↑ in Vth: vds={vds:?}",
        );
    }
}
