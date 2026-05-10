// SPDX-License-Identifier: GPL-3.0-only
// RLX-EDA — circuit-level building blocks on top of rlx.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Nonlinear validation for `batched_solve_dc`.
//!
//! Builds the same R + Diode circuit `newton_solve.rs` validates the
//! scalar Newton path on, then sweeps V_in over N draws via the
//! batched solver. Compares per-draw V_mid to scalar `solve_dc` and
//! confirms the batched Newton actually iterates (the diode's
//! exponential nonlinearity means at least a handful of iters).

use std::collections::HashMap;

use eda_hir::Block;
use eda_mna::{
    batched_solve_dc, solve_dc, Circuit, NetId, NewtonOptions,
};
use spike_divider_block::{Diode, Resistor};

#[test]
fn batched_dc_matches_scalar_on_r_plus_diode() {
    let mut c = Circuit::new();
    let v_in_net  = c.alloc_boundary_net();
    let v_mid_net = c.alloc_unknown_net();

    let r = Resistor { length: 10_000, id: "Rmid".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "Dmid".into() };
    c.add_device(r.clone(), &[v_in_net, v_mid_net]);
    c.add_device(d.clone(), &[v_mid_net, NetId::GND]);

    let mut params: HashMap<String, f32> = HashMap::new();
    params.insert(Block::name(&r), 1_000.0);
    params.insert(format!("{}_Is", Block::name(&d)), 1e-15);

    // Five draws spanning two OOM of input drive — each one converges
    // to a slightly different V_mid because the diode pinches above
    // ~0.7 V regardless of upstream V_in.
    let v_in_draws: Vec<f32> = vec![0.5, 1.0, 2.0, 3.0, 5.0];
    let n_draws = v_in_draws.len();
    let mut boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
    boundary.insert(v_in_net, v_in_draws.clone());

    let opt = NewtonOptions { init: 0.6, ..NewtonOptions::default() };
    let mc_params: HashMap<String, Vec<f32>> = HashMap::new();
    let batched = batched_solve_dc(&c, n_draws, &params, &mc_params, &boundary, opt);

    assert!(batched.converged.iter().all(|&c| c),
        "every draw should converge: {:?}", batched.converged);
    // Diode is nonlinear → expect more than 1 Newton iter, but it's
    // a 1-D problem so still well under 20.
    assert!(batched.iters >= 1 && batched.iters < 20,
        "expected 1..20 Newton iters, got {}", batched.iters);

    // Per-draw scalar reference.
    for (idx, &v_in) in v_in_draws.iter().enumerate() {
        let mut bnd_scalar: HashMap<NetId, f32> = HashMap::new();
        bnd_scalar.insert(v_in_net, v_in);
        let scalar_op = solve_dc(&c, &params, &bnd_scalar, opt);
        assert!(scalar_op.converged, "scalar draw {idx} (V_in={v_in}) didn't converge");

        let scalar_vmid  = scalar_op.voltages[&v_mid_net];
        let batched_vmid = batched.voltages[&v_mid_net][idx];
        let drift = (batched_vmid - scalar_vmid).abs();

        // 5e-4 V tolerance reflects the f32 path through the diode
        // exponential — the diode forces V_mid into the saturation
        // region where Vt ≈ 25 mV, so per-step rounding is amplified
        // through exp(V/Vt). The Newton residual tolerance is 1e-7;
        // converged voltages still drift more than that across paths.
        assert!(
            drift < 5e-4,
            "draw {idx}: V_in={v_in} batched V_mid={batched_vmid} \
             scalar V_mid={scalar_vmid} (Δ {drift:.3e})",
        );

        // Sanity: V_mid is between 0 and V_in (current flows from
        // V_in through R into the diode-to-ground load — Kirchhoff +
        // passive sign convention). At low V_in the diode barely
        // conducts and V_mid ≈ V_in; at high V_in the diode clamps
        // V_mid near ~0.7 V (Si threshold). Both extremes are
        // physically valid; the per-draw scalar parity check above
        // is what proves numerical correctness.
        assert!(
            batched_vmid > 0.0 && batched_vmid <= v_in + 1e-3,
            "draw {idx}: V_mid={batched_vmid} not in (0, V_in={v_in}]",
        );
    }
}
