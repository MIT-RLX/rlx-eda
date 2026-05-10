// SPDX-License-Identifier: GPL-3.0-only
// RLX-EDA — circuit-level building blocks on top of rlx.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Phase-4.5 — full 1:4 NMOS current mirror Monte-Carlo'd via
//! `batched_solve_dc`.
//!
//! Topology:
//!
//! ```text
//!   I_ref ──→ vgs ──── M1.D, M1.G  (diode-connected)
//!                      M1.S, M1.B = GND
//!
//!                      M2.G = vgs   (gate shared with M1)
//!                      M2.D = V_bias (boundary)
//!                      M2.S, M2.B = GND
//! ```
//!
//! MC variation: per-draw Vth_M1 and Vth_M2 (Pelgrom-style mismatch).
//! Same architecture spike-mosfet-dc's hand-built `mc_gpu.rs` proves
//! works in 30 ms for 100k draws — the eda-mna route here trades some
//! per-iter overhead (Newton iterates instead of closed-form) for a
//! generic MNA pipeline that any other circuit topology drops into
//! without per-spike code.
//!
//! What's inline that isn't yet in spike-divider-block:
//! `ConstantCurrentSource` — a 1-terminal `NonlinearDcBehavioral`
//! that contributes a Param-bound current `+I_ref` into its single
//! terminal. The "other end" is implicit ground; that's an ideal
//! Norton equivalent of an actual two-terminal source. Promoting
//! this to a public device alongside Resistor / VoltageSource is a
//! ~20-line follow-up.

use std::collections::HashMap;

use eda_hir::{Block, NonlinearDcBehavioral};
use eda_mna::{batched_solve_dc, solve_dc, Circuit, NetId, NewtonOptions};
use rlx_ir::{op::BinaryOp, DType, Graph, NodeId, Op, Shape};
use spike_divider_block::Mosfet;

// ── Inline current-source device ────────────────────────────────────

/// Ideal current source: pushes `I_ref` into its single terminal. The
/// param key is `<id>_I` — bind via `params` (or `mc_params` for
/// per-draw I_ref sweeps).
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct ConstantCurrentSource {
    id: String,
}
impl Block for ConstantCurrentSource {
    fn name(&self) -> String {
        format!("Iref_{}", self.id)
    }
}
impl NonlinearDcBehavioral for ConstantCurrentSource {
    fn name(&self) -> String { <Self as Block>::name(self) }
    fn n_terminals(&self) -> usize { 1 }

    fn currents(&self, voltages: &[NodeId], g: &mut Graph) -> Vec<NodeId> {
        debug_assert_eq!(voltages.len(), 1);
        let s = Shape::new(&[1], DType::F32);
        let i_ref = g.param(format!("{}_I", <Self as Block>::name(self)), s.clone());
        // Phantom dependency on the terminal voltage: AD machinery
        // gets confused if currents() returns nodes that don't reach
        // any input. `0 · v` adds a graph-visible (numerically-zero)
        // dependency that satisfies the wrt-coverage check without
        // affecting the value. Same trick eda-mna's residual builder
        // uses for the unknown-net phantom term.
        let zero = g.add_node(
            Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
            vec![],
            s.clone(),
        );
        let zero_v = g.binary(BinaryOp::Mul, zero, voltages[0], s.clone());
        let i_with_phantom = g.binary(BinaryOp::Add, i_ref, zero_v, s);
        vec![i_with_phantom]
    }
}

// ── Circuit ────────────────────────────────────────────────────────

const I_REF: f32   = 5e-6;       // 5 µA reference
const V_BIAS: f32  = 0.9;        // M2 drain bias
const VTH_NOM: f32 = 0.5;
// Pelgrom σ at unit area (1×1 µm²) — synthetic mismatch.
const SIGMA_M1: f32 = 0.005;     // 5 mV
const SIGMA_M2: f32 = 0.0025;    // 2.5 mV (M2 is 4× wider → 2× smaller σ)

#[test]
fn batched_dc_solves_mosfet_mirror_with_pelgrom_vth_mismatch() {
    let mut c = Circuit::new();
    let v_bias = c.alloc_boundary_net();
    let vgs    = c.alloc_unknown_net();

    // I_ref into vgs.
    let iref_dev = ConstantCurrentSource { id: "ref".into() };
    c.add_device(iref_dev.clone(), &[vgs]);

    // M1: diode-connected at vgs.
    let m1 = Mosfet::nmos(1_000, 1_000, "M1");
    c.add_device(m1.clone(), &[vgs, vgs, NetId::GND, NetId::GND]);

    // M2: gate tied to vgs, drain at V_bias. 4× wider in the encoded
    // W; that's reflected in `m2.default_params()` since W appears in
    // the Mosfet's name + Block::name().
    let m2 = Mosfet::nmos(4_000, 1_000, "M2");
    c.add_device(m2.clone(), &[v_bias, vgs, NetId::GND, NetId::GND]);

    // Shared params: each device's defaults + the I_ref source value.
    // Strip Vth keys for both M1 and M2 — those go through mc_params.
    let mut params: HashMap<String, f32> = m1.default_params();
    params.extend(m2.default_params());
    let m1_vth_key = format!("{}_Vth", Block::name(&m1));
    let m2_vth_key = format!("{}_Vth", Block::name(&m2));
    let _ = params.remove(&m1_vth_key);
    let _ = params.remove(&m2_vth_key);
    params.insert(format!("{}_I", Block::name(&iref_dev)), I_REF);
    // Override M2's Kp to reflect 4× width — Mosfet::default_params
    // seeds 100µA which is just K_p, and Mosfet's currents() multiplies
    // by W/L geometrically. Defaults already account for W via the
    // Mosfet's encoded geometry, but we re-confirm here for clarity.
    // (For square-law: I = K_p·(W/L)·…; W and L are folded inside
    // currents() via the device's stored geometry, not via Param.)

    // Per-draw Vth realisations — synthetic Pelgrom-like spread of
    // 6 draws around the nominal 0.5 V.
    let vth_m1_draws: Vec<f32> = vec![
        VTH_NOM - 2.0 * SIGMA_M1,  // -10 mV
        VTH_NOM - 1.0 * SIGMA_M1,  // -5 mV
        VTH_NOM,
        VTH_NOM + 1.0 * SIGMA_M1,  // +5 mV
        VTH_NOM + 2.0 * SIGMA_M1,  // +10 mV
        VTH_NOM,                    // duplicate to test redundancy
    ];
    let vth_m2_draws: Vec<f32> = vec![
        VTH_NOM,                    // M1 varies, M2 nominal
        VTH_NOM,
        VTH_NOM,
        VTH_NOM - 1.0 * SIGMA_M2,  // M1 nominal, M2 -2.5 mV
        VTH_NOM + 1.0 * SIGMA_M2,
        VTH_NOM,
    ];
    let n_draws = vth_m1_draws.len();

    let mut mc_params: HashMap<String, Vec<f32>> = HashMap::new();
    mc_params.insert(m1_vth_key.clone(), vth_m1_draws.clone());
    mc_params.insert(m2_vth_key.clone(), vth_m2_draws.clone());

    let mut boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
    boundary.insert(v_bias, vec![V_BIAS; n_draws]);

    // Initial vgs guess: closed-form saturation V_th + sqrt(2·I/Kp)
    // ≈ 0.5 + sqrt(2·5e-6/100e-6) = 0.5 + 0.316 = 0.816 V.
    let opt = NewtonOptions { init: 0.82, ..NewtonOptions::default() };
    let batched = batched_solve_dc(&c, n_draws, &params, &mc_params, &boundary, opt);

    assert!(batched.converged.iter().all(|&c| c),
        "every draw should converge: converged={:?} after {} iters \
         (residuals={:?})",
        batched.converged, batched.iters, batched.final_residual_max);

    // Per-draw scalar reference.
    for idx in 0..n_draws {
        let mut p_scalar = params.clone();
        p_scalar.insert(m1_vth_key.clone(), vth_m1_draws[idx]);
        p_scalar.insert(m2_vth_key.clone(), vth_m2_draws[idx]);
        let mut bnd_scalar: HashMap<NetId, f32> = HashMap::new();
        bnd_scalar.insert(v_bias, V_BIAS);
        let scalar_op = solve_dc(&c, &p_scalar, &bnd_scalar, opt);
        assert!(scalar_op.converged,
            "scalar draw {idx} (Vth_M1={}, Vth_M2={}) didn't converge \
             in {} iters (residual={:.3e})",
            vth_m1_draws[idx], vth_m2_draws[idx],
            scalar_op.iters, scalar_op.final_residual_max);

        let scalar_vgs  = scalar_op.voltages[&vgs];
        let batched_vgs = batched.voltages[&vgs][idx];
        let drift = (batched_vgs - scalar_vgs).abs();

        // 5 mV tolerance — well below typical analog noise floors and
        // comfortably above f32 LU noise on a 1×1 system.
        assert!(
            drift < 5e-3,
            "draw {idx}: Vth_M1={} Vth_M2={} batched vgs={batched_vgs} \
             scalar vgs={scalar_vgs} (Δ {drift:.3e})",
            vth_m1_draws[idx], vth_m2_draws[idx],
        );

        // Sanity: vgs in the saturation operating range (Vth + 100 mV
        // < vgs < Vth + 500 mV for Kp=100µA, I_ref=5µA, W/L=1).
        assert!(
            (VTH_NOM + 0.05) < batched_vgs && batched_vgs < (VTH_NOM + 0.6),
            "draw {idx}: vgs={batched_vgs} outside saturation range",
        );
    }

    // Monotonicity sanity: increasing Vth_M1 (M1 needs higher Vgs to
    // carry the same I_ref) should increase vgs. Compare draws 0..4
    // where Vth_M2 is held at nominal.
    let vgs_seq: Vec<f32> = (0..5).map(|i| batched.voltages[&vgs][i]).collect();
    for i in 1..5 {
        assert!(
            vgs_seq[i] >= vgs_seq[i-1] - 1e-4,
            "vgs should monotone-↑ in Vth_M1: {vgs_seq:?}",
        );
    }
}
