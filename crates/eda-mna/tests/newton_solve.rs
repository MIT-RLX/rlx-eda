//! End-to-end Newton solver test on the R+D circuit.
//!
//! Define the circuit topologically — `Resistor` between V_in and Vmid,
//! `Diode` between Vmid and GND. Call `solve_dc`. Assert converged Vmid
//! matches the operating point from `spike-diode` (a hand-coded Newton
//! on the same equations).

use std::collections::HashMap;
use eda_mna::{solve_dc, Circuit, NetId, NewtonOptions};
use spike_divider_block::{Diode, Resistor};

#[test]
fn newton_solves_r_plus_d_circuit() {
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();

    let r = Resistor { length: 10_000, id: "Rmid".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "Dmid".into() };
    c.add_device(r.clone(), &[v_in, vmid]);
    c.add_device(d.clone(), &[vmid, NetId::GND]);

    // Param values keyed by Block::name() (Resistor) or
    // <Block::name>_Is (Diode), matching the trait impls.
    let mut params: HashMap<String, f32> = HashMap::new();
    params.insert(eda_hir::Block::name(&r), 1_000.0);
    params.insert(format!("{}_Is", eda_hir::Block::name(&d)), 1e-15);

    let mut boundary: HashMap<NetId, f32> = HashMap::new();
    boundary.insert(v_in, 1.0);

    let op = solve_dc(&c, &params, &boundary, NewtonOptions::default());

    assert!(op.converged,
        "Newton did not converge: residual_max = {:.3e}, iters = {}",
        op.final_residual_max, op.iters);

    // Vmid should match the spike-diode operating point ≈ 0.6845 V.
    let vmid_solved = op.voltages[&vmid];
    let vmid_expected = 0.684_494_8_f32;
    assert!((vmid_solved - vmid_expected).abs() < 1e-3,
        "Vmid solved = {vmid_solved}, expected ≈ {vmid_expected}");

    // V_in should be reflected back in the result map (identity
    // for boundary nets).
    assert!((op.voltages[&v_in] - 1.0).abs() < 1e-9);

    // Should converge fast — the system is 1×1 and well-conditioned.
    assert!(op.iters < 20, "Newton took {} iters; expected < 20", op.iters);
}

#[test]
fn newton_returns_diagnostics_when_not_converged() {
    // Force non-convergence by capping max_iters at 1.
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();
    let r = Resistor { length: 10_000, id: "R".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "D".into() };
    c.add_device(r.clone(), &[v_in, vmid]);
    c.add_device(d.clone(), &[vmid, NetId::GND]);

    let mut params: HashMap<String, f32> = HashMap::new();
    params.insert(eda_hir::Block::name(&r), 1_000.0);
    params.insert(format!("{}_Is", eda_hir::Block::name(&d)), 1e-15);
    let mut boundary = HashMap::new();
    boundary.insert(v_in, 1.0);

    let opt = NewtonOptions { max_iters: 1, tol: 1e-12, vntol: 1e-6, init: 0.6, max_backtracks: 0 };
    let op = solve_dc(&c, &params, &boundary, opt);
    assert!(!op.converged);
    assert!(op.final_residual_max.is_finite());
    assert_eq!(op.iters, 1);
}

#[test]
fn newton_handles_two_resistor_divider() {
    // No diodes — pure linear system. R1 between V_in and Vmid,
    // R2 between Vmid and GND. Expected Vmid = V_in · R2/(R1+R2).
    // Trivial Newton convergence (one step, since linear).
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();
    let r1 = Resistor { length: 10_000, id: "R1".into() };  // R = 1000 Ω
    let r2 = Resistor { length: 30_000, id: "R2".into() };  // R = 3000 Ω
    c.add_device(r1.clone(), &[v_in, vmid]);
    c.add_device(r2.clone(), &[vmid, NetId::GND]);

    let mut params: HashMap<String, f32> = HashMap::new();
    params.insert(eda_hir::Block::name(&r1), 1_000.0);
    params.insert(eda_hir::Block::name(&r2), 3_000.0);
    let mut boundary = HashMap::new();
    boundary.insert(v_in, 1.0);

    let op = solve_dc(&c, &params, &boundary, NewtonOptions::default());
    assert!(op.converged);

    // Vmid = 1 · 3000 / 4000 = 0.75
    let vmid_solved = op.voltages[&vmid];
    assert!((vmid_solved - 0.75).abs() < 1e-5,
        "Vmid = {vmid_solved}, expected 0.75");
}
