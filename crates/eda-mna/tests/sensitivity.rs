//! Sensitivity test: at the converged DC operating point of the R+D
//! circuit, `sensitivities(...)` returns `∂Vmid/∂R` and `∂Vmid/∂Is`.
//! These should match `spike-diode`'s analytic implicit-function-theorem
//! gradients to f32 precision.
//!
//! This is the architectural payoff of the framework: a circuit defined
//! topologically yields not just an operating point but **gradients of
//! operating-point voltages w.r.t. device parameters** — the ingredient
//! every inverse-design / autotuning workflow needs.

use std::collections::HashMap;
use eda_mna::{sensitivities, solve_dc, Circuit, NetId, NewtonOptions};
use spike_divider_block::{Diode, Resistor};

#[test]
fn r_plus_d_sensitivities_match_ift_analytic() {
    // Build R+D circuit topologically.
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();
    let r = Resistor { length: 10_000, id: "Rmid".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "Dmid".into() };
    c.add_device(r.clone(), &[v_in, vmid]);
    c.add_device(d.clone(), &[vmid, NetId::GND]);

    let r_name  = eda_hir::Block::name(&r);
    let is_name = format!("{}_Is", eda_hir::Block::name(&d));

    let r_val   = 1_000.0_f32;
    let is_val  = 1e-15_f32;
    let v_in_val = 1.0_f32;

    let mut params = HashMap::new();
    params.insert(r_name.clone(),  r_val);
    params.insert(is_name.clone(), is_val);
    let mut boundary = HashMap::new();
    boundary.insert(v_in, v_in_val);

    let op = solve_dc(&c, &params, &boundary, NewtonOptions::default());
    assert!(op.converged, "Newton did not converge");

    // Compute sensitivities through the framework.
    let sens = sensitivities(
        &c, &params, &boundary, &op,
        &[r_name.clone(), is_name.clone()],
    );

    // Reference: IFT analytic (from spike-diode's hand-derived closed form).
    let vmid_op = op.voltages[&vmid];
    let vt = spike_diode::VT;
    let analytic_dv_dr  = spike_diode::analytic_dvmid_dr (v_in_val, r_val, is_val, vt, vmid_op);
    let analytic_dv_dis = spike_diode::analytic_dvmid_dis(v_in_val, r_val, is_val, vt, vmid_op);

    // Each `sens[name]` is a Vec<f32> in unknown-net order. Here that's
    // just `[vmid]`, so index 0 is the relevant gradient.
    let dv_dr  = sens[&r_name][0];
    let dv_dis = sens[&is_name][0];

    println!("∂Vmid/∂R  framework: {dv_dr:+.6e}   analytic: {analytic_dv_dr:+.6e}");
    println!("∂Vmid/∂Is framework: {dv_dis:+.6e}   analytic: {analytic_dv_dis:+.6e}");

    let rel_err_r  = (dv_dr  - analytic_dv_dr ).abs() / analytic_dv_dr.abs();
    let rel_err_is = (dv_dis - analytic_dv_dis).abs() / analytic_dv_dis.abs();
    assert!(rel_err_r  < 1e-3, "∂Vmid/∂R rel err = {rel_err_r:.2e}");
    assert!(rel_err_is < 1e-3, "∂Vmid/∂Is rel err = {rel_err_is:.2e}");
}

#[test]
fn linear_divider_sensitivities_match_closed_form() {
    // R+R divider: V_in=1V, R1=1kΩ, R2=3kΩ → Vmid = 0.75 V.
    // ∂Vmid/∂R1 = -V_in · R2 / (R1+R2)² = -1·3000/16e6 = -1.875e-4
    // ∂Vmid/∂R2 = +V_in · R1 / (R1+R2)² = +1·1000/16e6 = +6.25e-5
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();
    let r1 = Resistor { length: 10_000, id: "R1".into() };
    let r2 = Resistor { length: 30_000, id: "R2".into() };
    c.add_device(r1.clone(), &[v_in, vmid]);
    c.add_device(r2.clone(), &[vmid, NetId::GND]);

    let n1 = eda_hir::Block::name(&r1);
    let n2 = eda_hir::Block::name(&r2);

    let mut params = HashMap::new();
    params.insert(n1.clone(), 1_000.0_f32);
    params.insert(n2.clone(), 3_000.0_f32);
    let mut boundary = HashMap::new();
    boundary.insert(v_in, 1.0_f32);

    let op = solve_dc(&c, &params, &boundary, NewtonOptions::default());
    assert!(op.converged);

    let sens = sensitivities(
        &c, &params, &boundary, &op,
        &[n1.clone(), n2.clone()],
    );

    let dv_dr1 = sens[&n1][0];
    let dv_dr2 = sens[&n2][0];

    let exp_dr1 = -1.0_f32 * 3000.0 / 16e6;   // = -1.875e-4
    let exp_dr2 =  1.0_f32 * 1000.0 / 16e6;   // = +6.25e-5

    assert!((dv_dr1 - exp_dr1).abs() < 1e-7, "∂R1: got {dv_dr1}, expected {exp_dr1}");
    assert!((dv_dr2 - exp_dr2).abs() < 1e-7, "∂R2: got {dv_dr2}, expected {exp_dr2}");
}
