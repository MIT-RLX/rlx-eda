//! `optimize_to_target` end-to-end test: drive the R+D circuit's Vmid
//! to a chosen value by optimizing R via SGD on top of the framework's
//! solve_dc + sensitivities loop.

use std::collections::HashMap;
use eda_mna::{
    optimize_to_target, Circuit, NetId, NewtonOptions, OptimizeTargetOptions,
};
use spike_divider_block::{Diode, Resistor};

#[test]
fn optimize_r_to_hit_target_vmid() {
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();
    let r = Resistor { length: 5_000, id: "R".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "D".into() };
    c.add_device(r.clone(), &[v_in, vmid]);
    c.add_device(d.clone(), &[vmid, NetId::GND]);

    let r_name  = eda_hir::Block::name(&r);
    let is_name = format!("{}_Is", eda_hir::Block::name(&d));

    let mut params = HashMap::new();
    params.insert(r_name.clone(),  500.0_f32);     // initial R = 500 Ω
    params.insert(is_name,         1e-15_f32);
    let mut boundary = HashMap::new();
    boundary.insert(v_in, 1.0_f32);

    // Pick a Vmid target slightly above the natural diode forward drop
    // at R=500. The framework should walk R up until Vmid = 0.55 V.
    let target_vmid = 0.55_f32;

    let opt = OptimizeTargetOptions::default();

    let res = optimize_to_target(&c, &params, &boundary, vmid, target_vmid, &r_name, opt);

    println!("converged: {}, iters: {}", res.converged, res.iters);
    println!("final R = {} Ω", res.final_params[&r_name]);
    println!("final Vmid = {} V (target {})", res.final_v_target, target_vmid);

    assert!(res.converged,
        "optimizer didn't converge in {} iters; final loss = {:.3e}",
        res.iters, res.final_loss);
    assert!((res.final_v_target - target_vmid).abs() < 1e-3,
        "final Vmid {} not within tol of target {}", res.final_v_target, target_vmid);
    // R should have moved well off its 500 Ω starting point.
    let final_r = res.final_params[&r_name];
    assert!(final_r > 100.0, "final R = {final_r} Ω, suspiciously small");
}
