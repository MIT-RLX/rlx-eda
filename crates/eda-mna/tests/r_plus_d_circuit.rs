//! End-to-end test of the MNA assembler on the R+D circuit:
//!
//! ```text
//!     V_in ─[R]─ Vmid ─[D]─ GND
//! ```
//!
//! Build the circuit through the `Circuit` API (devices added by net,
//! not by hand-coded residual). `build_residual_graph` produces an rlx
//! graph that takes per-net voltages as inputs and outputs the KCL
//! residual at unknown nets. Evaluate at the operating point from
//! `spike-diode` (`Vmid = 0.6844948 V`) — assert the residual is
//! essentially zero.

use eda_mna::{build_residual_graph, net_input_name, Circuit, NetId};
use rlx_runtime::{Device, Session};
use spike_divider_block::{Diode, Resistor};

#[test]
fn assembler_residual_is_zero_at_known_operating_point() {
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();   // boundary — user supplies V at runtime
    let vmid = c.alloc_unknown_net();    // unknown — solver iterates

    let r = Resistor { length: 10_000, id: "Rmid".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "Dmid".into() };

    // Resistor terminals: a=v_in, b=vmid.
    c.add_device(r.clone(), &[v_in, vmid]);
    // Diode: a=vmid (anode), b=GND (cathode).
    c.add_device(d.clone(), &[vmid, NetId::GND]);

    let rg = build_residual_graph(&c);

    // Output structure:
    //   - one residual per unknown net (here: just `vmid`)
    assert_eq!(rg.unknown_nets, vec![vmid]);
    // All nets get an Op::Input named v_<id>:
    assert_eq!(rg.all_nets.len(), 2);

    let mut compiled = Session::new(Device::Cpu).compile(rg.graph);
    compiled.set_param(&eda_hir::Block::name(&r), &[1_000.0_f32]);
    let is_name = format!("{}_Is", eda_hir::Block::name(&d));
    compiled.set_param(&is_name, &[1e-15_f32]);

    // Evaluate at the operating point from spike-diode.
    let vmid_op = 0.684_494_8_f32;
    let v_in_val = 1.0_f32;
    let outs = compiled.run(&[
        (net_input_name(v_in).as_str(),  &[v_in_val][..]),
        (net_input_name(vmid).as_str(),  &[vmid_op][..]),
    ]);
    let residual = outs[0][0];

    assert!(residual.abs() < 1e-6,
        "KCL residual at OP = {residual:+.3e}, expected ~0");
}

#[test]
fn assembler_residual_is_nonzero_off_operating_point() {
    // Counter-test: residual at the wrong Vmid should NOT be zero.
    // Catches "trait wired up identically zero by accident."
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();
    let r = Resistor { length: 10_000, id: "R".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "D".into() };
    c.add_device(r.clone(), &[v_in, vmid]);
    c.add_device(d.clone(), &[vmid, NetId::GND]);

    let rg = build_residual_graph(&c);
    let mut compiled = Session::new(Device::Cpu).compile(rg.graph);
    compiled.set_param(&eda_hir::Block::name(&r), &[1_000.0_f32]);
    compiled.set_param(&format!("{}_Is", eda_hir::Block::name(&d)), &[1e-15_f32]);

    // 0.5 V is off the OP — residual should be sizeable (~3e-4 A scale).
    let outs = compiled.run(&[
        (net_input_name(v_in).as_str(), &[1.0_f32][..]),
        (net_input_name(vmid).as_str(), &[0.5_f32][..]),
    ]);
    assert!(outs[0][0].abs() > 1e-5,
        "residual at off-OP Vmid = 0.5 should be non-trivial; got {}", outs[0][0]);
}

#[test]
fn n_unknowns_excludes_boundary_and_ground() {
    let mut c = Circuit::new();
    let _b1 = c.alloc_boundary_net();
    let _u1 = c.alloc_unknown_net();
    let _u2 = c.alloc_unknown_net();
    let _b2 = c.alloc_boundary_net();
    let _u3 = c.alloc_unknown_net();
    assert_eq!(c.n_nets(),    5);
    assert_eq!(c.n_unknowns(), 3);
}
