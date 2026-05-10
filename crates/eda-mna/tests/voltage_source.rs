//! Voltage source as a real MNA device — replace boundary nets with
//! a `VoltageSource` device that contributes a branch-current unknown
//! plus an algebraic constraint.
//!
//! Circuit:
//!   `gnd ──[VS=1V]── vplus ──[R1]── vmid ──[R2]── gnd`
//!
//! All three internal nets (`vplus`, `vmid`, `gnd`-as-NetId-ref) are
//! framework-managed. The voltage source's branch current `i_VS` is
//! also a Newton unknown.
//!
//! Expected: `vplus = 1 V`, `vmid = 0.75 V` (linear divider with
//! R1=1k, R2=3k), `i_VS = 1V/4kΩ = 0.25 mA`.

use std::collections::HashMap;
use eda_mna::{solve_dc, Circuit, NetId, NewtonOptions};
use spike_divider_block::{Resistor, VoltageSource};

#[test]
fn divider_with_explicit_voltage_source() {
    let mut c = Circuit::new();
    let vplus = c.alloc_unknown_net();    // VS+ — solver finds this
    let vmid  = c.alloc_unknown_net();
    let r1 = Resistor { length: 10_000, id: "R1".into() };
    let r2 = Resistor { length: 30_000, id: "R2".into() };
    let vs = VoltageSource::from_volts(1.0, "VS");

    c.add_mna_device(vs.clone(), &[vplus, NetId::GND]);
    c.add_device(r1.clone(),     &[vplus, vmid]);
    c.add_device(r2.clone(),     &[vmid, NetId::GND]);

    // Sanity on the Circuit's bookkeeping: 2 unknown nets + 1 branch = 3 unknowns.
    assert_eq!(c.n_unknowns(), 3);

    let mut params = HashMap::new();
    params.insert(eda_hir::Block::name(&r1), 1_000.0_f32);
    params.insert(eda_hir::Block::name(&r2), 3_000.0_f32);
    let boundary: HashMap<NetId, f32> = HashMap::new();    // no boundary nets!

    let op = solve_dc(&c, &params, &boundary, NewtonOptions::default());
    assert!(op.converged, "Newton failed: residual_max = {:.3e}", op.final_residual_max);

    // vplus drops the full source → 1.0 V.
    assert!((op.voltages[&vplus] - 1.0).abs() < 1e-5,
        "vplus = {}, expected 1.0", op.voltages[&vplus]);
    // vmid = 1·R2/(R1+R2) = 0.75.
    assert!((op.voltages[&vmid] - 0.75).abs() < 1e-5,
        "vmid = {}, expected 0.75", op.voltages[&vmid]);

    // Branch current = (vplus - 0)/(R1+R2) = 0.25 mA. The framework
    // returns it through op.branch_currents. Get its BranchId — there's
    // only one in this circuit, and we can find it as the only entry.
    let (_, i_vs) = op.branch_currents.iter().next().expect("one branch unknown");
    let expected_i = 1.0 / 4_000.0;
    assert!((i_vs - expected_i).abs() < 1e-7,
        "i_VS = {} A, expected {}", i_vs, expected_i);
}
