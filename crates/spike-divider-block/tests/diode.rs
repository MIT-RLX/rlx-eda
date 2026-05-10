//! Diode block: layout under multiple PDKs + NonlinearDcBehavioral
//! (Resistor and Diode both expose terminal currents from terminal
//! voltages via the same trait).

use eda_hir::{Layout, NonlinearDcBehavioral};
use klayout_core::LayerInfo;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use spike_divider_block::pdks::{Gf180Lite, Sky130Lite};
use spike_divider_block::pdks_foundry::{HAS_GF180MCU, HAS_SKY130};
use spike_divider_block::*;

const VT: f32 = 0.025_852;

#[test]
fn diode_lays_out_under_all_pdks() {
    let d = Diode { size: 2_000, is_value: 1e-15, id: "D1".into() };

    // RcDemo: RES = (50, 0)
    let lib = RcDemo::new_library("d_rcdemo");
    let pdk = RcDemo::register(&lib);
    let id = d.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert_eq!(cell.ports().len(), 2);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(50, 0))).count() > 0);

    // Sky130Lite: poly = (66, 20)
    let lib = Sky130Lite::new_library("d_sky130");
    let pdk = Sky130Lite::register(&lib);
    let id = d.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(66, 20))).count() > 0);

    // Gf180Lite: poly2 = (30, 0)
    let lib = Gf180Lite::new_library("d_gf180");
    let pdk = Gf180Lite::register(&lib);
    let id = d.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(30, 0))).count() > 0);

    // Auto-generated foundry PDKs (when present)
    if HAS_SKY130 {
        use spike_divider_block::pdks_foundry::Sky130;
        let lib = Sky130::new_library("d_sky130_foundry");
        let pdk = Sky130::register(&lib);
        let id = d.layout(&lib, &pdk);
        let cell = lib.get(id);
        assert!(cell.shapes_on(lib.layer(LayerInfo::gds(66, 20))).count() > 0);
    }
    if HAS_GF180MCU {
        use spike_divider_block::pdks_foundry::Gf180mcu;
        let lib = Gf180mcu::new_library("d_gf180_foundry");
        let pdk = Gf180mcu::register(&lib);
        let id = d.layout(&lib, &pdk);
        let cell = lib.get(id);
        assert!(cell.shapes_on(lib.layer(LayerInfo::gds(30, 0))).count() > 0);
    }
}

#[test]
fn resistor_currents_obey_ohms_law_with_correct_signs() {
    // Build a tiny standalone graph: inputs v_a, v_b → outputs r.currents().
    let r = Resistor { length: 10_000, id: "Rtest".into() };
    let mut g = Graph::new("r_currents_eval");
    let s = Shape::new(&[1], DType::F32);
    let v_a = g.input("v_a", s.clone());
    let v_b = g.input("v_b", s);
    let cs = r.currents(&[v_a, v_b], &mut g);
    g.set_outputs(cs);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param(&<Resistor as eda_hir::Block>::name(&r), &[1_000.0_f32]);
    let outs = compiled.run(&[("v_a", &[1.0_f32][..]), ("v_b", &[0.0_f32][..])]);

    // (v_a - v_b)/R = 1V/1kΩ = 1 mA. Anode loses, cathode gains.
    let i_a = outs[0][0];
    let i_b = outs[1][0];
    assert!((i_a - (-1e-3)).abs() < 1e-6, "i_a = {i_a}, expected -1e-3");
    assert!((i_b -  1e-3).abs() < 1e-6, "i_b = {i_b}, expected +1e-3");
    assert!((i_a + i_b).abs() < 1e-9, "KCL violated: i_a + i_b = {}", i_a + i_b);
}

#[test]
fn diode_currents_obey_shockley_with_correct_signs() {
    let d = Diode { size: 2_000, is_value: 1e-15, id: "Dtest".into() };
    let mut g = Graph::new("d_currents_eval");
    let s = Shape::new(&[1], DType::F32);
    let v_a = g.input("v_a", s.clone());
    let v_b = g.input("v_b", s);
    let cs = d.currents(&[v_a, v_b], &mut g);
    g.set_outputs(cs);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let is_param_name = format!("{}_Is", <Diode as eda_hir::Block>::name(&d));
    compiled.set_param(&is_param_name, &[1e-15_f32]);

    // Forward bias V_ab = 0.6 V → I_D = 1e-15·(exp(0.6/Vt) − 1).
    let outs = compiled.run(&[("v_a", &[0.6_f32][..]), ("v_b", &[0.0_f32][..])]);
    let i_d_expected = 1e-15 * ((0.6_f32 / VT).exp() - 1.0);
    let i_a = outs[0][0];
    let i_b = outs[1][0];
    assert!((i_a + i_d_expected).abs() / i_d_expected.abs() < 1e-3,
        "forward i_a: {i_a}, expected {} (-I_D)", -i_d_expected);
    assert!((i_b - i_d_expected).abs() / i_d_expected.abs() < 1e-3,
        "forward i_b: {i_b}, expected {i_d_expected} (+I_D)");

    // Reverse bias V_ab = -0.5 V → I_D ≈ -Is (saturation at -1e-15 A).
    // Build a fresh graph (run() expects matching inputs).
    let outs = compiled.run(&[("v_a", &[0.0_f32][..]), ("v_b", &[0.5_f32][..])]);
    let i_d_rev = 1e-15 * ((-0.5_f32 / VT).exp() - 1.0); // ≈ -1e-15
    let i_a_rev = outs[0][0];
    assert!((i_a_rev - (-i_d_rev)).abs() < 1e-14,
        "reverse i_a: {i_a_rev}, expected {} (-I_D = +Is)", -i_d_rev);
}

#[test]
fn r_plus_d_residual_sums_to_zero_at_operating_point() {
    // Architectural integration test: build the residual for the
    // R+D circuit using the trait abstraction (Resistor::currents +
    // Diode::currents), evaluate at the known operating point from
    // spike-diode (Vmid ≈ 0.6845 V). The two contributions to the Vmid
    // node should sum to zero (KCL).

    let r = Resistor { length: 10_000, id: "Rmid".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "Dmid".into() };

    let mut g = Graph::new("rd_kcl_check");
    let s = Shape::new(&[1], DType::F32);
    let v_in = g.input("V",    s.clone());
    let vmid = g.input("Vmid", s.clone());
    let gnd  = g.add_node(
        rlx_ir::Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() },
        vec![], s.clone(),
    );

    // Resistor between V (terminal a) and Vmid (terminal b).
    let r_cs = r.currents(&[v_in, vmid], &mut g);
    // Diode between Vmid (anode) and gnd (cathode).
    let d_cs = d.currents(&[vmid, gnd], &mut g);

    // KCL at Vmid: r_cs[1] (current INTO Vmid from R) + d_cs[0] (current
    // INTO Vmid from D) must equal zero at the operating point.
    let kcl = g.binary(rlx_ir::op::BinaryOp::Add, r_cs[1], d_cs[0], s);
    g.set_outputs(vec![kcl]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param(&<Resistor as eda_hir::Block>::name(&r), &[1_000.0_f32]);
    let is_name = format!("{}_Is", <Diode as eda_hir::Block>::name(&d));
    compiled.set_param(&is_name, &[1e-15_f32]);

    // Operating point from spike-diode: Vmid ≈ 0.6844948 V at V=1, R=1k,
    // Is=1e-15. KCL residual should be ~0 there.
    let vmid_op = 0.684_494_8_f32;
    let outs = compiled.run(&[("V", &[1.0_f32][..]), ("Vmid", &[vmid_op][..])]);
    let residual = outs[0][0];
    // I_R and I_D are O(3e-4) at this operating point; KCL residual
    // should be small in absolute terms.
    assert!(residual.abs() < 1e-6,
        "KCL residual at OP = {residual:+.3e}, expected ~0");
}
