//! `TransientStorage` trait — Capacitor exposes its capacitance as an
//! rlx Param. The MVP guarantee we need: each capacitor instance gets
//! a distinct, named, settable Param keyed `<Block::name>_C`. Once that
//! holds, the BE-step assembler can pin each instance's stamp from a
//! flat `params: HashMap<String, f32>` exactly like resistors do today.
//!
//! Two checks:
//!
//! 1. Single instance — value flows through compile/set_param/run.
//! 2. Two distinct instances — keys don't collide (different `id`s
//!    yield different param names, both setting independently).

use eda_hir::{Block, TransientStorage};
use rlx_ir::Graph;
use rlx_runtime::{Device, Session};
use spike_divider_block::Capacitor;

#[test]
fn capacitor_capacitance_param_round_trips_through_rlx() {
    let cap = Capacitor { plate_size: 5_000, id: "C1".into() };
    let mut g = Graph::new("cap_param_test");
    let c_node = cap.capacitance(&mut g);
    g.set_outputs(vec![c_node]);

    let key = format!("{}_C", <Capacitor as Block>::name(&cap));
    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param(&key, &[1.5e-12]);
    let outs = compiled.run(&[]);
    assert!((outs[0][0] - 1.5e-12).abs() < 1e-18,
        "capacitance round-trip: got {}, expected 1.5e-12", outs[0][0]);
}

#[test]
fn two_capacitors_have_independent_param_keys() {
    let c1 = Capacitor { plate_size: 5_000, id: "C1".into() };
    let c2 = Capacitor { plate_size: 5_000, id: "C2".into() };

    let mut g = Graph::new("two_caps");
    let n1 = c1.capacitance(&mut g);
    let n2 = c2.capacitance(&mut g);
    g.set_outputs(vec![n1, n2]);

    let k1 = format!("{}_C", <Capacitor as Block>::name(&c1));
    let k2 = format!("{}_C", <Capacitor as Block>::name(&c2));
    assert_ne!(k1, k2, "instance keys must be distinct");

    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param(&k1, &[1.0e-12]);
    compiled.set_param(&k2, &[3.3e-9]);
    let outs = compiled.run(&[]);
    assert!((outs[0][0] - 1.0e-12).abs() < 1e-18, "C1 leaked: got {}", outs[0][0]);
    assert!((outs[1][0] - 3.3e-9 ).abs() < 1e-15, "C2 leaked: got {}", outs[1][0]);
}
