//! Tier 1: trait dispatch produces a structurally-correct layout.

use eda_hir::{Block, Layout};
use spike_divider_block::*;

#[test]
fn divider_block_geometry_matches_free_function_version() {
    let (lib, _pdk, top) = make_divider_layout(10_000, 30_000);
    let cell = lib.get(top);

    // Three ports, named, on METAL1.
    assert_eq!(cell.ports().len(), 3);
    assert!(cell.port("vin").is_some());
    assert!(cell.port("vout").is_some());
    assert!(cell.port("gnd").is_some());

    // Two child instances + the routed wire shape.
    assert_eq!(cell.instances().len(), 2);

    // Bbox identical to the free-function spike.
    let b = cell.full_bbox(&lib);
    assert_eq!((b.min.x, b.min.y, b.max.x, b.max.y), (-1000, -3500, 46000, 1500));
}

#[test]
fn block_name_derives_from_parameters() {
    // Block::name() drives CellName + diagnostic identity. Different
    // params → different name.
    let r_short = Resistor { length: 1_000, id: "R".into() };
    let r_long  = Resistor { length: 9_999, id: "R".into() };
    assert_eq!(r_short.name(), "Resistor_R_L1000");
    assert_eq!(r_long.name(),  "Resistor_R_L9999");
    assert_ne!(r_short.name(), r_long.name());

    let div = RcDivider::new(r_short, r_long);
    assert_eq!(div.name(), "RcDivider_R_R");
}

#[test]
fn equal_blocks_are_eq_and_have_equal_hashes() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let a = Resistor { length: 10_000, id: "X".into() };
    let b = Resistor { length: 10_000, id: "X".into() };
    assert_eq!(a, b);

    let mut h_a = DefaultHasher::new(); a.hash(&mut h_a);
    let mut h_b = DefaultHasher::new(); b.hash(&mut h_b);
    assert_eq!(h_a.finish(), h_b.finish());
}

#[test]
fn resistor_layout_dispatch_inserts_two_ports() {
    let lib = RcDemo::new_library("rd");
    let pdk = RcDemo::register(&lib);
    let r = Resistor { length: 5_000, id: "test".into() };
    let id = r.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert_eq!(cell.ports().len(), 2);
}
