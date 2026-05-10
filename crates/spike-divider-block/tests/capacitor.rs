//! Capacitor primitive — a second block type going through the same
//! `Layout<P: RcLikePdk>` plumbing as `Resistor`.

use eda_hir::Layout;
use klayout_core::LayerInfo;
use spike_divider_block::pdks::{Gf180Lite, Sky130Lite};
use spike_divider_block::*;

#[test]
fn capacitor_lays_out_under_each_pdk_with_pdk_specific_metal1() {
    let cap = Capacitor { plate_size: 5_000, id: "C1".into() };

    // RcDemo: METAL1 = (10, 0)
    let lib = RcDemo::new_library("cap_rcdemo");
    let pdk = RcDemo::register(&lib);
    let id = cap.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(10, 0))).count() > 0,
        "RcDemo METAL1 (10,0) empty");

    // Sky130Lite: MET1 = (68, 20)
    let lib = Sky130Lite::new_library("cap_sky130");
    let pdk = Sky130Lite::register(&lib);
    let id = cap.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(68, 20))).count() > 0,
        "Sky130 MET1 (68,20) empty");

    // Gf180Lite: METAL1 = (34, 0)
    let lib = Gf180Lite::new_library("cap_gf180");
    let pdk = Gf180Lite::register(&lib);
    let id = cap.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(34, 0))).count() > 0,
        "Gf180 METAL1 (34,0) empty");
}

#[test]
fn capacitor_has_two_ports_and_correct_bbox() {
    use klayout_core::Bbox;
    let lib = RcDemo::new_library("cap_test");
    let pdk = RcDemo::register(&lib);
    let cap = Capacitor { plate_size: 3_000, id: "test".into() };
    let id = cap.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert_eq!(cell.ports().len(), 2);
    assert_eq!(cell.local_bbox(), Bbox::new(
        klayout_core::Point::new(0, 0),
        klayout_core::Point::new(3_000, 3_000),
    ));
}
