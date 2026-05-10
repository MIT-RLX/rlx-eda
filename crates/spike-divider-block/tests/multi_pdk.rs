//! Cross-PDK genericity — same `Resistor` / `RcDivider` types lay out
//! under three different PDKs (`RcDemo`, `Sky130Lite`, `Gf180Lite`),
//! each producing geometry on the PDK's own GDS layer numbers.
//!
//! This is the architectural payoff of `Layout<P: RcLikePdk>`: write
//! the block once, retarget by swapping the PDK at the call site.
//!
//! Each test:
//! 1. Declares a fresh Library with the PDK's DBU.
//! 2. Lays out the canonical divider (R1=10 µm, R2=30 µm).
//! 3. Verifies every shape lands on the **expected GDS (layer, datatype)
//!    pair** for that PDK — the literal numbers from the foundry tech
//!    decks.

use eda_hir::Layout;
use klayout_core::{LayerInfo, Library, Shape};
use spike_divider_block::pdks::{Gf180Lite, Sky130Lite};
use spike_divider_block::*;

fn count_shapes_on(lib: &Library, top: klayout_core::CellId, info: LayerInfo) -> usize {
    let layer = lib.layer(info);
    let cell  = lib.get(top);
    cell.shapes_on(layer).count()
        + cell.instances().iter().map(|i| {
            lib.get(i.cell).shapes_on(layer).count()
        }).sum::<usize>()
}

#[test]
fn divider_lays_out_under_rc_demo_with_50_10_20_layers() {
    let lib = RcDemo::new_library("rcdemo_test");
    let pdk = RcDemo::register(&lib);
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let top = div.layout(&lib, &pdk);

    // RcDemo: RES=(50,0), METAL1=(10,0), VIA1=(20,0).
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(50, 0)) > 0, "RES (50,0) empty");
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(10, 0)) > 0, "METAL1 (10,0) empty");
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(20, 0)) > 0, "VIA1 (20,0) empty");
}

#[test]
fn divider_lays_out_under_sky130_with_66_68_layers() {
    let lib = Sky130Lite::new_library("sky130_test");
    let pdk = Sky130Lite::register(&lib);
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let top = div.layout(&lib, &pdk);

    // Sky130Lite: POLY=(66,20), MET1=(68,20), LICON1=(66,44).
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(66, 20)) > 0, "POLY (66,20) empty");
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(68, 20)) > 0, "MET1 (68,20) empty");
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(66, 44)) > 0, "LICON1 (66,44) empty");

    // And the RcDemo numbers should NOT appear — the same Resistor type
    // is fully retargeted, not bleeding RcDemo-numbered shapes.
    assert_eq!(count_shapes_on(&lib, top, LayerInfo::gds(50, 0)), 0,
        "stale RcDemo RES layer (50,0) leaked into Sky130 layout");
}

#[test]
fn divider_lays_out_under_gf180_with_30_34_33_layers() {
    let lib = Gf180Lite::new_library("gf180_test");
    let pdk = Gf180Lite::register(&lib);
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let top = div.layout(&lib, &pdk);

    // Gf180Lite: POLY2=(30,0), METAL1=(34,0), CONTACT=(33,0).
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(30, 0)) > 0, "POLY2 (30,0) empty");
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(34, 0)) > 0, "METAL1 (34,0) empty");
    assert!(count_shapes_on(&lib, top, LayerInfo::gds(33, 0)) > 0, "CONTACT (33,0) empty");
}

#[test]
fn same_block_value_yields_same_shape_count_across_pdks() {
    // Geometry is PDK-independent (same width / pad sizes / via sizes —
    // only the layer assignments change). Total shape count per layer
    // role should match across PDKs for the same divider parameters.
    let make = |id_label: &str| RcDivider::new(
        Resistor { length: 10_000, id: format!("R1_{id_label}") },
        Resistor { length: 30_000, id: format!("R2_{id_label}") },
    );

    let lib_a = RcDemo::new_library("a"); let pdk_a = RcDemo::register(&lib_a);
    let lib_b = Sky130Lite::new_library("b"); let pdk_b = Sky130Lite::register(&lib_b);
    let lib_c = Gf180Lite::new_library("c"); let pdk_c = Gf180Lite::register(&lib_c);
    let top_a = make("a").layout(&lib_a, &pdk_a);
    let top_b = make("b").layout(&lib_b, &pdk_b);
    let top_c = make("c").layout(&lib_c, &pdk_c);

    // Total polygon+box count (excluding paths since stylizer emits Boxes
    // for the wire) should match across PDKs.
    let count = |lib: &Library, top| {
        let cell = lib.get(top);
        let mut n = 0;
        for layer in cell.layers() {
            n += cell
                .shapes_on(layer)
                .filter(|s| matches!(s, Shape::Polygon(_) | Shape::Box(_)))
                .count();
        }
        for inst in cell.instances() {
            let child = lib.get(inst.cell);
            for layer in child.layers() {
                n += child
                    .shapes_on(layer)
                    .filter(|s| matches!(s, Shape::Polygon(_) | Shape::Box(_)))
                    .count();
            }
        }
        n
    };

    let na = count(&lib_a, top_a);
    let nb = count(&lib_b, top_b);
    let nc = count(&lib_c, top_c);
    assert_eq!(na, nb, "RcDemo vs Sky130Lite shape count: {na} vs {nb}");
    assert_eq!(nb, nc, "Sky130Lite vs Gf180Lite shape count: {nb} vs {nc}");
}
