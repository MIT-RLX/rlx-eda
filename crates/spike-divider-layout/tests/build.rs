//! Tier 1: builds-without-panic + structural sanity (counts, bbox, ports).

use spike_divider_layout::*;

#[test]
fn divider_builds_with_expected_geometry() {
    let r1_len = 10_000_i64;
    let r2_len = 30_000_i64;
    let (lib, _pdk, top) = make_divider_layout(r1_len, r2_len);

    let cell = lib.get(top);

    // Three top-level ports — vin / vout / gnd.
    assert_eq!(cell.ports().len(), 3);
    assert!(cell.port("vin").is_some());
    assert!(cell.port("vout").is_some());
    assert!(cell.port("gnd").is_some());

    // Two resistor instances + the routed wire shape on METAL1.
    assert_eq!(cell.instances().len(), 2);

    // Hierarchical bbox spans from the leftmost pad of R1 to the rightmost
    // pad of R2 (with the y-drop). With R2 translated by (r1_len+5000, -3000)
    // and a pad of 2 µm, expect:
    //   x_min = -PAD/2          = -1000
    //   x_max = r1_len + 5000 + r2_len + PAD/2  = 10000+5000+30000+1000 = 46000
    //   y_min = -3000 - PAD/2   = -3500          (R2's bottom pad edge)
    //   y_max = +PAD/2          = +1500          (R1's top pad edge)
    let b = cell.full_bbox(&lib);
    assert_eq!((b.min.x, b.min.y, b.max.x, b.max.y), (-1000, -3500, 46000, 1500),
        "unexpected hierarchical bbox: {:?}", b);
}

#[test]
fn resistor_cell_has_two_ports() {
    let lib = RcDemo::new_library("test");
    let pdk = RcDemo::register(&lib);
    let r = build_resistor_cell(&lib, &pdk, 10_000, "Rprimitive");
    let cell = lib.get(r);
    assert_eq!(cell.ports().len(), 2);
    assert!(cell.port("a").is_some());
    assert!(cell.port("b").is_some());
}

#[test]
fn pdk_layer_indices_are_distinct() {
    let lib = RcDemo::new_library("test");
    let pdk = RcDemo::register(&lib);
    // Three layers were declared — each must come back as a unique LayerIndex.
    assert_ne!(pdk.RES, pdk.METAL1);
    assert_ne!(pdk.METAL1, pdk.VIA1);
    assert_ne!(pdk.RES, pdk.VIA1);
    // And they must have the GDS pairs we declared.
    assert_eq!(lib.layer_info(pdk.RES).layer, 50);
    assert_eq!(lib.layer_info(pdk.METAL1).layer, 10);
    assert_eq!(lib.layer_info(pdk.VIA1).layer, 20);
}
