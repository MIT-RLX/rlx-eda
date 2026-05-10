//! End-to-end PNR smoke: two resistor cells + a wire between them.
//!
//! Exercises [`Netlist`] → [`ManualPlacer`] → [`ManhattanRouter`] →
//! [`PnrFlow::run`] on a Library + a tiny hand-built Pdk.
//! Verifies:
//!
//! 1. The flow produces a top cell.
//! 2. At least one box on the wire layer (the router actually emitted).
//! 3. The external `vin` / `vout` ports show up on the top cell.
//! 4. A `< 2`-pin net is reported in `failed_nets` rather than panicking.

use eda_pnr::{ManhattanRouter, ManualPlacer, Netlist, PnrFlow, WireStyle};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape, Trans, Vec2,
};
use klayout_pdk::pdk;

pdk! {
    pub TestPdk {
        dbu: 1000,
        layers: {
            RES    = (50, 0),
            METAL1 = (10, 0),
        },
        ports: { Electrical },
    }
}

fn build_resistor(lib: &Library, pdk: &TestPdk, name: &str, length_dbu: i64) -> CellId {
    let mut cb = CellBuilder::new(name);
    let rect = Rect::new(Bbox::new(
        Point::new(0, -500),
        Point::new(length_dbu, 500),
    ));
    cb.add_shape(pdk.RES, Shape::Box(rect));
    let kind = TestPdk::Electrical;
    cb.add_port(
        Port::new("a", pdk.METAL1, Point::new(0, 0), Angle90::W, 1_000)
            .with_kind(kind),
    );
    cb.add_port(
        Port::new("b", pdk.METAL1, Point::new(length_dbu, 0), Angle90::E, 1_000)
            .with_kind(kind),
    );
    lib.insert(cb)
}

#[test]
fn place_route_two_resistors_end_to_end() {
    let lib = TestPdk::new_library("pnr_smoke");
    let pdk = TestPdk::register(&lib);

    let r1 = build_resistor(&lib, &pdk, "R1", 5_000);
    let r2 = build_resistor(&lib, &pdk, "R2", 5_000);

    let mut nl = Netlist::new("pnr_top").with_default_signal_layer(pdk.METAL1);
    let i1 = nl.add_instance("R1", r1);
    let i2 = nl.add_instance("R2", r2);
    nl.connect("vmid", i1, "b");
    nl.connect("vmid", i2, "a");
    nl.expose("vin", "vin_net", Some(eda_hir::PinDirection::Input));
    nl.connect("vin_net", i1, "a");
    nl.expose("vout", "vout_net", Some(eda_hir::PinDirection::Output));
    nl.connect("vout_net", i2, "b");
    // Single-pin net to exercise the failed-net path.
    nl.connect("dangling", i1, "a");
    // Strip the second pin so it stays solo:
    if let Some(net) = nl.nets.iter_mut().find(|n| n.name == "dangling") {
        net.pins.truncate(1);
    }

    let placer = ManualPlacer::new(vec![
        Trans::translate(Vec2::new(0, 0)),
        Trans::translate(Vec2::new(10_000, 4_000)), // gap + offset to force a real bend
    ]);
    let router = ManhattanRouter::new(WireStyle::Polygon);
    let flow = PnrFlow::new(placer, router);

    let result = flow.run(&nl, &lib);
    let cell = lib.get(result.top);

    // Top cell instantiates both resistors.
    assert_eq!(cell.instances().len(), 2);

    // Top cell has at least one `Shape::Box` on METAL1 — the
    // router put down at least one polygonised wire segment.
    let has_metal1_wire = cell
        .shapes_on(pdk.METAL1)
        .any(|s| matches!(s, Shape::Box(_)));
    assert!(
        has_metal1_wire,
        "expected at least one Box on METAL1 from ManhattanRouter (Polygon style)",
    );

    // External pins promoted by the flow.
    assert!(cell.port("vin").is_some(), "vin external pin missing");
    assert!(cell.port("vout").is_some(), "vout external pin missing");

    // Single-pin net is reported as failed, not panicked-on.
    assert!(
        result.routed.failed_nets.iter().any(|n| n == "dangling"),
        "single-pin net should appear in failed_nets",
    );

    // Three nets declared total (vmid, vin_net, vout_net) plus
    // dangling = 4. Two routable (vmid + each external promotion's
    // 2-pin net are routable when they have ≥ 2 pins; here vin_net
    // and vout_net are 1-pin → fail). Just sanity-check the
    // failed-net count.
    assert!(
        result.routed.failed_nets.len() >= 1,
        "at least the dangling net should fail",
    );
}
