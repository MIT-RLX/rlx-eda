//! Voltage divider laid out as physical geometry via klayout-rs.
//!
//! Where the rlx spikes proved that the **MIR** half (residual + autodiff)
//! works on the divider, this spike proves that the **LIR** half — typed
//! PDK + cells + ports + routing + GDS export — works on the same circuit.
//!
//! ## What we exercise from klayout-rs
//!
//! - `klayout_pdk::pdk!`  — declarative PDK macro: layers + port-kind ids.
//! - `klayout_core::{Library, CellBuilder, Port, Rect, Bbox, Trans, ...}`
//!                         — frozen-cell layout DB.
//! - `klayout_geom::Region` — flatten cell-on-layer to a deterministic
//!                            polygon set; basis for DRC + LVS extraction.
//! - `klayout_route::{ManhattanPlanner, WirePathStylizer}` — Planner →
//!                    Stylizer pattern that produces a routed `Shape`.
//! - `klayout_io::write_gds_path`           — final GDS export.
//!
//! ## The geometry
//!
//! Two rectangular resistors (R1, R2) on the `RES` layer, each with two
//! `METAL1` contact pads + a `VIA1` square per pad. R2 sits to the right
//! and below R1 so the inter-resistor route bends. A single `METAL1`
//! `Path` connects R1's right port to R2's left port. Top cell exposes
//! `vin`, `vout`, `gnd` ports.
//!
//! All coordinates are i64 DBU (`dbu = 1000` → 1 nm/DBU).

use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Instance, Library, Point, Port, Rect, Trans, Vec2,
};
use klayout_pdk::pdk;
use klayout_route::{ManhattanPlanner, Obstacles, Planner, Stylizer, WirePathStylizer};

pdk! {
    pub RcDemo {
        dbu: 1000,
        layers: {
            RES    = (50, 0),
            METAL1 = (10, 0),
            VIA1   = (20, 0),
        },
        ports: { Electrical },
    }
}

// Geometry constants — DBU (1000 DBU = 1 µm).
const RES_WIDTH: i64 = 1_000;     // 1 µm body
const PAD: i64       = 2_000;     // 2 µm contact-pad square
const VIA: i64       = 500;       // 0.5 µm via square

/// Build a single rectangular-resistor primitive cell of the given body
/// length. Pads sit at both ends, ports `a` (W-facing) and `b` (E-facing).
pub fn build_resistor_cell(lib: &Library, pdk: &RcDemo, length: i64, name: &str) -> CellId {
    let mut b = CellBuilder::new(name);

    // RES body — `(0, 0)..(length, RES_WIDTH)`.
    b.add_shape(pdk.RES, Rect::new(Bbox::new(
        Point::new(0, 0), Point::new(length, RES_WIDTH),
    )));

    // Contact pad + via at each end. Pads centered vertically on the body
    // midline; vias inset within the pads.
    let cy        = RES_WIDTH / 2;
    let half_pad  = PAD / 2;
    let half_via  = VIA / 2;
    for &x in &[0_i64, length] {
        b.add_shape(pdk.METAL1, Rect::new(Bbox::new(
            Point::new(x - half_pad, cy - half_pad),
            Point::new(x + half_pad, cy + half_pad),
        )));
        b.add_shape(pdk.VIA1, Rect::new(Bbox::new(
            Point::new(x - half_via, cy - half_via),
            Point::new(x + half_via, cy + half_via),
        )));
    }

    // Routing ports on METAL1, one DBU outside each pad's outer edge so
    // the planner emits METAL1 wires that overlap the pad.
    b.add_port(
        Port::new("a", pdk.METAL1, Point::new(-half_pad, cy), Angle90::W, PAD)
            .with_kind(RcDemo::Electrical),
    );
    b.add_port(
        Port::new("b", pdk.METAL1, Point::new(length + half_pad, cy), Angle90::E, PAD)
            .with_kind(RcDemo::Electrical),
    );

    lib.insert(b)
}

/// Build the divider top cell: two resistor instances + a routed METAL1
/// wire between them. Returns the top cell id.
///
/// `r1_len`, `r2_len` set the resistor body lengths (DBU). The R1↔R2 ratio
/// determines the divider ratio at the schematic level; geometry-wise it's
/// just length on the RES layer.
pub fn build_divider(lib: &Library, pdk: &RcDemo, r1_len: i64, r2_len: i64) -> CellId {
    let r1 = build_resistor_cell(lib, pdk, r1_len, "R1");
    let r2 = build_resistor_cell(lib, pdk, r2_len, "R2");

    // Place R1 at origin; R2 to the right with a deliberate vertical drop
    // so the ManhattanPlanner emits an L-bend (not a degenerate straight line).
    let t1 = Trans::IDENTITY;
    let r1_to_r2_dx: i64 = 5_000;   // 5 µm horizontal gap between pads
    let r1_to_r2_dy: i64 = -3_000;  // 3 µm drop
    let t2 = Trans::translate(Vec2::new(r1_len + r1_to_r2_dx, r1_to_r2_dy));

    // Resolve absolute-coordinate ports for the inter-resistor route.
    let r1_cell = lib.get(r1);
    let r2_cell = lib.get(r2);
    let r1_b_abs = r1_cell.port("b").expect("R1.b").transform(t1);
    let r2_a_abs = r2_cell.port("a").expect("R2.a").transform(t2);

    // Plan + stylize — ManhattanPlanner gives a centerline `Path`, then
    // WirePathStylizer wraps it as a `Shape::Path` on METAL1.
    let path = ManhattanPlanner.plan(&r1_b_abs, &r2_a_abs, &Obstacles::default());
    let stylized = WirePathStylizer.stylize(pdk.METAL1, path);

    // Top cell: instances + routed shapes + exposed ports.
    let mut top = CellBuilder::new("divider");
    top.add_instance(Instance::new(r1, t1));
    top.add_instance(Instance::new(r2, t2));
    for (layer, shape) in stylized {
        top.add_shape(layer, shape);
    }

    // Top-level ports map to the divider's three external nets:
    //   vin  ↔ R1.a   (input)
    //   vout ↔ R1.b   (mid-net, electrically equal to R2.a)
    //   gnd  ↔ R2.b   (return)
    let vin  = r1_cell.port("a").expect("R1.a").transform(t1);
    let gnd  = r2_cell.port("b").expect("R2.b").transform(t2);
    top.add_port(
        Port::new("vin", pdk.METAL1, vin.center, vin.angle, vin.width)
            .with_kind(RcDemo::Electrical),
    );
    top.add_port(
        Port::new("vout", pdk.METAL1, r1_b_abs.center, r1_b_abs.angle, r1_b_abs.width)
            .with_kind(RcDemo::Electrical),
    );
    top.add_port(
        Port::new("gnd", pdk.METAL1, gnd.center, gnd.angle, gnd.width)
            .with_kind(RcDemo::Electrical),
    );

    lib.insert(top)
}

/// Convenience: build a fresh PDK + Library + divider in one call.
/// Returns `(library, pdk, top_cell_id)`.
pub fn make_divider_layout(r1_len: i64, r2_len: i64) -> (Library, RcDemo, CellId) {
    let lib = RcDemo::new_library("rc_demo");
    let pdk = RcDemo::register(&lib);
    let top = build_divider(&lib, &pdk, r1_len, r2_len);
    (lib, pdk, top)
}
