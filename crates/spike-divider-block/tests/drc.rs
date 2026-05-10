//! DRC tier — exercise klayout_drc rules against the divider's geometry.
//!
//! We declare a small rule deck that captures intent for the spike PDK:
//!
//!   - RES min width:           1 µm
//!   - METAL1 min width:        0.5 µm
//!   - METAL1 min space:        0.4 µm
//!   - VIA1 min width:          0.5 µm
//!   - VIA1 enclosed by METAL1: 0.25 µm on every side
//!
//! All values in DBU (1 DBU = 1 nm here). Dimensions in the spike are
//! sized to pass these rules with healthy margin — every rule should
//! return an empty `Region`. A regression that violates any rule
//! (someone shrinks a pad, a router places wires too close) trips the
//! corresponding assert immediately.

use klayout_drc::{enclosing, separation, space, violations_from_region, width};
use klayout_geom::Region;
use spike_divider_block::*;

fn no_violations(rule: &str, region: &Region) {
    if !region.is_empty() {
        let v = violations_from_region(rule, region);
        panic!("[{rule}] {} violations: {:?}", v.len(), v);
    }
}

#[test]
fn divider_passes_drc() {
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);

    let res    = Region::from_cell_layer(&lib, top, pdk.RES);
    let metal1 = Region::from_cell_layer(&lib, top, pdk.METAL1);
    let via1   = Region::from_cell_layer(&lib, top, pdk.VIA1);

    // Minimum widths.
    no_violations("RES.W>=1um",    &width(&res,    1_000));
    no_violations("METAL1.W>=0.5um", &width(&metal1,  500));
    no_violations("VIA1.W>=0.5um", &width(&via1,    500));

    // METAL1 spacing: 0.4 µm. Wire fattening + pad geometry is
    // dimensioned so the merged METAL1 spacing is comfortable.
    no_violations("METAL1.S>=0.4um", &space(&metal1, 400));

    // RES↔METAL1 separation: RES extends only inside the resistor body
    // and METAL1 pads cap the ends; they overlap by design, so
    // `separation` (across-layer min spacing) is N/A here. We instead
    // assert RES↔VIA1 separation = 0 (they should not coexist; vias
    // sit on METAL1 pads only).
    no_violations("RES~VIA1>0", &separation(&res, &via1, 1));

    // VIA1 enclosed by METAL1 by ≥ 0.25 µm on every side.
    no_violations("VIA1 enc M1>=0.25um", &enclosing(&metal1, &via1, 250));
}

#[test]
fn drc_catches_a_too_narrow_metal_segment() {
    // Sanity-check that the DRC engine actually flags real violations:
    // build a Region with a single 100 nm-wide rectangle and run a
    // 500 nm width rule against it.
    use klayout_core::{Bbox, Point, Polygon};
    let narrow = Polygon::rect(Bbox::new(
        Point::new(0, 0), Point::new(10_000, 100),
    ));
    let r = Region::from_polygons(std::iter::once(narrow));
    let violations = width(&r, 500);
    assert!(!violations.is_empty(), "expected width violation; engine missed it");
}
