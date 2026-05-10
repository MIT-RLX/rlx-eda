//! Tier 2: per-layer flatten via klayout-geom matches the free-function
//! spike's expected counts (2 RES bodies, 4 contact pads, 4 vias).

use klayout_geom::Region;
use spike_divider_block::*;

#[test]
fn flatten_per_layer_polygon_counts() {
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);

    let res    = Region::from_cell_layer(&lib, top, pdk.RES);
    let metal1 = Region::from_cell_layer(&lib, top, pdk.METAL1);
    let via1   = Region::from_cell_layer(&lib, top, pdk.VIA1);

    assert_eq!(res.len(),    2, "RES bodies");
    // METAL1 = 4 contact pads + 2 wire segments (PolygonWireStylizer
    // emits one Box per Path segment; ManhattanPlanner gave us an L-bend
    // → 2 segments).
    assert_eq!(metal1.len(), 6, "METAL1 pads (4) + wire segments (2)");
    assert_eq!(via1.len(),   4, "VIA1 squares");
}

#[test]
fn merge_collapses_overlapping_metal1_into_three_polygons() {
    use klayout_geom::merge;
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);
    let metal1 = Region::from_cell_layer(&lib, top, pdk.METAL1);
    let merged = merge(&metal1);
    // After merge: R1.left pad (isolated); R1.right + seg1 + seg2 + R2.left
    // (joined L-shape over the routed wire); R2.right pad (isolated). → 3.
    assert_eq!(merged.len(), 3, "merged METAL1: 2 isolated pads + 1 wire-joined L");
}
