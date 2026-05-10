//! Tier 3: klayout-geom `Region::from_cell_layer` flattens the hierarchy
//! into per-layer polygon sets. We use it as a "what landed where?" check
//! that doesn't trust shape-counting on the cell directly.
//!
//! Note: `Region` only collects `Shape::Polygon` and `Shape::Box`. Path
//! shapes (the routed wire) are intentionally not present — DRC and LVS
//! flows that need wires lifted to polygons run a stylize-as-polygon pass
//! first. We document this here as the expected behavior, not a bug.

use klayout_geom::{merge, Region};
use spike_divider_layout::*;

#[test]
fn flatten_per_layer_polygon_counts() {
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);

    let res    = Region::from_cell_layer(&lib, top, pdk.RES);
    let metal1 = Region::from_cell_layer(&lib, top, pdk.METAL1);
    let via1   = Region::from_cell_layer(&lib, top, pdk.VIA1);

    // 2 RES bodies (one per resistor instance).
    assert_eq!(res.len(), 2, "RES polygon count: {}", res.len());

    // 4 contact pads (2 per resistor × 2 resistors). The routed wire is a
    // Path shape and is not collected here — see the module doc above.
    assert_eq!(metal1.len(), 4, "METAL1 polygon count: {}", metal1.len());

    // 4 vias (mirrors METAL1).
    assert_eq!(via1.len(), 4, "VIA1 polygon count: {}", via1.len());

    // Bboxes should be non-empty.
    assert!(!res.bbox().is_empty());
    assert!(!metal1.bbox().is_empty());
    assert!(!via1.bbox().is_empty());
}

#[test]
fn merge_does_not_combine_disjoint_resistors() {
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);
    let res = Region::from_cell_layer(&lib, top, pdk.RES);
    let merged = merge(&res);
    // Two physically-separated bodies → still 2 polygons after merge.
    assert_eq!(merged.len(), 2, "merge result polygon count: {}", merged.len());
}
