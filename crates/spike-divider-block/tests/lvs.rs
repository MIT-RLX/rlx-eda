//! LVS tier — extract connectivity from the laid-out divider and assert
//! the net structure matches the intended schematic.
//!
//! ## Choice of conductor set
//!
//! `klayout_connect::extract_hierarchical` is geometric-connectivity-only:
//! shapes overlapping on a conductor layer (or stitched via a Via) join
//! into one net. There is no notion yet of a *device* boundary — RES is
//! one continuous strip per resistor, so if we declared RES a conductor
//! the body would short its own two ports together and the whole divider
//! would collapse to a single net.
//!
//! For a primitive-resistor LVS the canonical move is: declare only the
//! interconnect layer as the conductor (here METAL1), and let the
//! resistor primitive be modelled at the schematic level as a 2-pin
//! device. That gives us the three nets we expect:
//!
//!   - **vin**  — R1's left METAL1 pad (isolated)
//!   - **vout** — R1's right pad ⨄ wire-seg-1 ⨄ wire-seg-2 ⨄ R2's left pad
//!                (joined by the routed wire's polygon segments)
//!   - **gnd**  — R2's right METAL1 pad (isolated)

use klayout_connect::{extract_hierarchical, Conductor, ExtractConfig};
use spike_divider_block::*;

#[test]
fn divider_extracts_three_nets() {
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);

    let cfg = ExtractConfig {
        conductors: vec![Conductor {
            layer: pdk.METAL1,
            // No Text labels in the spike's geometry — labels would
            // need to come from a separate "label" layer. With no
            // labels, the extractor auto-names nets `net_0`, `net_1`,
            // `net_2`. We verify by content / bbox below.
            label_layer: pdk.METAL1,
        }],
        vias: vec![],
    };

    let nl = extract_hierarchical(&lib, top, &cfg);
    assert_eq!(nl.nets().len(), 3,
        "expected 3 METAL1 nets (vin, vout, gnd); got {}: {:?}",
        nl.nets().len(),
        nl.nets().iter().map(|n| (n.name.as_str(), n.bbox)).collect::<Vec<_>>(),
    );
}

#[test]
fn each_top_port_lands_inside_a_net() {
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);
    let top_cell = lib.get(top);
    let nl = extract_hierarchical(&lib, top, &ExtractConfig {
        conductors: vec![Conductor { layer: pdk.METAL1, label_layer: pdk.METAL1 }],
        vias: vec![],
    });

    // Every top-level port (vin/vout/gnd) must lie within exactly one net's
    // bounding box. This is the structural assertion that connects "what
    // the block claims its terminals are" to "what the layout extracted."
    for port in top_cell.ports() {
        let p = port.center;
        let containing: Vec<&klayout_connect::Net> = nl
            .nets()
            .iter()
            .filter(|n|
                n.bbox.min.x <= p.x && p.x <= n.bbox.max.x &&
                n.bbox.min.y <= p.y && p.y <= n.bbox.max.y
            )
            .collect();
        assert_eq!(
            containing.len(), 1,
            "port {} at ({}, {}) lies in {} nets (expected exactly 1)",
            port.name, p.x, p.y, containing.len(),
        );
    }
}

#[test]
fn vout_net_spans_routed_wire_extent() {
    // Structural marker: the middle net (vout) is the joined L-shape that
    // includes the routed wire, so its bbox spans both the elbow and both
    // resistor's inner pads. The endpoint nets (vin / gnd) are single
    // pads — small bboxes. Compare areas directly.
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);
    let nl = extract_hierarchical(&lib, top, &ExtractConfig {
        conductors: vec![Conductor { layer: pdk.METAL1, label_layer: pdk.METAL1 }],
        vias: vec![],
    });

    let mut areas: Vec<i128> = nl
        .nets()
        .iter()
        .map(|n| (n.bbox.max.x - n.bbox.min.x) as i128 * (n.bbox.max.y - n.bbox.min.y) as i128)
        .collect();
    areas.sort();
    // Endpoint pads: 2 µm × 2 µm = 4_000_000 DBU². Joined-L net: spans
    // ~5 µm × ~4 µm minimum (much larger). Order-of-magnitude assertion.
    assert!(areas[0] <= 6_000_000, "smallest net bbox too big — pads not isolated?");
    assert!(areas[1] <= 6_000_000, "second-smallest net bbox too big");
    assert!(areas[2] >= 10_000_000,
        "largest net bbox should span the routed wire; got {} DBU²", areas[2]);
}
