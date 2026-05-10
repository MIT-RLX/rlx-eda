//! Tier 3: write GDS via the trait-driven layout, read back, polygon
//! counts per layer match.

use klayout_geom::Region;
use klayout_io::{read_gds_bytes, write_gds_bytes};
use spike_divider_block::*;

#[test]
fn divider_block_gds_roundtrip_matches() {
    let (lib_a, pdk_a, top_a) = make_divider_layout(10_000, 30_000);

    let bytes = write_gds_bytes(&lib_a).expect("write");
    assert!(!bytes.is_empty());
    let lib_b = read_gds_bytes(&bytes).expect("read");

    // Look up the top cell by its Block::name() — derived deterministically
    // from the divider's resistor ids.
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let top_b = lib_b.by_name(&eda_hir::Block::name(&div))
        .expect("top cell name from Block::name should be present");

    // Layers in the read-back library are looked up by GDS pair (no PDK).
    let res_b    = lib_b.layer(klayout_core::LayerInfo::gds(50, 0));
    let metal1_b = lib_b.layer(klayout_core::LayerInfo::gds(10, 0));
    let via1_b   = lib_b.layer(klayout_core::LayerInfo::gds(20, 0));

    let counts = |lib, top, layer| Region::from_cell_layer(lib, top, layer).len();
    assert_eq!(counts(&lib_a, top_a, pdk_a.RES),    counts(&lib_b, top_b, res_b));
    assert_eq!(counts(&lib_a, top_a, pdk_a.METAL1), counts(&lib_b, top_b, metal1_b));
    assert_eq!(counts(&lib_a, top_a, pdk_a.VIA1),   counts(&lib_b, top_b, via1_b));
}
