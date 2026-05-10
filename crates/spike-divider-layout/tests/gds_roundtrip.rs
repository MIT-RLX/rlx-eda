//! Tier 4: GDS write → read structural roundtrip.
//!
//! Writes the divider layout to in-memory GDS bytes via klayout-io, reads
//! them back into a fresh `Library`, then checks layer-by-layer polygon
//! counts using `Region::from_cell_layer`.

use klayout_geom::Region;
use klayout_io::{read_gds_bytes, write_gds_bytes};
use spike_divider_layout::*;

#[test]
fn divider_gds_roundtrip_preserves_polygon_counts() {
    let (lib_a, pdk_a, _top_a) = make_divider_layout(10_000, 30_000);

    // Round-trip through GDS bytes.
    let bytes = write_gds_bytes(&lib_a).expect("write_gds_bytes");
    assert!(!bytes.is_empty());
    let lib_b = read_gds_bytes(&bytes).expect("read_gds_bytes");

    // After roundtrip we don't get the typed PDK back; we look up layers by
    // their GDS (layer, datatype) pairs. The values here mirror the
    // declarations inside the `pdk!` invocation.
    let res_b    = lib_b.layer(klayout_core::LayerInfo::gds(50, 0));
    let metal1_b = lib_b.layer(klayout_core::LayerInfo::gds(10, 0));
    let via1_b   = lib_b.layer(klayout_core::LayerInfo::gds(20, 0));

    // GDS has no notion of a "top cell" beyond the named cells; we look up
    // ours by name.
    let top_b = lib_b.by_name("divider")
        .expect("read library should contain `divider` cell");

    let count_per_layer = |lib, top, layer| Region::from_cell_layer(lib, top, layer).len();

    let res_a    = count_per_layer(&lib_a, _top_a, pdk_a.RES);
    let metal1_a = count_per_layer(&lib_a, _top_a, pdk_a.METAL1);
    let via1_a   = count_per_layer(&lib_a, _top_a, pdk_a.VIA1);

    let res_b_n    = count_per_layer(&lib_b, top_b, res_b);
    let metal1_b_n = count_per_layer(&lib_b, top_b, metal1_b);
    let via1_b_n   = count_per_layer(&lib_b, top_b, via1_b);

    assert_eq!((res_a, metal1_a, via1_a), (res_b_n, metal1_b_n, via1_b_n),
        "polygon counts diverged across GDS roundtrip: \
         pre=(RES={res_a}, METAL1={metal1_a}, VIA1={via1_a}) \
         post=(RES={res_b_n}, METAL1={metal1_b_n}, VIA1={via1_b_n})");
}
