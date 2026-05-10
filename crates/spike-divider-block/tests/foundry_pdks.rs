//! Auto-generated foundry PDKs (`Sky130`, `Gf180mcu`) — same divider
//! lays out under each, on the foundry's actual GDS layer pairs.
//!
//! These PDKs are generated at **build time** by `build.rs`, which reads
//! the foundry's `.lyp` file and emits a `klayout_pdk::pdk! { ... }`
//! invocation into `OUT_DIR`. The mapping `RES → polydrawing_m` etc. is
//! picked once in `build.rs`; the rest of the geometry code stays
//! foundry-agnostic via `RcLikePdk`.
//!
//! Each test soft-skips if the corresponding `.lyp` wasn't present at
//! build time — `HAS_SKY130` / `HAS_GF180MCU` constants are emitted by
//! the build script.

use eda_hir::Layout;
use klayout_core::{LayerInfo, Library};
use spike_divider_block::pdks_foundry::{HAS_GF180MCU, HAS_SKY130};
use spike_divider_block::*;

fn count_shapes_recursive(lib: &Library, top: klayout_core::CellId, info: LayerInfo) -> usize {
    let layer = lib.layer(info);
    let cell = lib.get(top);
    cell.shapes_on(layer).count()
        + cell.instances().iter()
            .map(|i| lib.get(i.cell).shapes_on(layer).count())
            .sum::<usize>()
}

/// Layer-2 conformance helper: lay out a 10k/30k voltage divider under
/// any `RcLikePdk` and assert each of `expected` GDS pairs has ≥1 shape
/// in the resulting cell hierarchy.
///
/// Centralizing this means a new CMOS PDK gets one ~5-line test calling
/// in here, instead of duplicating the divider-construct + per-layer
/// shape-count loop.
fn assert_divider_lays_out_with_expected_gds_pairs<P: RcLikePdk>(
    lib: &Library,
    pdk: &P,
    pdk_label: &str,
    expected: &[(u16, u16, &str)],
) {
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let top = div.layout(lib, pdk);
    for (l, d, layer_name) in expected {
        let n = count_shapes_recursive(lib, top, LayerInfo::gds(*l, *d));
        assert!(
            n > 0,
            "{}: {} at GDS ({},{}) empty",
            pdk_label, layer_name, l, d,
        );
    }
}

#[test]
fn divider_under_generated_sky130_uses_real_foundry_gds_pairs() {
    if !HAS_SKY130 {
        eprintln!("skipping: sky130 lyp not present at build time");
        return;
    }
    use spike_divider_block::pdks_foundry::Sky130;
    let lib = Sky130::new_library("sky130_foundry_test");
    let pdk = Sky130::register(&lib);
    // Foundry GDS pairs from sky130 layers.lyp:
    //   polydrawing_m   = (66, 20)
    //   met1            = (68, 20)
    //   licon1drawing_m = (66, 44)
    assert_divider_lays_out_with_expected_gds_pairs(
        &lib, &pdk, "Sky130",
        &[(66, 20, "RES"), (68, 20, "METAL1"), (66, 44, "VIA1")],
    );
}

#[test]
fn divider_under_generated_gf180mcu_uses_real_foundry_gds_pairs() {
    if !HAS_GF180MCU {
        eprintln!("skipping: gf180mcu lyp not present at build time");
        return;
    }
    use spike_divider_block::pdks_foundry::Gf180mcu;
    let lib = Gf180mcu::new_library("gf180_foundry_test");
    let pdk = Gf180mcu::register(&lib);
    // Foundry GDS pairs from gf180mcu.lyp:
    //   poly2   = (30, 0)
    //   metal1  = (34, 0)
    //   contact = (33, 0)
    assert_divider_lays_out_with_expected_gds_pairs(
        &lib, &pdk, "Gf180mcu",
        &[(30, 0, "RES"), (34, 0, "METAL1"), (33, 0, "VIA1")],
    );
}

#[test]
fn generated_and_handcoded_pdks_yield_matching_layer_pairs() {
    // Sanity: the auto-generated `Sky130` should produce shapes on the
    // same GDS pairs as the hand-coded `Sky130Lite`. If they diverge,
    // either build.rs picked the wrong lyp short_name or the foundry
    // file changed — either way it's a regression.
    if !HAS_SKY130 { return; }
    use spike_divider_block::pdks::Sky130Lite;
    use spike_divider_block::pdks_foundry::Sky130;

    let make = || RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );

    let lib_a = Sky130Lite::new_library("hand"); let pdk_a = Sky130Lite::register(&lib_a);
    let lib_b = Sky130::new_library("gen");      let pdk_b = Sky130::register(&lib_b);
    let top_a = make().layout(&lib_a, &pdk_a);
    let top_b = make().layout(&lib_b, &pdk_b);

    for (l, d) in [(66_u16, 20_u16), (68, 20), (66, 44)] {
        let na = count_shapes_recursive(&lib_a, top_a, LayerInfo::gds(l, d));
        let nb = count_shapes_recursive(&lib_b, top_b, LayerInfo::gds(l, d));
        assert_eq!(na, nb,
            "GDS ({l},{d}) shape count: hand={na}, generated={nb}");
    }
}
