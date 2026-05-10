//! Cross-PDK Layout conformance: the same `Lna` Rust value lays out
//! cleanly on the local `RfDemo` PDK and any foundry PDK enabled by
//! Cargo features. Mirrors `spike-waveguide-block::tests::layout`.

use eda_hir::Layout;
use spike_lna::{Lna, Mosfet, RfDemo, SpiralInductor};

#[test]
fn mosfet_lays_out_on_demo_pdk() {
    let lib = RfDemo::new_library("lna_layout_test");
    let pdk = RfDemo::register(&lib);
    let m = Mosfet { width_dbu: 100_000, length_dbu: 180, id: "m_test".into() };
    let _cid = m.layout(&lib, &pdk);
    // CellId opaque; existence + non-panic is the conformance.
}

#[test]
fn spiral_inductance_is_sane_for_24ghz_design() {
    // 6-turn spiral, ~280 µm outer, 4 µm trace, 2 µm spacing.
    let l = SpiralInductor {
        outer_dbu: 280_000, width_dbu: 4_000, spacing_dbu: 2_000,
        n_turns: 6, id: "lg".into(),
    };
    let nh = l.inductance_nh();
    // Mohan square-spiral with these dims lands at ~16-22 nH —
    // matches the canonical 2.4 GHz LNA Lg target (≈ 17 nH).
    assert!(
        nh > 10.0 && nh < 30.0,
        "Lg geometry should give ~10-30 nH from Mohan formula, got {nh:.3} nH",
    );
}

#[test]
fn spiral_lays_out_on_demo_pdk() {
    let lib = RfDemo::new_library("lna_layout_test");
    let pdk = RfDemo::register(&lib);
    let l = SpiralInductor {
        outer_dbu: 100_000, width_dbu: 4_000, spacing_dbu: 2_000,
        n_turns: 4, id: "spiral_test".into(),
    };
    let _cid = l.layout(&lib, &pdk);
}

#[test]
fn lna_lays_out_on_demo_pdk() {
    let lib = RfDemo::new_library("lna_layout_test");
    let pdk = RfDemo::register(&lib);
    let lna = Lna::lna_24ghz("test_lay");
    let _cid = lna.layout(&lib, &pdk);
}

#[test]
fn lna_layout_produces_metal1_routing() {
    // Phase 2 regression: Lna::layout now drives `eda-pnr`, so the
    // top cell must contain *more* metal1 shapes than just the 5
    // pad rectangles — at least one extra Box per routed net.
    use klayout_core::Shape;
    let lib = RfDemo::new_library("lna_routing_test");
    let pdk = RfDemo::register(&lib);
    let lna = Lna::lna_24ghz("test_route");
    let top_id = lna.layout(&lib, &pdk);
    let cell = lib.get(top_id);
    let m1_box_count = cell
        .shapes_on(pdk.METAL1)
        .filter(|s| matches!(s, Shape::Box(_)))
        .count();
    // 7 nets are declared; some are 2-pin (one route segment), some
    // 3-pin (star → two segments). Lower bound ≥ 7 wire rects.
    assert!(
        m1_box_count >= 7,
        "expected ≥ 7 metal1 boxes from PNR routing, got {m1_box_count}",
    );
}

#[cfg(feature = "sky130")]
#[test]
fn lna_lays_out_on_sky130() {
    if !eda_pdks::HAS_SKY130 {
        eprintln!("skipping: sky130 .lyp not present at build time");
        return;
    }
    let lib = eda_pdks::Sky130::new_library("lna_sky130_test");
    let pdk = eda_pdks::Sky130::register(&lib);
    let lna = Lna::lna_24ghz("test_sky");
    let _cid = lna.layout(&lib, &pdk);
}

#[cfg(feature = "gf180mcu")]
#[test]
fn lna_lays_out_on_gf180mcu() {
    if !eda_pdks::HAS_GF180MCU {
        eprintln!("skipping: gf180mcu .lyp not present at build time");
        return;
    }
    let lib = eda_pdks::Gf180mcu::new_library("lna_gf180_test");
    let pdk = eda_pdks::Gf180mcu::register(&lib);
    let lna = Lna::lna_24ghz("test_gf");
    let _cid = lna.layout(&lib, &pdk);
}
