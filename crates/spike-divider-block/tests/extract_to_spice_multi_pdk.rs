//! Multi-PDK leg of the layout-extraction tier — the same `RcDivider`
//! lays out *and extracts* to the same SPICE topology under five PDK
//! flavors (RcDemo, Sky130Lite, Gf180Lite, plus the auto-generated
//! foundry-derived `Sky130` and `Gf180mcu`).
//!
//! The point: `eda-extract` is PDK-agnostic by construction
//! (`ExtractConfig` takes `LayerIndex` values, recognizers match on
//! PDK-independent cell names like `Resistor_R1_L10000`). This test is
//! the witness that the API actually delivers on that — change PDK at
//! the call site, the extracted netlist's vin/vout/gnd shape stays the
//! same, the emitted SPICE deck is byte-identical.
//!
//! `Sky130` / `Gf180mcu` tests soft-skip when the foundry `.lyp` wasn't
//! present at build time (matches `tests/foundry_pdks.rs`).

use eda_extract::{extract, to_spice_deck, Conductor, DeviceKind, DeviceSpec};
use eda_hir::Layout;
use klayout_connect::ExtractConfig;
use klayout_core::{CellId, Library};
use spike_divider_block::*;

fn divider_recognizer(cell_name: &str) -> Option<DeviceSpec> {
    let rest = cell_name.strip_prefix("Resistor_")?;
    let (_id, len_str) = rest.rsplit_once("_L")?;
    let length_dbu: i64 = len_str.parse().ok()?;
    Some(DeviceSpec {
        kind: DeviceKind::R,
        value: length_to_resistance(length_dbu) as f64,
        terminals: vec!["a".into(), "b".into()],
    })
}

/// Lay the canonical 1k/3k divider out under `pdk`, extract it, and
/// assert the topology is the divider's canonical vin → R1 → vout → R2
/// → gnd shape regardless of which foundry-numbered METAL1 the geometry
/// landed on.
fn assert_divider_extracts_to_canonical_topology<P: RcLikePdk>(
    lib: &Library, pdk: &P, top: CellId, pdk_label: &str,
) {
    let metal1 = <P as RcLikePdk>::metal1(pdk);
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: metal1, label_layer: metal1 }],
        vias: vec![],
    };
    let design = extract(lib, top, &cfg, &divider_recognizer)
        .unwrap_or_else(|e| panic!("{pdk_label}: extraction failed: {e}"));

    assert_eq!(design.devices.len(), 2,
        "{pdk_label}: expected 2 resistors, got {}", design.devices.len());
    assert_eq!(design.top_ports.len(), 3,
        "{pdk_label}: expected vin/vout/gnd top ports");

    let r1 = &design.devices[0];
    let r2 = &design.devices[1];
    assert!((r1.value - 1_000.0).abs() < 1e-3, "{pdk_label}: R1 = {}", r1.value);
    assert!((r2.value - 3_000.0).abs() < 1e-3, "{pdk_label}: R2 = {}", r2.value);

    let r1_a = r1.terminals.iter().find(|t| t.port == "a").unwrap();
    let r1_b = r1.terminals.iter().find(|t| t.port == "b").unwrap();
    let r2_a = r2.terminals.iter().find(|t| t.port == "a").unwrap();
    let r2_b = r2.terminals.iter().find(|t| t.port == "b").unwrap();
    assert_eq!(r1_a.net, "vin",  "{pdk_label}: R1.a → {}",  r1_a.net);
    assert_eq!(r1_b.net, "vout", "{pdk_label}: R1.b → {}", r1_b.net);
    assert_eq!(r2_a.net, "vout", "{pdk_label}: R2.a → {}", r2_a.net);
    assert_eq!(r2_b.net, "gnd",  "{pdk_label}: R2.b → {}",  r2_b.net);
}

fn build_divider<P: RcLikePdk>(lib: &Library, pdk: &P) -> CellId {
    RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    )
    .layout(lib, pdk)
}

#[test]
fn extracts_under_rc_demo() {
    let lib = RcDemo::new_library("rcdemo_extract");
    let pdk = RcDemo::register(&lib);
    let top = build_divider(&lib, &pdk);
    assert_divider_extracts_to_canonical_topology(&lib, &pdk, top, "RcDemo");
}

#[test]
fn extracts_under_sky130_lite() {
    use spike_divider_block::pdks::Sky130Lite;
    let lib = Sky130Lite::new_library("sky130lite_extract");
    let pdk = Sky130Lite::register(&lib);
    let top = build_divider(&lib, &pdk);
    assert_divider_extracts_to_canonical_topology(&lib, &pdk, top, "Sky130Lite");
}

#[test]
fn extracts_under_gf180_lite() {
    use spike_divider_block::pdks::Gf180Lite;
    let lib = Gf180Lite::new_library("gf180lite_extract");
    let pdk = Gf180Lite::register(&lib);
    let top = build_divider(&lib, &pdk);
    assert_divider_extracts_to_canonical_topology(&lib, &pdk, top, "Gf180Lite");
}

#[test]
fn extracts_under_generated_sky130() {
    use spike_divider_block::pdks_foundry::{Sky130, HAS_SKY130};
    if !HAS_SKY130 {
        eprintln!("skipping: sky130 lyp not present at build time");
        return;
    }
    let lib = Sky130::new_library("sky130_foundry_extract");
    let pdk = Sky130::register(&lib);
    let top = build_divider(&lib, &pdk);
    assert_divider_extracts_to_canonical_topology(&lib, &pdk, top, "Sky130");
}

#[test]
fn extracts_under_generated_gf180mcu() {
    use spike_divider_block::pdks_foundry::{Gf180mcu, HAS_GF180MCU};
    if !HAS_GF180MCU {
        eprintln!("skipping: gf180mcu lyp not present at build time");
        return;
    }
    let lib = Gf180mcu::new_library("gf180mcu_foundry_extract");
    let pdk = Gf180mcu::register(&lib);
    let top = build_divider(&lib, &pdk);
    assert_divider_extracts_to_canonical_topology(&lib, &pdk, top, "Gf180mcu");
}

/// PDK-invariance proof: the deck text `eda_extract::to_spice_deck`
/// produces is identical across every PDK flavor, because at the SPICE
/// level the foundry's METAL1 layer number is invisible — only the
/// device topology matters. If a future change sneaks PDK-specific
/// content into the emitted deck, this fires.
#[test]
fn extracted_spice_deck_is_pdk_invariant() {
    use spike_divider_block::pdks::{Gf180Lite, Sky130Lite};

    let extract_deck = |build_lib_pdk: &dyn Fn() -> (Library, ExtractConfig, CellId)| {
        let (lib, cfg, top) = build_lib_pdk();
        let design = extract(&lib, top, &cfg, &divider_recognizer).expect("extract");
        to_spice_deck(&design, "rc_divider").deck()
    };

    let cfg_for = |layer| ExtractConfig {
        conductors: vec![Conductor { layer, label_layer: layer }],
        vias: vec![],
    };

    // Use the trait method `pdk.metal1()` rather than the PDK-specific
    // field name (`METAL1` vs `MET1` vs `METAL1` again) — that's the
    // whole point of `RcLikePdk`.
    let deck_demo = extract_deck(&|| {
        let lib = RcDemo::new_library("a");
        let pdk = RcDemo::register(&lib);
        let top = build_divider(&lib, &pdk);
        let cfg = cfg_for(RcLikePdk::metal1(&pdk));
        (lib, cfg, top)
    });
    let deck_sky = extract_deck(&|| {
        let lib = Sky130Lite::new_library("b");
        let pdk = Sky130Lite::register(&lib);
        let top = build_divider(&lib, &pdk);
        let cfg = cfg_for(RcLikePdk::metal1(&pdk));
        (lib, cfg, top)
    });
    let deck_gf = extract_deck(&|| {
        let lib = Gf180Lite::new_library("c");
        let pdk = Gf180Lite::register(&lib);
        let top = build_divider(&lib, &pdk);
        let cfg = cfg_for(RcLikePdk::metal1(&pdk));
        (lib, cfg, top)
    });

    assert_eq!(deck_demo, deck_sky,
        "RcDemo vs Sky130Lite deck diverged:\n--- RcDemo ---\n{deck_demo}\n--- Sky130Lite ---\n{deck_sky}");
    assert_eq!(deck_sky, deck_gf,
        "Sky130Lite vs Gf180Lite deck diverged:\n--- Sky130Lite ---\n{deck_sky}\n--- Gf180Lite ---\n{deck_gf}");
}
