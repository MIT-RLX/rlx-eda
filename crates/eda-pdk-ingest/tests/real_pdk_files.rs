//! Real-PDK file ingestion. Skips when the foundry distribution isn't
//! present in any candidate location; runs and verifies when it is.
//!
//! Search order (mirrors `eda-pdks/build.rs::resolve_lyp`):
//!   1. `RLX_EDA_PDK_<NAME>_LYP` env var.
//!   2. `$PDK_ROOT` / `~/.ciel` / `~/.volare` install trees.
//!   3. Legacy hardcoded dev paths (transitional).

use eda_pdk_ingest::parse_lyp;
use std::path::PathBuf;

/// Try every candidate path in `candidates`, plus an env-var override
/// keyed on `name`. Returns the first lyp contents that resolve.
fn read_lyp(name: &str, install_relpaths: &[&str], legacy: &[&str]) -> Option<String> {
    if let Ok(p) = std::env::var(format!("RLX_EDA_PDK_{}_LYP", name)) {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(std::fs::read_to_string(&path).expect("read env-pointed lyp"));
        }
    }
    // Probe install trees. We don't know the family/variant at this
    // layer — test caller passes install_relpaths with `<variant>/<rel>`
    // baked in so we just glob versions/* below each root.
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("PDK_ROOT") {
        roots.push(PathBuf::from(p));
    }
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(&home).join(".ciel"));
        roots.push(PathBuf::from(&home).join(".volare"));
    }
    for root in &roots {
        for bin in &["ciel", "volare"] {
            for rel in install_relpaths {
                // `<root>/<bin>/<family>/versions/<ver>/<rel>` glob.
                // We don't know `<family>` here, so walk one extra level.
                let bin_dir = root.join(bin);
                let Ok(family_iter) = std::fs::read_dir(&bin_dir) else { continue };
                for fam in family_iter.flatten() {
                    let versions = fam.path().join("versions");
                    let Ok(ver_iter) = std::fs::read_dir(&versions) else { continue };
                    for ver in ver_iter.flatten() {
                        let p = ver.path().join(rel);
                        if p.is_file() {
                            return Some(std::fs::read_to_string(&p).expect("read installed lyp"));
                        }
                    }
                }
            }
        }
    }
    for p in legacy {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(std::fs::read_to_string(&path).expect("read legacy lyp"));
        }
    }
    None
}

#[test]
fn sky130_layers_lyp_yields_known_layers() {
    let Some(xml) = read_lyp(
        "SKY130",
        &[
            "sky130A/libs.tech/klayout/tech/sky130A.lyp",
            "sky130B/libs.tech/klayout/tech/sky130B.lyp",
        ],
        &["/Users/Shared/mtl/skywater130/sky130/klayout/layers.lyp"],
    ) else {
        eprintln!("skipping: sky130 lyp not present in any known location");
        return;
    };
    let layers = parse_lyp(&xml).expect("parse");
    // Should be hundreds of layers in a real PDK distribution.
    assert!(layers.len() >= 100, "got {} layers, expected ≥100", layers.len());

    // Spot-check the layers our `Sky130Lite` uses: poly = (66, 20),
    // met1 = (68, 20), licon1 = (66, 44).
    let has = |layer: u16, datatype: u16| {
        layers.iter().any(|p| p.layer == layer && p.datatype == datatype)
    };
    assert!(has(66, 20), "no poly / (66,20) layer in real sky130 lyp");
    assert!(has(68, 20), "no met1 / (68,20) layer in real sky130 lyp");
    assert!(has(66, 44), "no licon1 / (66,44) layer in real sky130 lyp");
}

#[test]
fn gf180mcu_lyp_yields_known_layers() {
    let Some(xml) = read_lyp(
        "GF180MCU",
        &[
            "gf180mcuA/libs.tech/klayout/tech/gf180mcu.lyp",
            "gf180mcuB/libs.tech/klayout/tech/gf180mcu.lyp",
            "gf180mcuC/libs.tech/klayout/tech/gf180mcu.lyp",
            "gf180mcuD/libs.tech/klayout/tech/gf180mcu.lyp",
        ],
        &["/Users/Shared/mtl/gf180mcu/gf180mcu/klayout/tech/gf180mcu.lyp"],
    ) else {
        eprintln!("skipping: gf180mcu lyp not present in any known location");
        return;
    };
    let layers = parse_lyp(&xml).expect("parse");
    assert!(layers.len() >= 50, "got {} layers", layers.len());

    // Check the Gf180Lite layers: poly2=(30,0), metal1=(34,0), contact=(33,0).
    let has = |layer: u16, datatype: u16| {
        layers.iter().any(|p| p.layer == layer && p.datatype == datatype)
    };
    assert!(has(30, 0), "no (30,0) layer (Gf180Lite POLY2)");
    assert!(has(34, 0), "no (34,0) layer (Gf180Lite METAL1)");
    assert!(has(33, 0), "no (33,0) layer (Gf180Lite CONTACT)");
}

// ── Photonic PDKs ────────────────────────────────────────────────────
//
// These exercise the same parser against the open-source PDKs that gdsfactory
// ships / publishes. They establish that the ingest path generalizes off the
// CMOS PDKs it was first designed against, and pin down the GDS pairs the
// downstream `pdk!` codegen will see.

#[test]
fn gdsfactory_generic_lyp_yields_known_photonic_layers() {
    // The "generic" tech bundled inside the gdsfactory Python package — the
    // reference PDK every gdsfactory tutorial uses.
    let Some(xml) = read_lyp(
        "GDSFACTORY_GENERIC",
        &[],
        &["/Users/Shared/mtl/gdsfactory/gdsfactory/generic_tech/klayout/layers.lyp"],
    ) else {
        eprintln!("skipping: gdsfactory generic lyp not present");
        return;
    };
    let layers = parse_lyp(&xml).expect("parse");
    // Real distribution has dozens of layers across waveguide, slab, doping, metal.
    assert!(layers.len() >= 30, "got {} layers, expected ≥30", layers.len());

    let has = |layer: u16, datatype: u16| {
        layers.iter().any(|p| p.layer == layer && p.datatype == datatype)
    };
    // Canonical photonic layers in gdsfactory.generic_tech:
    //   Waveguide = (1, 0), SLAB150 = (2, 0), SLAB90 = (3, 0)
    assert!(has(1, 0), "no (1,0) Waveguide layer");
    assert!(has(2, 0), "no (2,0) SLAB150 layer");
    assert!(has(3, 0), "no (3,0) SLAB90 layer");
    // And a metal/heater layer used for thermal phase shifters: M1 = (41, 0).
    assert!(has(41, 0), "no (41,0) M1 layer");
}

#[test]
fn siepic_ebeam_lyp_yields_known_photonic_layers() {
    // SiEPIC EBeam (UBC) — open-source SOI photonic PDK widely used in
    // university tape-outs. Its .lyp uses nested <group-members> under
    // wildcard-source group headers; we exercise the parser's grouped
    // path here.
    let Some(xml) = read_lyp(
        "SIEPIC_EBEAM",
        &[],
        &["/Users/Shared/mtl/siepic-ebeam/klayout/EBeam.lyp"],
    ) else {
        eprintln!("skipping: siepic-ebeam lyp not present");
        return;
    };
    let layers = parse_lyp(&xml).expect("parse");
    assert!(layers.len() >= 15, "got {} layers, expected ≥15", layers.len());

    let has = |layer: u16, datatype: u16| {
        layers.iter().any(|p| p.layer == layer && p.datatype == datatype)
    };
    // Canonical EBeam layers across the Waveguides / Text / Metal groups.
    assert!(has(1, 0),  "no (1,0) Si waveguide layer");
    assert!(has(4, 0),  "no (4,0) SiN waveguide layer");
    assert!(has(10, 0), "no (10,0) Text layer");
    assert!(has(11, 0), "no (11,0) M1_heater layer");
    assert!(has(12, 0), "no (12,0) M2_router layer");
}

#[test]
fn cornerstone_si220_lyp_yields_known_photonic_layers() {
    // Cornerstone (UK SiPh foundry, MPW). gdsfactory exposes this via the
    // cspdk Python package; the .lyp is small and flat.
    let Some(xml) = read_lyp(
        "CORNERSTONE_SI220",
        &[],
        &["/Users/Shared/mtl/cornerstone-si220/klayout/layers.lyp"],
    ) else {
        eprintln!("skipping: cornerstone-si220 lyp not present");
        return;
    };
    let layers = parse_lyp(&xml).expect("parse");
    assert!(layers.len() >= 8, "got {} layers, expected ≥8", layers.len());

    let has = |layer: u16, datatype: u16| {
        layers.iter().any(|p| p.layer == layer && p.datatype == datatype)
    };
    // Cornerstone si220 canonical pairs: WG=(3,0), SLAB=(5,0), HEATER=(39,0).
    assert!(has(3, 0),  "no (3,0) WG layer");
    assert!(has(5, 0),  "no (5,0) SLAB layer");
    assert!(has(39, 0), "no (39,0) HEATER layer");
}
