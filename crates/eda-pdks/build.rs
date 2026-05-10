//! Build script: for each enabled foundry feature, parse the foundry's
//! `.lyp` and generate a `klayout_pdk::pdk! { ... }` invocation into
//! `OUT_DIR/pdks_generated.rs`. The crate's `lib.rs` `include!`s that file.
//!
//! ## Path resolution
//!
//! `.lyp` files are *not* committed to this repo — they ship with the
//! foundry's PDK. Contributors install PDKs via `rlx-eda-cli pdk install
//! <variant>` (which wraps `ciel` / `volare`); installs land under one
//! of:
//!
//! - `$PDK_ROOT/<bin>/<family>/versions/<ver>/<variant>/<lyp_subpath>`
//! - `~/.ciel/<bin>/<family>/versions/<ver>/<variant>/<lyp_subpath>`
//! - `~/.volare/<bin>/<family>/versions/<ver>/<variant>/<lyp_subpath>`
//!
//! …matching the layout `rlx-eda-cli/src/pdk/install.rs::scan_ciel_root`
//! probes at runtime. Contributors who already keep a foundry git
//! checkout outside the install tree can point to it via:
//!
//! - `RLX_EDA_PDK_<UPPER>_LYP` — single absolute path override.
//!
//! When *no* candidate resolves, the foundry falls back to a stub PDK
//! (`HAS_<UPPER> = false`) so unrelated workspace builds aren't
//! blocked on a missing foundry checkout.
//!
//! ## What lives here
//!
//! Logical → short-name mappings — they define the cross-PDK API
//! surface (`pdk.RES`, `pdk.WG`, …). The GDS pair each maps to is read
//! from the `.lyp` so the foundry stays the source of truth for layer
//! numbers.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

struct Foundry {
    /// Cargo feature gate. The generated `cfg_attr(feature = "...")`
    /// wraps each PDK so feature-disabled foundries have zero footprint.
    feature: &'static str,
    /// Constant name suffix: `HAS_<UPPER>`. Also drives the env-var
    /// override name `RLX_EDA_PDK_<UPPER>_LYP`.
    upper: &'static str,
    /// Generated Rust struct name.
    struct_name: &'static str,
    /// How to resolve this foundry's `.lyp` at build time.
    lyp: LypSource,
    /// Logical → candidate-short-names mapping. Multiple candidates
    /// per logical let one foundry support both upstream raw `.lyp`
    /// names (`polydrawing_m`) and open_pdks-built names
    /// (`poly.drawing`) — same GDS pair, different naming convention.
    mapping: &'static [(&'static str, &'static [&'static str])],
    /// Port-kind identifiers for the generated `ports: { ... }` clause.
    ports: &'static [&'static str],
}

/// Knobs the resolver uses to find a `.lyp` on a given developer's
/// machine. Mirrors `rlx-eda-cli`'s install-tree convention.
struct LypSource {
    /// `ciel`/`volare` family name (e.g., `"sky130"`, `"gf180mcu"`,
    /// `"ihp-sg13g2"`). Used to construct the install-tree path.
    /// Empty for foundries that don't ship through ciel/volare (the
    /// open photonic PDKs); those fall back to legacy paths only.
    family: &'static str,
    /// Variant under the family (`"sky130A"`, `"gf180mcuB"`, …). For
    /// foundries with one variant, equals `family`.
    variant: &'static str,
    /// `.lyp` path relative to `<install>/<variant>/`. The first
    /// candidate that exists wins — supports open_pdks-built layouts
    /// (e.g. `libs.tech/klayout/tech/sky130A.lyp`) AND raw foundry
    /// git checkouts (e.g. `klayout/layers.lyp`).
    install_relpaths: &'static [&'static str],
    /// Hardcoded paths kept as a transitional fallback for the original
    /// dev machine. New PDKs should leave this empty — the resolver
    /// finds them through `$PDK_ROOT` / `~/.ciel` / `~/.volare` /
    /// `RLX_EDA_PDK_<UPPER>_LYP`.
    legacy_dev_paths: &'static [&'static str],
}

// ── Mappings (foundry-agnostic API surface; layer numbers come from .lyp) ──

// Each logical layer carries multiple candidate short names: upstream
// foundry-repo naming first (`polydrawing_m`), open_pdks-built naming
// second (`poly.drawing`). The lyp resolver finds whichever lyp the
// developer has installed; the codegen tries candidates in order.
const SKY130_MAPPING: &[(&str, &[&str])] = &[
    ("RES",    &["polydrawing_m",     "poly.drawing"]),
    ("METAL1", &["met1",              "met1.drawing"]),
    ("VIA1",   &["licon1drawing_m",   "licon1.drawing"]),
    ("DIFF",   &["diffdrawing_m",     "diff.drawing"]),
    ("NWELL",  &["nwelldrawing_m",    "nwell.drawing"]),
    ("NPLUS",  &["nsdmdrawing_m",     "nsdm.drawing"]),
    ("PPLUS",  &["psdmdrawing_m",     "psdm.drawing"]),
];

const GF180MCU_MAPPING: &[(&str, &[&str])] = &[
    // Capitalisation differs between upstream gf180mcu repo (`poly2`,
    // `metal1`) and the open_pdks-built lyp (`Poly2`, `Metal1`).
    ("RES",    &["poly2",   "Poly2"]),
    ("METAL1", &["metal1",  "Metal1"]),
    ("VIA1",   &["contact", "Contact"]),
    ("DIFF",   &["COMP"]),
    ("NWELL",  &["NWell",   "Nwell"]),
    ("NPLUS",  &["nplus",   "Nplus"]),
    ("PPLUS",  &["pplus",   "Pplus"]),
];

// ── Foundry table ────────────────────────────────────────────────

const FOUNDRIES: &[Foundry] = &[
    // ── CMOS ────────────────────────────────────────────────────
    Foundry {
        feature: "sky130", upper: "SKY130", struct_name: "Sky130",
        lyp: LypSource {
            family: "sky130", variant: "sky130A",
            install_relpaths: &[
                "libs.tech/klayout/tech/sky130A.lyp",
                "klayout/layers.lyp",
            ],
            legacy_dev_paths: &[
                "/Users/Shared/mtl/skywater130/sky130/klayout/layers.lyp",
            ],
        },
        mapping: SKY130_MAPPING, ports: &["Electrical"],
    },
    Foundry {
        feature: "sky130b", upper: "SKY130B", struct_name: "Sky130B",
        lyp: LypSource {
            family: "sky130", variant: "sky130B",
            install_relpaths: &[
                "libs.tech/klayout/tech/sky130B.lyp",
                "libs.tech/klayout/tech/sky130A.lyp", // upstream layer numbers identical
                "klayout/layers.lyp",
            ],
            legacy_dev_paths: &[
                "/Users/Shared/mtl/skywater130/sky130/klayout/layers.lyp",
            ],
        },
        mapping: SKY130_MAPPING, ports: &["Electrical"],
    },
    Foundry {
        feature: "gf180mcu", upper: "GF180MCU", struct_name: "Gf180mcu",
        lyp: LypSource {
            family: "gf180mcu", variant: "gf180mcuA",
            install_relpaths: &[
                "libs.tech/klayout/tech/gf180mcu.lyp",
                "klayout/tech/gf180mcu.lyp",
            ],
            legacy_dev_paths: &[
                "/Users/Shared/mtl/gf180mcu/gf180mcu/klayout/tech/gf180mcu.lyp",
            ],
        },
        mapping: GF180MCU_MAPPING, ports: &["Electrical"],
    },
    Foundry {
        feature: "gf180mcu_a", upper: "GF180MCU_A", struct_name: "Gf180mcuA",
        lyp: LypSource {
            family: "gf180mcu", variant: "gf180mcuA",
            install_relpaths: &["libs.tech/klayout/tech/gf180mcu.lyp"],
            legacy_dev_paths: &["/Users/Shared/mtl/gf180mcu/gf180mcu/klayout/tech/gf180mcu.lyp"],
        },
        mapping: GF180MCU_MAPPING, ports: &["Electrical"],
    },
    Foundry {
        feature: "gf180mcu_b", upper: "GF180MCU_B", struct_name: "Gf180mcuB",
        lyp: LypSource {
            family: "gf180mcu", variant: "gf180mcuB",
            install_relpaths: &["libs.tech/klayout/tech/gf180mcu.lyp"],
            legacy_dev_paths: &["/Users/Shared/mtl/gf180mcu/gf180mcu/klayout/tech/gf180mcu.lyp"],
        },
        mapping: GF180MCU_MAPPING, ports: &["Electrical"],
    },
    Foundry {
        feature: "gf180mcu_c", upper: "GF180MCU_C", struct_name: "Gf180mcuC",
        lyp: LypSource {
            family: "gf180mcu", variant: "gf180mcuC",
            install_relpaths: &["libs.tech/klayout/tech/gf180mcu.lyp"],
            legacy_dev_paths: &["/Users/Shared/mtl/gf180mcu/gf180mcu/klayout/tech/gf180mcu.lyp"],
        },
        mapping: GF180MCU_MAPPING, ports: &["Electrical"],
    },
    Foundry {
        feature: "ihp_sg13g2", upper: "IHP_SG13G2", struct_name: "IhpSg13g2",
        lyp: LypSource {
            family: "ihp-sg13g2", variant: "ihp-sg13g2",
            install_relpaths: &["libs.tech/klayout/tech/sg13g2.lyp"],
            legacy_dev_paths: &[],
        },
        // IHP layer naming follows the open_pdks dotted convention
        // (`Activ.drawing`, `GatPoly.drawing`, `Metal1.drawing`,
        // `NWell.drawing`, `pSD.drawing`, `nSD.drawing`, `Cont.drawing`).
        // Note: IHP doesn't expose a "RES" layer per se — `GatPoly` is
        // both the gate and the natural poly-resistor draw. Same shape
        // as sky130's `polydrawing_m` aliasing.
        mapping: &[
            ("RES",    &["GatPoly.drawing"]),
            ("METAL1", &["Metal1.drawing"]),
            ("VIA1",   &["Cont.drawing"]),
            ("DIFF",   &["Activ.drawing"]),
            ("NWELL",  &["NWell.drawing"]),
            ("NPLUS",  &["nSD.drawing"]),
            ("PPLUS",  &["pSD.drawing"]),
        ],
        ports: &["Electrical"],
    },
    Foundry {
        feature: "gf180mcu_d", upper: "GF180MCU_D", struct_name: "Gf180mcuD",
        lyp: LypSource {
            family: "gf180mcu", variant: "gf180mcuD",
            install_relpaths: &["libs.tech/klayout/tech/gf180mcu.lyp"],
            legacy_dev_paths: &["/Users/Shared/mtl/gf180mcu/gf180mcu/klayout/tech/gf180mcu.lyp"],
        },
        mapping: GF180MCU_MAPPING, ports: &["Electrical"],
    },
    // ── Photonic ─────────────────────────────────────────────────
    // These don't ship through ciel/volare today, so the install
    // path is empty and the resolver falls through to env override
    // / legacy paths.
    Foundry {
        feature: "gdsfactory-generic", upper: "GDSFACTORY_GENERIC",
        struct_name: "GdsfactoryGeneric",
        lyp: LypSource {
            family: "", variant: "",
            install_relpaths: &[],
            legacy_dev_paths: &[
                "/Users/Shared/mtl/gdsfactory/gdsfactory/generic_tech/klayout/layers.lyp",
            ],
        },
        mapping: &[
            ("WG",     &["Waveguide"]),
            ("SLAB",   &["SLAB150"]),
            ("HEATER", &["MH"]),
            ("M1",     &["M1"]),
        ],
        ports: &["Optical", "Electrical"],
    },
    Foundry {
        feature: "cornerstone-si220", upper: "CORNERSTONE_SI220",
        struct_name: "CornerstoneSi220",
        lyp: LypSource {
            family: "", variant: "",
            install_relpaths: &[],
            legacy_dev_paths: &["/Users/Shared/mtl/cornerstone-si220/klayout/layers.lyp"],
        },
        mapping: &[
            ("WG",     &["WG"]),
            ("SLAB",   &["SLAB"]),
            ("HEATER", &["HEATER"]),
            ("M1",     &["PAD"]),
        ],
        ports: &["Optical", "Electrical"],
    },
    Foundry {
        feature: "siepic-ebeam", upper: "SIEPIC_EBEAM",
        struct_name: "SiepicEbeam",
        lyp: LypSource {
            family: "", variant: "",
            install_relpaths: &[],
            legacy_dev_paths: &["/Users/Shared/mtl/siepic-ebeam/klayout/EBeam.lyp"],
        },
        // EBeam is a strip-waveguide-only stack — no general slab layer
        // exposed under a unique short name (the partial-etch layer is
        // <name>Si - 90 nm rib</name>, which `short_name()` collapses to
        // `"Si"` and collides with the strip waveguide). Cross-PDK
        // generic code that needs slab should be trait-bounded on a
        // `SlabLayer` trait once that exists; until then EBeam exposes
        // WG / HEATER / M1 only.
        mapping: &[
            ("WG",     &["Si"]),
            ("HEATER", &["M1_heater"]),
            ("M1",     &["M2_router"]),
        ],
        ports: &["Optical", "Electrical"],
    },
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run when any of the path-resolver inputs change so a freshly
    // installed PDK is picked up without a clean.
    println!("cargo:rerun-if-env-changed=PDK_ROOT");
    println!("cargo:rerun-if-env-changed=HOME");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let dst = PathBuf::from(&out_dir).join("pdks_generated.rs");
    let conf_dst = PathBuf::from(&out_dir).join("pdks_conformance.rs");

    let mut src = String::new();
    src.push_str("// Auto-generated by eda-pdks/build.rs.\n\n");
    let mut conf = String::new();
    conf.push_str("// Auto-generated conformance tests (eda-pdks/build.rs).\n\n");

    for f in FOUNDRIES {
        // Per-foundry env-var override re-trigger.
        println!("cargo:rerun-if-env-changed=RLX_EDA_PDK_{}_LYP", f.upper);

        // Skip code emission when feature is off.
        if env::var(format!("CARGO_FEATURE_{}", feature_env(f.feature))).is_err() {
            continue;
        }
        let resolved = resolve_lyp(f);
        // Re-run if the resolved file changes (no-op when stub-pathed).
        if let Some(p) = &resolved {
            println!("cargo:rerun-if-changed={}", p.display());
        }
        let present = emit_foundry(&mut src, f, resolved.as_deref());
        emit_conformance(&mut conf, f, resolved.as_deref(), present);
    }

    fs::write(&dst, src).expect("write pdks_generated.rs");
    fs::write(&conf_dst, conf).expect("write pdks_conformance.rs");
}

/// Walk every candidate path for `f` and return the first that exists.
/// Order:
///   1. `$RLX_EDA_PDK_<UPPER>_LYP` (single absolute override).
///   2. `$PDK_ROOT` install tree (matches `rlx-eda-cli pdk install`).
///   3. `~/.ciel` install tree.
///   4. `~/.volare` install tree.
///   5. `legacy_dev_paths` (transitional).
fn resolve_lyp(f: &Foundry) -> Option<PathBuf> {
    // 1. Per-foundry env override.
    if let Ok(p) = env::var(format!("RLX_EDA_PDK_{}_LYP", f.upper)) {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
        // Set but missing — emit a warning and continue searching.
        println!(
            "cargo:warning=RLX_EDA_PDK_{}_LYP set to {} but file not found; \
             falling back to install-tree search",
            f.upper,
            path.display(),
        );
    }

    // 2-4. Install-tree search.
    if !f.lyp.family.is_empty() && !f.lyp.variant.is_empty() {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Ok(p) = env::var("PDK_ROOT") {
            roots.push(PathBuf::from(p));
        }
        if let Ok(home) = env::var("HOME") {
            roots.push(PathBuf::from(&home).join(".ciel"));
            roots.push(PathBuf::from(&home).join(".volare"));
        }
        for root in &roots {
            for bin in &["ciel", "volare"] {
                if let Some(p) = locate_in_install_tree(
                    root, bin, f.lyp.family, f.lyp.variant, f.lyp.install_relpaths,
                ) {
                    return Some(p);
                }
            }
            // Some users `git clone` a foundry directly under `$PDK_ROOT`
            // skipping the `<bin>/<family>/versions/<ver>/<variant>/`
            // wrapper (this is what the legacy dev FS layout was).
            // Try `<root>/<variant>/<relpath>` and `<root>/<relpath>`.
            for rel in f.lyp.install_relpaths {
                let p = root.join(f.lyp.variant).join(rel);
                if p.is_file() { return Some(p); }
                let p = root.join(rel);
                if p.is_file() { return Some(p); }
            }
        }
    }

    // 5. Legacy dev paths — last resort.
    for p in f.lyp.legacy_dev_paths {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }

    None
}

/// Walk `<root>/<bin>/<family>/versions/*/<variant>/<rel>` for each
/// `rel` candidate, return the first hit. Picks the most-recently-
/// modified version directory on ties (latest install wins).
fn locate_in_install_tree(
    root: &Path,
    bin: &str,
    family: &str,
    variant: &str,
    relpaths: &[&str],
) -> Option<PathBuf> {
    let versions_dir = root.join(bin).join(family).join("versions");
    let entries = fs::read_dir(&versions_dir).ok()?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in entries.flatten() {
        for rel in relpaths {
            let candidate = entry.path().join(variant).join(rel);
            if !candidate.is_file() { continue; }
            let mtime = entry.metadata().and_then(|m| m.modified()).ok()
                .unwrap_or(std::time::UNIX_EPOCH);
            if best.as_ref().map_or(true, |(prev, _)| mtime > *prev) {
                best = Some((mtime, candidate));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Emits the `pdk!` macro invocation and `HAS_<FOUNDRY>` const for one
/// foundry. Returns whether the `.lyp` was actually present.
fn emit_foundry(src: &mut String, f: &Foundry, lyp: Option<&Path>) -> bool {
    let Some(lyp) = lyp else {
        let hint = format!(
            "  hint: install with `rlx-eda-cli pdk install {}` or set \
             RLX_EDA_PDK_{}_LYP=/path/to/foundry.lyp",
            f.lyp.variant, f.upper,
        );
        src.push_str(&format!(
            "// {} `.lyp` not found at build time — emitting empty stub PDK.\n\
             //{}\n\
             klayout_pdk::pdk! {{ pub {} {{ dbu: 1000, layers: {{ }} }} }}\n\
             pub const HAS_{}: bool = false;\n\n",
            f.feature, hint, f.struct_name, f.upper,
        ));
        return false;
    };
    let xml = match fs::read_to_string(lyp) {
        Ok(s) => s,
        Err(e) => panic!("read {}: {}", lyp.display(), e),
    };
    let layers = match eda_pdk_ingest::parse_lyp(&xml) {
        Ok(l) => l,
        Err(e) => panic!("parse {}: {}", lyp.display(), e),
    };
    let code = match eda_pdk_ingest::generate_pdk_macro_with_candidates(
        f.struct_name, &layers, f.mapping, f.ports,
    ) {
        Ok(c) => c,
        Err(missing) => panic!("{}: {}", lyp.display(), missing),
    };
    src.push_str(&code);
    src.push_str(&format!("pub const HAS_{}: bool = true;\n\n", f.upper));
    true
}

/// Emits one `#[test] fn <foundry>_invariants()` per foundry. Soft-skips
/// inside the test body when the lyp wasn't resolvable at build time.
fn emit_conformance(conf: &mut String, f: &Foundry, lyp: Option<&Path>, present: bool) {
    use std::fmt::Write as _;
    let test_fn = format!("{}_invariants", f.upper.to_lowercase());
    let _ = writeln!(conf, "#[test]");
    let _ = writeln!(conf, "fn {}() {{", test_fn);
    let Some(lyp) = lyp.filter(|_| present) else {
        let _ = writeln!(conf, "    // {} `.lyp` unavailable at build time — soft-skip.", f.feature);
        let _ = writeln!(conf, "    eprintln!(\"skipping {}: .lyp absent at build time\");", test_fn);
        let _ = writeln!(conf, "    return;");
        let _ = writeln!(conf, "}}\n");
        return;
    };

    let xml = fs::read_to_string(lyp).expect("read lyp for conformance");
    let layers = eda_pdk_ingest::parse_lyp(&xml).expect("parse lyp for conformance");
    use std::collections::HashMap;
    let by_short: HashMap<&str, &eda_pdk_ingest::LayerProps> =
        layers.iter().map(|p| (p.short_name(), p)).collect();
    let resolved: Vec<(&str, &str, u16, u16)> = f.mapping.iter().map(|(logical, candidates)| {
        let (matched_short, p) = candidates.iter()
            .find_map(|s| by_short.get(s).map(|p| (*s, *p)))
            .unwrap_or_else(|| panic!(
                "missing layer for {logical} (tried {:?}) in {}",
                candidates, lyp.display(),
            ));
        (*logical, matched_short, p.layer, p.datatype)
    }).collect();

    let _ = writeln!(conf, "    let lib = super::{}::new_library(\"conformance_{}\");", f.struct_name, test_fn);
    let _ = writeln!(conf, "    let pdk = super::{}::register(&lib);", f.struct_name);
    let _ = writeln!(conf, "    let rows = [");
    for (logical, _short, l, d) in &resolved {
        let _ = writeln!(
            conf,
            "        super::__conformance::LayerRow {{ name: {:?}, layer: {}, datatype: {}, idx: pdk.{} }},",
            logical, l, d, logical,
        );
    }
    let _ = writeln!(conf, "    ];");
    let _ = writeln!(conf, "    super::__conformance::check_layers_match_expected(&lib, &rows);");
    let _ = writeln!(conf, "    super::__conformance::check_pairwise_distinct_gds_pairs(&rows);");

    let lyp_str = lyp.display().to_string();
    let _ = writeln!(conf, "    super::__conformance::check_lyp_drift(");
    let _ = writeln!(conf, "        {:?},", lyp_str);
    let _ = writeln!(conf, "        &[");
    for (logical, short, l, d) in &resolved {
        let _ = writeln!(conf, "            ({:?}, {:?}, {}, {}),", logical, short, l, d);
    }
    let _ = writeln!(conf, "        ],");
    let _ = writeln!(conf, "    );");

    if !f.ports.is_empty() {
        let _ = writeln!(conf, "    super::__conformance::check_port_kinds_distinct(&[");
        for p in f.ports {
            let _ = writeln!(conf, "        ({:?}, super::{}::{}),", p, f.struct_name, p);
        }
        let _ = writeln!(conf, "    ]);");
    }

    let _ = writeln!(conf, "}}\n");
}

/// Cargo translates `foo-bar` features to `CARGO_FEATURE_FOO_BAR`.
fn feature_env(name: &str) -> String {
    name.to_uppercase().replace('-', "_")
}
