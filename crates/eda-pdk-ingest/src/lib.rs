//! `eda-pdk-ingest` — read foundry layer-property files and lift them
//! into Rust-side `LayerProps` structs ready to drive a `pdk!`
//! declaration.
//!
//! ## What this is
//!
//! Real PDKs ship layer-info as KLayout `.lyp` (XML) files: hundreds of
//! `<properties>` blocks each carrying a layer name and a `<source>L/D@N</source>`
//! tuple. The architectural value of this crate is **not** to be a
//! complete XML parser — it's to demonstrate that we can ingest a
//! foundry's actual file and produce a structurally typed PDK
//! description ready for the rest of the rlx-eda flow.
//!
//! ## What this is not
//!
//! - Not a full XML parser (no namespaces, no entities, no schema check).
//! - Not a `pdk!` macro emitter — `parse_lyp` returns a plain `Vec` of
//!   `LayerProps`; turning that into a `pdk!` invocation is a separate
//!   step (a build script, a procedural macro, or hand transcription).
//! - Not a tech-file (`.tech`) parser — that's a sibling effort with a
//!   different format.
//!
//! ## Format quirks worth knowing
//!
//! - `<name>` strings are noisy: `pwelldrawing_m 64/44` (purpose + GDS).
//!   Caller should typically split on whitespace and keep the leading
//!   token as the canonical name.
//! - `<source>` strings include a frame index: `64/44@1`. The frame is
//!   KLayout-display-internal; we preserve it but rarely use it.
//! - Empty / comment-only `<properties>` blocks (e.g. group headers)
//!   exist; we skip them.
//! - Photonic PDKs (gdsfactory generic, SiEPIC EBeam, …) nest real
//!   layers inside `<group-members>` under a parent `<properties>`
//!   whose own `<source>` is a wildcard like `*/*@*`. We treat each
//!   `<group-members>` as a nested frame and silently skip parent
//!   frames whose source is a wildcard.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerProps {
    /// Raw `<name>` text. Often suffixed with the GDS pair, e.g.
    /// `"pwelldrawing_m 64/44"`. Caller may want `.split_whitespace().next()`.
    pub name: String,
    /// GDS layer number from `<source>L/D@N</source>`.
    pub layer: u16,
    /// GDS datatype from `<source>L/D@N</source>`.
    pub datatype: u16,
    /// Frame index `N` after `@`. Defaults to 1 when omitted.
    pub frame: u16,
    /// `<fill-color>#RRGGBB</fill-color>` if present in the lyp frame.
    /// This is the colour KLayout fills hatched-shape interiors with —
    /// the natural source for SVG fills in `eda-viz`.
    pub fill_color: Option<String>,
    /// `<frame-color>#RRGGBB</frame-color>` if present. KLayout's outline
    /// colour; useful when a renderer wants frame ≠ fill.
    pub frame_color: Option<String>,
}

#[derive(Debug, Error)]
pub enum LypError {
    #[error("malformed source attribute: {0}")]
    BadSource(String),
}

/// Parse a `.lyp` file's contents into `LayerProps` entries. Tolerant
/// of whitespace and missing fields; silently skips frames that lack a
/// `<source>` tag and frames whose source is a wildcard
/// (`*/*@*`-style group headers).
///
/// Both `<properties>` and `<group-members>` push frames onto a stack:
/// the closing tag flushes the frame's `(name, source)` pair. This lets
/// nested photonic-PDK layouts (parent group + N child layers) emit
/// every child layer rather than only the last one.
pub fn parse_lyp(xml: &str) -> Result<Vec<LayerProps>, LypError> {
    #[derive(Default)]
    struct Frame {
        name: Option<String>,
        source: Option<String>,
        fill: Option<String>,
        frame: Option<String>,
    }
    let mut out = Vec::new();
    let mut stack: Vec<Frame> = Vec::new();

    for raw in xml.lines() {
        let line = raw.trim();
        if line.starts_with("<properties>") || line.starts_with("<group-members>") {
            stack.push(Frame::default());
        } else if line.starts_with("</properties>") || line.starts_with("</group-members>") {
            if let Some(frame) = stack.pop() {
                if let Some(src) = frame.source {
                    // Wildcards mark KLayout group headers, not real GDS
                    // pairs — skip silently.
                    if !src.contains('*') {
                        let (layer, datatype, frame_ix) = parse_source(&src)?;
                        // gf180mcu open_pdks lyps leave `<name/>` empty
                        // and embed the name in `<source>`. When the
                        // explicit name is missing, recover it from the
                        // source's leading whitespace-token.
                        let name = match frame.name {
                            Some(n) if !n.is_empty() => n,
                            _ => embedded_name_in_source(&src).to_string(),
                        };
                        out.push(LayerProps {
                            name,
                            layer, datatype,
                            frame: frame_ix,
                            fill_color: frame.fill,
                            frame_color: frame.frame,
                        });
                    }
                }
            }
        } else if let Some(top) = stack.last_mut() {
            if let Some(text) = strip_tag(line, "name") {
                top.name = Some(text.to_string());
            } else if let Some(text) = strip_tag(line, "source") {
                top.source = Some(text.to_string());
            } else if let Some(text) = strip_tag(line, "fill-color") {
                top.fill = Some(text.to_string());
            } else if let Some(text) = strip_tag(line, "frame-color") {
                top.frame = Some(text.to_string());
            }
        }
    }
    Ok(out)
}

/// `"pass_mk 2/222@1"` → `"pass_mk"`. Returns `""` when the source has
/// no embedded name (the common shape: `"2/222@1"`).
fn embedded_name_in_source(src: &str) -> &str {
    let mut tokens = src.split_whitespace();
    let first = tokens.next().unwrap_or("");
    // If the first token already looks like a GDS pair, there's no name.
    if first.chars().next().map_or(false, |c| c.is_ascii_digit()) {
        ""
    } else {
        first
    }
}

fn strip_tag<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    line.strip_prefix(&open)?.strip_suffix(&close)
}

/// Extract `(layer, datatype, frame)` from a string like `64/44@1` or
/// `66/20`. The `@N` frame is optional; defaults to 1.
///
/// Some foundry-shipped lyps (notably gf180mcu's open_pdks build) put
/// the layer name *inside* the source tag:
/// `<source>pass_mk 2/222@1</source>`. We tolerate that by taking the
/// last whitespace-delimited token before parsing.
fn parse_source(s: &str) -> Result<(u16, u16, u16), LypError> {
    let bad = || LypError::BadSource(s.into());
    // `*` is KLayout's "any" indicator; bail rather than guess.
    if s.contains('*') { return Err(bad()); }
    // Tolerate "name L/D@N"-style source tags by keeping the last token.
    let s = s.split_whitespace().last().unwrap_or(s);
    let (head, frame) = match s.split_once('@') {
        Some((h, f)) => (h, f.parse::<u16>().map_err(|_| bad())?),
        None         => (s, 1),
    };
    let (l, d) = head.split_once('/').ok_or_else(bad)?;
    let layer    = l.parse::<u16>().map_err(|_| bad())?;
    let datatype = d.parse::<u16>().map_err(|_| bad())?;
    Ok((layer, datatype, frame))
}

/// Codegen: emit a `klayout_pdk::pdk! { ... }` Rust source string given
/// a parsed lyp + a logical→short-name mapping + a list of port-kind
/// identifiers.
///
/// Each `(logical_name, short_name)` entry in `mapping` becomes one
/// `LOGICAL_NAME = (layer, datatype),` line in the generated layers
/// block, with the GDS pair filled in from the matched `LayerProps`.
/// Field names use `logical_name` so downstream code (trait impls) stays
/// foundry-agnostic — `Sky130` and `Gf180mcu` PDKs both expose
/// `pdk.RES`, just at different GDS pairs.
///
/// `ports` becomes the `ports: { ... }` clause. Pass `&["Electrical"]`
/// for CMOS PDKs; `&["Optical", "Electrical"]` for photonic ones.
///
/// Each generated layer is preceded by a `// from lyp: <name>` comment
/// so the foundry-side provenance is visible in the output.
///
/// Returns `Err(missing)` if any logical entry's short name isn't present
/// in `all_layers`.
pub fn generate_pdk_macro_with_mapping(
    pdk_name: &str,
    all_layers: &[LayerProps],
    mapping: &[(&str, &str)],
    ports: &[&str],
) -> Result<String, String> {
    // Wrap each short into a single-candidate slice and delegate.
    let owned: Vec<(&str, &[&str])> = mapping.iter()
        .map(|(logical, short)| (*logical, std::slice::from_ref(short)))
        .collect();
    generate_pdk_macro_with_candidates(pdk_name, all_layers, &owned, ports)
}

/// Like [`generate_pdk_macro_with_mapping`] but each logical layer
/// accepts a list of candidate short names. The first candidate that
/// matches a layer in the lyp wins. Useful when the same foundry ships
/// multiple `.lyp` flavours that differ only in naming convention —
/// e.g. sky130's upstream `polydrawing_m` vs. the open_pdks-built
/// `poly.drawing` for the same `(66, 20)` GDS pair.
pub fn generate_pdk_macro_with_candidates(
    pdk_name: &str,
    all_layers: &[LayerProps],
    mapping: &[(&str, &[&str])],
    ports: &[&str],
) -> Result<String, String> {
    use std::collections::HashMap;
    let by_short: HashMap<&str, &LayerProps> =
        all_layers.iter().map(|p| (p.short_name(), p)).collect();

    let mut out = String::new();
    out.push_str(&format!(
        "// Auto-generated from foundry lyp file. {} of {} lyp layers selected.\n",
        mapping.len(), all_layers.len(),
    ));
    out.push_str("klayout_pdk::pdk! {\n");
    out.push_str(&format!("    pub {pdk_name} {{\n"));
    out.push_str("        dbu: 1000,\n");
    out.push_str("        layers: {\n");
    for (logical, candidates) in mapping {
        let p = candidates.iter()
            .find_map(|short| by_short.get(short).copied())
            .ok_or_else(|| format!(
                "no layer matching {logical} (tried {:?}) in lyp",
                candidates,
            ))?;
        out.push_str(&format!("            // from lyp: {}\n", p.name));
        out.push_str(&format!("            {logical} = ({}, {}),\n", p.layer, p.datatype));
    }
    out.push_str("        },\n");
    out.push_str(&format!("        ports: {{ {} }},\n", ports.join(", ")));
    out.push_str("    }\n");
    out.push_str("}\n");
    Ok(out)
}

/// Helpers commonly used after `parse_lyp`.
impl LayerProps {
    /// First whitespace-delimited token of `name` — usually the canonical
    /// layer identifier (`"pwelldrawing_m"` instead of
    /// `"pwelldrawing_m 64/44"`).
    pub fn short_name(&self) -> &str {
        self.name.split_whitespace().next().unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_LYP: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
 <properties>
  <frame-color>#ccccd9</frame-color>
  <name>prBoundary_m 235/4</name>
  <source>235/4@1</source>
 </properties>
 <properties>
  <name>poly 66/20</name>
  <source>66/20@1</source>
 </properties>
 <properties>
  <name>met1 68/20</name>
  <source>68/20@1</source>
 </properties>
 <properties>
  <!-- block with no source — should be skipped -->
  <name>category-header</name>
 </properties>
</layer-properties>
"#;

    #[test]
    fn tolerates_name_prefix_in_source_tag() {
        // gf180mcu open_pdks-built lyp shape: source carries `name L/D@N`.
        let (l, d, f) = parse_source("pass_mk 2/222@1").unwrap();
        assert_eq!((l, d, f), (2, 222, 1));
    }

    #[test]
    fn extracts_fill_and_frame_colors() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
 <properties>
  <frame-color>#ff0000</frame-color>
  <fill-color>#aa0000</fill-color>
  <name>poly 66/20</name>
  <source>66/20@1</source>
 </properties>
</layer-properties>
"#;
        let layers = parse_lyp(xml).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].fill_color.as_deref(), Some("#aa0000"));
        assert_eq!(layers[0].frame_color.as_deref(), Some("#ff0000"));
    }

    #[test]
    fn parses_layer_source_pairs() {
        let layers = parse_lyp(SAMPLE_LYP).expect("parse");
        assert_eq!(layers.len(), 3, "expected 3 layers, got {}", layers.len());

        assert_eq!(layers[0].name, "prBoundary_m 235/4");
        assert_eq!(layers[0].layer, 235);
        assert_eq!(layers[0].datatype, 4);
        assert_eq!(layers[0].frame, 1);

        assert_eq!(layers[1].layer, 66);
        assert_eq!(layers[1].datatype, 20);

        assert_eq!(layers[2].layer, 68);
        assert_eq!(layers[2].datatype, 20);
    }

    #[test]
    fn short_name_strips_gds_suffix() {
        let p = LayerProps {
            name: "poly 66/20".into(), layer: 66, datatype: 20, frame: 1,
            fill_color: None, frame_color: None,
        };
        assert_eq!(p.short_name(), "poly");
    }

    #[test]
    fn source_without_frame_defaults_to_1() {
        let (l, d, f) = parse_source("10/0").unwrap();
        assert_eq!((l, d, f), (10, 0, 1));
    }

    #[test]
    fn rejects_wildcards() {
        assert!(parse_source("*/0").is_err());
        assert!(parse_source("66/*").is_err());
    }

    #[test]
    fn codegen_emits_well_formed_pdk_macro_source() {
        let layers = parse_lyp(SAMPLE_LYP).unwrap();
        let src = generate_pdk_macro_with_mapping(
            "DemoFoundry",
            &layers,
            &[("RES", "poly"), ("METAL1", "met1")],
            &["Electrical"],
        ).unwrap();
        assert!(src.contains("klayout_pdk::pdk! {"));
        assert!(src.contains("pub DemoFoundry"));
        assert!(src.contains("RES = (66, 20)"));
        assert!(src.contains("METAL1 = (68, 20)"));
        assert!(src.contains("from lyp: poly 66/20"));
        assert!(src.contains("ports: { Electrical }"));
    }

    #[test]
    fn codegen_emits_multi_port_clause_for_photonic_pdk() {
        let layers = parse_lyp(SAMPLE_LYP).unwrap();
        let src = generate_pdk_macro_with_mapping(
            "PhotonicFoundry",
            &layers,
            &[("WG", "poly")],
            &["Optical", "Electrical"],
        ).unwrap();
        assert!(src.contains("ports: { Optical, Electrical }"));
    }

    #[test]
    fn codegen_errors_on_missing_short_name() {
        let layers = parse_lyp(SAMPLE_LYP).unwrap();
        let res = generate_pdk_macro_with_mapping(
            "X", &layers,
            &[("RES", "nonexistent_layer")],
            &["Electrical"],
        );
        assert!(res.is_err());
    }

    #[test]
    fn parses_nested_group_members_and_skips_wildcards() {
        // Photonic-PDK shape: parent <properties> with wildcard source acts
        // as a group header; real layers live inside <group-members>.
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
 <properties>
  <name>Doping</name>
  <source>*/*@*</source>
  <group-members>
   <name>N 20/0</name>
   <source>20/0@1</source>
  </group-members>
  <group-members>
   <name>P 21/0</name>
   <source>21/0@1</source>
  </group-members>
 </properties>
 <properties>
  <name>Text 10/0</name>
  <source>10/0@1</source>
 </properties>
</layer-properties>
"#;
        let layers = parse_lyp(xml).expect("parse");
        // Expect 3: N, P, Text — Doping group header is skipped (wildcard).
        assert_eq!(layers.len(), 3, "got {} layers", layers.len());
        let by_name: std::collections::HashMap<&str, &LayerProps> =
            layers.iter().map(|p| (p.short_name(), p)).collect();
        assert_eq!((by_name["N"].layer,    by_name["N"].datatype),    (20, 0));
        assert_eq!((by_name["P"].layer,    by_name["P"].datatype),    (21, 0));
        assert_eq!((by_name["Text"].layer, by_name["Text"].datatype), (10, 0));
        assert!(!by_name.contains_key("Doping"), "Doping group header should be skipped");
    }

    #[test]
    fn parses_real_sky130_layers_lyp_subset() {
        // First few entries from the actual sky130 layers.lyp shipped
        // with the open PDK — proves we work on real foundry input.
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
 <properties>
  <name>pwelldrawing_m 64/44</name>
  <source>64/44@1</source>
 </properties>
 <properties>
  <name>poly_m 66/20</name>
  <source>66/20@1</source>
 </properties>
 <properties>
  <name>licon1_m 66/44</name>
  <source>66/44@1</source>
 </properties>
 <properties>
  <name>met1_m 68/20</name>
  <source>68/20@1</source>
 </properties>
</layer-properties>
"#;
        let layers = parse_lyp(xml).unwrap();
        assert_eq!(layers.len(), 4);
        let by_name: std::collections::HashMap<&str, &LayerProps> =
            layers.iter().map(|p| (p.short_name(), p)).collect();
        assert_eq!((by_name["poly_m"].layer,    by_name["poly_m"].datatype),    (66, 20));
        assert_eq!((by_name["licon1_m"].layer,  by_name["licon1_m"].datatype),  (66, 44));
        assert_eq!((by_name["met1_m"].layer,    by_name["met1_m"].datatype),    (68, 20));
    }
}
