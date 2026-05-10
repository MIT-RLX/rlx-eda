//! `eda-drc` — tier-1 DRC over `klayout-drc` primitives.
//!
//! Drives `klayout_drc::{width, space, ...}` with a [`Ruleset`] and
//! returns a sorted `Vec<klayout_drc::Violation>` per rule, ready to
//! drop into `eda_viz::highlights_from_drc` for rendering.
//!
//! ## Scope
//!
//! Tier-1: minimum-width, minimum-space. Each foundry ships defaults
//! via [`sky130a_default_limits`] etc.
//!
//! ## Not in scope (for now)
//!
//! - **Density rules** — `klayout_drc::density_fill` covers the fill
//!   side; checking density bounds is a separate primitive.
//! - **LFD / OPC** — pattern-litho rules; `klayout_drc::lfd` and
//!   `klayout_drc::opc` are exposed if you need them.
//! - **Edge-direction-specific rules** — `klayout_drc::edge_rules` is
//!   available; not wired into the [`Ruleset`] DSL yet.
//! - **Deck-language parsing** — sign-off decks (Calibre SVRF, KLayout
//!   DRC scripts) live in DSL files; this crate is the Rust-API path,
//!   not a deck importer.
//!
//! ## Sky130A defaults
//!
//! Numbers track the open `sky130_fd_pr` foundry rule deck. They cover
//! the rules a typical analog spike ought to pass (M1-M5 width and
//! space, poly width). Tape-out flows must validate against the
//! foundry's actual deck — these defaults are a sanity net, not
//! sign-off.

use klayout_core::{CellId, LayerIndex, Library};
use klayout_drc::Violation;
use klayout_geom::Region;
use serde::{Deserialize, Serialize};

/// One rule to run. `layer` is the [`klayout_core::LayerIndex`]
/// resolved against the test library's PDK; the caller produces this
/// from `pdk.METAL1` etc.
#[derive(Clone, Debug)]
pub struct Rule {
    /// Human-readable rule name (`"M1.W.1"`, `"M2.S.1"`, …). Surfaces
    /// in [`Violation::rule`] and in any rendered overlay.
    pub name: String,
    pub layer: LayerIndex,
    pub kind: RuleKind,
}

/// Built-in rule primitives. Each maps to one `klayout_drc` call.
#[derive(Clone, Debug, Copy)]
pub enum RuleKind {
    /// Min-width: anything narrower than `min_dbu` flags.
    MinWidth { min_dbu: i64 },
    /// Min-space: any pair of polygons closer than `min_dbu` flags.
    MinSpace { min_dbu: i64 },
}

#[derive(Clone, Debug, Default)]
pub struct Ruleset {
    pub rules: Vec<Rule>,
}

impl Ruleset {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, rule: Rule) -> &mut Self {
        self.rules.push(rule);
        self
    }

    /// Run every rule against `top` and collect the union of
    /// violations, sorted by `(rule_name, bbox)` for deterministic
    /// output across runs.
    ///
    /// `lib`/`top` are the test design; the rule-side `layer`s must
    /// have been allocated against the same library (`lib.layer(...)`
    /// or via the PDK's `register(&lib)`).
    pub fn check(&self, lib: &Library, top: CellId) -> Vec<Violation> {
        let mut out: Vec<Violation> = Vec::new();
        for rule in &self.rules {
            let region = Region::from_cell_layer(lib, top, rule.layer);
            let violations_region = match rule.kind {
                RuleKind::MinWidth { min_dbu } => klayout_drc::width(&region, min_dbu),
                RuleKind::MinSpace { min_dbu } => klayout_drc::space(&region, min_dbu),
            };
            out.extend(klayout_drc::violations_from_region(
                rule.name.as_str(),
                &violations_region,
            ));
        }
        out.sort_by(|a, b| {
            a.rule.cmp(&b.rule)
                .then((a.bbox.min.x, a.bbox.min.y).cmp(&(b.bbox.min.x, b.bbox.min.y)))
        });
        out
    }
}

/// Foundry `(layer, min-width, min-space)` triple in DBU.
#[derive(Clone, Debug)]
pub struct LayerLimits {
    pub name: &'static str,
    pub layer: LayerIndex,
    pub min_width_dbu: i64,
    pub min_space_dbu: i64,
}

/// Build a `Ruleset` from a list of `LayerLimits`, two rules per
/// layer (`<name>.W` for width, `<name>.S` for space).
pub fn ruleset_from_limits(limits: &[LayerLimits]) -> Ruleset {
    let mut r = Ruleset::new();
    for l in limits {
        r.push(Rule {
            name: format!("{}.W", l.name),
            layer: l.layer,
            kind: RuleKind::MinWidth { min_dbu: l.min_width_dbu },
        });
        r.push(Rule {
            name: format!("{}.S", l.name),
            layer: l.layer,
            kind: RuleKind::MinSpace { min_dbu: l.min_space_dbu },
        });
    }
    r
}

/// Sky130A defaults (DBU; `dbu = 1000` ⇒ 1 µm = 1000 DBU). Numbers
/// from the open `sky130_fd_pr` rules:
///
/// | layer | min W | min S |
/// | ----- | ----- | ----- |
/// | poly  | 0.150 | 0.210 |
/// | met1  | 0.140 | 0.140 |
/// | met2  | 0.140 | 0.140 |
/// | met3  | 0.300 | 0.300 |
/// | met4  | 0.300 | 0.300 |
/// | met5  | 1.600 | 1.600 |
///
/// Pass only the layers your design uses — `met2..5` are `Option<>`
/// so analog spikes can stop at met1.
pub fn sky130a_default_limits(
    poly: LayerIndex,
    met1: LayerIndex,
    met2: Option<LayerIndex>,
    met3: Option<LayerIndex>,
    met4: Option<LayerIndex>,
    met5: Option<LayerIndex>,
) -> Vec<LayerLimits> {
    let mut out = vec![
        LayerLimits { name: "POLY", layer: poly, min_width_dbu: 150, min_space_dbu: 210 },
        LayerLimits { name: "MET1", layer: met1, min_width_dbu: 140, min_space_dbu: 140 },
    ];
    if let Some(l) = met2 { out.push(LayerLimits { name: "MET2", layer: l, min_width_dbu: 140, min_space_dbu: 140 }); }
    if let Some(l) = met3 { out.push(LayerLimits { name: "MET3", layer: l, min_width_dbu: 300, min_space_dbu: 300 }); }
    if let Some(l) = met4 { out.push(LayerLimits { name: "MET4", layer: l, min_width_dbu: 300, min_space_dbu: 300 }); }
    if let Some(l) = met5 { out.push(LayerLimits { name: "MET5", layer: l, min_width_dbu: 1600, min_space_dbu: 1600 }); }
    out
}

/// One-shot helper: build the sky130A default ruleset, then run it.
pub fn check_sky130a(
    lib: &Library,
    top: CellId,
    poly: LayerIndex,
    met1: LayerIndex,
    met2: Option<LayerIndex>,
    met3: Option<LayerIndex>,
    met4: Option<LayerIndex>,
    met5: Option<LayerIndex>,
) -> Vec<Violation> {
    let limits = sky130a_default_limits(poly, met1, met2, met3, met4, met5);
    ruleset_from_limits(&limits).check(lib, top)
}

/// Row form for CSV / JSON dumps.
#[derive(Debug, Serialize, Deserialize)]
pub struct ViolationRow {
    pub rule: String,
    pub min_x: i64,
    pub min_y: i64,
    pub max_x: i64,
    pub max_y: i64,
}

pub fn rows(vs: &[Violation]) -> Vec<ViolationRow> {
    vs.iter()
        .map(|v| ViolationRow {
            rule: v.rule.to_string(),
            min_x: v.bbox.min.x,
            min_y: v.bbox.min.y,
            max_x: v.bbox.max.x,
            max_y: v.bbox.max.y,
        })
        .collect()
}
