//! Realistic-layer-numbers PDKs for `Resistor` / `RcDivider`.
//!
//! Each PDK declares the three layers + an `Electrical` port-kind via the
//! `klayout_pdk::pdk!` macro and implements [`crate::RcLikePdk`] in a
//! 4-line block. With these in place the same `Resistor` / `RcDivider`
//! Rust types lay out under any of `RcDemo` / `Sky130Lite` / `Gf180Lite`,
//! producing the same geometry on each PDK's distinct GDS layer numbers.
//!
//! ## Scope
//!
//! These are **not full PDKs** — they declare just enough layers to lay
//! out a poly-resistor + metal1-routed divider. A real PDK would derive
//! its `pdk!` declaration from the foundry's `.lyp` file, declare every
//! layer, and provide foundry-specific cell parameters (sheet rho,
//! taper rules, etc.). For our spike, "enough to lay out the divider
//! with realistic layer numbers" is the bar.
//!
//! ## Layer numbers
//!
//! | Role     | RcDemo  | Sky130Lite       | Gf180Lite       |
//! |----------|---------|------------------|-----------------|
//! | RES body | (50, 0) | poly = (66, 20)  | poly2 = (30, 0) |
//! | METAL1   | (10, 0) | met1 = (68, 20)  | metal1 = (34, 0)|
//! | VIA1     | (20, 0) | licon1 = (66, 44)| contact = (33, 0)|
//!
//! Sky130 numbers track the SkyWater open PDK's GDS layer pairs. gf180mcu
//! numbers track the GlobalFoundries 180 nm MCU PDK distribution. Both
//! are widely cited in open-source flows.

use crate::{MosfetPdk, RcLikePdk};
use klayout_core::{LayerIndex, PortKindId};
use klayout_pdk::pdk;

// ── Sky130Lite ─────────────────────────────────────────────────────────

pdk! {
    pub Sky130Lite {
        dbu: 1000,
        layers: {
            POLY    = (66, 20),
            MET1    = (68, 20),
            LICON1  = (66, 44),
            DIFF    = (65, 20),
            NWELL   = (64, 20),
            NSDM    = (93, 44),
            PSDM    = (94, 20),
        },
        ports: { Electrical },
    }
}

impl RcLikePdk for Sky130Lite {
    fn res(&self) -> LayerIndex { self.POLY }
    fn metal1(&self) -> LayerIndex { self.MET1 }
    fn via1(&self) -> LayerIndex { self.LICON1 }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

impl MosfetPdk for Sky130Lite {
    fn diff(&self) -> LayerIndex { self.DIFF }
    fn poly(&self) -> LayerIndex { self.POLY }
    fn metal1(&self) -> LayerIndex { self.MET1 }
    fn via1(&self) -> LayerIndex { self.LICON1 }
    fn nwell(&self) -> LayerIndex { self.NWELL }
    fn nplus(&self) -> LayerIndex { self.NSDM }
    fn pplus(&self) -> LayerIndex { self.PSDM }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

// ── Gf180Lite ──────────────────────────────────────────────────────────

pdk! {
    pub Gf180Lite {
        dbu: 1000,
        layers: {
            POLY2   = (30, 0),
            METAL1  = (34, 0),
            CONTACT = (33, 0),
            COMP    = (22, 0),
            NWELL   = (21, 0),
            NPLUS   = (32, 0),
            PPLUS   = (31, 0),
        },
        ports: { Electrical },
    }
}

impl RcLikePdk for Gf180Lite {
    fn res(&self) -> LayerIndex { self.POLY2 }
    fn metal1(&self) -> LayerIndex { self.METAL1 }
    fn via1(&self) -> LayerIndex { self.CONTACT }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}

impl MosfetPdk for Gf180Lite {
    fn diff(&self) -> LayerIndex { self.COMP }
    fn poly(&self) -> LayerIndex { self.POLY2 }
    fn metal1(&self) -> LayerIndex { self.METAL1 }
    fn via1(&self) -> LayerIndex { self.CONTACT }
    fn nwell(&self) -> LayerIndex { self.NWELL }
    fn nplus(&self) -> LayerIndex { self.NPLUS }
    fn pplus(&self) -> LayerIndex { self.PPLUS }
    fn electrical_kind(&self) -> PortKindId { Self::Electrical }
}
