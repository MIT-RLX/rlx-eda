//! Schematic IR — the small, framework-neutral data structure that a
//! `Block` produces when asked for its symbolic schematic. Renderers
//! (`eda-viz`), netlist exporters (future SPICE writer), and verifiers
//! all consume this IR; nobody consumes another crate's renderer-side
//! type.
//!
//! ## Why this exists
//!
//! Without it, the schematic the user *sees* and the layout the
//! framework *produces* are two independently-authored data structures
//! that drift the moment a real user adds a third resistor. With it,
//! both come from the same `Block` impl: `Layout::layout()` and
//! `Schematic::schematic()` consume the same Rust fields and so cannot
//! disagree about topology.
//!
//! ## Coordinates
//!
//! Schematic coordinates are float, y-up, in arbitrary "schematic
//! units." The renderer in `eda-viz` applies its own pixel scale.
//! Composition uses [`SchematicIr::translate`] / [`SchematicIr::merge`]
//! to glue children into a parent's frame.

use crate::Block;

/// Symbol orientation. Only orthogonal axes — schematics traditionally
/// don't rotate by arbitrary angles.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum SchemOrient {
    #[default]
    Horizontal,
    Vertical,
}

/// Canonical symbol glyphs. Renderers map each variant to a fixed
/// drawing; netlist emitters map each to a SPICE primitive.
///
/// Each variant carries the optional human-facing value text the
/// renderer will draw next to the symbol; it does NOT carry the
/// numerical value, which lives separately on the simulation graph
/// (parameterized by `Block::name()`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Resistor,
    Capacitor,
    Diode,
    /// Independent voltage source (DC for the MVP).
    Vsource,
    /// Ground reference. Single-terminal symbol.
    Ground,
    /// N-channel MOSFET — 4 terminals: D, G, S, B.
    Nmos,
    /// P-channel MOSFET — 4 terminals: D, G, S, B.
    Pmos,
    /// Hierarchical subcircuit — drawn as a labeled rectangle with
    /// named pins on its sides. Use when a child block should NOT be
    /// expanded inline (e.g. an op-amp inside a larger amplifier
    /// schematic). Netlist emitters output a `.subckt` reference.
    Subcircuit { pin_names: Vec<String> },
}

impl SymbolKind {
    /// Number of terminals. Drives wire-routing logic.
    pub fn n_terminals(&self) -> usize {
        match self {
            SymbolKind::Ground => 1,
            SymbolKind::Nmos | SymbolKind::Pmos => 4,
            SymbolKind::Subcircuit { pin_names } => pin_names.len(),
            _ => 2,
        }
    }
}

/// One placed symbol in a schematic.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemSymbol {
    /// Stable identifier (`R1`, `R2`, `V`, …). Used as the visible
    /// label and as a key for cross-referencing — netlist emitters
    /// will line this up with `Block::name()` of the producing block.
    pub label: String,
    /// Optional value text next to the label (`"10 kΩ"`, `"5 V"`).
    pub value: Option<String>,
    pub kind: SymbolKind,
    pub anchor: (f64, f64),
    pub orient: SchemOrient,
    /// Per-pin net assignment, parallel to the symbol's pin order
    /// (`[a, b]` for 2-terminal, `[D, G, S, B]` for MOSFETs, etc).
    /// `Some("vin")` ties pin `i` to net `vin`; `None` leaves the pin
    /// unbound at the IR level (renderers may still infer it from
    /// spatial coincidence with wire endpoints).
    ///
    /// Empty `Vec` means "all pins unbound" — the historical
    /// pre-net-IR shape, kept for backward compat. Netlist exporters
    /// (SPICE, LVS) require this field populated; renderers read it
    /// only for tooltips/highlighting.
    pub pin_nets: Vec<Option<String>>,
}

impl SchemSymbol {
    /// Default constructor — `pin_nets` left empty.
    pub fn new(
        label: impl Into<String>,
        kind: SymbolKind,
        anchor: (f64, f64),
        orient: SchemOrient,
    ) -> Self {
        Self {
            label: label.into(),
            value: None,
            kind,
            anchor,
            orient,
            pin_nets: Vec::new(),
        }
    }

    pub fn with_value(mut self, v: impl Into<String>) -> Self {
        self.value = Some(v.into());
        self
    }

    pub fn with_pin_nets<I, S>(mut self, nets: I) -> Self
    where
        I: IntoIterator<Item = Option<S>>,
        S: Into<String>,
    {
        self.pin_nets = nets.into_iter().map(|o| o.map(Into::into)).collect();
        self
    }
}

/// A polyline wire connecting two or more grid points, optionally
/// labeled with a net name.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemWire {
    pub points: Vec<(f64, f64)>,
    pub net: Option<String>,
}

/// External port (top-level pin) of the schematic.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemPort {
    pub name: String,
    pub at: (f64, f64),
}

/// Complete schematic for a block — composable via `translate` +
/// `merge`. Optional `title` is rendered as a header.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SchematicIr {
    pub title: Option<String>,
    pub symbols: Vec<SchemSymbol>,
    pub wires: Vec<SchemWire>,
    pub ports: Vec<SchemPort>,
}

impl SchematicIr {
    pub fn new() -> Self { Self::default() }

    pub fn with_title(mut self, t: impl Into<String>) -> Self {
        self.title = Some(t.into());
        self
    }

    /// Translate every coordinate in-place. Used by parent blocks to
    /// place a child block's IR at a chosen offset.
    pub fn translate(mut self, dx: f64, dy: f64) -> Self {
        for s in &mut self.symbols {
            s.anchor.0 += dx;
            s.anchor.1 += dy;
        }
        for w in &mut self.wires {
            for p in &mut w.points { p.0 += dx; p.1 += dy; }
        }
        for p in &mut self.ports { p.at.0 += dx; p.at.1 += dy; }
        self
    }

    /// Append `other`'s symbols, wires, and ports. Title is preserved
    /// from `self`. Coordinates are not transformed — call
    /// [`Self::translate`] on `other` first if needed.
    pub fn merge(mut self, mut other: SchematicIr) -> Self {
        self.symbols.append(&mut other.symbols);
        self.wires.append(&mut other.wires);
        self.ports.append(&mut other.ports);
        self
    }

    pub fn add_wire(&mut self, points: impl IntoIterator<Item = (f64, f64)>, net: Option<String>) {
        self.wires.push(SchemWire { points: points.into_iter().collect(), net });
    }

    pub fn add_port(&mut self, name: impl Into<String>, at: (f64, f64)) {
        self.ports.push(SchemPort { name: name.into(), at });
    }

    pub fn add_symbol(&mut self, sym: SchemSymbol) {
        self.symbols.push(sym);
    }
}

/// A block that can describe itself as a symbolic schematic.
///
/// Mirrors [`crate::Layout`] for the schematic flow: the same Rust
/// value that produces a layout (`fn layout(&self, lib, pdk) ->
/// CellId`) also produces a schematic (`fn schematic(&self, pdk) ->
/// SchematicIr`). Composition: a parent block calls each child's
/// `schematic`, translates the result, and merges into its own IR —
/// adding the wires that connect children at the parent level.
///
/// `P` is the same PDK type used in `Layout<P>`. We pass it in so a
/// schematic emitter can read PDK-level conventions (e.g., the symbol
/// glyph for an NMOS in a foundry that distinguishes NMOS_LV / NMOS_HV
/// — for the MVP this isn't exercised but the slot is reserved).
pub trait Schematic<P>: Block {
    fn schematic(&self, pdk: &P) -> SchematicIr;
}
