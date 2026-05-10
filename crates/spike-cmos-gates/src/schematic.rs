//! Schematic IR producers for the gate-level standard cells.
//!
//! Each cell in the parent module is a `SpiceEmit` composite — i.e. a
//! flat collection of `Nmos`/`Pmos` primitives wired into supplies and
//! signal nets. The same Rust value can produce a symbolic schematic
//! by calling its `schematic()` method here. Topology comes straight
//! from the cell's fields, so the layout the user *sees* and the
//! netlist the framework *emits* cannot drift apart.
//!
//! ## Why free functions instead of `Schematic<P>` impls
//!
//! `eda_hir::Schematic<P>` requires the implementor to be `Block`
//! (which gives a stable name + Hash + Eq). The standard cells here
//! deliberately stay `SpiceEmit`-only — they're pure SPICE-side
//! composites, no layout/sim presentation yet. Adding `Block` impls
//! would force a `Hash + Eq` constraint that buys us nothing for the
//! current SPICE-only tests. When (and if) the gates pick up
//! `Layout<P: MosfetPdk>` implementations they can grow `Block` /
//! `Schematic<P>` impls at the same time. Until then this module
//! exposes plain methods that return [`SchematicIr`].
//!
//! ## Coordinate convention
//!
//! Each gate's IR is produced in its **canonical local frame**: input
//! pins on the left at integer x = 0, output pin on the right at the
//! gate's natural width, supplies running horizontally above and
//! below the cell. The parent block (or test harness) translates the
//! IR to its global position via [`SchematicIr::translate`].
//!
//! ## What's drawn
//!
//! - **Combinational** (Inverter / Nand2 / Nand3 / And2): the NMOS
//!   pull-down stack on the bottom row, the PMOS pull-up on the top
//!   row, gates wired together, drains routed to the output rail.
//!   Supply ports are emitted as [`SchemPort`] anchors so the parent
//!   knows where to connect Vdd / Gnd buses.
//! - **Sequential** (DLatch / Dff / DLatchSR / DffSR): the underlying
//!   gate composites are NOT redrawn — instead each sequential cell
//!   is emitted as a single *box-with-pins* placeholder via the
//!   `SymbolKind::Subcircuit` variant. Drawing the full SR-latch
//!   structure would clutter any hierarchical schematic; you can
//!   always inspect the SPICE deck for transistor-level detail.

use eda_hir::{
    SchemOrient, SchemSymbol, SchematicIr, SymbolKind,
};

use crate::{And2, DLatch, DLatchSR, Dff, DffSR, Inverter, Nand2, Nand3};

/// One transistor placement in a combinational-cell schematic.
fn place_mos(label: &str, kind: SymbolKind, x: f64, y: f64) -> SchemSymbol {
    SchemSymbol {
        label: label.into(),
        value: None,
        kind,
        anchor: (x, y),
        orient: SchemOrient::Vertical,
        pin_nets: Vec::new(),
    }
}

impl Inverter {
    /// CMOS inverter schematic. Two transistors:
    /// - PMOS at top, drain on `out`, source/bulk on Vdd.
    /// - NMOS at bottom, drain on `out`, source/bulk on Gnd.
    /// Gates tied together at the input pin.
    pub fn schematic(&self) -> SchematicIr {
        let mut ir = SchematicIr::new().with_title("Inverter");
        ir.add_symbol(place_mos("Mp", SymbolKind::Pmos, 0.0,  2.0));
        ir.add_symbol(place_mos("Mn", SymbolKind::Nmos, 0.0, -2.0));
        // Wires:
        //   in  → both gates       (gate stub at x = -1.5 in vertical glyph)
        //   out → both drains      (drain stubs at y = ±0.5 from anchor)
        //   Vdd → PMOS source pin  (top of PMOS body)
        //   Gnd → NMOS source pin  (bottom of NMOS body)
        ir.add_wire([(-2.0, 2.0), (-1.5,  2.0)], Some("in".into()));
        ir.add_wire([(-2.0, -2.0), (-1.5, -2.0)], Some("in".into()));
        ir.add_wire([( 0.0, 0.5), ( 0.0, -0.5)], Some("out".into()));
        ir.add_port("in",  (-2.0, 0.0));
        ir.add_port("out", ( 0.0, 0.0));
        ir.add_port("vdd", ( 0.0, 4.0));
        ir.add_port("gnd", ( 0.0, -4.0));
        ir
    }
}

impl Nand2 {
    /// 2-input NAND. PMOS pull-up pair (parallel) on top row, NMOS
    /// pull-down stack (series) on bottom row. Gates `a`/`b` enter
    /// from the left at the y-position of each transistor.
    pub fn schematic(&self) -> SchematicIr {
        let mut ir = SchematicIr::new().with_title("NAND2");
        // PMOS row at y = +2 (parallel — two boxes side by side).
        ir.add_symbol(place_mos("Mpa", SymbolKind::Pmos,  0.0,  2.0));
        ir.add_symbol(place_mos("Mpb", SymbolKind::Pmos,  3.0,  2.0));
        // NMOS stack at y = -1 (top of stack) and y = -4 (bottom).
        ir.add_symbol(place_mos("Mna", SymbolKind::Nmos,  0.0, -1.0));
        ir.add_symbol(place_mos("Mnb", SymbolKind::Nmos,  0.0, -4.0));
        // Gate inputs.
        ir.add_wire([(-2.0,  2.0), (-1.5,  2.0)], Some("a".into()));
        ir.add_wire([(-2.0, -1.0), (-1.5, -1.0)], Some("a".into()));
        ir.add_wire([(-2.0, -4.0), (-1.5, -4.0)], Some("b".into()));
        ir.add_wire([( 1.5,  2.0), ( 4.5,  2.0)], Some("b".into()));
        // Output node — both PMOS drains and the NMOS top drain.
        ir.add_wire([( 0.0,  2.5), ( 3.0,  2.5)], Some("out".into()));
        ir.add_wire([( 0.0,  2.5), ( 0.0, -0.5)], Some("out".into()));
        ir.add_port("a",   (-2.0,  2.0));
        ir.add_port("b",   (-2.0, -4.0));
        ir.add_port("out", ( 0.0,  2.5));
        ir.add_port("vdd", ( 1.5,  4.0));
        ir.add_port("gnd", ( 0.0, -5.5));
        ir
    }
}

impl Nand3 {
    /// 3-input NAND. PMOS triplet on top (parallel); NMOS triplet
    /// stacked below.
    pub fn schematic(&self) -> SchematicIr {
        let mut ir = SchematicIr::new().with_title("NAND3");
        ir.add_symbol(place_mos("Mpa", SymbolKind::Pmos,  0.0,  2.0));
        ir.add_symbol(place_mos("Mpb", SymbolKind::Pmos,  3.0,  2.0));
        ir.add_symbol(place_mos("Mpc", SymbolKind::Pmos,  6.0,  2.0));
        ir.add_symbol(place_mos("Mna", SymbolKind::Nmos,  0.0, -1.0));
        ir.add_symbol(place_mos("Mnb", SymbolKind::Nmos,  0.0, -4.0));
        ir.add_symbol(place_mos("Mnc", SymbolKind::Nmos,  0.0, -7.0));
        ir.add_wire([(-2.0,  2.0), (-1.5,  2.0)], Some("a".into()));
        ir.add_wire([(-2.0, -1.0), (-1.5, -1.0)], Some("a".into()));
        ir.add_wire([(-2.0, -4.0), (-1.5, -4.0)], Some("b".into()));
        ir.add_wire([( 1.5,  2.0), ( 4.5,  2.0)], Some("b".into()));
        ir.add_wire([(-2.0, -7.0), (-1.5, -7.0)], Some("c".into()));
        ir.add_wire([( 4.5,  2.0), ( 7.5,  2.0)], Some("c".into()));
        // Output rail across the PMOS drains and down to the top of
        // the NMOS stack.
        ir.add_wire([( 0.0,  2.5), ( 6.0,  2.5)], Some("out".into()));
        ir.add_wire([( 0.0,  2.5), ( 0.0, -0.5)], Some("out".into()));
        ir.add_port("a",   (-2.0,  2.0));
        ir.add_port("b",   (-2.0, -4.0));
        ir.add_port("c",   (-2.0, -7.0));
        ir.add_port("out", ( 0.0,  2.5));
        ir.add_port("vdd", ( 3.0,  4.0));
        ir.add_port("gnd", ( 0.0, -8.5));
        ir
    }
}

impl And2 {
    /// AND2 = NAND2 + Inverter. Composed by translating the Inverter's
    /// IR to the right of the NAND2 and merging.
    pub fn schematic(&self) -> SchematicIr {
        let nand_ir = self.nand.schematic();
        let inv_ir  = self.inv.schematic().translate(8.0, 0.0);
        let mut ir = nand_ir.merge(inv_ir).with_title("AND2");
        // Connect NAND2.out to Inverter.in via a horizontal wire.
        ir.add_wire([(0.0, 2.5), (8.0 - 2.0, 0.0)], Some("nand_out".into()));
        ir
    }
}

// ── Sequential cells: rendered as Subcircuit boxes ────────────────────

fn subckt_symbol(label: &str, pin_names: &[&str]) -> SchemSymbol {
    SchemSymbol {
        label: label.into(),
        value: None,
        kind: SymbolKind::Subcircuit {
            pin_names: pin_names.iter().map(|s| s.to_string()).collect(),
        },
        anchor: (0.0, 0.0),
        orient: SchemOrient::Horizontal,
        pin_nets: Vec::new(),
    }
}

impl DLatch {
    pub fn schematic(&self) -> SchematicIr {
        let mut ir = SchematicIr::new().with_title("DLatch");
        ir.add_symbol(subckt_symbol("DLatch", &["d", "en", "q", "qb", "vdd", "gnd"]));
        ir
    }
}

impl Dff {
    pub fn schematic(&self) -> SchematicIr {
        let mut ir = SchematicIr::new().with_title("Dff");
        ir.add_symbol(subckt_symbol("Dff", &["d", "clk", "q", "qb", "vdd", "gnd"]));
        ir
    }
}

impl DLatchSR {
    pub fn schematic(&self) -> SchematicIr {
        let mut ir = SchematicIr::new().with_title("DLatchSR");
        ir.add_symbol(subckt_symbol(
            "DLatchSR",
            &["d", "en", "set_b", "reset_b", "q", "qb", "vdd", "gnd"],
        ));
        ir
    }
}

impl DffSR {
    pub fn schematic(&self) -> SchematicIr {
        let mut ir = SchematicIr::new().with_title("DffSR");
        ir.add_symbol(subckt_symbol(
            "DffSR",
            &["d", "clk", "set_b", "reset_b", "q", "qb", "vdd", "gnd"],
        ));
        ir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_kind(ir: &SchematicIr, want: &SymbolKind) -> usize {
        ir.symbols.iter().filter(|s| &s.kind == want).count()
    }

    #[test]
    fn inverter_schematic_has_one_nmos_one_pmos() {
        let ir = Inverter::default().schematic();
        assert_eq!(count_kind(&ir, &SymbolKind::Nmos), 1);
        assert_eq!(count_kind(&ir, &SymbolKind::Pmos), 1);
        assert!(ir.ports.iter().any(|p| p.name == "in"));
        assert!(ir.ports.iter().any(|p| p.name == "out"));
    }

    #[test]
    fn nand2_schematic_has_two_each() {
        let ir = Nand2::default().schematic();
        assert_eq!(count_kind(&ir, &SymbolKind::Nmos), 2);
        assert_eq!(count_kind(&ir, &SymbolKind::Pmos), 2);
    }

    #[test]
    fn nand3_schematic_has_three_each() {
        let ir = Nand3::default().schematic();
        assert_eq!(count_kind(&ir, &SymbolKind::Nmos), 3);
        assert_eq!(count_kind(&ir, &SymbolKind::Pmos), 3);
    }

    #[test]
    fn and2_schematic_merges_nand_plus_inverter() {
        let ir = And2::default().schematic();
        // NAND2 (2N + 2P) + Inverter (1N + 1P) = 3 of each.
        assert_eq!(count_kind(&ir, &SymbolKind::Nmos), 3);
        assert_eq!(count_kind(&ir, &SymbolKind::Pmos), 3);
    }

    #[test]
    fn sequential_cells_are_subcircuit_boxes() {
        let dlatch = DLatch::default().schematic();
        assert_eq!(dlatch.symbols.len(), 1);
        assert!(matches!(dlatch.symbols[0].kind,
            SymbolKind::Subcircuit { .. }));

        let dff = Dff::default().schematic();
        if let SymbolKind::Subcircuit { ref pin_names } = dff.symbols[0].kind {
            assert_eq!(pin_names.len(), 6);
            assert!(pin_names.iter().any(|n| n == "clk"));
        } else { panic!("Dff not a Subcircuit"); }

        let dffsr = DffSR::default().schematic();
        if let SymbolKind::Subcircuit { ref pin_names } = dffsr.symbols[0].kind {
            assert_eq!(pin_names.len(), 8);
            assert!(pin_names.iter().any(|n| n == "set_b"));
            assert!(pin_names.iter().any(|n| n == "reset_b"));
        } else { panic!("DffSR not a Subcircuit"); }
    }
}
