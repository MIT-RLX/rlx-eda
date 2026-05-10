//! Schematic round-trip: the `Schematic<P>` IR a block emits must agree
//! with its `Layout<P>` cell on the externally-visible facts that
//! downstream tools (LVS, netlist, render) consume.
//!
//! Layout ports and schematic ports are the contract surface — they
//! have to match by name, or LVS will report fictional missing nets
//! and a netlist exporter will emit a SPICE file that doesn't connect
//! to the simulator's expectations. Symbol labels are the cross-link
//! to the simulation graph: `Block::name()` is the parameter key the
//! MNA assembler stamps in, and the schematic must use the same id so
//! a viewer's "click → see param" affordance stays honest.

use eda_hir::{Block, Layout, Schematic, SchemOrient, SchematicIr, SymbolKind};
use spike_divider_block::*;
use std::collections::HashSet;

fn divider() -> RcDivider {
    RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 20_000, id: "R2".into() },
    )
}

#[test]
fn schematic_ports_match_layout_ports() {
    let d = divider();
    let lib = RcDemo::new_library("schem_ports");
    let pdk = RcDemo::register(&lib);

    let top = d.layout(&lib, &pdk);
    let cell = lib.get(top);
    let layout_ports: HashSet<String> = cell.ports().iter()
        .map(|p| p.name.to_string()).collect();

    let ir: SchematicIr = <RcDivider as Schematic<RcDemo>>::schematic(&d, &pdk);
    let schem_ports: HashSet<String> = ir.ports.iter()
        .map(|p| p.name.clone()).collect();

    assert_eq!(schem_ports, layout_ports,
        "schematic ports {schem_ports:?} must match layout ports {layout_ports:?}");
}

#[test]
fn schematic_resistor_labels_match_child_block_names() {
    let d = divider();
    let lib = RcDemo::new_library("schem_labels");
    let pdk = RcDemo::register(&lib);
    let ir: SchematicIr = <RcDivider as Schematic<RcDemo>>::schematic(&d, &pdk);

    let r_labels: HashSet<&str> = ir.symbols.iter()
        .filter(|s| s.kind == SymbolKind::Resistor)
        .map(|s| s.label.as_str())
        .collect();
    // The divider uses the bare `id` (`"R1"`/`"R2"`) as the schematic
    // label — short and human-readable. The simulation-side parameter
    // key uses `Block::name()`. Both live in the block, so neither can
    // drift without the other landing in this assertion.
    assert!(r_labels.contains("R1"), "missing R1 in {r_labels:?}");
    assert!(r_labels.contains("R2"), "missing R2 in {r_labels:?}");
    // Block::name() must contain the id — proves the MNA-side key
    // round-trips through the schematic-side label without renaming.
    assert!(<Resistor as Block>::name(&d.r1).contains("R1"));
    assert!(<Resistor as Block>::name(&d.r2).contains("R2"));
}

#[test]
fn translate_then_inverse_is_identity() {
    let d = divider();
    let lib = RcDemo::new_library("schem_translate");
    let pdk = RcDemo::register(&lib);
    let ir: SchematicIr = <RcDivider as Schematic<RcDemo>>::schematic(&d, &pdk);

    let round = ir.clone().translate(3.5, -7.25).translate(-3.5, 7.25);
    assert_eq!(round, ir, "translate by (dx,dy) then (-dx,-dy) must be identity");
}

#[test]
fn merge_preserves_self_title() {
    // `merge` keeps `self.title`, drops `other.title` — necessary so
    // composing children doesn't clobber the parent's header.
    let parent = SchematicIr::new().with_title("parent");
    let child = SchematicIr::new().with_title("child");
    let merged = parent.merge(child);
    assert_eq!(merged.title.as_deref(), Some("parent"));
}

#[test]
fn schematic_wire_endpoints_meet_a_symbol_or_port() {
    // Every wire in the divider's IR must terminate at either a
    // symbol pin or a port location. Catches a class of typo bugs in
    // the hand-coded schematic where a wire leaves a "dangling stub"
    // that visually looks fine but doesn't actually wire anything up.
    let d = divider();
    let lib = RcDemo::new_library("schem_wires");
    let pdk = RcDemo::register(&lib);
    let ir: SchematicIr = <RcDivider as Schematic<RcDemo>>::schematic(&d, &pdk);

    // Collect anchorable points: every symbol pin (computed below) +
    // every declared port.
    let mut anchors: Vec<(f64, f64)> = ir.ports.iter().map(|p| p.at).collect();
    for s in &ir.symbols {
        for pin_local in symbol_pins_local(&s.kind) {
            let oriented = match s.orient {
                SchemOrient::Horizontal => pin_local,
                SchemOrient::Vertical   => (pin_local.1, -pin_local.0),
            };
            anchors.push((s.anchor.0 + oriented.0, s.anchor.1 + oriented.1));
        }
    }

    let close = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6;
    for (wi, w) in ir.wires.iter().enumerate() {
        let endpoints = [w.points.first().copied(), w.points.last().copied()];
        for end in endpoints.into_iter().flatten() {
            assert!(anchors.iter().any(|a| close(*a, end)),
                "wire #{wi} endpoint {end:?} doesn't meet any symbol pin or port \
                 (anchors = {anchors:?})");
        }
    }
}

/// Pin offsets in the symbol's own (horizontal) frame. Mirrors the
/// renderer's [`Symbol::pins`] in `eda-viz`. Encoded here too rather
/// than depending on `eda-viz` from this test crate, because the
/// schematic→layout contract shouldn't bring in the SVG renderer.
fn symbol_pins_local(kind: &SymbolKind) -> Vec<(f64, f64)> {
    match kind {
        SymbolKind::Ground => vec![(0.0, 1.0)],
        _ => vec![(-2.0, 0.0), (2.0, 0.0)],
    }
}
