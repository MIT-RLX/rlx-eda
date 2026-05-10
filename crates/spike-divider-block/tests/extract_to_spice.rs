//! Layout-extraction → SPICE deck round-trip for `RcDivider`.
//!
//! This is the *structural* leg of the new layout-vs-behavioral
//! validation tier — it runs without ngspice and asserts that the
//! extracted-from-geometry deck has the topology the block claims to
//! have. Concretely:
//!
//!   - exactly two `R` element lines (one per `Resistor` instance), and
//!   - they wire `vin → vout` and `vout → 0` (i.e. R1 from supply to
//!     mid-node, R2 from mid-node to ground).
//!
//! That's the structural shape of a divider; if `Layout::layout` ever
//! drops the routing wire (the LNA / MZI failure mode), the middle
//! `vout` net splits and one of the resistor terminals lands on a
//! different net — the deck stops looking like a divider, this test
//! fires.
//!
//! The numerical leg (`extract_to_spice_ngspice.rs`) actually solves the
//! deck and checks Vout = 0.75 V, but is gated on `ngspice` availability.
//! This test is the always-on guardrail.

use eda_extract::{extract, to_spice_deck, Conductor, DeviceKind, DeviceSpec};
use klayout_connect::ExtractConfig;
use spike_divider_block::*;

/// Recognizer for `RcDivider`'s children: every cell named
/// `Resistor_<id>_L<length>` is a 2-terminal R with value computed from
/// the layout-derived `length_to_resistance`. Anything else is
/// `None` (skipped — there's nothing else in the divider).
fn divider_recognizer(cell_name: &str) -> Option<DeviceSpec> {
    let rest = cell_name.strip_prefix("Resistor_")?;
    // "<id>_L<length>" → split on the last `_L`.
    let (_id, len_str) = rest.rsplit_once("_L")?;
    let length_dbu: i64 = len_str.parse().ok()?;
    let r = length_to_resistance(length_dbu) as f64;
    Some(DeviceSpec {
        kind: DeviceKind::R,
        value: r,
        terminals: vec!["a".into(), "b".into()],
    })
}

#[test]
fn divider_extracts_two_resistors_with_canonical_topology() {
    // R1 = 10_000 DBU → 1 kΩ; R2 = 30_000 DBU → 3 kΩ. Vout = 0.75 V at Vin=1.
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);

    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: pdk.METAL1, label_layer: pdk.METAL1 }],
        vias: vec![],
    };
    let design = extract(&lib, top, &cfg, &divider_recognizer)
        .expect("extraction failed (layout dropped a wire?)");

    assert_eq!(design.devices.len(), 2,
        "expected 2 resistors in extracted divider, got {}: {:?}",
        design.devices.len(),
        design.devices.iter().map(|d| (&d.cell_name, &d.kind, d.value)).collect::<Vec<_>>(),
    );
    assert_eq!(design.top_ports.len(), 3, "expected vin/vout/gnd top ports");

    // Resistor instance 0 = R1 (1 kΩ); instance 1 = R2 (3 kΩ).
    let r1 = &design.devices[0];
    let r2 = &design.devices[1];
    assert!((r1.value - 1_000.0).abs() < 1e-3, "R1 value: {}", r1.value);
    assert!((r2.value - 3_000.0).abs() < 1e-3, "R2 value: {}", r2.value);

    // Topology: R1.a is on `vin`, R1.b is on `vout`; R2.a is on `vout`,
    // R2.b is on `gnd`. The divider mid-node *must* be the same net for
    // R1.b and R2.a — that's the wire the router laid down. If it's
    // missing, those two pins land on different nets and this assertion
    // fires (which is the point).
    let r1_a = r1.terminals.iter().find(|t| t.port == "a").unwrap();
    let r1_b = r1.terminals.iter().find(|t| t.port == "b").unwrap();
    let r2_a = r2.terminals.iter().find(|t| t.port == "a").unwrap();
    let r2_b = r2.terminals.iter().find(|t| t.port == "b").unwrap();

    assert_eq!(r1_a.net, "vin",  "R1.a should land on vin (got {})",  r1_a.net);
    assert_eq!(r1_b.net, "vout", "R1.b should land on vout (got {})", r1_b.net);
    assert_eq!(r2_a.net, "vout", "R2.a should land on vout (got {})", r2_a.net);
    assert_eq!(r2_b.net, "gnd",  "R2.b should land on gnd (got {})",  r2_b.net);
    assert_eq!(r1_b.net, r2_a.net,
        "divider mid-node split: R1.b on '{}' but R2.a on '{}' — routing wire missing?",
        r1_b.net, r2_a.net);
}

#[test]
fn extracted_spice_deck_has_two_r_lines_with_right_nets() {
    // Same divider, but check the *emitted SPICE deck* — this is what
    // ngspice / LTspice would actually run.
    let (lib, pdk, top) = make_divider_layout(10_000, 30_000);
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: pdk.METAL1, label_layer: pdk.METAL1 }],
        vias: vec![],
    };
    let design = extract(&lib, top, &cfg, &divider_recognizer)
        .expect("extraction failed");
    let net = to_spice_deck(&design, "rc_divider_extracted");
    let deck = net.deck();

    // Two device lines, both starting with `R` (instance index suffix
    // is the only thing that differs between R0 / R1).
    let r_lines: Vec<&str> = deck.lines().filter(|l| l.starts_with('R')).collect();
    assert_eq!(r_lines.len(), 2, "expected 2 R lines, got {}: deck =\n{}", r_lines.len(), deck);

    // Topology in the deck: one line is `R0 vin vout 1.0e3`,
    // the other is `R1 vout 0 3.0e3`. (Order matches instance order.)
    assert!(r_lines.iter().any(|l| l.contains(" vin vout ")),
        "no R wired vin→vout in deck:\n{}", deck);
    assert!(r_lines.iter().any(|l| l.contains(" vout 0 ")),
        "no R wired vout→gnd (gnd remapped to '0') in deck:\n{}", deck);
}
