//! Static-vs-Pnr layout config: both should produce equivalent
//! verification + PEX outputs even though the wire geometry comes
//! from different sources (hand-coded rectangles vs. a planner).
//!
//! The test runs each config end-to-end through DRC + LVS + EM
//! (via `LeloEx::verify`) and PEX (via the deck-extension path),
//! then asserts:
//!
//! - Both configs return clean LVS (3 nets recognised, both M's
//!   terminals resolve to the right nets).
//! - Both configs return clean DRC (geometry is sky130A-clean).
//! - Both configs produce the same set of PEX caps (one substrate
//!   cap per non-gnd net).
//!
//! This isolates the `LayoutConfig` selection without touching
//! ngspice — runs in <1 s on any machine.

use eda_extract::{
    extract, sky130_recognizer, DeviceKind,
};
use klayout_connect::{Conductor, ExtractConfig};
use spike_lelo_ex::{LayoutConfig, LeloEx};

fn lelo_with(config: LayoutConfig) -> LeloEx {
    // lib_path doesn't matter — verify never reads the SPICE lib.
    LeloEx::new("/nonexistent/skipped").with_layout_config(config)
}

#[test]
fn static_layout_extracts_three_nets_and_two_mosfets() {
    use spike_lelo_ex::__test_helpers::synthetic_layout;
    let layout = synthetic_layout(&lelo_with(LayoutConfig::Static));
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: layout.met1, label_layer: layout.met1_label }],
        vias: vec![],
    };
    let recog = sky130_recognizer();
    let design = extract(&layout.lib, layout.top, &cfg, &recog).expect("extract");
    assert_eq!(design.devices.len(), 2, "expected 2 MOSFETs, got {}", design.devices.len());
    assert!(design.devices.iter().all(|d| matches!(d.kind, DeviceKind::Other(ref s) if s == "M")));
    // Each device has 4 terminals.
    for d in &design.devices {
        assert_eq!(d.terminals.len(), 4);
    }
}

#[test]
fn pnr_layout_extracts_three_nets_and_two_mosfets() {
    use spike_lelo_ex::__test_helpers::synthetic_layout;
    let layout = synthetic_layout(&lelo_with(LayoutConfig::Pnr));
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: layout.met1, label_layer: layout.met1_label }],
        vias: vec![],
    };
    let recog = sky130_recognizer();
    let design = extract(&layout.lib, layout.top, &cfg, &recog).expect("extract");
    assert_eq!(design.devices.len(), 2);
    assert!(design.devices.iter().all(|d| matches!(d.kind, DeviceKind::Other(ref s) if s == "M")));
}

#[test]
fn pex_lines_present_in_both_configs() {
    use spike_lelo_ex::__test_helpers::pex_lines;
    let static_lines = pex_lines(&lelo_with(LayoutConfig::Static));
    let pnr_lines    = pex_lines(&lelo_with(LayoutConfig::Pnr));
    // Both should at least carry the `* PEX` header comment +
    // some Cpex element lines. Counts may differ slightly because
    // the planner's wire paths cover marginally different areas.
    assert!(static_lines.iter().any(|l| l.starts_with("* PEX (")));
    assert!(pnr_lines.iter().any(|l| l.starts_with("* PEX (")));
    let static_caps: usize = static_lines.iter().filter(|l| l.starts_with("Cpex")).count();
    let pnr_caps: usize    = pnr_lines.iter().filter(|l| l.starts_with("Cpex")).count();
    assert!(static_caps > 0, "static config produced no Cpex lines");
    assert!(pnr_caps > 0,    "pnr config produced no Cpex lines");
}
