//! Numerical leg of the layout-extraction pyramid: emit the divider's
//! extracted-from-geometry SPICE deck, hand it to ngspice, and assert
//! the operating-point Vout matches the analytic divider response
//! V·R₂/(R₁+R₂).
//!
//! Skipped when ngspice is not on PATH (matches the convention used by
//! `spike-pulse-rc/tests/ngspice.rs`). The deck-shape assertions in
//! `extract_to_spice.rs` already gate on the same layout structure
//! without ngspice, so this test is the over-the-line numerical
//! confirmation rather than the only line of defense.

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};
use eda_extract::{extract, to_spice_deck, Conductor, DeviceKind, DeviceSpec};
use eda_validate::LayoutVsBehavioralReport;
use klayout_connect::ExtractConfig;
use spike_divider_block::*;
use std::collections::HashMap;

fn divider_recognizer(cell_name: &str) -> Option<DeviceSpec> {
    let rest = cell_name.strip_prefix("Resistor_")?;
    let (_id, len_str) = rest.rsplit_once("_L")?;
    let length_dbu: i64 = len_str.parse().ok()?;
    Some(DeviceSpec {
        kind: DeviceKind::R,
        value: length_to_resistance(length_dbu) as f64,
        terminals: vec!["a".into(), "b".into()],
    })
}

#[test]
fn extracted_divider_dc_matches_analytic_in_ngspice() {
    let ng = match LocalBinary::from_env() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping (no ngspice): {e}"); return; }
    };

    // R1 = 1 kΩ, R2 = 3 kΩ, V_in = 1.0 V → V_out = R2/(R1+R2) = 0.75 V.
    let r1_dbu: i64 = 10_000;
    let r2_dbu: i64 = 30_000;
    let v_in: f64 = 1.0;

    let (lib, pdk, top) = make_divider_layout(r1_dbu, r2_dbu);
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: pdk.METAL1, label_layer: pdk.METAL1 }],
        vias: vec![],
    };
    let design = extract(&lib, top, &cfg, &divider_recognizer)
        .expect("extraction failed");

    // Build the deck: extracted devices + a DC source on `vin`. The
    // emitter already remaps `gnd` → `0`, so R2 sits between vout and 0.
    let mut deck = to_spice_deck(&design, "rc_divider_extracted_dc");
    deck.add_dc_source("in", "vin", "0", v_in);

    let result = ng.run_dc(&deck.deck(), &[OutputRequest::NodeVoltage("vout".into())])
        .expect("ngspice run_dc failed");

    let v_out_extracted = result.node_voltages["vout"];
    let r1 = length_to_resistance(r1_dbu) as f64;
    let r2 = length_to_resistance(r2_dbu) as f64;
    let v_out_analytic = v_in * r2 / (r1 + r2);

    let mut ex = HashMap::new();
    ex.insert("vout".into(), v_out_extracted);
    let mut be = HashMap::new();
    be.insert("vout".into(), v_out_analytic);

    // Linear DC, no models — ngspice should match the analytic value to
    // many digits. 1e-6 rtol with a tiny atol floor is comfortable.
    LayoutVsBehavioralReport::dc(&ex, &be)
        .assert_within(1e-6, 1e-9, "rc_divider extracted DC vs analytic");
}
