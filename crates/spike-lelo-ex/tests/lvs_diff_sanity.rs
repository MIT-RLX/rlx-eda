//! Sanity check that the LVS diff actually flags an intentional
//! schematic-vs-layout mismatch — not just that "0 violations"
//! happens to be the boring case.
//!
//! Reuses the synthetic 2-instance layout from `LeloEx::verify` via
//! eda-extract directly; declares a *wrong* schematic (M1 and M2
//! swapped) and confirms `lvs_compare` reports mismatches.

use eda_extract::{
    extract, lvs_compare, sky130_recognizer, DeviceKind, LvsMismatch, SchematicDevice,
};
use klayout_connect::{Conductor, ExtractConfig};
use klayout_core::{
    Angle90, Bbox, CellBuilder, Instance, LayerInfo, Library, Point, Port, Rect, Trans, Vec2,
};

fn build_lelo_layout() -> (Library, klayout_core::CellId, klayout_core::LayerIndex, klayout_core::LayerIndex) {
    let lib = Library::new("lvs_test", 1000);
    let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
    let met1_label = lib.layer(LayerInfo::named("met1.label", 68, 5));
    let poly = lib.layer(LayerInfo::named("poly", 66, 20));

    let nfet = {
        let mut cb = CellBuilder::new("nfet_01v8");
        let pads = [
            ("d", Point::new(2_000, 4_000)),
            ("g", Point::new(    0, 2_000)),
            ("s", Point::new(2_000,    0)),
            ("b", Point::new(4_000,   500)),
        ];
        for (name, c) in &pads {
            cb.add_shape(met1, Rect::new(Bbox::new(
                Point::new(c.x - 250, c.y - 250),
                Point::new(c.x + 250, c.y + 250),
            )));
            cb.add_port(Port::new(*name, met1, *c, Angle90::E, 500));
        }
        cb.add_shape(poly, Rect::new(Bbox::new(Point::new(1_900, 0), Point::new(2_100, 4_000))));
        lib.insert(cb)
    };

    let mut tb = CellBuilder::new("lelo_top");
    tb.add_instance(Instance::new(nfet, Trans::IDENTITY));
    tb.add_instance(Instance::new(nfet, Trans::translate(Vec2::new(8_000, 0))));
    tb.add_shape(met1, Rect::new(Bbox::new(Point::new(1_750, 1_750), Point::new(2_250, 4_250))));
    tb.add_shape(met1, Rect::new(Bbox::new(Point::new(  -250, 1_750), Point::new(8_250, 2_250))));
    tb.add_shape(met1, Rect::new(Bbox::new(Point::new( 9_750, 3_750), Point::new(15_000, 4_250))));
    tb.add_shape(met1, Rect::new(Bbox::new(Point::new(  -250,  -250), Point::new(15_000, 750))));
    tb.add_port(Port::new("gate", met1_label, Point::new(4_000, 2_000), Angle90::E, 500));
    tb.add_port(Port::new("mout", met1_label, Point::new(12_500, 4_000), Angle90::E, 500));
    tb.add_port(Port::new("gnd",  met1_label, Point::new(7_500, 250),   Angle90::E, 500));
    let top = lib.insert(tb);
    (lib, top, met1, met1_label)
}

fn correct_schematic() -> Vec<SchematicDevice> {
    vec![
        SchematicDevice {
            instance_index: 0,
            kind: DeviceKind::Other("M".into()),
            value: 0.0,
            terminals: vec![
                ("d".into(), "gate".into()),
                ("g".into(), "gate".into()),
                ("s".into(), "gnd".into()),
                ("b".into(), "gnd".into()),
            ],
        },
        SchematicDevice {
            instance_index: 1,
            kind: DeviceKind::Other("M".into()),
            value: 0.0,
            terminals: vec![
                ("d".into(), "mout".into()),
                ("g".into(), "gate".into()),
                ("s".into(), "gnd".into()),
                ("b".into(), "gnd".into()),
            ],
        },
    ]
}

#[test]
fn correct_schematic_passes_lvs() {
    let (lib, top, met1, met1_label) = build_lelo_layout();
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: met1, label_layer: met1_label }],
        vias: vec![],
    };
    let design = extract(&lib, top, &cfg, &sky130_recognizer()).expect("extract");
    let mismatches = lvs_compare(&correct_schematic(), &design.devices, 0.0);
    assert!(mismatches.is_empty(), "unexpected LVS mismatches: {mismatches:?}");
}

#[test]
fn swapping_m1_and_m2_drains_flags_lvs_mismatch() {
    // Same layout, but the schematic side claims M2 is the diode-
    // connected device and M1 is the mirror output. That's wrong —
    // the layout has it the other way. LVS must catch this.
    let (lib, top, met1, met1_label) = build_lelo_layout();
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: met1, label_layer: met1_label }],
        vias: vec![],
    };
    let design = extract(&lib, top, &cfg, &sky130_recognizer()).expect("extract");

    let mut wrong = correct_schematic();
    // Swap the d-terminal nets between M1 and M2.
    wrong[0].terminals[0].1 = "mout".into();
    wrong[1].terminals[0].1 = "gate".into();

    let mismatches = lvs_compare(&wrong, &design.devices, 0.0);
    // Both M1.d and M2.d disagree.
    assert_eq!(mismatches.len(), 2, "expected exactly two terminal-net mismatches: {mismatches:?}");
    assert!(matches!(mismatches[0], LvsMismatch::TerminalNetDiffers { ref port, .. } if port == "d"));
    assert!(matches!(mismatches[1], LvsMismatch::TerminalNetDiffers { ref port, .. } if port == "d"));
}

#[test]
fn missing_device_in_schematic_flags_count_mismatch() {
    let (lib, top, met1, met1_label) = build_lelo_layout();
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: met1, label_layer: met1_label }],
        vias: vec![],
    };
    let design = extract(&lib, top, &cfg, &sky130_recognizer()).expect("extract");

    // Schematic says only one device; layout has two.
    let mut short = correct_schematic();
    short.pop();
    let mismatches = lvs_compare(&short, &design.devices, 0.0);
    assert!(mismatches.iter().any(|m| matches!(m,
        LvsMismatch::DeviceCountDiffers { schematic: 1, layout: 2 })),
        "expected count mismatch in: {mismatches:?}");
}
