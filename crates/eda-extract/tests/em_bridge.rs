//! End-to-end EM bridge: build a tiny two-net layout, run the
//! `klayout-connect` extractor, hand the nets + currents to
//! `em_segments_from_nets`, run `eda_em::check`, and confirm the
//! over-current segment fires.

use eda_em::{check, Jmax, Layer, LayerThickness};
use eda_extract::em_segments_from_nets;
use klayout_connect::{extract_hierarchical, Conductor, ExtractConfig};
use klayout_core::{Bbox, CellBuilder, LayerInfo, Library, Point, Rect};

fn rect(x0: i64, y0: i64, x1: i64, y1: i64) -> Rect {
    Rect::new(Bbox::new(Point::new(x0, y0), Point::new(x1, y1)))
}

#[test]
fn over_current_segment_flagged_via_extract_bridge() {
    // dbu = 1000 ⇒ 1 µm = 1000 DBU.
    let lib = Library::new("em_bridge", 1000);
    let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
    let label = lib.layer(LayerInfo::named("met1.label", 68, 5));

    // Two disjoint met1 stripes — two separate nets after extraction.
    // Stripe A: 1µm wide, 5µm long. Stripe B: 0.3µm wide, 5µm long.
    let mut b = CellBuilder::new("em_top");
    b.add_shape(met1, rect(0,    0,    5_000, 1_000));   // wide stripe
    b.add_shape(met1, rect(0,    2_000, 5_000, 2_300));  // narrow stripe
    let top = lib.insert(b);

    // Run the conductor extractor on met1.
    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: met1, label_layer: label }],
        vias: vec![],
    };
    let result = extract_hierarchical(&lib, top, &cfg);

    // We expect 2 nets (one per stripe), both anonymous (no labels).
    let nets: Vec<_> = result.nets().to_vec();
    assert_eq!(nets.len(), 2, "expected two disjoint nets, got {}", nets.len());

    // Pretend both nets carry 5 mA peak. The wide stripe (1 µm × 0.36 µm
    // met1 thickness) gets J = 5e-3 / (1.0 × 0.36) ≈ 13.9 mA/µm² — over
    // the 4.17 mA/µm² sky130 met1 limit. The narrow stripe is even
    // worse (0.3 µm × 0.36 µm) so will also flag. Therefore *both* fire.
    // The wide one is meant to flag at 5 mA — try a smaller current
    // for "passes" coverage in a second test.
    let segments = em_segments_from_nets(
        &nets,
        lib.dbu(),
        |_| 5e-3,
        |_| Some(Layer::Metal1), // single-conductor test ⇒ all nets on met1
    );
    assert_eq!(segments.len(), 2, "expected 2 segments, got {segments:?}");

    let v = check(&segments, &Jmax::sky130_metal(), &LayerThickness::sky130_metal()).unwrap();
    // Both segments are over-current at 5 mA on met1 — sanity-check.
    assert_eq!(v.len(), 2, "expected 2 violations, got {v:?}");
    // The narrower stripe should have the higher margin_ratio.
    let widths: Vec<f64> = v.iter().map(|x| x.width_um).collect();
    assert!(widths.contains(&1.0));
    assert!(widths.contains(&0.3));
}

#[test]
fn within_limit_segments_pass() {
    // 1µm wide stripe at 0.5 mA — well under the 1.5 mA peak / µm.
    let lib = Library::new("em_ok", 1000);
    let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
    let label = lib.layer(LayerInfo::named("met1.label", 68, 5));
    let mut b = CellBuilder::new("ok");
    b.add_shape(met1, rect(0, 0, 5_000, 1_000));
    let top = lib.insert(b);

    let cfg = ExtractConfig {
        conductors: vec![Conductor { layer: met1, label_layer: label }],
        vias: vec![],
    };
    let result = extract_hierarchical(&lib, top, &cfg);
    let nets: Vec<_> = result.nets().to_vec();

    let segments = em_segments_from_nets(
        &nets, lib.dbu(),
        |_| 0.5e-3,
        |_| Some(Layer::Metal1),
    );
    let v = check(&segments, &Jmax::sky130_metal(), &LayerThickness::sky130_metal()).unwrap();
    assert!(v.is_empty(), "unexpected violations: {v:?}");
}
