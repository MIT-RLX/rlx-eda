//! Known-bad layout fires DRC. Confirms the rule wiring lands.

use eda_drc::{check_sky130a, ruleset_from_limits, LayerLimits, Rule, RuleKind, Ruleset};
use klayout_core::{Bbox, CellBuilder, LayerInfo, Library, Point, Rect};

fn rect(x0: i64, y0: i64, x1: i64, y1: i64) -> Rect {
    Rect::new(Bbox::new(Point::new(x0, y0), Point::new(x1, y1)))
}

#[test]
fn too_narrow_met1_fires_min_width_rule() {
    // dbu = 1000 ⇒ 1 µm = 1000 DBU. Sky130A met1 min width = 140 DBU.
    // We draw a 50-DBU-wide stripe → must violate.
    let lib = Library::new("drc_test", 1000);
    let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
    let mut b = CellBuilder::new("bad");
    b.add_shape(met1, rect(0, 0, 5_000, 50));
    let top = lib.insert(b);

    let v = check_sky130a(&lib, top, met1 /* unused as poly */, met1, None, None, None, None);
    assert!(v.iter().any(|x| x.rule == "MET1.W"), "missing MET1.W: {v:?}");
}

#[test]
fn ok_met1_passes() {
    // 200 DBU wide, well above the 140 DBU minimum.
    let lib = Library::new("drc_test", 1000);
    let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
    let mut b = CellBuilder::new("ok");
    b.add_shape(met1, rect(0, 0, 5_000, 200));
    let top = lib.insert(b);

    let limits = vec![LayerLimits {
        name: "MET1", layer: met1, min_width_dbu: 140, min_space_dbu: 140,
    }];
    let v = ruleset_from_limits(&limits).check(&lib, top);
    assert!(v.is_empty(), "unexpected violations: {v:?}");
}

#[test]
fn too_close_met1_pair_fires_min_space_rule() {
    // Two 200-DBU stripes 50 DBU apart — under the 140 DBU min space.
    let lib = Library::new("drc_test", 1000);
    let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
    let mut b = CellBuilder::new("close");
    b.add_shape(met1, rect(0,    0, 1_000, 200));
    b.add_shape(met1, rect(0,  250, 1_000, 450));
    let top = lib.insert(b);

    let limits = vec![LayerLimits {
        name: "MET1", layer: met1, min_width_dbu: 140, min_space_dbu: 140,
    }];
    let v = ruleset_from_limits(&limits).check(&lib, top);
    assert!(v.iter().any(|x| x.rule == "MET1.S"), "missing MET1.S: {v:?}");
}

#[test]
fn violations_are_sorted_for_determinism() {
    // Two violations that would otherwise come back in geometry-order;
    // confirm the (rule_name, bbox) sort key produces stable output.
    let lib = Library::new("drc_test", 1000);
    let met1 = lib.layer(LayerInfo::named("met1", 68, 20));
    let mut b = CellBuilder::new("two_bad");
    b.add_shape(met1, rect(10_000, 0,    11_000, 50));
    b.add_shape(met1, rect(0,      0,     1_000, 50));
    let top = lib.insert(b);

    let r = Ruleset {
        rules: vec![Rule {
            name: "MET1.W".into(), layer: met1,
            kind: RuleKind::MinWidth { min_dbu: 140 },
        }],
    };
    let v = r.check(&lib, top);
    assert_eq!(v.len(), 2);
    assert!(v[0].bbox.min.x <= v[1].bbox.min.x);
}
