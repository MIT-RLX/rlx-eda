//! Smoke tests for the layout and schematic renderers.
//!
//! Layout test builds a tiny `(Library, Cell)` directly via `klayout-core`
//! to avoid pulling `spike-divider-block` into eda-viz's dep tree just
//! for the test. Schematic test exercises the canned voltage-divider
//! builder.

use eda_viz::{layout, schematic, Style};
use klayout_core::{
    Bbox, CellBuilder, LayerInfo, Library, Point, Rect,
};

#[test]
fn layout_smoke_renders_basic_rects() {
    let lib = Library::new("smoke", 1000);
    let res = lib.layer(LayerInfo::named("RES", 50, 0));
    let met = lib.layer(LayerInfo::named("METAL1", 10, 0));

    let mut b = CellBuilder::new("rect_pair");
    b.add_shape(res, Rect::new(Bbox::new(Point::new(0, 0), Point::new(10_000, 1_000))));
    b.add_shape(met, Rect::new(Bbox::new(Point::new(-1_000, -500), Point::new(1_000, 1_500))));
    b.add_shape(met, Rect::new(Bbox::new(Point::new( 9_000, -500), Point::new(11_000, 1_500))));
    let top = lib.insert(b);

    let svg = layout::render_to_svg(&lib, top, &Style::default());

    // Sanity: contains an <svg> root, the layer color palette, and a
    // <rect> for at least one shape.
    assert!(svg.starts_with("<?xml"), "missing xml header");
    assert!(svg.contains("<svg "), "missing svg root");
    assert!(svg.contains("<rect "), "no rects emitted");
    assert!(svg.contains("RES"), "legend missing layer name");
}

#[cfg(feature = "png")]
#[test]
fn png_feature_rasterizes_schematic() {
    let s = schematic::voltage_divider("R1", None, "R2", None, "V", None);
    let svg = schematic::render_to_svg(&s, &schematic::SchemStyle::default());
    let bytes = eda_viz::png::svg_to_png(&svg, 1.0).expect("png rasterize");
    // PNG magic: 89 50 4E 47 0D 0A 1A 0A
    assert_eq!(&bytes[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
}

#[test]
fn schematic_smoke_renders_divider() {
    let s = schematic::voltage_divider(
        "R1", Some("10 kΩ"),
        "R2", Some("30 kΩ"),
        "V",  Some("5 V"),
    );
    let svg = schematic::render_to_svg(&s, &schematic::SchemStyle::default());
    assert!(svg.contains("<svg "));
    assert!(svg.contains("R1"));
    assert!(svg.contains("R2"));
    assert!(svg.contains("vout"));
}

/// **Connectivity-bound rendering test.**
///
/// Drives the renderer with the *real* `make_divider_layout` output
/// from `spike-divider-block` and re-asserts the same LVS invariant
/// that crate's `tests/lvs.rs` checks: exactly 3 METAL1 nets (vin,
/// vout, gnd). If the layout regresses (dangling wire, missing pad,
/// off-by-one routing offset, …) the test fails *before* anyone sees
/// the rendered output, so a broken layout can never silently slip
/// into a demo PNG.
///
/// The point isn't that this duplicates spike-divider-block's tests —
/// it's that a renderer demo binds the visualization to the
/// LVS-verified geometry, so we can't drift back to a hand-built
/// geometry that bypasses verification (the failure mode that hid the
/// dangling-wire bug earlier).
#[test]
fn renderer_runs_only_on_lvs_verified_layout() {
    use klayout_connect::{extract_hierarchical, Conductor, ExtractConfig};

    let (lib, pdk, top) = spike_divider_block::make_divider_layout(10_000, 30_000);

    // 1. LVS check — same conductor set as spike-divider-block/tests/lvs.rs.
    //    METAL1 only: declaring RES a conductor would short the
    //    resistor body's two terminals, collapsing the divider to a
    //    single net (see that file for the rationale).
    let nl = extract_hierarchical(&lib, top, &ExtractConfig {
        conductors: vec![Conductor { layer: pdk.METAL1, label_layer: pdk.METAL1 }],
        vias: vec![],
    });
    assert_eq!(
        nl.nets().len(), 3,
        "rendered layout has {} METAL1 nets, expected 3 (vin, vout, gnd) — \
         a dangling routed wire or missing pad would land here first",
        nl.nets().len(),
    );

    // 2. Render: must produce non-empty SVG with all three layers
    //    showing up in the legend.
    let svg = layout::render_to_svg(&lib, top, &Style::default());
    assert!(svg.contains("<rect "), "no rects emitted");
    for layer_name in ["RES", "METAL1", "VIA1"] {
        assert!(
            svg.contains(layer_name),
            "rendered SVG missing '{layer_name}' in legend",
        );
    }
}

#[test]
fn schematic_renders_nmos_and_pmos_glyphs() {
    use schematic::{Orient, Schematic, SchemStyle, Symbol};

    let mut s = Schematic::new();
    s.title = Some("MOSFET smoke".into());
    s.place(
        Symbol::Nmos { label: "M1".into(), value: Some("W/L=2/1".into()) },
        (0.0, 0.0),
        Orient::Vertical,
    );
    s.place(
        Symbol::Pmos { label: "M2".into(), value: Some("W/L=4/1".into()) },
        (5.0, 0.0),
        Orient::Vertical,
    );

    let svg = schematic::render_to_svg(&s, &SchemStyle::default());
    assert!(svg.contains("<svg "));
    assert!(svg.contains("M1"), "NMOS label missing");
    assert!(svg.contains("M2"), "PMOS label missing");
    // Each MOSFET emits 4 lead lines + 4 body strokes (channel, gate
    // stripe, gate stub, body stub) + 1 polygon (source arrow).
    // Polygon count must be ≥ 2 — one per MOSFET.
    let polys = svg.matches("<polygon").count();
    assert!(polys >= 2,
        "expected ≥2 polygons (1 source arrow per MOSFET), got {polys}");
}

#[test]
fn schematic_mosfet_pin_count_is_four() {
    use schematic::Symbol;
    let n = Symbol::Nmos { label: "N".into(), value: None };
    let p = Symbol::Pmos { label: "P".into(), value: None };
    assert_eq!(n.pins().len(), 4, "NMOS must have 4 pins");
    assert_eq!(p.pins().len(), 4, "PMOS must have 4 pins");
}

/// Render a divider laid out in a foundry-realistic PDK (`Sky130Lite`)
/// instead of the toy `RcDemo` PDK. Catches the class of "works on the
/// demo, breaks on real PDKs" bug — different layer numbers, different
/// dbu, different layer count.
#[test]
fn renders_under_sky130lite_pdk() {
    use eda_hir::Layout as _;
    use spike_divider_block::{pdks::Sky130Lite, RcDivider, Resistor};

    let lib = Sky130Lite::new_library("sky130_demo");
    let pdk = Sky130Lite::register(&lib);
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let top = div.layout(&lib, &pdk);

    let svg = layout::render_to_svg(&lib, top, &Style::default());
    assert!(svg.contains("<rect "), "no rects emitted");
    // Sky130's poly layer (`66/20`) is the resistor body in `Sky130Lite`.
    // The legend must include it.
    assert!(
        svg.contains("66/20") || svg.contains("poly"),
        "Sky130 poly layer not in legend",
    );
}

#[test]
fn repetition_array_stamps_correctly() {
    use klayout_core::{
        Bbox, CellBuilder, Instance, LayerInfo, Library, Point, Rect, Repetition, Trans, Vec2,
    };

    // Child cell: a 1×1 µm METAL1 square.
    let lib = Library::new("rep_test", 1000);
    let met = lib.layer(LayerInfo::named("METAL1", 10, 0));
    let mut child = CellBuilder::new("dot");
    child.add_shape(met, Rect::new(Bbox::new(Point::new(0, 0), Point::new(1_000, 1_000))));
    let dot = lib.insert(child);

    // Top cell: a 3×3 grid of `dot`, spaced 5 µm apart.
    let mut top = CellBuilder::new("grid");
    let inst = Instance::new(dot, Trans::IDENTITY).with_repetition(Repetition::Regular {
        col: Vec2::new(5_000, 0),
        row: Vec2::new(0, 5_000),
        n_cols: 3,
        n_rows: 3,
    });
    top.add_instance(inst);
    let top_id = lib.insert(top);

    let svg = layout::render_to_svg(&lib, top_id, &Style::default());
    // Each replicated dot becomes its own <rect>. 3×3 = 9 squares; the
    // legend swatch adds 1, and the background adds 1 → 11 total.
    let rect_count = svg.matches("<rect ").count();
    assert!(
        rect_count >= 9,
        "expected at least 9 stamped rects from 3x3 repetition; got {rect_count}\n{svg}",
    );
}

#[test]
fn waveform_renders_two_traces() {
    use eda_viz::waveform::{render_to_svg as render_wave, Trace, WaveformStyle};

    let v_in: Vec<(f64, f64)> = (0..200).map(|i| {
        let t = i as f64 * 1e-9;
        (t, 5.0 * (2.0 * std::f64::consts::PI * 1e6 * t).sin())
    }).collect();
    let v_out: Vec<(f64, f64)> = v_in.iter().map(|&(t, v)| (t, v * 0.75)).collect();

    let svg = render_wave(
        &[
            Trace { label: "vin".into(),  points: v_in },
            Trace { label: "vout".into(), points: v_out },
        ],
        &WaveformStyle { title: Some("Divider transient".into()), ..Default::default() },
    );
    assert!(svg.contains("<svg "));
    assert!(svg.contains("vin"));
    assert!(svg.contains("vout"));
    assert!(svg.contains("Divider transient"));
    // 2 traces → at least 2 polylines for the data + grid-line group.
    assert!(svg.matches("<polyline ").count() >= 2);
}

#[test]
fn schematic_pin_nets_round_trip() {
    use eda_hir::Schematic as _;
    use spike_divider_block::{RcDemo, RcDivider, Resistor};

    let lib = RcDemo::new_library("pin_nets");
    let pdk = RcDemo::register(&lib);
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );
    let ir = div.schematic(&pdk);

    // Find R1 and R2 in the IR; assert their pin_nets are wired up.
    let r1 = ir.symbols.iter().find(|s| s.label == "R1").expect("R1 missing");
    let r2 = ir.symbols.iter().find(|s| s.label == "R2").expect("R2 missing");
    assert_eq!(
        r1.pin_nets,
        vec![Some("vin".to_string()), Some("vout".to_string())],
        "R1 pins should be (vin, vout) — divider input + tap",
    );
    assert_eq!(
        r2.pin_nets,
        vec![Some("vout".to_string()), Some("gnd".to_string())],
        "R2 pins should be (vout, gnd)",
    );
    let v = ir.symbols.iter().find(|s| s.label == "V").expect("V missing");
    assert_eq!(v.pin_nets, vec![Some("vin".to_string()), Some("gnd".to_string())]);
}

#[test]
fn highlights_overlay_renders() {
    use eda_viz::Highlight;
    use klayout_core::{Bbox, CellBuilder, LayerInfo, Library, Point, Rect};

    let lib = Library::new("highlight_demo", 1000);
    let met = lib.layer(LayerInfo::named("METAL1", 10, 0));
    let mut b = CellBuilder::new("with_violation");
    b.add_shape(met, Rect::new(Bbox::new(Point::new(0, 0), Point::new(5_000, 1_000))));
    let top = lib.insert(b);

    let mut style = Style::default();
    style.highlights.push(Highlight {
        bbox: Bbox::new(Point::new(1_000, -200), Point::new(2_000, 1_200)),
        color: "#e74c3c".into(),
        label: "DRC: spacing".into(),
    });
    let svg = layout::render_to_svg(&lib, top, &style);
    assert!(svg.contains("DRC: spacing"), "highlight label missing");
    assert!(svg.contains("#e74c3c"), "highlight color missing");
}
