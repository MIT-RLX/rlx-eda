//! End-to-end demo: drives every renderer eda-viz exposes from a
//! single `RcDivider` value, plus a synthetic transient waveform.
//!
//! Emits into `target/eda-viz-demo/`:
//!
//! - `divider_layout.svg` / `.png`     — layout with custom palette,
//!                                        tooltips, CSS classes,
//!                                        instance labels, and a
//!                                        DRC-style highlight overlay.
//! - `divider_layout.svgz`             — gzipped variant.
//! - `divider_schematic.svg` / `.png`  — derived from
//!                                        `RcDivider::schematic`
//!                                        (with Symbol→Pin→Net IR).
//! - `divider_waveform.svg` / `.png`   — synthetic V_in / V_out
//!                                        transient.
//!
//! Every artifact comes from the SAME `RcDivider` value, so any field
//! change (e.g. `r1.length = 20_000`) propagates to all three views.

use eda_hir::{Layout as _, Schematic as _};
use eda_viz::{layout, png::svg_to_png, schematic, waveform, Highlight, LayerPalette, Style};
use klayout_core::{Bbox, Point};
use spike_divider_block::{RcDemo, RcDivider, Resistor};

fn main() {
    let out = std::path::PathBuf::from("target/eda-viz-demo");
    std::fs::create_dir_all(&out).unwrap();

    // ── Single source of truth ───────────────────────────────────────
    let divider = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );

    // ── 1. Layout — full feature exercise ────────────────────────────
    let lib = RcDemo::new_library("rc_demo");
    let pdk = RcDemo::register(&lib);
    let top = divider.layout(&lib, &pdk);

    // Custom palette: foundry-canonical-ish colors keyed by GDS pair.
    let mut palette = LayerPalette::new();
    palette
        .set(50, 0, "#c0392b")  // RES — brick red
        .set(10, 0, "#2980b9")  // METAL1 — blue
        .set(20, 0, "#f39c12"); // VIA1 — amber

    let style = Style {
        layer_palette: Some(palette),
        tooltips: true,
        emit_css_classes: true,
        show_instance_labels: true,
        // Synthetic DRC-style annotation: pretend the routing wire's
        // elbow has a min-width violation. Bbox roughly covers the
        // L-bend corner where R1.b's via meets the routed strip.
        highlights: vec![Highlight {
            bbox: Bbox::new(Point::new(9_500, -200), Point::new(11_500, 1_500)),
            color: "#9b59b6".into(),
            label: "DRC: M1.W demo".into(),
        }],
        ..Style::default()
    };

    let layout_svg = layout::render_to_svg(&lib, top, &style);
    std::fs::write(out.join("divider_layout.svg"), &layout_svg).unwrap();
    std::fs::write(
        out.join("divider_layout.png"),
        svg_to_png(&layout_svg, 3.0).unwrap(),
    ).unwrap();
    eda_viz::write_svgz(&layout_svg, &out.join("divider_layout.svgz")).unwrap();

    // ── 2. Schematic — derived from the SAME RcDivider ───────────────
    let ir = divider.schematic(&pdk);
    let schem_svg = schematic::render_to_svg(
        &schematic::from_ir(&ir),
        &schematic::SchemStyle::default(),
    );
    std::fs::write(out.join("divider_schematic.svg"), &schem_svg).unwrap();
    std::fs::write(
        out.join("divider_schematic.png"),
        svg_to_png(&schem_svg, 3.0).unwrap(),
    ).unwrap();

    // ── 3. Waveform — synthetic transient that matches the divider ratio ─
    // V_out = V_in * R2 / (R1 + R2). With R1=1kΩ, R2=3kΩ, ratio = 0.75.
    use std::f64::consts::PI;
    let r1_ohm = 1000.0;
    let r2_ohm = 3000.0;
    let ratio = r2_ohm / (r1_ohm + r2_ohm);
    let v_in: Vec<(f64, f64)> = (0..400).map(|i| {
        let t = i as f64 * 5e-9;
        (t, 5.0 * (2.0 * PI * 1e6 * t).sin())
    }).collect();
    let v_out: Vec<(f64, f64)> = v_in.iter().map(|&(t, v)| (t, v * ratio as f64)).collect();

    let wave_svg = waveform::render_to_svg(
        &[
            waveform::Trace { label: "vin".into(),  points: v_in },
            waveform::Trace { label: "vout".into(), points: v_out },
        ],
        &waveform::WaveformStyle {
            title: Some("Divider transient response".into()),
            x_label: "t (s)".into(),
            y_label: "V".into(),
            x_start_zero: true,
            ..Default::default()
        },
    );
    std::fs::write(out.join("divider_waveform.svg"), &wave_svg).unwrap();
    std::fs::write(
        out.join("divider_waveform.png"),
        svg_to_png(&wave_svg, 2.0).unwrap(),
    ).unwrap();

    for f in [
        "divider_layout.svg", "divider_layout.png", "divider_layout.svgz",
        "divider_schematic.svg", "divider_schematic.png",
        "divider_waveform.svg", "divider_waveform.png",
    ] {
        println!("wrote {}", out.join(f).display());
    }
}
