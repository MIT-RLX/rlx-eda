//! End-to-end "report card" — every view of the divider in one image.
//!
//! Layout:
//!
//!   ┌──────────────────────────────────────────────────────┐
//!   │  Title                                               │
//!   ├───────────────────────┬──────────────────────────────┤
//!   │  Schematic            │  Layout (with DRC overlay)   │
//!   ├───────────────────────┴──────────────────────────────┤
//!   │  Transient waveform                                  │
//!   ├──────────────┬──────────────┬────────────────────────┤
//!   │ Performance  │ Accuracy     │ Parasitics             │
//!   ├──────────────┼──────────────┼────────────────────────┤
//!   │ DRC errors   │ Noise        │ LVS                    │
//!   └──────────────┴──────────────┴────────────────────────┘
//!
//! Composed by nesting each panel SVG via SVG's native `<svg>` element
//! nesting: each child keeps its own viewBox, the outer attributes
//! (`x`, `y`, `width`, `height`) place and size it. No string-replace
//! tricks beyond a tiny `extract_viewbox` / `extract_inner` pair.
//!
//! Numbers in the metrics panel are *computed* from the same
//! `RcDivider` value that drives layout + schematic + waveform — so
//! changing `R1.length` propagates to every cell of the report.

// ── Tiny math layout engine (inline; specific to this dashboard) ─────
//
// Builds a tree of [`MathExpr`] nodes, computes a (width, ascent,
// descent) bbox for each, then emits SVG with absolute positions for
// fractions and radicals. Inline parts (italic vars, subscripts) use
// `<tspan>` inside one `<text>`. Widths are estimated from character
// count × em-width per font style — good enough at the resolutions
// this dashboard renders.

#[derive(Clone)]
enum MathExpr {
    /// Plain text. `italic = true` for variable names, false for
    /// numbers and operators.
    T(String, bool),
    /// Row of expressions, laid out left-to-right on a shared baseline.
    Row(Vec<MathExpr>),
    /// Base + subscript (subscript shifts down-right, smaller font).
    Sub(Box<MathExpr>, Box<MathExpr>),
    /// Stacked fraction with horizontal bar.
    Frac(Box<MathExpr>, Box<MathExpr>),
    /// Square root: `√` with overbar across the radicand.
    Sqrt(Box<MathExpr>),
    /// Parenthesized group `(expr)`. Parens are non-italic.
    Paren(Box<MathExpr>),
}

fn t_var(s: &str) -> MathExpr { MathExpr::T(s.into(), true) }
fn t_op(s: &str) -> MathExpr { MathExpr::T(s.into(), false) }
fn row(parts: Vec<MathExpr>) -> MathExpr { MathExpr::Row(parts) }
fn sub(b: MathExpr, sub_: MathExpr) -> MathExpr { MathExpr::Sub(Box::new(b), Box::new(sub_)) }
fn frac(n: MathExpr, d: MathExpr) -> MathExpr { MathExpr::Frac(Box::new(n), Box::new(d)) }
fn sqrt_(e: MathExpr) -> MathExpr { MathExpr::Sqrt(Box::new(e)) }
fn paren(e: MathExpr) -> MathExpr { MathExpr::Paren(Box::new(e)) }

/// Approximate width of a single character at `em` size, calibrated
/// against **Latin Modern Math**'s OpenType advance widths.
///
/// LM Math's serif glyphs are slightly narrower overall than DejaVu
/// Sans, but italic letters (which are "math italic" — the LM Roman
/// italic forms) are wider for letters with descenders/swashes
/// (especially e, f, g, p, y).
///
/// Conservative overestimates (~5 %) keep `Frac`/`Sqrt` from landing
/// on the trailing character of an inline run.
fn char_em(c: char, italic: bool) -> f64 {
    match c {
        ' '  => 0.30,
        '·'  | '⋅' => 0.30,
        ','  | '.'  => 0.28,
        '−'  | '-'  => 0.60,
        '+'  | '='  => 0.60,
        '/'  | '|'  => 0.28,
        '('  | ')'  => 0.38,
        'i'  | 'l'  | 'I' | '!' | 'j' => 0.32,
        '√'  => 0.55,
        'Σ'  => 0.78,
        // Wide capitals (LM Math italic capitals are even wider than
        // DejaVu's; the math italic cap M/W/H have flowing strokes).
        'M' | 'W' => if italic { 0.96 } else { 0.88 },
        'A'..='Z' => if italic { 0.74 } else { 0.72 },
        // Lowercase. Math-italic lowercase is wider than text italic.
        'f' | 'g' | 'y' if italic => 0.60,
        'a'..='z' => if italic { 0.55 } else { 0.55 },
        '0'..='9' => 0.50,
        _ => 0.62,
    }
}
fn text_w(s: &str, italic: bool, em: f64) -> f64 {
    s.chars().map(|c| char_em(c, italic)).sum::<f64>() * em
}

impl MathExpr {
    /// True iff this expression is one-dimensional (no fraction bar
    /// or radical). Inline expressions can be fused into a single
    /// `<text>` element with `<tspan>` children, which keeps the
    /// font's native kerning and avoids the pixel-cracks you get
    /// when adjacent characters live in separate `<text>` elements.
    fn is_inline(&self) -> bool {
        match self {
            MathExpr::T(_, _) => true,
            MathExpr::Sub(b, s) => b.is_inline() && s.is_inline(),
            MathExpr::Row(parts) => parts.iter().all(|p| p.is_inline()),
            MathExpr::Paren(e) => e.is_inline(),
            MathExpr::Frac(_, _) | MathExpr::Sqrt(_) => false,
        }
    }
}

/// `(width, ascent, descent)` bbox of `e` at base font size `em`.
fn bbox(e: &MathExpr, em: f64) -> (f64, f64, f64) {
    match e {
        MathExpr::T(s, italic) => (text_w(s, *italic, em), em * 0.78, em * 0.22),
        MathExpr::Row(parts) => {
            let (mut w, mut a, mut d) = (0.0_f64, 0.0_f64, 0.0_f64);
            for p in parts {
                let (pw, pa, pd) = bbox(p, em);
                w += pw;
                if pa > a { a = pa; }
                if pd > d { d = pd; }
            }
            (w, a, d)
        }
        MathExpr::Sub(b, s) => {
            // Subscript is shifted DOWN by 0.30 em (its baseline);
            // reduces ascent contribution, increases descent.
            let (bw, ba, bd) = bbox(b, em);
            let sub_em = em * 0.72;
            let (sw, _, sd) = bbox(s, sub_em);
            (bw + sw, ba, bd.max(em * 0.30 + sd))
        }
        MathExpr::Frac(n, d) => {
            let (nw, na, nd) = bbox(n, em);
            let (dw, da, dd) = bbox(d, em);
            let w = nw.max(dw) + em * 0.40;
            // Numerator + bar above baseline, denominator below.
            // Math axis (bar position) sits 0.30 em above baseline.
            let a = na + nd + em * 0.20;       // num above bar
            let dep = da + dd + em * 0.20;     // den below bar
            (w, a, dep)
        }
        MathExpr::Sqrt(e) => {
            let (ew, ea, ed) = bbox(e, em);
            // Custom-drawn radical: tick (0.18) + diagonal (0.32) +
            // radicand + trailing pad (0.20).
            (ew + em * (0.18 + 0.32 + 0.20), ea + em * 0.18, ed)
        }
        MathExpr::Paren(e) => {
            let (ew, ea, ed) = bbox(e, em);
            (ew + em * 0.50, ea, ed)
        }
    }
}

/// Render `e` with baseline at `(x, y)` and base size `em`. Returns
/// the rendered width. Strategy: collapse contiguous inline subtrees
/// into a single `<text>` element so SVG's native kerning applies;
/// only Frac and Sqrt break the inline run.
fn render_math(out: &mut String, e: &MathExpr, x: f64, y: f64, em: f64) -> f64 {
    if e.is_inline() {
        emit_inline(out, e, x, y, em);
        bbox(e, em).0
    } else {
        match e {
            MathExpr::Row(parts) => {
                let mut cur = x;
                let mut buf: Vec<&MathExpr> = Vec::new();
                for p in parts {
                    if p.is_inline() {
                        buf.push(p);
                    } else {
                        if !buf.is_empty() {
                            cur += emit_inline_run(out, &buf, cur, y, em);
                            // Safety pad after an inline run, before a
                            // 2-D element (Frac / Sqrt). My
                            // per-character width estimates drift
                            // a few percent vs DejaVu Sans's actual
                            // metrics; cumulative drift over a
                            // multi-char run can be 3–4 px. 0.18 em
                            // is enough to absorb that without
                            // looking like a deliberate gap.
                            cur += em * 0.18;
                            buf.clear();
                        }
                        cur += render_math(out, p, cur, y, em);
                        cur += em * 0.18;
                    }
                }
                if !buf.is_empty() {
                    cur += emit_inline_run(out, &buf, cur, y, em);
                }
                cur - x
            }
            MathExpr::Frac(n, d) => render_frac(out, n, d, x, y, em),
            MathExpr::Sqrt(inner) => render_sqrt(out, inner, x, y, em),
            MathExpr::Paren(inner) => {
                // Tall parens not implemented; degrade to inline-style
                // parens around possibly-non-inline content. Rare in
                // our formulas.
                let lp = MathExpr::T("(".into(), false);
                let rp = MathExpr::T(")".into(), false);
                let mut cur = x;
                cur += render_math(out, &lp, cur, y, em);
                cur += render_math(out, inner, cur, y, em);
                cur += render_math(out, &rp, cur, y, em);
                cur - x
            }
            // Inline cases handled above.
            _ => unreachable!(),
        }
    }
}

/// Emit an inline expression as a single `<text>` element with
/// `<tspan>` children. Italic vars and subscripts become tspans.
fn emit_inline(out: &mut String, e: &MathExpr, x: f64, y: f64, em: f64) {
    out.push_str(&format!(
        "<text x=\"{x:.2}\" y=\"{y:.2}\" font-family=\"Latin Modern Math, DejaVu Sans, sans-serif\" \
         font-size=\"{em:.2}\" fill=\"#222\" xml:space=\"preserve\">"
    ));
    push_inline_tspans(out, e);
    out.push_str("</text>");
}

/// Same as [`emit_inline`] but for a Vec of inline parts (a slice of
/// a Row). Single `<text>` element wrapping all the tspans.
fn emit_inline_run(out: &mut String, parts: &[&MathExpr], x: f64, y: f64, em: f64) -> f64 {
    out.push_str(&format!(
        "<text x=\"{x:.2}\" y=\"{y:.2}\" font-family=\"Latin Modern Math, DejaVu Sans, sans-serif\" \
         font-size=\"{em:.2}\" fill=\"#222\" xml:space=\"preserve\">"
    ));
    let mut total_w = 0.0;
    for p in parts {
        push_inline_tspans(out, p);
        total_w += bbox(p, em).0;
    }
    out.push_str("</text>");
    total_w
}

/// Append `<tspan>` children for an inline expression to `out`. Caller
/// is responsible for the surrounding `<text>` wrapper.
fn push_inline_tspans(out: &mut String, e: &MathExpr) {
    match e {
        MathExpr::T(s, italic) => {
            if *italic {
                out.push_str(&format!(
                    "<tspan font-style=\"italic\">{}</tspan>",
                    xml_escape(s),
                ));
            } else {
                // Plain operator/number — no tspan needed but we use
                // one anyway so all inline pieces are addressable from
                // CSS if a stylesheet wants to recolor operators.
                out.push_str(&xml_escape(s));
            }
        }
        MathExpr::Row(parts) => {
            for p in parts {
                push_inline_tspans(out, p);
            }
        }
        MathExpr::Sub(b, s) => {
            push_inline_tspans(out, b);
            // SVG `baseline-shift="sub"` automatically lowers and the
            // 0.7em font shrinks the subscript. The renderer
            // (resvg/usvg) honors both for `<tspan>`.
            out.push_str("<tspan baseline-shift=\"sub\" font-size=\"0.72em\">");
            push_inline_tspans(out, s);
            out.push_str("</tspan>");
        }
        MathExpr::Paren(e) => {
            out.push_str("(");
            push_inline_tspans(out, e);
            out.push_str(")");
        }
        // Frac/Sqrt aren't inline; this function shouldn't see them.
        MathExpr::Frac(_, _) | MathExpr::Sqrt(_) => {}
    }
}

fn render_frac(
    out: &mut String,
    n: &MathExpr, d: &MathExpr,
    x: f64, y: f64, em: f64,
) -> f64 {
    let (nw, _na, _nd) = bbox(n, em);
    let (dw, da, _dd) = bbox(d, em);
    let bar_w = nw.max(dw) + em * 0.40;
    let bar_x0 = x;
    let bar_x1 = x + bar_w;
    // Math axis at 0.30 em above text baseline — matches surrounding
    // inline text's vertical center reasonably well.
    let bar_y = y - em * 0.30;

    // Numerator: place its baseline far enough above the bar that
    // the numerator's whole height (ascent) is comfortably above.
    let num_x = bar_x0 + (bar_w - nw) * 0.5;
    let num_baseline = bar_y - em * 0.20;
    render_math(out, n, num_x, num_baseline, em);

    // Bar.
    out.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{bar_y:.2}\" x2=\"{:.2}\" y2=\"{bar_y:.2}\" \
         stroke=\"#222\" stroke-width=\"0.9\" stroke-linecap=\"butt\"/>",
        bar_x0, bar_x1,
    ));

    // Denominator: baseline below the bar = ascent + small gap.
    let den_x = bar_x0 + (bar_w - dw) * 0.5;
    let den_baseline = bar_y + em * 0.05 + da;
    render_math(out, d, den_x, den_baseline, em);

    bar_w
}

fn render_sqrt(out: &mut String, inner: &MathExpr, x: f64, y: f64, em: f64) -> f64 {
    let (iw, ia, _id) = bbox(inner, em);

    // We DON'T use the Unicode √ glyph — its width / vertical extent
    // varies by font, which made the overbar attachment unreliable.
    // Instead, draw a tick + a diagonal as two `<line>`s and tack on
    // the overbar — a custom radical glyph whose dimensions we own.
    //
    //         ____________  overbar (horizontal)
    //        /
    //   \   /  diagonal (rising right)
    //    \ /
    //     v   tick (short down-left arm)
    //
    // Geometry: the tip of the radical sits at `(x + tick_w, y)`;
    // the diagonal rises from there to `(x + tick_w + diag_w, bar_y)`;
    // the overbar continues from there to `bar_x1`.
    let tick_w = em * 0.18;
    let diag_w = em * 0.32;
    let bar_y = y - ia - em * 0.06;
    let bar_x0 = x + tick_w + diag_w;
    let bar_x1 = bar_x0 + iw + em * 0.18;
    let stroke = "stroke=\"#222\" stroke-width=\"1.1\" stroke-linecap=\"round\" \
                  stroke-linejoin=\"round\" fill=\"none\"";

    // Single polyline: tick → tip → diagonal → overbar.
    out.push_str(&format!(
        "<polyline points=\"{:.2},{:.2} {:.2},{:.2} {:.2},{:.2} {:.2},{:.2}\" {stroke}/>",
        x,                       y - em * 0.20,
        x + tick_w,              y,
        bar_x0,                  bar_y,
        bar_x1,                  bar_y,
    ));

    // Radicand starts just after the diagonal, with a small gap.
    render_math(out, inner, bar_x0 + em * 0.06, y, em);
    tick_w + diag_w + iw + em * 0.20
}

use eda_hir::{Layout as _, Schematic as _};
use eda_viz::{layout, png::svg_to_png, schematic, waveform, Highlight, LayerPalette, Style};
use klayout_connect::{extract_hierarchical, Conductor, ExtractConfig};
use klayout_core::{Bbox, Point};
use spike_divider_block::{length_to_resistance, pdks::Sky130Lite, RcDemo, RcDivider, Resistor};

const W: f64 = 1280.0;
const H: f64 = 1620.0;

const PANEL_BG: &str = "#fafafa";
const PANEL_BORDER: &str = "#bbb";
const HEADING: &str = "#111";
const SUBHEADING: &str = "#555";
const ACCENT_OK: &str = "#27ae60";
const ACCENT_WARN: &str = "#f39c12";

fn main() {
    let out = std::path::PathBuf::from("target/eda-viz-demo");
    std::fs::create_dir_all(&out).unwrap();

    // ── Single source of truth ───────────────────────────────────────
    let r1_len = 10_000_i64;
    let r2_len = 30_000_i64;
    let divider = RcDivider::new(
        Resistor { length: r1_len, id: "R1".into() },
        Resistor { length: r2_len, id: "R2".into() },
    );
    let v_supply = 5.0_f64;

    // ── Render the three child panels ────────────────────────────────
    let lib = RcDemo::new_library("rc_demo");
    let pdk = RcDemo::register(&lib);
    let top = divider.layout(&lib, &pdk);

    let mut palette = LayerPalette::new();
    palette
        .set(50, 0, "#c0392b")
        .set(10, 0, "#2980b9")
        .set(20, 0, "#f39c12");

    let layout_style = Style {
        layer_palette: Some(palette),
        show_instance_labels: true,
        highlights: vec![Highlight {
            bbox: Bbox::new(Point::new(9_500, -200), Point::new(11_500, 1_500)),
            color: "#9b59b6".into(),
            label: "DRC: M1.W demo".into(),
        }],
        ..Style::default()
    };
    let layout_svg = layout::render_to_svg(&lib, top, &layout_style);

    let ir = divider.schematic(&pdk);
    let schem_svg = schematic::render_to_svg(
        &schematic::from_ir(&ir),
        &schematic::SchemStyle::default(),
    );

    // Transient: drive a 1 MHz sinewave at v_supply through the
    // closed-form divider. Real users would feed sim output here.
    let r1_ohm = length_to_resistance(r1_len) as f64;
    let r2_ohm = length_to_resistance(r2_len) as f64;
    let ratio = r2_ohm / (r1_ohm + r2_ohm);
    let v_in: Vec<(f64, f64)> = (0..400).map(|i| {
        let t = i as f64 * 5e-9;
        (t, v_supply * (2.0 * std::f64::consts::PI * 1e6 * t).sin())
    }).collect();
    let v_out: Vec<(f64, f64)> = v_in.iter().map(|&(t, v)| (t, v * ratio)).collect();
    let wave_svg = waveform::render_to_svg(
        &[
            waveform::Trace { label: "vin".into(),  points: v_in.clone() },
            waveform::Trace { label: "vout".into(), points: v_out.clone() },
        ],
        &waveform::WaveformStyle {
            title: Some("Transient response (V vs t)".into()),
            x_label: "t (s)".into(),
            y_label: "V".into(),
            x_start_zero: true,
            ..Default::default()
        },
    );

    // ── Compute metrics ──────────────────────────────────────────────
    let r_eq = r1_ohm * r2_ohm / (r1_ohm + r2_ohm);
    let c_load_pf = 1.0_f64; // assumed pad capacitance
    let bw_hz = 1.0 / (2.0 * std::f64::consts::PI * r_eq * c_load_pf * 1e-12);

    let target = v_supply * ratio;
    let actual = target; // closed-form, no error in MVP
    let err_pct = ((actual - target).abs() / target.max(1e-12)) * 100.0;

    // Crude parasitic estimate: METAL1 sheet rho ≈ 0.05 Ω/sq, wire
    // length ≈ gap_x in DBU = 5 µm, width = pad/2. Pad cap ≈ 0.5 fF.
    let r_wire_ohm = 0.05 * 5.0; // 0.25 Ω, roughly
    let c_pad_ff = 0.5;

    let drc_violations = layout_style.highlights.len();

    // Thermal voltage noise at vout: √(4·kB·T·R_eq) [V/√Hz].
    let kb = 1.380649e-23_f64;
    let t_kelvin = 300.0_f64;
    let v_n_per_rthz = (4.0 * kb * t_kelvin * r_eq).sqrt();
    let v_n_per_nv = v_n_per_rthz * 1e9; // nV/√Hz
    let bw_for_total = 1e6_f64; // RMS over 1 MHz BW
    let v_n_total_uv = v_n_per_rthz * bw_for_total.sqrt() * 1e6; // µV RMS

    // LVS: extract netlist, count nets, compare to schematic IR.
    let nl = extract_hierarchical(&lib, top, &ExtractConfig {
        conductors: vec![Conductor { layer: pdk.METAL1, label_layer: pdk.METAL1 }],
        vias: vec![],
    });
    let layout_nets = nl.nets().len();
    let schem_nets = ["vin", "vout", "gnd"].len();
    let lvs_pass = layout_nets == schem_nets;

    // ── Compose dashboard SVG ────────────────────────────────────────
    let mut s = String::new();
    s.push_str(&format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <svg xmlns=\"http://www.w3.org/2000/svg\" \
         viewBox=\"0 0 {W} {H}\" width=\"{W}\" height=\"{H}\">"
    ));

    // Background.
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{W}\" height=\"{H}\" fill=\"white\"/>"
    ));

    // Title bar.
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{W}\" height=\"60\" fill=\"#2c3e50\"/>\
         <text x=\"30\" y=\"38\" font-family=\"sans-serif\" font-size=\"22\" \
         font-weight=\"bold\" fill=\"white\">RC Voltage Divider — Design Report</text>\
         <text x=\"{}\" y=\"38\" font-family=\"sans-serif\" font-size=\"13\" \
         fill=\"#bdc3c7\" text-anchor=\"end\">R1={:.0} Ω · R2={:.0} Ω · V={:.1} V</text>",
        W - 30.0, r1_ohm, r2_ohm, v_supply,
    ));

    // Row 1: schematic (left) + layout (right). y=80, h=460.
    panel_frame(&mut s, 20.0, 80.0, 480.0, 460.0, "Schematic");
    embed_panel(&mut s, &schem_svg, 30.0, 110.0, 460.0, 420.0);

    panel_frame(&mut s, 520.0, 80.0, 740.0, 460.0, "Layout (with DRC overlay)");
    embed_panel(&mut s, &layout_svg, 530.0, 110.0, 720.0, 420.0);

    // Row 2: waveform full width. y=560, h=340.
    panel_frame(&mut s, 20.0, 560.0, 1240.0, 340.0, "Simulation");
    embed_panel(&mut s, &wave_svg, 30.0, 590.0, 1220.0, 300.0);

    // Row 3: 3×2 metrics grid. Each cell now carries formula +
    // substitution + conclusion lines so a reader can audit how the
    // big number was reached.
    let mx = 20.0;
    let my = 920.0;
    let cell_w = (W - 2.0 * mx) / 3.0;
    let cell_h = 340.0;

    let cells = [
        Metric {
            col: 0, row: 0,
            title: "Performance".into(),
            big: format!("{:.1} MHz", bw_hz / 1e6),
            // BW = 1 / (2π · R_eq · C_load)
            formula: row(vec![
                t_var("BW"), t_op(" = "),
                frac(
                    t_op("1"),
                    row(vec![
                        t_op("2π · "), sub(t_var("R"), t_var("eq")), t_op(" · "),
                        sub(t_var("C"), t_var("load")),
                    ]),
                ),
            ]),
            calc: format!(
                "R_eq = R1‖R2 = {r1:.0}·{r2:.0}/({r1:.0}+{r2:.0}) = {req:.0} Ω\n\
                 BW   = 1 / (2π · {req:.0} · {c} pF) = {bw:.1} MHz",
                r1 = r1_ohm, r2 = r2_ohm, req = r_eq,
                c = c_load_pf, bw = bw_hz / 1e6,
            ),
            note: format!("First-order RC pole at the divider tap. C_load is assumed ({} pF pad cap).", c_load_pf),
            color: ACCENT_OK.into(),
        },
        Metric {
            col: 1, row: 0,
            title: "Accuracy".into(),
            big: format!("{:.2} V", actual),
            // V_out = V_in · R2 / (R1 + R2)
            formula: row(vec![
                sub(t_var("V"), t_var("out")), t_op(" = "),
                sub(t_var("V"), t_var("in")), t_op(" · "),
                frac(
                    sub(t_var("R"), t_op("2")),
                    row(vec![
                        sub(t_var("R"), t_op("1")), t_op(" + "),
                        sub(t_var("R"), t_op("2")),
                    ]),
                ),
            ]),
            calc: format!(
                "V_out = {vin:.1} · {r2:.0}/({r1:.0}+{r2:.0}) = {actual:.3} V\n\
                 err   = |V_out − target| / target = {err:.2} %",
                vin = v_supply, r1 = r1_ohm, r2 = r2_ohm,
                actual = actual, err = err_pct,
            ),
            note: format!(
                "Closed-form DC; target taken as ideal divider ratio (no input loading, no R tolerance)."
            ),
            color: ACCENT_OK.into(),
        },
        Metric {
            col: 2, row: 0,
            title: "Parasitics".into(),
            big: format!("{:.2} Ω · {:.1} fF", r_wire_ohm, c_pad_ff),
            // R_wire = ρ_s · ℓ/w   ;   C_pad = ε_0·ε_r·A/d
            // (Sheet resistivity convention: ρ_s, not ρ_□. The square
            // glyph used to render as a missing-glyph box in some
            // configurations of DejaVu Sans.)
            formula: row(vec![
                sub(t_var("R"), t_var("wire")), t_op(" = "),
                sub(t_var("ρ"), t_var("s")), t_op(" · "),
                frac(t_var("ℓ"), t_var("w")),
                t_op("    "),
                sub(t_var("C"), t_var("pad")), t_op(" = "),
                frac(
                    row(vec![sub(t_var("ε"), t_op("0")), t_op(" · "),
                              sub(t_var("ε"), t_var("r")), t_op(" · "), t_var("A")]),
                    t_var("d"),
                ),
            ]),
            calc: format!(
                "ρ_□ ≈ 0.05 Ω/sq, ℓ ≈ 5 µm (gap_x), w ≈ pad/2\n\
                 R_wire ≈ 0.05 · 5/1 ≈ {rw:.2} Ω\n\
                 C_pad assumed {cpf:.1} fF (1 µm² M1-to-substrate)",
                rw = r_wire_ohm, cpf = c_pad_ff,
            ),
            note: "Estimates only — real values come from extraction (PEX) on a finished layout.".into(),
            color: ACCENT_OK.into(),
        },
        Metric {
            col: 0, row: 1,
            title: "DRC".into(),
            big: format!("{} violation{}",
                drc_violations,
                if drc_violations == 1 { "" } else { "s" }),
            // violations = Σ_{rule r} |result_region(r)|
            formula: row(vec![
                t_var("violations"), t_op(" = "),
                t_op("Σ "),
                t_op("|"), t_var("result"), t_op("("), t_var("r"), t_op(")|"),
            ]),
            calc: format!(
                "Style::highlights carries {n} entr{y}\n\
                 (synthetic 'M1.W demo' overlay; no real DRC run wired here yet)",
                n = drc_violations,
                y = if drc_violations == 1 { "y" } else { "ies" },
            ),
            note: "Replace the synthetic highlight with klayout-drc rule output via highlights_from_drc().".into(),
            color: if drc_violations == 0 { ACCENT_OK.into() } else { ACCENT_WARN.into() },
        },
        Metric {
            col: 1, row: 1,
            title: "Noise".into(),
            big: format!("{:.1} nV/√Hz", v_n_per_nv),
            // v_n = √(4·k_B·T·R_eq)   v_n,RMS = v_n·√BW
            formula: row(vec![
                sub(t_var("v"), t_var("n")), t_op(" = "),
                sqrt_(row(vec![
                    t_op("4 · "), sub(t_var("k"), t_var("B")), t_op(" · "),
                    t_var("T"), t_op(" · "), sub(t_var("R"), t_var("eq")),
                ])),
                t_op("   "),
                sub(t_var("v"), t_var("n,RMS")), t_op(" = "),
                sub(t_var("v"), t_var("n")), t_op(" · "),
                sqrt_(t_var("BW")),
            ]),
            calc: format!(
                "T = {tk:.0} K, k_B = 1.381e−23 J/K, R_eq = {req:.0} Ω\n\
                 v_n = √(4 · 1.381e−23 · {tk:.0} · {req:.0}) = {vn:.1} nV/√Hz\n\
                 v_n,RMS over {bwm:.0} MHz = {vn:.1} nV/√Hz · √({bw_hz:.0e} Hz) = {tot:.1} µV",
                tk = t_kelvin, req = r_eq, vn = v_n_per_nv,
                bwm = bw_for_total / 1e6, bw_hz = bw_for_total,
                tot = v_n_total_uv,
            ),
            note: "Johnson noise at the open-circuit divider tap; ignores op-amp / sense-amp loading.".into(),
            color: ACCENT_OK.into(),
        },
        Metric {
            col: 2, row: 1,
            title: "LVS".into(),
            big: (if lvs_pass { "PASS" } else { "FAIL" }).to_string(),
            // |layout_nets| = |schematic_nets|  AND  pin_nets agree
            formula: row(vec![
                t_op("| "), sub(t_var("nets"), t_var("layout")), t_op(" |"),
                t_op(" = "),
                t_op("| "), sub(t_var("nets"), t_var("schem")), t_op(" |"),
            ]),
            calc: format!(
                "Layout: extract_hierarchical(METAL1) → {layn} nets\n\
                 Schematic: pin_nets distinct labels → {{vin, vout, gnd}} = {sn} nets\n\
                 Counts match → PASS (full LVS would also match per-net pin sets)",
                layn = layout_nets, sn = schem_nets,
            ),
            note: "MVP LVS = net-count match. Real LVS compares per-pin net assignments via SchemSymbol::pin_nets.".into(),
            color: if lvs_pass { ACCENT_OK.into() } else { String::from("#e74c3c") },
        },
    ];
    for m in &cells {
        let cx = mx + m.col as f64 * cell_w;
        let cy = my + m.row as f64 * cell_h;
        metric_cell(&mut s,
            cx + 6.0, cy + 6.0,
            cell_w - 12.0, cell_h - 12.0,
            m,
        );
    }

    s.push_str("</svg>");

    std::fs::write(out.join("divider_dashboard.svg"), &s).unwrap();
    std::fs::write(
        out.join("divider_dashboard.png"),
        svg_to_png(&s, 2.0).unwrap(),
    ).unwrap();

    // ── Foundry-PDK variant: Sky130Lite ─────────────────────────────
    //
    // Same divider value, run through `Sky130Lite` instead of
    // `RcDemo`. This exercises the renderer against real foundry
    // GDS pairs (POLY 66/20 for the resistor body, MET1 68/20 for
    // routing, mcon 67/44 for vias) and confirms the .lyp-driven
    // palette plumbing works on a non-toy layer set.
    let sky_lib = Sky130Lite::new_library("sky130_demo");
    let sky_pdk = Sky130Lite::register(&sky_lib);
    let sky_top = divider.layout(&sky_lib, &sky_pdk);

    // Foundry-canonical Sky130 colors — picked to match the typical
    // KLayout `.lyp` defaults users see when opening a Sky130 GDS.
    let mut sky_palette = LayerPalette::new();
    sky_palette
        .set(66, 20, "#e74c3c")  // poly (resistor body)
        .set(68, 20, "#9b59b6")  // mcon
        .set(68, 44, "#9b59b6")  // mcon (alt datatype)
        .set(67, 20, "#3498db"); // li1 / metal1

    let sky_style = Style {
        layer_palette: Some(sky_palette),
        show_instance_labels: true,
        ..Style::default()
    };
    let sky_layout_svg = layout::render_to_svg(&sky_lib, sky_top, &sky_style);
    std::fs::write(out.join("divider_sky130_layout.svg"), &sky_layout_svg).unwrap();
    std::fs::write(
        out.join("divider_sky130_layout.png"),
        svg_to_png(&sky_layout_svg, 3.0).unwrap(),
    ).unwrap();
    println!("wrote {}", out.join("divider_sky130_layout.svg").display());
    println!("wrote {}", out.join("divider_sky130_layout.png").display());

    // ── Sky130 dashboard (re-uses everything else from the RcDemo
    //    variant above; only the layout panel changes). ──────────────
    let mut s_sky = String::new();
    s_sky.push_str(&format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <svg xmlns=\"http://www.w3.org/2000/svg\" \
         viewBox=\"0 0 {W} {H}\" width=\"{W}\" height=\"{H}\">\
         <rect x=\"0\" y=\"0\" width=\"{W}\" height=\"{H}\" fill=\"white\"/>\
         <rect x=\"0\" y=\"0\" width=\"{W}\" height=\"60\" fill=\"#34495e\"/>\
         <text x=\"30\" y=\"38\" font-family=\"sans-serif\" font-size=\"22\" \
         font-weight=\"bold\" fill=\"white\">RC Voltage Divider — Sky130 PDK</text>\
         <text x=\"{:.0}\" y=\"38\" font-family=\"sans-serif\" font-size=\"13\" \
         fill=\"#bdc3c7\" text-anchor=\"end\">Sky130Lite · poly resistor · {:.0} Ω + {:.0} Ω · {:.1} V</text>",
        W - 30.0, r1_ohm, r2_ohm, v_supply,
    ));
    panel_frame(&mut s_sky, 20.0, 80.0, 480.0, 460.0, "Schematic");
    embed_panel(&mut s_sky, &schem_svg, 30.0, 110.0, 460.0, 420.0);
    panel_frame(&mut s_sky, 520.0, 80.0, 740.0, 460.0,
        "Layout — Sky130Lite (poly + li1 + mcon)");
    embed_panel(&mut s_sky, &sky_layout_svg, 530.0, 110.0, 720.0, 420.0);
    panel_frame(&mut s_sky, 20.0, 560.0, 1240.0, 340.0, "Simulation");
    embed_panel(&mut s_sky, &wave_svg, 30.0, 590.0, 1220.0, 300.0);
    // Reuse the same metrics — they're computed from R1/R2 ohms,
    // which don't depend on the PDK choice.
    for m in &cells {
        let cx = mx + m.col as f64 * cell_w;
        let cy = my + m.row as f64 * cell_h;
        metric_cell(&mut s_sky,
            cx + 6.0, cy + 6.0,
            cell_w - 12.0, cell_h - 12.0,
            m,
        );
    }
    s_sky.push_str("</svg>");
    std::fs::write(out.join("divider_sky130_dashboard.svg"), &s_sky).unwrap();
    std::fs::write(
        out.join("divider_sky130_dashboard.png"),
        svg_to_png(&s_sky, 2.0).unwrap(),
    ).unwrap();
    println!("wrote {}", out.join("divider_sky130_dashboard.svg").display());
    println!("wrote {}", out.join("divider_sky130_dashboard.png").display());

    // Also emit a tight crop of the metrics row, rendered at 3× so
    // the formulas are readable at thumbnail size.
    let crop_y = my - 6.0;
    let crop_h = cell_h * 2.0 + 12.0;
    let crop_svg = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <svg xmlns=\"http://www.w3.org/2000/svg\" \
              viewBox=\"0 {crop_y} {W} {crop_h}\" \
              width=\"{W}\" height=\"{crop_h}\">\
              <rect x=\"0\" y=\"{crop_y}\" width=\"{W}\" height=\"{crop_h}\" fill=\"white\"/>\
              {body}\
         </svg>",
        body = &s[s.find("<rect x=\"20\"").unwrap_or(0)..s.rfind("</svg>").unwrap_or(s.len())],
    );
    std::fs::write(out.join("metrics_crop.svg"), &crop_svg).unwrap();
    std::fs::write(
        out.join("metrics_crop.png"),
        svg_to_png(&crop_svg, 2.5).unwrap(),
    ).unwrap();
    println!("wrote {}", out.join("divider_dashboard.svg").display());
    println!("wrote {}", out.join("divider_dashboard.png").display());
}

/// Empty bordered panel + a small header bar with the panel title.
fn panel_frame(s: &mut String, x: f64, y: f64, w: f64, h: f64, title: &str) {
    s.push_str(&format!(
        "<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" \
         fill=\"{PANEL_BG}\" stroke=\"{PANEL_BORDER}\" stroke-width=\"1\"/>\
         <rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"24\" \
         fill=\"#34495e\"/>\
         <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" \
         font-size=\"13\" font-weight=\"bold\" fill=\"white\">{}</text>",
        x + 10.0, y + 17.0, xml_escape(title),
    ));
}

struct Metric {
    col: usize,
    row: usize,
    title: String,
    big: String,
    /// Symbolic formula — built as a `MathExpr` tree so the renderer
    /// can produce real fractions, italic vars, and proper √.
    formula: MathExpr,
    /// Concrete substitution. Multi-line; `\n` becomes a new SVG row.
    calc: String,
    /// Short caveat / interpretation.
    note: String,
    color: String,
}

/// Render a single metric cell.
///
/// Layout:
///
///   ┌───────────────────────── color strip ─┐
///   │ TITLE                                  │
///   │                                        │
///   │  BIG VALUE                             │
///   │                                        │
///   │  Formula:  v_n = √(4·k_B·T·R_eq) ...   │
///   │  Calc:     T = 300 K, R_eq = 750 Ω    │
///   │            v_n = √(...) = 3.5 nV/√Hz   │
///   │                                        │
///   │  *Note italicized*                     │
///   └────────────────────────────────────────┘
fn metric_cell(s: &mut String, x: f64, y: f64, w: f64, h: f64, m: &Metric) {
    s.push_str(&format!(
        "<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" \
         fill=\"white\" stroke=\"{PANEL_BORDER}\" stroke-width=\"1\" rx=\"4\"/>\
         <rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"6\" fill=\"{c}\" rx=\"4\"/>",
        c = m.color,
    ));

    // Title.
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"13\" \
         font-weight=\"bold\" fill=\"{HEADING}\">{}</text>",
        x + 14.0, y + 30.0, xml_escape(&m.title),
    ));

    // Big value.
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"30\" \
         font-weight=\"bold\" fill=\"{}\">{}</text>",
        x + 14.0, y + 76.0, m.color, xml_escape(&m.big),
    ));

    // Formula label + math-typeset body.
    let mut yy = y + 110.0;
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"10\" \
         font-weight=\"bold\" fill=\"{SUBHEADING}\">FORMULA</text>",
        x + 14.0, yy,
    ));
    let math_em = 14.5;
    let (_, ascent, descent) = bbox(&m.formula, math_em);
    // Place the formula's baseline so its top stays clear of the
    // "FORMULA" header above and its descent doesn't bleed into the
    // CALCULATION block below.
    let math_baseline = yy + ascent + 6.0;
    render_math(s, &m.formula, x + 14.0, math_baseline, math_em);
    yy = math_baseline + descent + 10.0;

    // Calc label + multi-line body.
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"10\" \
         font-weight=\"bold\" fill=\"{SUBHEADING}\">CALCULATION</text>",
        x + 14.0, yy,
    ));
    yy += 16.0;
    for line in m.calc.lines() {
        s.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"DejaVu Sans Mono, monospace\" \
             font-size=\"10.5\" fill=\"#222\">{}</text>",
            x + 14.0, yy, xml_escape(line),
        ));
        yy += 14.0;
    }
    yy += 6.0;

    // Note.
    let note_lines = wrap_text(&m.note, 56);
    for line in &note_lines {
        s.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"10.5\" \
             font-style=\"italic\" fill=\"{SUBHEADING}\">{}</text>",
            x + 14.0, yy, xml_escape(line),
        ));
        yy += 13.0;
    }
}

/// Greedy wrap: pack words into lines up to `max_chars`. Doesn't
/// break mid-word; only useful for short, well-controlled strings.
fn wrap_text(s: &str, max_chars: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        if cur.is_empty() {
            cur.push_str(word);
        } else if cur.chars().count() + 1 + word.chars().count() <= max_chars {
            cur.push(' ');
            cur.push_str(word);
        } else {
            out.push(std::mem::take(&mut cur));
            cur.push_str(word);
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

/// Embed a panel SVG into the master document at a given slot. The
/// child SVG keeps its own viewBox; the outer attributes resize it.
fn embed_panel(s: &mut String, panel_svg: &str, x: f64, y: f64, w: f64, h: f64) {
    let viewbox = extract_viewbox(panel_svg).unwrap_or("0 0 100 100");
    let inner = extract_inner(panel_svg);
    s.push_str(&format!(
        "<svg x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" viewBox=\"{viewbox}\" \
         preserveAspectRatio=\"xMidYMid meet\">{inner}</svg>"
    ));
}

/// Pull the value of the first `viewBox=` attribute in `svg`.
fn extract_viewbox(svg: &str) -> Option<&str> {
    let i = svg.find("viewBox=\"")? + "viewBox=\"".len();
    let j = i + svg[i..].find('"')?;
    Some(&svg[i..j])
}

/// Slice out the inner body of `svg` (everything between `<svg ...>`
/// and `</svg>`).
fn extract_inner(svg: &str) -> &str {
    let svg_start = svg.find("<svg ").unwrap_or(0);
    let body_start = svg_start + 5 + svg[svg_start + 5..].find('>').unwrap_or(0) + 1;
    let body_end = svg.rfind("</svg>").unwrap_or(svg.len());
    &svg[body_start..body_end]
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<'  => out.push_str("&lt;"),
            '>'  => out.push_str("&gt;"),
            '&'  => out.push_str("&amp;"),
            '"'  => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _    => out.push(c),
        }
    }
    out
}
