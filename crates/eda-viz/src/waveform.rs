//! Waveform renderer — XY traces with axes, gridlines, and a legend.
//!
//! Built for time-domain simulation output (V vs t) but works for any
//! `[(x, y)]` data — Bode plots (frequency vs gain dB), DC sweeps
//! (Vin vs Vout), eye diagrams, …
//!
//! ## Design
//!
//! - Auto-scale axes from the union of all traces' bboxes, with a
//!   small padding fraction.
//! - Each trace gets a distinct palette color from
//!   [`crate::svg::palette_color`].
//! - Per-trace `<g>` so external CSS can restyle by class
//!   (`class="trace trace-{label}"`).
//! - Gridlines at "nice" round numbers (1, 2, 5 × 10^k).
//!
//! ## Example
//!
//! ```no_run
//! use eda_viz::waveform::{Trace, render_to_svg, WaveformStyle};
//!
//! let v_in  = (0..1000).map(|i| {
//!     let t = i as f64 * 1e-9;
//!     (t, 5.0 * (2.0 * std::f64::consts::PI * 1e6 * t).sin())
//! }).collect();
//! let v_out: Vec<(f64, f64)> = vec![]; // ... from sim
//!
//! let svg = render_to_svg(&[
//!     Trace { label: "vin".into(),  points: v_in },
//!     Trace { label: "vout".into(), points: v_out },
//! ], &WaveformStyle::default());
//! ```

use crate::svg::{palette_color, xml_escape, SvgDoc};

/// One labeled XY trace.
#[derive(Clone, Debug)]
pub struct Trace {
    pub label: String,
    pub points: Vec<(f64, f64)>,
}

/// Visual config for a waveform plot.
#[derive(Clone, Debug)]
pub struct WaveformStyle {
    /// Plot pixel dimensions (the SVG width/height).
    pub width: f64,
    pub height: f64,
    /// Margins around the data area: `[top, right, bottom, left]`.
    /// Bottom + left margins hold the axis tick labels; top holds the
    /// title; right holds nothing by default but reserved.
    pub margin: [f64; 4],
    pub background: Option<String>,
    pub axis_color: String,
    pub grid_color: String,
    pub label_color: String,
    pub stroke_width: f64,
    pub font_size: f64,
    pub title: Option<String>,
    pub x_label: String,
    pub y_label: String,
    /// If true, force x range to start at 0 even if data does not. Common
    /// for time-domain plots.
    pub x_start_zero: bool,
}

impl Default for WaveformStyle {
    fn default() -> Self {
        Self {
            width: 600.0,
            height: 320.0,
            margin: [30.0, 20.0, 40.0, 60.0],
            background: Some("white".into()),
            axis_color: "#222".into(),
            grid_color: "#ddd".into(),
            label_color: "#000".into(),
            stroke_width: 1.4,
            font_size: 11.0,
            title: None,
            x_label: "t".into(),
            y_label: "V".into(),
            x_start_zero: false,
        }
    }
}

/// Render `traces` to a complete SVG document.
pub fn render_to_svg(traces: &[Trace], style: &WaveformStyle) -> String {
    // 1. Auto-range across all traces.
    let (mut x_min, mut x_max, mut y_min, mut y_max) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    let mut have_any = false;
    for tr in traces {
        for &(x, y) in &tr.points {
            have_any = true;
            if x < x_min { x_min = x; }
            if x > x_max { x_max = x; }
            if y < y_min { y_min = y; }
            if y > y_max { y_max = y; }
        }
    }
    if !have_any {
        return SvgDoc::new(0.0, 0.0, style.width, style.height).finish();
    }
    if style.x_start_zero { x_min = x_min.min(0.0); }
    // Pad ranges by 5% so traces don't touch the axes.
    let x_pad = (x_max - x_min).abs() * 0.02;
    let y_pad = (y_max - y_min).abs() * 0.05;
    x_min -= x_pad; x_max += x_pad;
    y_min -= y_pad; y_max += y_pad;
    if (x_max - x_min).abs() < f64::EPSILON { x_max = x_min + 1.0; }
    if (y_max - y_min).abs() < f64::EPSILON { y_max = y_min + 1.0; }

    // 2. Layout.
    let [m_top, m_right, m_bot, m_left] = style.margin;
    let plot_w = (style.width - m_left - m_right).max(1.0);
    let plot_h = (style.height - m_top - m_bot).max(1.0);
    let plot_x0 = m_left;
    let plot_y0 = m_top;

    let mut doc = SvgDoc::new(0.0, 0.0, style.width, style.height);

    if let Some(bg) = &style.background {
        doc.rect(0.0, 0.0, style.width, style.height,
                 &format!("fill=\"{bg}\" stroke=\"none\""));
    }

    let to_px = |x: f64, y: f64| {
        let px = plot_x0 + (x - x_min) / (x_max - x_min) * plot_w;
        let py = plot_y0 + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h;
        (px, py)
    };

    // 3. Gridlines + axis labels.
    let x_ticks = nice_ticks(x_min, x_max, 6);
    let y_ticks = nice_ticks(y_min, y_max, 5);
    doc.group_open(&format!(
        "stroke=\"{}\" stroke-width=\"0.5\" fill=\"none\"", style.grid_color,
    ));
    for &x in &x_ticks {
        let (px, _) = to_px(x, y_min);
        doc.line(px, plot_y0, px, plot_y0 + plot_h, "");
    }
    for &y in &y_ticks {
        let (_, py) = to_px(x_min, y);
        doc.line(plot_x0, py, plot_x0 + plot_w, py, "");
    }
    doc.group_close();

    // 4. Axis box.
    doc.rect(
        plot_x0, plot_y0, plot_w, plot_h,
        &format!("fill=\"none\" stroke=\"{}\" stroke-width=\"1\"", style.axis_color),
    );

    // 5. Axis tick labels.
    let label_attrs = format!(
        "font-family=\"sans-serif\" font-size=\"{:.1}\" fill=\"{}\" stroke=\"none\"",
        style.font_size, style.label_color,
    );
    doc.group_open(&format!("{label_attrs} text-anchor=\"middle\""));
    for &x in &x_ticks {
        let (px, _) = to_px(x, y_min);
        doc.text(px, plot_y0 + plot_h + 14.0, "", &fmt_tick(x));
    }
    doc.group_close();
    doc.group_open(&format!("{label_attrs} text-anchor=\"end\" dominant-baseline=\"central\""));
    for &y in &y_ticks {
        let (_, py) = to_px(x_min, y);
        doc.text(plot_x0 - 6.0, py, "", &fmt_tick(y));
    }
    doc.group_close();

    // 6. Axis names.
    let axis_label_attrs = format!(
        "font-family=\"sans-serif\" font-size=\"{:.1}\" fill=\"{}\" \
         text-anchor=\"middle\" font-weight=\"bold\" stroke=\"none\"",
        style.font_size + 1.0, style.label_color,
    );
    doc.text(
        plot_x0 + plot_w * 0.5,
        plot_y0 + plot_h + 32.0,
        &axis_label_attrs,
        &style.x_label,
    );
    // y-label rotated 90° to the left of the plot.
    doc.raw(&format!(
        "<g transform=\"translate({:.1},{:.1}) rotate(-90)\">\
         <text {axis_label_attrs}>{}</text></g>",
        plot_x0 - 36.0, plot_y0 + plot_h * 0.5,
        xml_escape(&style.y_label),
    ));

    // 7. Title.
    if let Some(t) = &style.title {
        doc.text(
            style.width * 0.5, m_top * 0.6,
            &format!(
                "font-family=\"sans-serif\" font-size=\"{:.1}\" fill=\"{}\" \
                 text-anchor=\"middle\" font-weight=\"bold\" stroke=\"none\"",
                style.font_size + 3.0, style.label_color,
            ),
            t,
        );
    }

    // 8. Traces. Clip to the plot area so a trace going out of range
    // doesn't bleed onto axes.
    doc.raw(&format!(
        "<defs><clipPath id=\"plot-clip\"><rect x=\"{plot_x0:.1}\" y=\"{plot_y0:.1}\" \
         width=\"{plot_w:.1}\" height=\"{plot_h:.1}\"/></clipPath></defs>"
    ));
    doc.raw("<g clip-path=\"url(#plot-clip)\">");
    for (i, tr) in traces.iter().enumerate() {
        if tr.points.len() < 2 { continue; }
        let color = palette_color(i as u32);
        let pts: Vec<(f64, f64)> = tr.points.iter().map(|&(x, y)| to_px(x, y)).collect();
        doc.polyline(&pts, &format!(
            "fill=\"none\" stroke=\"{color}\" stroke-width=\"{:.2}\" \
             stroke-linejoin=\"round\" stroke-linecap=\"round\"",
            style.stroke_width,
        ));
    }
    doc.raw("</g>");

    // 9. Legend (top-right corner inside the plot area). White panel
    // behind the rows so traces near the legend don't bleed into the
    // text. Color swatch is on the LEFT of the label so it doesn't
    // pass through the text baseline.
    if !traces.is_empty() {
        let line_h = style.font_size + 6.0;
        let panel_w = traces.iter()
            .map(|t| t.label.chars().count())
            .max().unwrap_or(4) as f64
            * style.font_size * 0.6 + 32.0;
        let panel_h = (traces.len() as f64) * line_h + 6.0;
        let panel_x = plot_x0 + plot_w - panel_w - 6.0;
        let panel_y = plot_y0 + 6.0;
        doc.rect(
            panel_x, panel_y, panel_w, panel_h,
            "fill=\"white\" fill-opacity=\"0.85\" stroke=\"#aaa\" stroke-width=\"0.5\"",
        );
        for (i, tr) in traces.iter().enumerate() {
            let color = palette_color(i as u32);
            let row_y = panel_y + 3.0 + (i as f64 + 0.5) * line_h;
            // Swatch: short colored line, on the left side of the row.
            let sw_x1 = panel_x + 6.0;
            let sw_x2 = sw_x1 + 18.0;
            doc.line(sw_x1, row_y, sw_x2, row_y,
                &format!("stroke=\"{color}\" stroke-width=\"{:.2}\"", style.stroke_width));
            // Text: just to the right of the swatch.
            doc.text(
                sw_x2 + 5.0, row_y,
                &format!(
                    "font-family=\"sans-serif\" font-size=\"{:.1}\" fill=\"{}\" \
                     text-anchor=\"start\" dominant-baseline=\"central\" stroke=\"none\"",
                    style.font_size, style.label_color,
                ),
                &tr.label,
            );
        }
    }

    doc.finish()
}

/// Choose ~`target` "nice" tick positions in `[lo, hi]`.
/// Spacing is rounded to 1, 2, 2.5, or 5 × 10^k so labels stay short.
fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<f64> {
    if !(hi > lo) || target == 0 {
        return vec![lo, hi];
    }
    let raw = (hi - lo) / target as f64;
    let mag = 10f64.powi(raw.log10().floor() as i32);
    let normalized = raw / mag;
    let nice = if      normalized < 1.5 { 1.0 }
               else if normalized < 3.0 { 2.0 }
               else if normalized < 4.0 { 2.5 }
               else if normalized < 7.0 { 5.0 }
               else                     { 10.0 };
    let step = nice * mag;
    let first = (lo / step).ceil() * step;
    let mut t = first;
    let mut out = Vec::new();
    while t <= hi + step * 1e-9 {
        out.push(t);
        t += step;
    }
    out
}

fn fmt_tick(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".into()
    } else if abs < 1e-3 || abs >= 1e4 {
        format!("{v:.1e}")
    } else if abs < 1.0 {
        format!("{v:.3}")
    } else if abs < 100.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.0}")
    }
}
