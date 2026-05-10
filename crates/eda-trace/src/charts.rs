//! Declarative chart specs + a self-contained line-chart SVG
//! renderer. The harness owns this rather than calling into
//! `eda-viz::waveform` because every existing trace bin needed a
//! log-y toggle that `waveform` doesn't expose, plus consistent
//! gridlines / legend styling.

use crate::trace::Trace;

/// Chart kind. Most spikes use `Line`; `BeforeAfterOverlay` covers
/// the "spectrum before / spectrum after Adam" panel that
/// `mzi_ml_trace` made canonical.
#[derive(Clone, Debug)]
pub enum ChartKind {
    /// Single panel, one or more lines plotted against the same x.
    /// X is the step counter unless [`ChartSpec::with_x_series`]
    /// switches it to a named series.
    Line,
    /// Two precomputed sweep arrays drawn on the same axes — used
    /// for "before / after" comparisons where the x-axis is some
    /// physical quantity (wavelength, frequency, voltage), not the
    /// step counter.
    BeforeAfter {
        x_label: String,
        before_label: String,
        before: Vec<(f64, f64)>,
        after_label: String,
        after: Vec<(f64, f64)>,
    },
}

/// One Y-axis series. `series_name` keys into the [`Trace`]'s row
/// map; `label` is what shows up in the chart legend.
#[derive(Clone, Debug)]
pub struct YSeries {
    pub series_name: String,
    pub label: String,
    /// SVG color string. `None` picks from a default rotation.
    pub color: Option<String>,
}

/// Declarative chart spec. The harness renders each one to
/// `<assets>/<file_slug>.svg` (and `.png` if the `png` feature is on).
#[derive(Clone, Debug)]
pub struct ChartSpec {
    /// Output filename without extension; also the in-report image
    /// reference slug.
    pub file_slug: String,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    /// What's on the X axis. `None` means the trace step counter.
    /// `Some(name)` uses a named series instead — useful for
    /// "param vs param" scatter / parametric plots.
    pub x_series: Option<String>,
    pub y_series: Vec<YSeries>,
    pub log_y: bool,
    pub kind: ChartKind,
}

impl ChartSpec {
    pub fn line(
        file_slug: impl Into<String>,
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        Self {
            file_slug: file_slug.into(),
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            x_series: None,
            y_series: Vec::new(),
            log_y: false,
            kind: ChartKind::Line,
        }
    }

    pub fn add_series(
        mut self,
        series_name: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        self.y_series.push(YSeries {
            series_name: series_name.into(),
            label: label.into(),
            color: None,
        });
        self
    }

    pub fn add_colored_series(
        mut self,
        series_name: impl Into<String>,
        label: impl Into<String>,
        color: impl Into<String>,
    ) -> Self {
        self.y_series.push(YSeries {
            series_name: series_name.into(),
            label: label.into(),
            color: Some(color.into()),
        });
        self
    }

    pub fn with_y_log(mut self, log_y: bool) -> Self {
        self.log_y = log_y;
        self
    }

    pub fn with_x_series(mut self, name: impl Into<String>) -> Self {
        self.x_series = Some(name.into());
        self
    }

    pub fn before_after(
        file_slug: impl Into<String>,
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
        before_label: impl Into<String>,
        before: Vec<(f64, f64)>,
        after_label: impl Into<String>,
        after: Vec<(f64, f64)>,
    ) -> Self {
        let x_label_s: String = x_label.into();
        Self {
            file_slug: file_slug.into(),
            title: title.into(),
            x_label: x_label_s.clone(),
            y_label: y_label.into(),
            x_series: None,
            y_series: Vec::new(),
            log_y: false,
            kind: ChartKind::BeforeAfter {
                x_label: x_label_s,
                before_label: before_label.into(),
                before,
                after_label: after_label.into(),
                after,
            },
        }
    }
}

// ── Rendering ──────────────────────────────────────────────────────────
//
// One `render_chart_svg(spec, trace) -> String` entry point. The
// renderer is self-contained — no external SVG library — because the
// surface we need (axes, gridlines, polylines, legend, log-y) is
// small and the existing per-bin renderers were already reinventing
// it. Output dimensions match the de-facto standard from
// `mzi_ml_trace`: 920 × 480 with a 60 / 30 / 50 / 60 margin.

const W: f64 = 920.0;
const H: f64 = 480.0;
const M_TOP: f64 = 30.0;
const M_RIGHT: f64 = 30.0;
const M_BOTTOM: f64 = 60.0;
const M_LEFT: f64 = 80.0;
const PLOT_W: f64 = W - M_LEFT - M_RIGHT;
const PLOT_H: f64 = H - M_TOP - M_BOTTOM;

const DEFAULT_PALETTE: &[&str] = &[
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#ff7f0e", "#8c564b", "#e377c2", "#17becf",
];

/// Render one [`ChartSpec`] to a self-contained SVG string. The
/// caller decides what to do with it (write to disk, embed in HTML,
/// rasterize via the optional `png` helper).
pub fn render_chart_svg(spec: &ChartSpec, trace: &Trace) -> String {
    match &spec.kind {
        ChartKind::Line => render_line(spec, trace),
        ChartKind::BeforeAfter { .. } => render_before_after(spec),
    }
}

fn render_line(spec: &ChartSpec, trace: &Trace) -> String {
    // Resolve x/y data.
    let xs: Vec<f64> = if let Some(xn) = &spec.x_series {
        trace.rows.iter().map(|r| r.get(xn)).collect()
    } else {
        trace.rows.iter().map(|r| r.step as f64).collect()
    };
    let ys: Vec<Vec<f64>> = spec
        .y_series
        .iter()
        .map(|y| {
            trace
                .rows
                .iter()
                .map(|r| r.get(&y.series_name))
                .collect()
        })
        .collect();

    let labels: Vec<&str> = spec.y_series.iter().map(|y| y.label.as_str()).collect();
    let colors: Vec<String> = spec
        .y_series
        .iter()
        .enumerate()
        .map(|(i, y)| {
            y.color
                .clone()
                .unwrap_or_else(|| DEFAULT_PALETTE[i % DEFAULT_PALETTE.len()].to_string())
        })
        .collect();

    write_chart_svg(
        &spec.title,
        &spec.x_label,
        &spec.y_label,
        &xs,
        &ys,
        &labels,
        &colors,
        spec.log_y,
    )
}

fn render_before_after(spec: &ChartSpec) -> String {
    let ChartKind::BeforeAfter {
        x_label: _,
        before_label,
        before,
        after_label,
        after,
    } = &spec.kind
    else {
        unreachable!("render_before_after called with non-BeforeAfter kind")
    };

    // Combine the two sweeps into a single x grid + per-line ys.
    // Each sweep keeps its own x positions; we draw them separately
    // by piping through `write_chart_svg` with two pre-resolved
    // (xs, ys) pairs.
    let xs_b: Vec<f64> = before.iter().map(|(x, _)| *x).collect();
    let ys_b: Vec<f64> = before.iter().map(|(_, y)| *y).collect();
    let xs_a: Vec<f64> = after.iter().map(|(x, _)| *x).collect();
    let ys_a: Vec<f64> = after.iter().map(|(_, y)| *y).collect();

    write_two_curve_svg(
        &spec.title,
        &spec.x_label,
        &spec.y_label,
        (&xs_b, &ys_b, before_label, "#d62728"),
        (&xs_a, &ys_a, after_label, "#1f77b4"),
        spec.log_y,
    )
}

fn write_chart_svg(
    title: &str,
    x_label: &str,
    y_label: &str,
    xs: &[f64],
    ys: &[Vec<f64>],
    labels: &[&str],
    colors: &[String],
    log_y: bool,
) -> String {
    if xs.is_empty() || ys.is_empty() {
        return empty_chart_svg(title);
    }
    let (x_min, x_max) = minmax(xs);
    let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for ys_one in ys {
        let (lo, hi) = minmax_filtered(ys_one, log_y);
        if lo < y_min { y_min = lo; }
        if hi > y_max { y_max = hi; }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        return empty_chart_svg(title);
    }
    let (y_min, y_max) = pad_range(y_min, y_max);

    let mut out = String::new();
    open_svg(&mut out, title);
    draw_axes(&mut out, x_label, y_label, x_min, x_max, y_min, y_max, log_y);

    for (i, ys_one) in ys.iter().enumerate() {
        let color = &colors[i];
        let mut path = String::from("M");
        let mut started = false;
        for (j, &y) in ys_one.iter().enumerate() {
            let (px, py) = project(xs[j], y, x_min, x_max, y_min, y_max, log_y);
            if !py.is_finite() { continue; }
            if !started {
                path.push_str(&format!("{px:.2},{py:.2}"));
                started = true;
            } else {
                path.push_str(&format!(" L{px:.2},{py:.2}"));
            }
        }
        out.push_str(&format!(
            "<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>\n"
        ));
    }

    if labels.len() > 1 {
        draw_legend(&mut out, labels, colors);
    }
    out.push_str("</svg>\n");
    out
}

fn write_two_curve_svg(
    title: &str,
    x_label: &str,
    y_label: &str,
    a: (&[f64], &[f64], &str, &str),
    b: (&[f64], &[f64], &str, &str),
    log_y: bool,
) -> String {
    let (xs_a, ys_a, lbl_a, col_a) = a;
    let (xs_b, ys_b, lbl_b, col_b) = b;
    if xs_a.is_empty() && xs_b.is_empty() {
        return empty_chart_svg(title);
    }
    let (mut x_min, mut x_max) = minmax(xs_a);
    let (xb0, xb1) = minmax(xs_b);
    if xb0 < x_min { x_min = xb0; }
    if xb1 > x_max { x_max = xb1; }
    let (mut y_min, mut y_max) = minmax_filtered(ys_a, log_y);
    let (yb0, yb1) = minmax_filtered(ys_b, log_y);
    if yb0 < y_min { y_min = yb0; }
    if yb1 > y_max { y_max = yb1; }
    let (y_min, y_max) = pad_range(y_min, y_max);

    let mut out = String::new();
    open_svg(&mut out, title);
    draw_axes(&mut out, x_label, y_label, x_min, x_max, y_min, y_max, log_y);

    for (xs, ys, color) in [(xs_a, ys_a, col_a), (xs_b, ys_b, col_b)] {
        let mut path = String::from("M");
        let mut started = false;
        for (j, &y) in ys.iter().enumerate() {
            let (px, py) = project(xs[j], y, x_min, x_max, y_min, y_max, log_y);
            if !py.is_finite() { continue; }
            if !started {
                path.push_str(&format!("{px:.2},{py:.2}"));
                started = true;
            } else {
                path.push_str(&format!(" L{px:.2},{py:.2}"));
            }
        }
        out.push_str(&format!(
            "<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>\n"
        ));
    }

    draw_legend(&mut out, &[lbl_a, lbl_b], &[col_a.to_string(), col_b.to_string()]);
    out.push_str("</svg>\n");
    out
}

// ── Low-level SVG helpers ─────────────────────────────────────────────

fn open_svg(out: &mut String, title: &str) {
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" \
        width=\"{W}\" height=\"{H}\" viewBox=\"0 0 {W} {H}\" \
        font-family=\"-apple-system, system-ui, sans-serif\" font-size=\"13\">\n",
    ));
    out.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");
    out.push_str(&format!(
        "<text x=\"{:.0}\" y=\"20\" font-size=\"15\" font-weight=\"600\">{}</text>\n",
        W / 2.0,
        escape_xml(title),
    ));
    // Center title.
    let pos = out.rfind("<text x=\"").unwrap();
    out.replace_range(pos.., &out[pos..].replacen(
        "<text",
        "<text text-anchor=\"middle\"",
        1,
    ));
}

fn draw_axes(
    out: &mut String,
    x_label: &str,
    y_label: &str,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    log_y: bool,
) {
    let plot_x0 = M_LEFT;
    let plot_y0 = M_TOP;
    let plot_x1 = W - M_RIGHT;
    let plot_y1 = H - M_BOTTOM;

    // Plot area background + border.
    out.push_str(&format!(
        "<rect x=\"{plot_x0}\" y=\"{plot_y0}\" width=\"{PLOT_W}\" height=\"{PLOT_H}\" \
         fill=\"#fafafa\" stroke=\"#888\" stroke-width=\"0.5\"/>\n",
    ));

    // X gridlines + ticks.
    let x_ticks = nice_ticks(x_min, x_max, 8);
    for t in &x_ticks {
        let frac = (t - x_min) / (x_max - x_min);
        let px = plot_x0 + frac * PLOT_W;
        out.push_str(&format!(
            "<line x1=\"{px:.2}\" y1=\"{plot_y0}\" x2=\"{px:.2}\" y2=\"{plot_y1}\" \
             stroke=\"#e0e0e0\" stroke-width=\"0.5\"/>\n",
        ));
        out.push_str(&format!(
            "<text x=\"{px:.2}\" y=\"{:.0}\" text-anchor=\"middle\" fill=\"#444\">{}</text>\n",
            plot_y1 + 16.0,
            format_tick(*t),
        ));
    }

    // Y gridlines.
    let y_ticks = if log_y {
        log_ticks(y_min, y_max)
    } else {
        nice_ticks(y_min, y_max, 6)
    };
    for t in &y_ticks {
        let frac = if log_y {
            (t.log10() - y_min.log10()) / (y_max.log10() - y_min.log10())
        } else {
            (t - y_min) / (y_max - y_min)
        };
        let py = plot_y1 - frac * PLOT_H;
        if !py.is_finite() { continue; }
        out.push_str(&format!(
            "<line x1=\"{plot_x0}\" y1=\"{py:.2}\" x2=\"{plot_x1}\" y2=\"{py:.2}\" \
             stroke=\"#e0e0e0\" stroke-width=\"0.5\"/>\n",
        ));
        out.push_str(&format!(
            "<text x=\"{:.0}\" y=\"{py:.2}\" text-anchor=\"end\" \
             alignment-baseline=\"middle\" fill=\"#444\">{}</text>\n",
            plot_x0 - 6.0,
            format_tick(*t),
        ));
    }

    // Axis labels.
    out.push_str(&format!(
        "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" fill=\"#222\">{}</text>\n",
        plot_x0 + PLOT_W / 2.0,
        plot_y1 + 40.0,
        escape_xml(x_label),
    ));
    out.push_str(&format!(
        "<text x=\"20\" y=\"{:.0}\" text-anchor=\"middle\" fill=\"#222\" \
         transform=\"rotate(-90 20 {:.0})\">{}</text>\n",
        plot_y0 + PLOT_H / 2.0,
        plot_y0 + PLOT_H / 2.0,
        escape_xml(y_label),
    ));
}

fn draw_legend(out: &mut String, labels: &[&str], colors: &[String]) {
    let lx = W - M_RIGHT - 180.0;
    let ly0 = M_TOP + 12.0;
    out.push_str(&format!(
        "<rect x=\"{lx:.0}\" y=\"{:.0}\" width=\"170\" height=\"{}\" \
         fill=\"white\" stroke=\"#bbb\" stroke-width=\"0.5\" opacity=\"0.92\"/>\n",
        ly0 - 12.0,
        16 * labels.len() + 12,
    ));
    for (i, lbl) in labels.iter().enumerate() {
        let y = ly0 + (i as f64) * 16.0;
        out.push_str(&format!(
            "<line x1=\"{:.0}\" y1=\"{y:.0}\" x2=\"{:.0}\" y2=\"{y:.0}\" \
             stroke=\"{}\" stroke-width=\"3\"/>\n",
            lx + 8.0, lx + 28.0, colors[i],
        ));
        out.push_str(&format!(
            "<text x=\"{:.0}\" y=\"{:.0}\" fill=\"#222\">{}</text>\n",
            lx + 36.0,
            y + 4.0,
            escape_xml(lbl),
        ));
    }
}

fn project(
    x: f64,
    y: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    log_y: bool,
) -> (f64, f64) {
    let xf = (x - x_min) / (x_max - x_min);
    let yf = if log_y {
        if y <= 0.0 { return (0.0, f64::NAN); }
        (y.log10() - y_min.log10()) / (y_max.log10() - y_min.log10())
    } else {
        (y - y_min) / (y_max - y_min)
    };
    let px = M_LEFT + xf * PLOT_W;
    let py = (H - M_BOTTOM) - yf * PLOT_H;
    (px, py)
}

fn minmax(xs: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in xs {
        if !x.is_finite() { continue; }
        if x < lo { lo = x; }
        if x > hi { hi = x; }
    }
    (lo, hi)
}

fn minmax_filtered(xs: &[f64], log_y: bool) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in xs {
        if !x.is_finite() { continue; }
        if log_y && x <= 0.0 { continue; }
        if x < lo { lo = x; }
        if x > hi { hi = x; }
    }
    (lo, hi)
}

fn pad_range(lo: f64, hi: f64) -> (f64, f64) {
    if lo == hi {
        let pad = if lo == 0.0 { 1.0 } else { lo.abs() * 0.1 };
        return (lo - pad, hi + pad);
    }
    let span = hi - lo;
    (lo - 0.05 * span, hi + 0.05 * span)
}

fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<f64> {
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return vec![lo, hi];
    }
    let raw_step = (hi - lo) / (target as f64).max(1.0);
    let mag = 10f64.powf(raw_step.log10().floor());
    let norm = raw_step / mag;
    let step = mag * if norm < 1.5 {
        1.0
    } else if norm < 3.0 {
        2.0
    } else if norm < 7.0 {
        5.0
    } else {
        10.0
    };
    let start = (lo / step).ceil() * step;
    let mut out = Vec::new();
    let mut t = start;
    while t <= hi + step * 1e-9 {
        out.push(t);
        t += step;
    }
    out
}

fn log_ticks(lo: f64, hi: f64) -> Vec<f64> {
    if lo <= 0.0 || hi <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let lo_e = lo.log10().floor() as i32;
    let hi_e = hi.log10().ceil() as i32;
    for e in lo_e..=hi_e {
        let v = 10f64.powi(e);
        if v >= lo * 0.999 && v <= hi * 1.001 {
            out.push(v);
        }
    }
    out
}

fn format_tick(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if v.abs() < 1e-3 || v.abs() >= 1e4 {
        format!("{v:.2e}")
    } else if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.3}")
    }
}

fn empty_chart_svg(title: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{W}\" height=\"{H}\">\n\
         <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n\
         <text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" font-size=\"15\">\
         {} (empty trace)</text>\n</svg>\n",
        W / 2.0,
        H / 2.0,
        escape_xml(title),
    )
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{Trace, TraceCfg, TraceRow};

    fn synth_trace() -> Trace {
        let cfg = TraceCfg::new("t", 5);
        Trace::run(&cfg, |s| {
            TraceRow::new(s)
                .with("loss", 1.0 / ((s + 1) as f64))
                .with("param", s as f64 * 0.5)
        })
    }

    #[test]
    fn line_chart_renders_with_legend_and_log_y() {
        let trace = synth_trace();
        let spec = ChartSpec::line("loss", "Loss", "step", "|loss|")
            .with_y_log(true)
            .add_series("loss", "loss")
            .add_series("param", "param");
        let svg = render_chart_svg(&spec, &trace);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("Loss"));
        // Legend renders for >1 series.
        assert!(svg.contains("loss") && svg.contains("param"));
    }

    #[test]
    fn before_after_chart_uses_supplied_arrays() {
        let trace = synth_trace();
        let spec = ChartSpec::before_after(
            "spectrum", "Spectrum", "wavelength", "|T|",
            "before", vec![(1500.0, 0.1), (1550.0, 0.5), (1600.0, 0.9)],
            "after",  vec![(1500.0, 0.9), (1550.0, 0.05), (1600.0, 0.1)],
        );
        let svg = render_chart_svg(&spec, &trace);
        assert!(svg.contains("before") && svg.contains("after"));
    }

    #[test]
    fn empty_trace_does_not_panic() {
        let trace = Trace::default();
        let spec = ChartSpec::line("e", "Empty", "step", "y").add_series("loss", "loss");
        let svg = render_chart_svg(&spec, &trace);
        assert!(svg.contains("empty"));
    }
}
