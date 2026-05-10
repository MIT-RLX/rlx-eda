//! PNG and SVG rendering of [`Waveform`] via [`plotters`].
//!
//! Two output formats, one shared renderer:
//!
//! - PNG (`png_to_path`) — bitmap. Embeddable in markdown / GitHub
//!   issues, opens in any image viewer. Use for screenshots and CI
//!   artifacts.
//! - SVG (`svg_to_path` / `svg_to_string`) — vector. Scales without
//!   pixelation, the right choice for documentation and PR
//!   attachments where the reader may want to zoom into a transient
//!   edge. Also tiny on the wire.
//!
//! Both go through [`render`], which is generic over plotters'
//! `DrawingBackend`. So adding e.g. a Cairo PDF backend later means
//! one new wrapper, not a second renderer.
//!
//! ## Layout
//!
//! - [`Layout::Single`] (default): all real signals share one pane.
//!   Best for ≤3 signals on similar scales.
//! - [`Layout::Stacked`]: one row per signal, shared x-axis. Required
//!   for mixed digital/analog (a clock 0/1 plus a 0–1 V analog signal)
//!   and for the SAR ADC bit-trace use case (8 bits + clock).
//!
//! Complex waveforms always get magnitude (top) + phase (bottom) panes
//! sharing the log-frequency x-axis — the standard Bode layout. The
//! `Layout` setting is ignored for complex.
//!
//! ## Markers
//!
//! [`PlotConfig::markers`] holds vertical and horizontal annotation
//! lines. Each marker is drawn on every pane (so a `Vertical { x: 1e-6 }`
//! marker for a clock edge appears on every stacked row). Useful for
//! sampling instants, settling-time markers, threshold lines for ADC
//! comparator decisions.
//!
//! ## Defaults
//!
//! 800×600 px, white background, 11pt sans-serif labels. Override via
//! [`PlotConfig`] for higher-res CI artifacts.

use std::collections::BTreeMap;
use std::path::Path;

use plotters::backend::{BitMapBackend, SVGBackend};
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::Palette99;
use thiserror::Error;

use crate::Waveform;

#[derive(Debug, Error)]
pub enum PlotError {
    #[error("waveform has no samples")]
    Empty,
    #[error("plotters drawing error: {0}")]
    Draw(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Pane arrangement for real-valued waveforms.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Layout {
    /// All signals on one pane, color-coded with a legend. The default.
    #[default]
    Single,
    /// One signal per pane, vertically stacked, sharing the x-axis.
    /// Each pane y-scales to its own signal's data range — the right
    /// choice when signals have very different magnitudes (digital +
    /// analog).
    Stacked,
}

/// Annotation drawn on top of every pane.
#[derive(Debug, Clone)]
pub enum Marker {
    /// Vertical line at `x`. Useful for clock edges, sampling instants.
    Vertical { x: f64, label: Option<String> },
    /// Horizontal line at `y`. Useful for comparator thresholds, full-
    /// scale references. On stacked layouts the line draws on every
    /// pane — usually only meaningful when the panes share a y-scale,
    /// so use sparingly with `Stacked`.
    Horizontal { y: f64, label: Option<String> },
}

/// Render configuration. Pass `PlotConfig::default()` for 800×600 — the
/// CI-friendly default.
#[derive(Debug, Clone, Default)]
pub struct PlotConfig {
    pub width: u32,
    pub height: u32,
    pub title: Option<String>,
    pub layout: Layout,
    pub markers: Vec<Marker>,
}

impl PlotConfig {
    /// Builder convenience: 800×600, no title, single pane.
    pub fn new() -> Self {
        Self { width: 800, height: 600, title: None, layout: Layout::default(), markers: Vec::new() }
    }
    pub fn with_title(mut self, t: impl Into<String>) -> Self {
        self.title = Some(t.into()); self
    }
    pub fn with_layout(mut self, l: Layout) -> Self {
        self.layout = l; self
    }
    pub fn with_size(mut self, w: u32, h: u32) -> Self {
        self.width = w; self.height = h; self
    }
    pub fn add_marker(mut self, m: Marker) -> Self {
        self.markers.push(m); self
    }
}

// `Default` impl above gives 0×0 — fix to the documented defaults.
// (Manual impl conflicts with `#[derive(Default)]`; supply via a
// re-impl block below.)
impl PlotConfig {
    /// 800×600, single pane, no markers.
    pub const DEFAULT: PlotConfigConst = PlotConfigConst;
}
/// Marker type used only as the receiver for the doc-DEFAULT
/// associated constant; not directly constructible. Use
/// `PlotConfig::default()` instead.
#[doc(hidden)]
pub struct PlotConfigConst;

// ── Public renderers ───────────────────────────────────────────────────

pub fn png_to_path(
    wave: &Waveform,
    path: impl AsRef<Path>,
    cfg: &PlotConfig,
) -> Result<(), PlotError> {
    let cfg = with_defaults(cfg);
    let backend = BitMapBackend::new(path.as_ref(), (cfg.width, cfg.height));
    render(wave, backend, &cfg)
}

pub fn svg_to_path(
    wave: &Waveform,
    path: impl AsRef<Path>,
    cfg: &PlotConfig,
) -> Result<(), PlotError> {
    let cfg = with_defaults(cfg);
    let backend = SVGBackend::new(path.as_ref(), (cfg.width, cfg.height));
    render(wave, backend, &cfg)
}

pub fn svg_to_string(wave: &Waveform, cfg: &PlotConfig) -> Result<String, PlotError> {
    let cfg = with_defaults(cfg);
    let mut out = String::new();
    {
        let backend = SVGBackend::with_string(&mut out, (cfg.width, cfg.height));
        render(wave, backend, &cfg)?;
    }
    Ok(out)
}

/// Apply 800×600 defaults to fields the caller left at zero. Lets users
/// pass `PlotConfig { title: Some("…"), ..Default::default() }` without
/// remembering to set width/height.
fn with_defaults(cfg: &PlotConfig) -> PlotConfig {
    PlotConfig {
        width:  if cfg.width  == 0 { 800 } else { cfg.width  },
        height: if cfg.height == 0 { 600 } else { cfg.height },
        title: cfg.title.clone(),
        layout: cfg.layout,
        markers: cfg.markers.clone(),
    }
}

// ── Backend-generic dispatch ───────────────────────────────────────────

fn render<DB>(wave: &Waveform, backend: DB, cfg: &PlotConfig) -> Result<(), PlotError>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let root = backend.into_drawing_area();
    root.fill(&WHITE).map_err(draw_err)?;

    match wave {
        Waveform::Real { axis_name, axis, signals } => {
            if axis.is_empty() || signals.is_empty() {
                return Err(PlotError::Empty);
            }
            match cfg.layout {
                Layout::Single => draw_real_single(&root, axis_name, axis, signals, cfg)?,
                Layout::Stacked => draw_real_stacked(&root, axis_name, axis, signals, cfg)?,
            }
        }
        Waveform::Complex { axis_name, axis, signals } => {
            if axis.is_empty() || signals.is_empty() {
                return Err(PlotError::Empty);
            }
            draw_bode(&root, axis_name, axis, signals, cfg)?;
        }
    }

    root.present().map_err(draw_err)?;
    Ok(())
}

// ── Real single-pane ───────────────────────────────────────────────────

fn draw_real_single<DB>(
    root: &DrawingArea<DB, Shift>,
    axis_name: &str,
    axis: &[f64],
    signals: &BTreeMap<String, Vec<f64>>,
    cfg: &PlotConfig,
) -> Result<(), PlotError>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let (x_min, x_max) = axis_bounds(axis);
    let (y_min, y_max) = signals_bounds(signals.values());
    let title = cfg.title.clone().unwrap_or_else(|| format!("waveform ({axis_name})"));

    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(draw_err)?;
    let x_fmt = autoscale_formatter(x_max - x_min);
    let y_fmt = autoscale_formatter(y_max - y_min);
    chart
        .configure_mesh()
        .x_desc(axis_name)
        .y_desc("volts")
        .x_label_formatter(&x_fmt)
        .y_label_formatter(&y_fmt)
        .draw()
        .map_err(draw_err)?;

    for (idx, (name, samples)) in signals.iter().enumerate() {
        let color = Palette99::pick(idx).to_rgba();
        let series = LineSeries::new(
            axis.iter().copied().zip(samples.iter().copied()),
            color.stroke_width(2),
        );
        chart
            .draw_series(series)
            .map_err(draw_err)?
            .label(name.clone())
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 16, y)], color));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()
        .map_err(draw_err)?;

    draw_markers(&mut chart, &cfg.markers, x_min, x_max, y_min, y_max)?;
    Ok(())
}

// ── Real stacked panes ─────────────────────────────────────────────────

fn draw_real_stacked<DB>(
    root: &DrawingArea<DB, Shift>,
    axis_name: &str,
    axis: &[f64],
    signals: &BTreeMap<String, Vec<f64>>,
    cfg: &PlotConfig,
) -> Result<(), PlotError>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    if let Some(t) = &cfg.title {
        // Reserve a 30px strip at the top for the overall title; the
        // per-pane sub-areas split the rest.
        let (header, body) = root.split_vertically(30);
        header
            .titled(t, ("sans-serif", 18))
            .map_err(draw_err)?;
        draw_real_stacked_body(&body, axis_name, axis, signals, cfg)
    } else {
        draw_real_stacked_body(root, axis_name, axis, signals, cfg)
    }
}

fn draw_real_stacked_body<DB>(
    body: &DrawingArea<DB, Shift>,
    axis_name: &str,
    axis: &[f64],
    signals: &BTreeMap<String, Vec<f64>>,
    cfg: &PlotConfig,
) -> Result<(), PlotError>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let (x_min, x_max) = axis_bounds(axis);
    let panes = body.split_evenly((signals.len(), 1));

    for (idx, ((name, samples), pane)) in signals.iter().zip(panes.iter()).enumerate() {
        let (y_min, y_max) = signals_bounds(std::iter::once(samples));
        let is_last = idx + 1 == signals.len();
        let color = Palette99::pick(idx).to_rgba();

        let mut chart = ChartBuilder::on(pane)
            .margin_top(8)
            .margin_bottom(if is_last { 8 } else { 4 })
            .x_label_area_size(if is_last { 36 } else { 0 })
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .map_err(draw_err)?;

        let x_fmt = autoscale_formatter(x_max - x_min);
        let y_fmt = autoscale_formatter(y_max - y_min);
        let mut mesh = chart.configure_mesh();
        mesh.y_desc(name).y_label_formatter(&y_fmt);
        if is_last {
            mesh.x_desc(axis_name).x_label_formatter(&x_fmt);
        } else {
            // Hide x labels on non-bottom panes to keep the visual quiet.
            mesh.x_labels(0);
        }
        mesh.draw().map_err(draw_err)?;

        chart
            .draw_series(LineSeries::new(
                axis.iter().copied().zip(samples.iter().copied()),
                color.stroke_width(2),
            ))
            .map_err(draw_err)?;

        draw_markers(&mut chart, &cfg.markers, x_min, x_max, y_min, y_max)?;
    }
    Ok(())
}

// ── Bode (complex): magnitude + phase ──────────────────────────────────

fn draw_bode<DB>(
    root: &DrawingArea<DB, Shift>,
    axis_name: &str,
    axis: &[f64],
    signals: &BTreeMap<String, Vec<(f64, f64)>>,
    cfg: &PlotConfig,
) -> Result<(), PlotError>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let log_axis: Vec<f64> = axis.iter().map(|f| f.log10()).collect();

    let mag_db: Vec<Vec<f64>> = signals
        .values()
        .map(|samples| {
            samples
                .iter()
                .map(|(re, im)| 20.0 * (re * re + im * im).sqrt().max(1e-300).log10())
                .collect()
        })
        .collect();
    let phase_deg: Vec<Vec<f64>> = signals
        .values()
        .map(|samples| unwrap_phase(samples))
        .collect();

    // Top pane: mag. Bottom pane: phase. Optional title strip above.
    let (mag_area, phase_area) = if let Some(t) = &cfg.title {
        let (header, body) = root.split_vertically(30);
        header.titled(t, ("sans-serif", 18)).map_err(draw_err)?;
        body.split_vertically((body.dim_in_pixel().1 / 2) as i32)
    } else {
        root.split_vertically((root.dim_in_pixel().1 / 2) as i32)
    };

    // Magnitude pane
    let (x_min, x_max) = axis_bounds(&log_axis);
    let (mag_min, mag_max) = signals_bounds(mag_db.iter());
    let mut mag_chart = ChartBuilder::on(&mag_area)
        .margin(8)
        .x_label_area_size(0) // bottom labels live on the phase pane
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, mag_min..mag_max)
        .map_err(draw_err)?;
    mag_chart
        .configure_mesh()
        .x_labels(0)
        .y_desc("|H| (dB)")
        .draw()
        .map_err(draw_err)?;
    for (idx, (name, mag)) in signals.keys().zip(mag_db.iter()).enumerate() {
        let color = Palette99::pick(idx).to_rgba();
        mag_chart
            .draw_series(LineSeries::new(
                log_axis.iter().copied().zip(mag.iter().copied()),
                color.stroke_width(2),
            ))
            .map_err(draw_err)?
            .label(name.clone())
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 16, y)], color));
    }
    mag_chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()
        .map_err(draw_err)?;
    draw_markers_log_x(&mut mag_chart, &cfg.markers, x_min, x_max, mag_min, mag_max)?;

    // Phase pane
    let (phase_min, phase_max) = signals_bounds(phase_deg.iter());
    // Pad to nearest 45° so labels look clean.
    let phase_min = (phase_min / 45.0).floor() * 45.0;
    let phase_max = (phase_max / 45.0).ceil() * 45.0;
    let mut phase_chart = ChartBuilder::on(&phase_area)
        .margin(8)
        .x_label_area_size(36)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, phase_min..phase_max)
        .map_err(draw_err)?;
    let x_fmt: Box<dyn Fn(&f64) -> String> = Box::new(|v: &f64| format!("1e{}", v.round() as i32));
    phase_chart
        .configure_mesh()
        .x_desc(format!("{axis_name} (Hz)"))
        .x_label_formatter(&x_fmt)
        .y_desc("∠H (°)")
        .draw()
        .map_err(draw_err)?;
    for (idx, ph) in phase_deg.iter().enumerate() {
        let color = Palette99::pick(idx).to_rgba();
        phase_chart
            .draw_series(LineSeries::new(
                log_axis.iter().copied().zip(ph.iter().copied()),
                color.stroke_width(2),
            ))
            .map_err(draw_err)?;
    }
    draw_markers_log_x(&mut phase_chart, &cfg.markers, x_min, x_max, phase_min, phase_max)?;
    Ok(())
}

/// Continuous-phase unwrap: classical numpy.unwrap algorithm. Each
/// sample's raw atan2 is shifted by an integer multiple of 2π so the
/// step from the previous sample is at most π in magnitude. Returns
/// degrees.
fn unwrap_phase(samples: &[(f64, f64)]) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut out = Vec::with_capacity(samples.len());
    if samples.is_empty() { return out; }

    let raw0 = samples[0].1.atan2(samples[0].0);
    out.push(raw0.to_degrees());
    let mut prev_raw = raw0;
    let mut unwrapped = raw0;

    for &(re, im) in &samples[1..] {
        let raw = im.atan2(re);
        let mut diff = raw - prev_raw;
        while diff >  PI { diff -= 2.0 * PI; }
        while diff < -PI { diff += 2.0 * PI; }
        unwrapped += diff;
        out.push(unwrapped.to_degrees());
        prev_raw = raw;
    }
    out
}

// ── Marker drawing ─────────────────────────────────────────────────────

/// Draw markers on a linear-x chart.
fn draw_markers<DB>(
    chart: &mut ChartContext<DB, plotters::coord::cartesian::Cartesian2d<
        plotters::coord::types::RangedCoordf64,
        plotters::coord::types::RangedCoordf64,
    >>,
    markers: &[Marker],
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
) -> Result<(), PlotError>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    for m in markers {
        match m {
            Marker::Vertical { x, label } => {
                if *x < x_min || *x > x_max { continue; }
                chart.draw_series(LineSeries::new(
                    [(*x, y_min), (*x, y_max)],
                    BLACK.mix(0.4).stroke_width(1),
                )).map_err(draw_err)?;
                if let Some(text) = label {
                    chart.draw_series(std::iter::once(Text::new(
                        text.clone(),
                        (*x, y_max - (y_max - y_min) * 0.05),
                        ("sans-serif", 11).into_font().color(&BLACK.mix(0.7)),
                    ))).map_err(draw_err)?;
                }
            }
            Marker::Horizontal { y, label } => {
                if *y < y_min || *y > y_max { continue; }
                chart.draw_series(LineSeries::new(
                    [(x_min, *y), (x_max, *y)],
                    BLACK.mix(0.4).stroke_width(1),
                )).map_err(draw_err)?;
                if let Some(text) = label {
                    chart.draw_series(std::iter::once(Text::new(
                        text.clone(),
                        (x_min + (x_max - x_min) * 0.02, *y),
                        ("sans-serif", 11).into_font().color(&BLACK.mix(0.7)),
                    ))).map_err(draw_err)?;
                }
            }
        }
    }
    Ok(())
}

/// Like `draw_markers`, but the chart's x-axis is `log10(f)` — caller-
/// supplied marker x values (in Hz) are mapped to log10 before drawing.
fn draw_markers_log_x<DB>(
    chart: &mut ChartContext<DB, plotters::coord::cartesian::Cartesian2d<
        plotters::coord::types::RangedCoordf64,
        plotters::coord::types::RangedCoordf64,
    >>,
    markers: &[Marker],
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
) -> Result<(), PlotError>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let mapped: Vec<Marker> = markers
        .iter()
        .map(|m| match m {
            Marker::Vertical { x, label } => Marker::Vertical {
                x: x.max(1e-300).log10(),
                label: label.clone(),
            },
            other => other.clone(),
        })
        .collect();
    draw_markers(chart, &mapped, x_min, x_max, y_min, y_max)
}

// ── Helpers ────────────────────────────────────────────────────────────

fn draw_err<E: std::fmt::Display>(e: E) -> PlotError {
    PlotError::Draw(e.to_string())
}

/// Pick a tick-label formatter sized to the axis span.
///
/// plotters' default `{:.1}`-style formatter collapses to `0.0` on
/// micro-second time axes and to identical exponents on Bode plots.
/// Switch to scientific notation when the span falls outside
/// `[1e-2, 1e4)` — that window covers volts, milliseconds, and most
/// human-friendly engineering units; outside it (femtoseconds,
/// kilovolts, gigahertz) the `%.3e` form reads better.
fn autoscale_formatter(span: f64) -> Box<dyn Fn(&f64) -> String> {
    if span > 0.0 && (span < 1e-2 || span >= 1e4) {
        Box::new(|v: &f64| format!("{:.3e}", v))
    } else {
        Box::new(|v: &f64| format!("{:.3}", v))
    }
}

fn axis_bounds(axis: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in axis {
        if v < lo { lo = v; }
        if v > hi { hi = v; }
    }
    if (hi - lo).abs() < 1e-30 {
        (lo - 1.0, hi + 1.0)
    } else {
        let pad = (hi - lo) * 0.02;
        (lo - pad, hi + pad)
    }
}

fn signals_bounds<'a, I>(samples: I) -> (f64, f64)
where
    I: IntoIterator<Item = &'a Vec<f64>>,
{
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for series in samples {
        for &v in series {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
    }
    if (hi - lo).abs() < 1e-30 {
        (lo - 0.5, hi + 0.5)
    } else {
        let pad = (hi - lo) * 0.05;
        (lo - pad, hi + pad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_real() -> Waveform {
        let mut signals = BTreeMap::new();
        signals.insert("v(in)".into(), vec![0.0, 1.0, 1.0, 0.0]);
        signals.insert("v(out)".into(), vec![0.0, 0.5, 0.75, 0.0]);
        Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9, 2e-9, 3e-9],
            signals,
        }
    }

    fn fixture_complex() -> Waveform {
        let mut signals = BTreeMap::new();
        signals.insert(
            "v(out)".into(),
            vec![(1.0, 0.0), (0.7, -0.7), (0.3, -0.5)],
        );
        Waveform::Complex {
            axis_name: "frequency".into(),
            axis: vec![1e3, 1e4, 1e5],
            signals,
        }
    }

    #[test]
    fn png_writes_valid_magic() {
        let dir = tempdir();
        let path = dir.path().join("test.png");
        png_to_path(&fixture_real(), &path, &PlotConfig::default()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        assert!(bytes.len() > 1000);
    }

    #[test]
    fn svg_writes_well_formed() {
        let dir = tempdir();
        let path = dir.path().join("test.svg");
        svg_to_path(&fixture_real(), &path, &PlotConfig::default()).unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.starts_with("<svg") || s.contains("<svg"));
        assert!(s.contains("</svg>"));
        assert!(s.contains("v(in)"));
        assert!(s.contains("v(out)"));
    }

    #[test]
    fn svg_to_string_returns_same_content() {
        let s = svg_to_string(&fixture_real(), &PlotConfig::default()).unwrap();
        assert!(s.contains("v(in)"));
        assert!(s.contains("v(out)"));
    }

    #[test]
    fn complex_renders_bode_with_phase() {
        let s = svg_to_string(&fixture_complex(), &PlotConfig::default()).unwrap();
        assert!(s.contains("v(out)"));
        assert!(s.contains("dB"));
        // Phase pane title.
        assert!(s.contains("∠H") || s.contains("H ("), "phase axis label missing");
    }

    #[test]
    fn stacked_layout_emits_all_signals() {
        let cfg = PlotConfig { layout: Layout::Stacked, ..Default::default() };
        let s = svg_to_string(&fixture_real(), &cfg).unwrap();
        // In stacked layout, each signal name labels its own pane's y-axis.
        assert!(s.contains("v(in)"));
        assert!(s.contains("v(out)"));
    }

    #[test]
    fn marker_vertical_appears_in_svg() {
        let cfg = PlotConfig {
            markers: vec![Marker::Vertical { x: 1e-9, label: Some("clk".into()) }],
            ..Default::default()
        };
        let s = svg_to_string(&fixture_real(), &cfg).unwrap();
        assert!(s.contains("clk"));
    }

    #[test]
    fn marker_horizontal_threshold() {
        let cfg = PlotConfig {
            markers: vec![Marker::Horizontal { y: 0.5, label: Some("Vth".into()) }],
            ..Default::default()
        };
        let s = svg_to_string(&fixture_real(), &cfg).unwrap();
        assert!(s.contains("Vth"));
    }

    #[test]
    fn empty_signals_errors() {
        let w = Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1.0],
            signals: BTreeMap::new(),
        };
        let res = svg_to_string(&w, &PlotConfig::default());
        assert!(matches!(res, Err(PlotError::Empty)));
    }

    #[test]
    fn unwrap_phase_handles_pi_jump() {
        // Phase angles approaching -π: -170°, -179°, +179° (which is
        // really -181° unwrapped). Ensure unwrap doesn't reset.
        let samples = vec![
            ((-170.0f64).to_radians().cos(), (-170.0f64).to_radians().sin()),
            ((-179.0f64).to_radians().cos(), (-179.0f64).to_radians().sin()),
            (( 179.0f64).to_radians().cos(), ( 179.0f64).to_radians().sin()),
        ];
        let unwrapped = unwrap_phase(&samples);
        // After unwrap, third point should be roughly -181°, not +179°.
        assert!(unwrapped[2] < -90.0, "unwrap failed: {unwrapped:?}");
    }

    fn tempdir() -> tempfile::TempDir {
        tempfile::Builder::new().prefix("rlx-eda-plot-").tempdir().unwrap()
    }
}
