//! Markdown report assembly + on-disk writeout.
//!
//! [`Report::write`] is the harness's single side-effecting entry
//! point: it lays down `<out_dir>/<name>.md`, the SVG (and PNG, if
//! the `png` feature is on) for every chart, the floorplan SVG/PNG,
//! and the CSV of the trace.

use std::fs;
use std::path::{Path, PathBuf};

use crate::charts::{render_chart_svg, ChartSpec};
use crate::trace::Trace;

/// Domain tag — appears in the report header. The harness never
/// branches on it; it's there so a reader can tell at a glance
/// whether they're looking at an electrical, photonic, RF, quantum,
/// or layout-optimization run.
///
/// `Layout` covers the place-and-route side of the stack — HPWL
/// minimization, simulated-annealing placement, congestion-aware
/// routing — anything where the optimization target is geometric
/// rather than behavioral.
#[derive(Clone, Debug)]
pub enum Domain {
    Electrical,
    Photonic,
    Rf,
    Quantum,
    Neuromorphic,
    Layout,
    Other(String),
}

impl Domain {
    fn label(&self) -> &str {
        match self {
            Domain::Electrical => "Electrical",
            Domain::Photonic => "Photonic",
            Domain::Rf => "RF",
            Domain::Quantum => "Quantum",
            Domain::Neuromorphic => "Neuromorphic",
            Domain::Layout => "Layout",
            Domain::Other(s) => s,
        }
    }
}

/// One row of the literature-validation table — the same shape as
/// `mzi_ml_trace.md`'s "Validation against published references"
/// section, generalized so RF (Razavi/Lee), quantum (Nielsen/Chuang),
/// and electrical (Sedra/Smith) entries all fit.
#[derive(Clone, Debug)]
pub struct Reference {
    /// Markdown — citation with DOI link if available.
    pub citation: String,
    /// Markdown — formula or claim being checked, in TeX.
    pub formula: String,
    /// What the reference predicts (string so units / scientific
    /// notation render exactly as the author wrote them).
    pub predicted: String,
    /// What the simulator produced.
    pub simulated: String,
    /// Pass/fail tick for the `:-:` column.
    pub passed: bool,
}

/// A floorplan / device diagram. The harness embeds it; it never
/// renders one itself. Use [`FloorplanSource::Svg`] when the caller
/// has an SVG (from `eda-viz::layout::render_to_svg`, hand-built,
/// or otherwise); [`FloorplanSource::None`] for pure-behavioral
/// runs that don't have a layout.
#[derive(Clone, Debug)]
pub enum FloorplanSource {
    None,
    Svg {
        svg: String,
        /// Free-text caption rendered under the embedded image.
        caption: String,
    },
}

/// An optional report panel — for things the harness doesn't have a
/// first-class concept for (Mermaid diagrams, FFT plots, schematic
/// images, Bloch-sphere snapshots, …). Inserted between the chart
/// grid and the step-by-step trace table.
#[derive(Clone, Debug)]
pub struct Panel {
    pub heading: String,
    pub body: PanelBody,
}

#[derive(Clone, Debug)]
pub enum PanelBody {
    /// Raw markdown — useful for tables, blockquotes, Mermaid
    /// diagrams, anything Mathjax-rendered.
    Markdown(String),
    /// SVG body — the harness writes it to an asset file alongside
    /// the charts and embeds it via `![](...)`.
    Svg { file_slug: String, body: String },
}

/// Caller-supplied metadata. Everything in here is prose; the
/// harness fills in numbers from the trace.
#[derive(Clone, Debug)]
pub struct ReportMeta {
    pub title: String,
    /// Short identifier for the device under test — appears in the
    /// header as `Circuit: <circuit>`. Convention: `"Lna
    /// (spike-lna)"`.
    pub circuit: String,
    pub domain: Domain,
    /// Markdown — the "what is this device" intro paragraph.
    /// Rendered verbatim under a `## Background` heading.
    pub intro_md: String,
    /// Markdown — the optimization objective in math + words.
    /// Rendered verbatim under `## Objective`. Empty string skips
    /// the section.
    pub objective_md: String,
    /// Optional notes section (constraints, conventions, anything
    /// not tied to a chart). Rendered under `## Notes`.
    pub notes_md: String,
    /// Literature-validation rows. Empty list skips the table.
    pub references: Vec<Reference>,
}

impl ReportMeta {
    pub fn new(
        title: impl Into<String>,
        circuit: impl Into<String>,
        domain: Domain,
    ) -> Self {
        Self {
            title: title.into(),
            circuit: circuit.into(),
            domain,
            intro_md: String::new(),
            objective_md: String::new(),
            notes_md: String::new(),
            references: Vec::new(),
        }
    }

    pub fn with_intro(mut self, md: impl Into<String>) -> Self {
        self.intro_md = md.into();
        self
    }
    pub fn with_objective(mut self, md: impl Into<String>) -> Self {
        self.objective_md = md.into();
        self
    }
    pub fn with_notes(mut self, md: impl Into<String>) -> Self {
        self.notes_md = md.into();
        self
    }
    pub fn with_references(mut self, refs: Vec<Reference>) -> Self {
        self.references = refs;
        self
    }
}

/// One report bundle ready to write.
pub struct Report<'a> {
    pub trace: &'a Trace,
    pub meta: &'a ReportMeta,
    pub charts: &'a [ChartSpec],
    pub floorplan: FloorplanSource,
    pub extra_panels: Vec<Panel>,
}

impl Report<'_> {
    /// Write the markdown report and every asset to `out_dir`.
    /// Layout:
    ///
    /// ```text
    ///   <out_dir>/
    ///     <name>.md            ← the report
    ///     <name>.csv           ← trace CSV
    ///     assets/<name>/
    ///       floorplan.svg      ← if FloorplanSource::Svg
    ///       floorplan.png      ← if `png` feature on
    ///       <chart_slug>.svg   ← one per ChartSpec
    ///       <chart_slug>.png   ← if `png` feature on
    ///       <panel_slug>.svg   ← one per PanelBody::Svg
    /// ```
    ///
    /// `out_dir` is the *crate's `docs/` directory* by convention; the
    /// `assets/<name>/` subfolder keeps multiple runs from the same
    /// crate from clobbering each other's images. `name` is taken
    /// from the [`crate::TraceCfg`] used to build the trace and is
    /// passed in as `run_name` here.
    pub fn write(&self, out_dir: impl AsRef<Path>, run_name: &str) -> std::io::Result<()> {
        let out_dir = out_dir.as_ref();
        let assets_dir: PathBuf = out_dir.join("assets").join(run_name);
        fs::create_dir_all(&assets_dir)?;

        // Floorplan.
        let floorplan_md = match &self.floorplan {
            FloorplanSource::None => String::new(),
            FloorplanSource::Svg { svg, caption } => {
                let svg_path = assets_dir.join("floorplan.svg");
                fs::write(&svg_path, svg)?;
                rasterize_to_png(svg, &assets_dir.join("floorplan.png"))?;
                format!(
                    "## Floorplan\n\n\
                     ![{caption}](assets/{run_name}/floorplan.svg)\n\n\
                     {caption}\n\n",
                    caption = caption,
                    run_name = run_name,
                )
            }
        };

        // Charts.
        let mut chart_section = String::new();
        if !self.charts.is_empty() {
            chart_section.push_str("## Charts\n\n");
            for spec in self.charts {
                let svg = render_chart_svg(spec, self.trace);
                let svg_path = assets_dir.join(format!("{}.svg", spec.file_slug));
                fs::write(&svg_path, &svg)?;
                rasterize_to_png(&svg, &assets_dir.join(format!("{}.png", spec.file_slug)))?;
                chart_section.push_str(&format!(
                    "### {title}\n\n![{title}](assets/{run_name}/{slug}.svg)\n\n",
                    title = spec.title,
                    run_name = run_name,
                    slug = spec.file_slug,
                ));
            }
        }

        // Extra panels.
        let mut extras = String::new();
        for p in &self.extra_panels {
            extras.push_str(&format!("## {}\n\n", p.heading));
            match &p.body {
                PanelBody::Markdown(md) => {
                    extras.push_str(md);
                    if !md.ends_with('\n') {
                        extras.push('\n');
                    }
                    extras.push('\n');
                }
                PanelBody::Svg { file_slug, body } => {
                    let path = assets_dir.join(format!("{file_slug}.svg"));
                    fs::write(&path, body)?;
                    rasterize_to_png(body, &assets_dir.join(format!("{file_slug}.png")))?;
                    extras.push_str(&format!(
                        "![{heading}](assets/{run_name}/{slug}.svg)\n\n",
                        heading = p.heading,
                        run_name = run_name,
                        slug = file_slug,
                    ));
                }
            }
        }

        // Trace table + CSV.
        let csv = self.trace.to_csv();
        let csv_path = out_dir.join(format!("{run_name}.csv"));
        fs::write(&csv_path, &csv)?;
        let trace_table = render_trace_table(self.trace);

        // Outcome summary (first vs last row of the trace).
        let outcome = render_outcome(self.trace);

        // Compose markdown.
        let mut md = String::new();
        md.push_str(&format!("# {}\n\n", self.meta.title));
        md.push_str(&format!(
            "**Circuit:** {}  \n**Domain:** {}  \n**Steps:** {} ({} rows logged)\n\n",
            self.meta.circuit,
            self.meta.domain.label(),
            self.trace.rows.last().map(|r| r.step).unwrap_or(0),
            self.trace.rows.len(),
        ));
        if !self.meta.intro_md.is_empty() {
            md.push_str("## Background\n\n");
            md.push_str(&self.meta.intro_md);
            ensure_blankline(&mut md);
        }
        if !self.meta.objective_md.is_empty() {
            md.push_str("## Objective\n\n");
            md.push_str(&self.meta.objective_md);
            ensure_blankline(&mut md);
        }
        if !self.meta.notes_md.is_empty() {
            md.push_str("## Notes\n\n");
            md.push_str(&self.meta.notes_md);
            ensure_blankline(&mut md);
        }
        md.push_str(&floorplan_md);
        md.push_str("## Optimization outcome\n\n");
        md.push_str(&outcome);
        md.push_str("\n");
        md.push_str(&chart_section);
        md.push_str(&extras);
        if !self.meta.references.is_empty() {
            md.push_str("## Validation against published references\n\n");
            md.push_str(&render_reference_table(&self.meta.references));
            md.push_str("\n");
        }
        md.push_str("## Step-by-step trace\n\n");
        md.push_str(&trace_table);
        md.push_str(&format!(
            "\n_Full trace as CSV: [`{run_name}.csv`]({run_name}.csv)._\n",
            run_name = run_name,
        ));

        fs::write(out_dir.join(format!("{run_name}.md")), md)?;
        Ok(())
    }
}

fn ensure_blankline(md: &mut String) {
    if !md.ends_with("\n\n") {
        if !md.ends_with('\n') {
            md.push('\n');
        }
        md.push('\n');
    }
}

fn render_outcome(trace: &Trace) -> String {
    if trace.rows.is_empty() {
        return "_(no rows logged)_\n".to_string();
    }
    let first = &trace.rows[0];
    let last = trace.rows.last().unwrap();
    let mut out = String::new();
    out.push_str("| Series | Initial | Final | Δ |\n");
    out.push_str("| --- | ---: | ---: | ---: |\n");
    for s in &trace.series_order {
        let v0 = first.get(s);
        let v1 = last.get(s);
        let dv = v1 - v0;
        out.push_str(&format!(
            "| `{s}` | {v0} | {v1} | {dv} |\n",
            v0 = format_smart(v0),
            v1 = format_smart(v1),
            dv = format_smart(dv),
        ));
    }
    out
}

fn render_trace_table(trace: &Trace) -> String {
    if trace.rows.is_empty() {
        return "_(empty)_\n".to_string();
    }
    let mut out = String::new();
    out.push_str("| step |");
    for s in &trace.series_order {
        out.push_str(&format!(" `{s}` |"));
    }
    out.push('\n');
    out.push_str("| ---: |");
    for _ in &trace.series_order {
        out.push_str(" ---: |");
    }
    out.push('\n');
    for r in &trace.rows {
        out.push_str(&format!("| {} |", r.step));
        for s in &trace.series_order {
            out.push_str(&format!(" {} |", format_smart(r.get(s))));
        }
        out.push('\n');
    }
    out
}

fn render_reference_table(refs: &[Reference]) -> String {
    let mut out = String::new();
    out.push_str("| Reference | Formula | Predicted | Simulated | Pass |\n");
    out.push_str("| --- | --- | ---: | ---: | :---: |\n");
    for r in refs {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            r.citation,
            r.formula,
            r.predicted,
            r.simulated,
            if r.passed { "✓" } else { "✗" },
        ));
    }
    out
}

fn format_smart(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if !v.is_finite() {
        format!("{v}")
    } else if v.abs() < 1e-3 || v.abs() >= 1e5 {
        format!("{v:.4e}")
    } else if v.abs() >= 100.0 {
        format!("{v:.3}")
    } else {
        format!("{v:.6}")
    }
}

// ── PNG rasterization (feature-gated) ─────────────────────────────

#[cfg(feature = "png")]
fn rasterize_to_png(svg: &str, out: &Path) -> std::io::Result<()> {
    match eda_viz::png::svg_to_png(svg, 2.0) {
        Ok(bytes) => fs::write(out, bytes),
        Err(e) => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("svg_to_png failed: {e:?}"),
        )),
    }
}

#[cfg(not(feature = "png"))]
fn rasterize_to_png(_svg: &str, _out: &Path) -> std::io::Result<()> {
    // PNG rasterization disabled — silently skip. The SVG already
    // landed; consumers wanting PNGs enable the `png` feature.
    Ok(())
}
