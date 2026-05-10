//! Aggregating reporter — HTML + Markdown + PNG.
//!
//! Cicsim's `make summary` rolls per-corner outputs into:
//!  - `results/<tb>_<corner>.html` (single corner: meas + spec)
//!  - `README.md` (cross-corner summary table, pass/fail per spec)
//!  - PNG plots of each waveform.
//!
//! Mirror the same shape so cicsim users can drop our output into their
//! existing dashboards.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use eda_waveform::Waveform;

use crate::harness::RunOutcome;
use crate::measure::MeasurementValue;
use crate::spec::{SpecBundle, SpecCheck, SpecFail};

#[derive(Debug)]
pub struct Reporter<'a> {
    pub testbench_name: &'a str,
    pub specs: &'a SpecBundle,
    pub outcomes: &'a [RunOutcome],
    pub output_dir: PathBuf,
    /// How MC distributions are summarized in the cross-corner views
    /// and gated against specs. Default `ThreeStd` matches cicsim's
    /// production-grade method and is what gates `n_pass / n_total`.
    pub mc_style: McSummaryStyle,
}

#[derive(Debug, Clone)]
pub struct SummaryRow {
    pub spec_name: String,
    pub unit: Option<String>,
    /// Per-corner (label, check) entries in input order.
    pub per_corner: Vec<(String, SpecCheck)>,
}

impl<'a> Reporter<'a> {
    pub fn new(
        testbench_name: &'a str,
        specs: &'a SpecBundle,
        outcomes: &'a [RunOutcome],
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            testbench_name, specs, outcomes,
            output_dir: output_dir.into(),
            mc_style: McSummaryStyle::default(),
        }
    }

    /// Override the MC summary method (default is `ThreeStd`).
    pub fn mc_style(mut self, style: McSummaryStyle) -> Self {
        self.mc_style = style;
        self
    }

    /// Roll the per-corner spec checks into per-spec rows for the
    /// summary table.
    pub fn rows(&self) -> Vec<SummaryRow> {
        let mut rows = Vec::with_capacity(self.specs.specs.len());
        for spec in &self.specs.specs {
            let mut per_corner = Vec::with_capacity(self.outcomes.len());
            for o in self.outcomes {
                let check = o.spec_checks.iter()
                    .find(|(n, _)| n == &spec.name)
                    .map(|(_, c)| *c)
                    .unwrap_or(SpecCheck::Skipped);
                per_corner.push((o.corner.label.clone(), check));
            }
            rows.push(SummaryRow {
                spec_name: spec.name.clone(),
                unit: spec.unit.clone(),
                per_corner,
            });
        }
        rows
    }

    /// Write everything: per-corner HTML+PNG+deck+log, top-level
    /// Markdown summary, summary HTML, summary PDF (with embedded
    /// plots). Returns paths in write order.
    pub fn write_all(&self) -> std::io::Result<Vec<PathBuf>> {
        std::fs::create_dir_all(&self.output_dir)?;
        let mut written = Vec::new();
        let tb = self.testbench_name;
        let multi_view = self.outcomes.iter().map(|o| o.corner.view).collect::<std::collections::HashSet<_>>().len() > 1;

        for o in self.outcomes {
            let stem = corner_stem(tb, o, multi_view);
            // Per-corner HTML.
            let p = self.output_dir.join(format!("{stem}.html"));
            std::fs::write(&p, render_corner_html(tb, o, self.specs, multi_view))?;
            written.push(p);

            // Waveform PNG.
            if let Some(w) = &o.waveform {
                let p = self.output_dir.join(format!("{stem}.png"));
                if let Err(e) = render_waveform_png(w, &p, &format!("{tb} – {}", &stem[tb.len()+1..])) {
                    eprintln!("warn: png render failed for {}: {e}", o.corner.label);
                } else {
                    written.push(p);
                }
            }

            // Deck snapshot — exact text submitted to ngspice.
            let p = self.output_dir.join(format!("{stem}.deck.spice"));
            std::fs::write(&p, &o.deck)?;
            written.push(p);

            // ngspice stdout log.
            let p = self.output_dir.join(format!("{stem}.log"));
            std::fs::write(&p, &o.stdout)?;
            written.push(p);
        }

        let rows = self.rows();
        let p = self.output_dir.join("README.md");
        std::fs::write(&p, render_summary_markdown(tb, &rows, self.outcomes, self.specs, self.mc_style))?;
        written.push(p);

        let p = self.output_dir.join(format!("{tb}_summary.html"));
        std::fs::write(&p, render_summary_html(tb, &rows, self.outcomes, self.specs, self.mc_style))?;
        written.push(p);

        let p = self.output_dir.join(format!("{tb}_summary.pdf"));
        match render_summary_pdf(tb, &rows, self.outcomes, self.specs, self.mc_style, &self.output_dir, &p) {
            Ok(()) => written.push(p),
            Err(e) => eprintln!("warn: pdf render failed: {e}"),
        }

        // CSV per (view, kind) bucket — cicsim-compatible shape so
        // tooling that consumes `tran_Lay_typical.csv` can read ours.
        for csv_path in render_csv_per_bucket(tb, self.outcomes, &self.output_dir)? {
            written.push(csv_path);
        }
        Ok(written)
    }
}

/// Resolve `<manifest_dir>/docs/` for a given crate. Convenience for
/// testbench integration tests that publish their summary into the
/// owning crate's `docs/` directory.
///
/// Pass `env!("CARGO_MANIFEST_DIR")` from a test or example.
pub fn docs_dir_for_crate(manifest_dir: impl Into<PathBuf>) -> PathBuf {
    manifest_dir.into().join("docs")
}

use crate::corner::{CornerKind, View};
use crate::spec::{McSummaryStyle, Spec};

/// Engineering-notation pretty printer with a unit-prefix scale (m, µ,
/// n, p, k, M, G). Falls back to bare `{:.4}` for values inside ±[1, 1k]
/// and `{:.4e}` for the unscaled outliers (negative, NaN, …).
fn fmt_value(v: f64, unit: Option<&str>) -> String {
    let unit_s = unit.unwrap_or("");
    if !v.is_finite() { return format!("{v}"); }
    let av = v.abs();
    if av == 0.0 {
        return if unit_s.is_empty() { "0".into() } else { format!("0 {unit_s}") };
    }
    // Engineering prefix table.
    let prefixes: &[(f64, &str)] = &[
        (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
        (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f"),
    ];
    let (scale, prefix) = prefixes.iter().copied()
        .find(|(s, _)| av >= *s * 0.999)
        .unwrap_or((1e-15, "f"));
    let scaled = v / scale;
    let combined_unit = format!("{prefix}{unit_s}");
    let trimmed = combined_unit.trim();
    if trimmed.is_empty() {
        format!("{scaled:.3}")
    } else {
        format!("{scaled:.3} {trimmed}")
    }
}

fn check_cell(c: SpecCheck, unit: Option<&str>) -> (String, &'static str) {
    match c {
        SpecCheck::Pass { measured } => (fmt_value(measured, unit), "pass"),
        SpecCheck::Fail { measured, reason } => {
            let suffix = match reason {
                SpecFail::BelowMin(lo) => format!(" (< {})", fmt_value(lo, unit)),
                SpecFail::AboveMax(hi) => format!(" (> {})", fmt_value(hi, unit)),
            };
            (format!("{}{}", fmt_value(measured, unit), suffix), "fail")
        }
        SpecCheck::Skipped => ("—".into(), "skip"),
    }
}

/// Cicsim-compatible CSV per `(view, kind)` bucket. Filename is
/// `<tb>_<view>_<kind>.csv` (e.g. `lelo_ex_Sch_typical.csv`,
/// `lelo_ex_Lay_mc.csv`). Header row matches cicsim exactly:
///
/// ```text
/// ,<meas1>,<meas2>,...,name,type,time,OK
/// 0,<f64>,<f64>,...,<run_name>,Sch|Lay,<wallclock>,True|False
/// ```
///
/// Empty first column (the pandas integer index), measurement values
/// in raw SI floats, `name` = run identifier, `type` = view tag,
/// `time` = wallclock-at-run, `OK` = boolean per-run spec pass.
fn render_csv_per_bucket(
    tb: &str,
    outcomes: &[RunOutcome],
    output_dir: &Path,
) -> std::io::Result<Vec<PathBuf>> {
    use std::collections::BTreeMap;

    // Bucket: (view_tag, kind_tag) → list of outcomes.
    let mut buckets: BTreeMap<(String, &'static str), Vec<&RunOutcome>> = BTreeMap::new();
    for o in outcomes {
        let kind_tag = match o.corner.kind {
            CornerKind::Typical => "typical",
            CornerKind::Etc => "etc",
            CornerKind::Mc => "mc",
        };
        let view_tag = o.corner.view.as_str().to_string();
        buckets.entry((view_tag, kind_tag)).or_default().push(o);
    }

    let mut written = Vec::new();
    for ((view, kind), bucket) in buckets {
        let p = output_dir.join(format!("{tb}_{view}_{kind}.csv"));
        std::fs::write(&p, render_one_csv(&bucket))?;
        written.push(p);
    }
    Ok(written)
}

fn render_one_csv(outcomes: &[&RunOutcome]) -> String {
    use std::collections::BTreeSet;

    // Union of measurement names across the bucket — preserves a
    // stable column order regardless of which run defines which key.
    let mut all_names: BTreeSet<&str> = BTreeSet::new();
    for o in outcomes {
        for n in o.measures.values.keys() { all_names.insert(n.as_str()); }
    }
    let cols: Vec<&str> = all_names.into_iter().collect();

    let mut s = String::new();
    // Header: empty first cell (pandas index), measurements, then
    // verification counts (`-` for verifiers that didn't run, integer
    // for those that did), then cicsim-shape `name,type,time,OK`.
    let _ = std::fmt::Write::write_str(&mut s, "");
    for c in &cols {
        let _ = std::fmt::Write::write_fmt(&mut s, format_args!(",{c}"));
    }
    let _ = std::fmt::Write::write_str(&mut s, ",drc,lvs,em,name,type,time,OK\n");

    for (idx, o) in outcomes.iter().enumerate() {
        let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{idx}"));
        for c in &cols {
            let v = match o.measures.values.get(*c) {
                Some(MeasurementValue::Number(v)) => format!("{v}"),
                _ => "".into(),
            };
            let _ = std::fmt::Write::write_fmt(&mut s, format_args!(",{v}"));
        }
        // Verifier columns: dash when not run, count otherwise.
        let v_cell = |r: Option<&crate::VerifierResult>| -> String {
            r.map_or_else(|| "-".to_string(), |x| x.count.to_string())
        };
        let _ = std::fmt::Write::write_fmt(
            &mut s,
            format_args!(",{drc},{lvs},{em}",
                drc = v_cell(o.verify.drc.as_ref()),
                lvs = v_cell(o.verify.lvs.as_ref()),
                em  = v_cell(o.verify.em.as_ref()),
            ),
        );
        // Wallclock as a simple ISO-8601 string. We don't pull in
        // chrono just for this; raw seconds-since-epoch is honest and
        // round-trips through pandas without a parser.
        let secs = o.ran_at
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let _ = std::fmt::Write::write_fmt(
            &mut s,
            format_args!(",{name},{view},{secs},{ok}\n",
                name = o.corner.label,
                view = o.corner.view.as_str(),
                ok = if o.ok() { "True" } else { "False" }),
        );
    }
    s
}

/// Build the file-stem for one corner's artifacts. When `multi_view`
/// is true (the CornerSet contains both Sch and Lay corners), the
/// view tag is interleaved: `<tb>_<View>_<label>`. Otherwise we keep
/// the simpler `<tb>_<label>` form so single-view runs don't get
/// noisier filenames just because the View enum exists.
fn corner_stem(tb: &str, o: &RunOutcome, multi_view: bool) -> String {
    if multi_view {
        format!("{tb}_{}_{}", o.corner.view.as_str(), o.corner.label)
    } else {
        format!("{tb}_{}", o.corner.label)
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
        .replace('"', "&quot;").replace('\'', "&#39;")
}

/// Inline SVG showing where `measured` sits within `[min, max]`.
/// Width = 200 px. Used in the per-corner measurement table.
fn range_bar_svg(measured: f64, spec: &Spec) -> String {
    let lo = spec.min;
    let hi = spec.max;
    let mut s = String::from("<svg width=200 height=24 viewBox=\"0 0 200 24\" class=range>");
    s.push_str("<rect x=10 y=10 width=180 height=4 fill=#e1e4e8 rx=2/>");
    if let (Some(l), Some(h)) = (lo, hi) {
        let span = h - l;
        if span > 0.0 {
            // Spec-pass band: full bar.
            s.push_str("<rect x=10 y=10 width=180 height=4 fill=#c2f0c2 rx=2/>");
            let frac = ((measured - l) / span).clamp(-0.2, 1.2);
            let x = 10.0 + frac * 180.0;
            let cls = if (lo.map_or(true, |lo| measured >= lo))
                && (hi.map_or(true, |hi| measured <= hi))
            { "ok" } else { "bad" };
            let _ = std::fmt::Write::write_fmt(
                &mut s, format_args!("<circle cx={x:.1} cy=12 r=5 class=mark-{cls}/>"),
            );
            // Tick marks at typ if defined.
            if let Some(t) = spec.typ {
                let tx = 10.0 + ((t - l) / span).clamp(0.0, 1.0) * 180.0;
                let _ = std::fmt::Write::write_fmt(
                    &mut s, format_args!("<line x1={tx:.1} y1=6 x2={tx:.1} y2=18 stroke=#586069 stroke-width=1 stroke-dasharray=\"2,2\"/>"),
                );
            }
        }
    }
    s.push_str("</svg>");
    s
}

/// Inline SVG histogram of MC draws for a single spec. Returns None
/// when there aren't enough draws (< 3) to be meaningful.
fn mc_histogram_svg(values: &[f64], spec: &Spec) -> Option<String> {
    if values.len() < 3 { return None; }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let (min, max) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    // Domain padding: spec bounds, ± 1 σ around mean, observed range.
    let mut lo = min.min(mean - std);
    let mut hi = max.max(mean + std);
    if let Some(l) = spec.min { lo = lo.min(l); }
    if let Some(h) = spec.max { hi = hi.max(h); }
    let pad = (hi - lo) * 0.05;
    lo -= pad; hi += pad;
    if hi - lo < f64::EPSILON { return None; }

    let w = 320.0_f64;
    let h = 80.0_f64;
    let to_x = |v: f64| -> f64 { 10.0 + ((v - lo) / (hi - lo)) * (w - 20.0) };

    let mut s = String::new();
    let _ = std::fmt::Write::write_fmt(&mut s,
        format_args!("<svg width={w} height={h} viewBox=\"0 0 {w} {h}\" class=hist>"));
    // Spec range band.
    if let (Some(min_b), Some(max_b)) = (spec.min, spec.max) {
        let x1 = to_x(min_b); let x2 = to_x(max_b);
        let _ = std::fmt::Write::write_fmt(&mut s,
            format_args!("<rect x={x1:.1} y=20 width={width:.1} height=40 fill=#c2f0c2 opacity=0.45/>",
                width = (x2 - x1).max(0.0)));
    }
    // Baseline.
    let _ = std::fmt::Write::write_fmt(&mut s,
        format_args!("<line x1=10 y1=60 x2={right:.1} y2=60 stroke=#999 stroke-width=1/>", right = w - 10.0));
    // Individual draws.
    for v in values {
        let x = to_x(*v);
        let _ = std::fmt::Write::write_fmt(&mut s,
            format_args!("<circle cx={x:.1} cy=60 r=3 fill=#0366d6 opacity=0.7/>"));
    }
    // Mean line.
    let xm = to_x(mean);
    let _ = std::fmt::Write::write_fmt(&mut s,
        format_args!("<line x1={xm:.1} y1=15 x2={xm:.1} y2=65 stroke=#b31d28 stroke-width=2/>"));
    // ±σ band.
    let xl = to_x(mean - std); let xh = to_x(mean + std);
    let _ = std::fmt::Write::write_fmt(&mut s,
        format_args!("<rect x={xl:.1} y=55 width={width:.1} height=10 fill=#b31d28 opacity=0.18/>",
            width = (xh - xl).max(0.0)));
    // Labels.
    let unit = spec.unit.as_deref().unwrap_or("");
    let _ = std::fmt::Write::write_fmt(&mut s,
        format_args!("<text x=10 y=14 font-size=10 fill=#586069>µ={mean_s}, σ={std_s} ({n} draws)</text>",
            mean_s = fmt_value(mean, Some(unit)).replace(' ', "\u{00a0}"),
            std_s = fmt_value(std, Some(unit)).replace(' ', "\u{00a0}"),
            n = values.len()));
    s.push_str("</svg>");
    Some(s)
}

/// Pull the MC `f64` draws for `spec_name` out of the outcomes list.
fn mc_values(outcomes: &[RunOutcome], spec_name: &str) -> Vec<f64> {
    outcomes.iter()
        .filter(|o| o.corner.kind == CornerKind::Mc)
        .filter_map(|o| match o.measures.get(spec_name) {
            Some(MeasurementValue::Number(v)) => Some(v),
            _ => None,
        })
        .collect()
}

fn delta_from_typ(measured: f64, spec: &Spec) -> Option<(f64, &'static str)> {
    let t = spec.typ?;
    if t.abs() < f64::EPSILON { return None; }
    let rel = (measured - t) / t.abs();
    let cls = if rel.abs() <= 0.05 { "delta-ok" }
        else if rel.abs() <= 0.20 { "delta-warn" }
        else { "delta-bad" };
    Some((rel * 100.0, cls))
}

fn render_corner_html(tb_name: &str, o: &RunOutcome, specs: &SpecBundle, multi_view: bool) -> String {
    let mut s = String::new();
    s.push_str("<!doctype html><meta charset=utf-8>");
    let _ = writeln!(s, "<title>{} – {}</title>", tb_name, o.corner.label);
    s.push_str(STYLE);
    let view_chip = if multi_view {
        format!(" <span class=tag-{view_lc}>{}</span>", o.corner.view.as_str(), view_lc = o.corner.view.as_str().to_lowercase())
    } else { String::new() };
    let _ = writeln!(s, "<h1>{} <span class=corner-tag>{}</span>{view_chip}</h1>", tb_name, o.corner.label);

    // Run metadata card.
    s.push_str("<div class=card><table class=meta-tbl>");
    let _ = writeln!(s, "<tr><th>kind<td>{:?}<th>lib section<td>{}", o.corner.kind, o.corner.lib_section);
    let _ = writeln!(s, "<tr><th>vdd<td>{:.3} V<th>temp<td>{:.1} °C", o.corner.vdd, o.corner.temp_c);
    if let Some(seed) = o.corner.seed {
        let _ = writeln!(s, "<tr><th>seed<td>{seed}<th>sha<td><code>{}</code>", &o.sha[..12]);
    } else {
        let _ = writeln!(s, "<tr><th>sha<td colspan=3><code>{}</code>", &o.sha[..12]);
    }
    let stem = corner_stem(tb_name, o, multi_view);
    let _ = writeln!(s, "<tr><th>cached<td>{}<th>artifacts<td><a href=\"{stem}.deck.spice\">deck</a> · <a href=\"{stem}.log\">log</a> · <a href=\"{stem}.png\">png</a>",
        if o.from_cache { "yes" } else { "no" });
    s.push_str("</table></div>");

    // Measurements table with range bars.
    s.push_str("<h2>Measurements</h2><table class=meas><tr><th>name<th>measured<th>min<th>typ<th>max<th>Δ vs typ<th>position");
    for (name, value) in &o.measures.values {
        let spec = specs.find(name);
        let unit_s = spec.and_then(|s| s.unit.as_deref());
        let measured_s = match value {
            MeasurementValue::Number(v) => fmt_value(*v, unit_s),
            MeasurementValue::Failed => "<span class=fail>failed</span>".into(),
        };
        let measured_n = match value {
            MeasurementValue::Number(v) => Some(*v),
            MeasurementValue::Failed => None,
        };
        let (min_s, typ_s, max_s) = match spec {
            Some(sp) => (
                sp.min.map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into()),
                sp.typ.map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into()),
                sp.max.map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into()),
            ),
            None => ("—".into(), "—".into(), "—".into()),
        };
        let delta_cell = match (measured_n, spec) {
            (Some(m), Some(sp)) => match delta_from_typ(m, sp) {
                Some((pct, cls)) => format!("<span class={cls}>{:+.1}%</span>", pct),
                None => "—".into(),
            },
            _ => "—".into(),
        };
        let bar = match (measured_n, spec) {
            (Some(m), Some(sp)) => range_bar_svg(m, sp),
            _ => "".into(),
        };
        let _ = writeln!(s, "<tr><td>{name}<td>{measured_s}<td>{min_s}<td>{typ_s}<td>{max_s}<td>{delta_cell}<td>{bar}");
    }
    s.push_str("</table>");

    // Spec checks summary.
    s.push_str("<h2>Spec checks</h2><table class=checks><tr><th>name<th>result");
    for (name, check) in &o.spec_checks {
        let unit_s = specs.find(name).and_then(|sp| sp.unit.as_deref());
        let (txt, cls) = check_cell(*check, unit_s);
        let _ = writeln!(s, "<tr><td>{name}<td class={cls}>{txt}");
    }
    s.push_str("</table>");

    // Verification: only show the section when at least one verifier
    // ran. Empty `VerifyReport` renders nothing — single-source default
    // for testbenches that don't override `verify()`.
    let any_ran = o.verify.drc.is_some() || o.verify.lvs.is_some() || o.verify.em.is_some();
    if any_ran {
        s.push_str("<h2>Verification</h2><table class=checks><tr><th>verifier<th>count<th>first<th>result");
        let row = |s: &mut String, name: &str, r: Option<&crate::VerifierResult>| {
            match r {
                None => {
                    let _ = writeln!(s, "<tr><td>{name}<td>—<td>—<td class=skipped>not run");
                }
                Some(v) => {
                    let cls = if v.is_clean() { "pass" } else { "fail" };
                    let badge = if v.is_clean() { "clean" } else { "violations" };
                    let first = if v.first_message.is_empty() {
                        "—".to_string()
                    } else {
                        html_escape(&v.first_message)
                    };
                    let _ = writeln!(s, "<tr><td>{name}<td>{}<td>{first}<td class={cls}>{badge}",
                        v.count);
                }
            }
        };
        row(&mut s, "DRC", o.verify.drc.as_ref());
        row(&mut s, "LVS", o.verify.lvs.as_ref());
        row(&mut s, "EM",  o.verify.em.as_ref());
        s.push_str("</table>");
    }

    if o.waveform.is_some() {
        let _ = writeln!(s, "<h2>Waveform</h2><img class=wave src=\"{stem}.png\" alt=\"waveform for {lab}\">",
            lab = o.corner.label);
    }

    // Deck source — collapsible.
    let _ = writeln!(s, "<details><summary>SPICE deck ({} lines)</summary><pre class=src>{deck}</pre></details>",
        o.deck.lines().count(), deck = html_escape(&o.deck));

    // ngspice log — collapsible (truncated to 200 lines for sanity).
    let log_lines = o.stdout.lines().count();
    let log_clip: String = o.stdout.lines().take(400).collect::<Vec<_>>().join("\n");
    let truncated = if log_lines > 400 { format!(" (showing first 400 of {} lines — see <a href=\"{stem}.log\">full log</a>)", log_lines) } else { String::new() };
    let _ = writeln!(s, "<details><summary>ngspice log ({log_lines} lines{truncated})</summary><pre class=src>{log}</pre></details>",
        log = html_escape(&log_clip));

    s
}

fn render_summary_markdown(
    tb_name: &str,
    rows: &[SummaryRow],
    outcomes: &[RunOutcome],
    specs: &SpecBundle,
    mc_style: McSummaryStyle,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# {tb_name} – simulation summary");
    let _ = writeln!(s);
    if outcomes.is_empty() {
        let _ = writeln!(s, "_No corners ran._");
        return s;
    }

    // Top-line stats.
    let total_corners = outcomes.len();
    let mc_corners = outcomes.iter().filter(|o| o.corner.kind == CornerKind::Mc).count();
    let _ = writeln!(s, "**{total_corners} corners** ({} typical/etc, {mc_corners} Monte Carlo) · **{} specs** · `{}`",
        total_corners - mc_corners, rows.len(), &outcomes.first().unwrap().sha[..12]);
    let _ = writeln!(s);

    // Cross-corner table — non-MC corners only (MC gets aggregated below).
    let non_mc: Vec<&RunOutcome> = outcomes.iter().filter(|o| o.corner.kind != CornerKind::Mc).collect();
    if !non_mc.is_empty() {
        let _ = writeln!(s, "## Per-corner");
        let _ = writeln!(s);
        let _ = write!(s, "| spec | unit | min | typ | max |");
        for o in &non_mc { let _ = write!(s, " {} |", o.corner.label); }
        let _ = writeln!(s);
        let _ = write!(s, "|---|---|---|---|---|");
        for _ in &non_mc { let _ = write!(s, "---|"); }
        let _ = writeln!(s);

        for row in rows {
            let spec = specs.find(&row.spec_name);
            let unit_s = row.unit.as_deref();
            let min_s = spec.and_then(|s| s.min).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let typ_s = spec.and_then(|s| s.typ).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let max_s = spec.and_then(|s| s.max).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let _ = write!(s, "| {name} | {u} | {min_s} | {typ_s} | {max_s} |",
                name = row.spec_name, u = row.unit.as_deref().unwrap_or(""));
            for o in &non_mc {
                let check = o.spec_checks.iter().find(|(n, _)| n == &row.spec_name).map(|(_, c)| *c).unwrap_or(SpecCheck::Skipped);
                let (txt, cls) = check_cell(check, unit_s);
                let mark = match cls { "pass" => "✓", "fail" => "✗", _ => "·" };
                let _ = write!(s, " {mark} {txt} |");
            }
            let _ = writeln!(s);
        }
        let _ = writeln!(s);
    }

    // Monte Carlo stats per spec.
    if mc_corners > 0 {
        let _ = writeln!(s, "## Monte Carlo ({mc_corners} draws, summary={})", mc_style.as_str());
        let _ = writeln!(s);
        let (low_h, hi_h) = match mc_style {
            McSummaryStyle::ThreeStd => ("µ-3σ", "µ+3σ"),
            McSummaryStyle::MinMax => ("min", "max"),
        };
        let _ = writeln!(s, "| spec | unit | {low_h} | µ | {hi_h} | σ | gate |");
        let _ = writeln!(s, "|---|---|---|---|---|---|---|");
        for row in rows {
            let spec = specs.find(&row.spec_name).cloned().unwrap_or_else(|| Spec {
                name: row.spec_name.clone(), min: None, typ: None, max: None,
                unit: row.unit.clone(),
            });
            let unit_s = row.unit.as_deref();
            let vs = mc_values(outcomes, &row.spec_name);
            if vs.is_empty() {
                let _ = writeln!(s, "| {n} | {u} | — | — | — | — | — |", n = row.spec_name, u = unit_s.unwrap_or(""));
                continue;
            }
            let n = vs.len() as f64;
            let mean = vs.iter().sum::<f64>() / n;
            let std = if vs.len() >= 2 {
                (vs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
            } else { 0.0 };
            let (lo, _typ, hi) = mc_style.summarize(&vs).unwrap_or((mean, mean, mean));
            // Gate: does the worst-case representative still meet spec?
            let gate = match spec.check_mc(&vs, mc_style) {
                SpecCheck::Pass { .. } => "✓ pass",
                SpecCheck::Fail { .. } => "✗ FAIL",
                SpecCheck::Skipped => "·",
            };
            let _ = writeln!(s, "| {name} | {u} | {lo_s} | {mu_s} | {hi_s} | {sigma_s} | {gate} |",
                name = row.spec_name, u = unit_s.unwrap_or(""),
                lo_s = fmt_value(lo, unit_s),
                mu_s = fmt_value(mean, unit_s),
                hi_s = fmt_value(hi, unit_s),
                sigma_s = fmt_value(std, unit_s));
        }
        let _ = writeln!(s);
    }

    // Yield summary.
    let _ = writeln!(s, "## Yield (non-MC)");
    for row in rows {
        let non_mc_runs: Vec<_> = row.per_corner.iter().enumerate()
            .filter(|(i, _)| outcomes.get(*i).map_or(true, |o| o.corner.kind != CornerKind::Mc))
            .map(|(_, c)| c).collect();
        let total = non_mc_runs.len();
        let pass = non_mc_runs.iter().filter(|(_, c)| c.is_pass()).count();
        let _ = writeln!(s, "- **{}**: {pass}/{total} pass", row.spec_name);
    }
    s
}

fn render_summary_html(
    tb_name: &str,
    rows: &[SummaryRow],
    outcomes: &[RunOutcome],
    specs: &SpecBundle,
    mc_style: McSummaryStyle,
) -> String {
    let mut s = String::new();
    s.push_str("<!doctype html><meta charset=utf-8>");
    let _ = writeln!(s, "<title>{tb_name} – summary</title>");
    s.push_str(STYLE);
    let _ = writeln!(s, "<h1>{tb_name} – simulation summary</h1>");
    if outcomes.is_empty() {
        s.push_str("<p>No corners ran.</p>");
        return s;
    }

    let total_corners = outcomes.len();
    let mc_corners = outcomes.iter().filter(|o| o.corner.kind == CornerKind::Mc).count();
    let pass_overall = rows.iter().all(|r| r.per_corner.iter().all(|(_, c)| c.is_pass() || matches!(c, SpecCheck::Skipped)));
    let banner_cls = if pass_overall { "banner-ok" } else { "banner-bad" };
    let banner_txt = if pass_overall { "ALL CORNERS PASS" } else { "SPEC FAILURES" };
    let _ = writeln!(s, "<div class=\"banner {banner_cls}\">{banner_txt}</div>");
    let _ = writeln!(s, "<p class=meta>{total_corners} corners ({} typical/etc, {mc_corners} Monte Carlo) · {} specs</p>",
        total_corners - mc_corners, rows.len());

    // Non-MC cross-corner table with min/typ/max + per-corner Δ-from-typ.
    let non_mc: Vec<(usize, &RunOutcome)> = outcomes.iter().enumerate()
        .filter(|(_, o)| o.corner.kind != CornerKind::Mc).collect();
    if !non_mc.is_empty() {
        s.push_str("<h2>Per-corner specs</h2>");
        s.push_str("<table class=summary><thead><tr><th>spec<th>unit<th>min<th>typ<th>max");
        for (_, o) in &non_mc { let _ = write!(s, "<th>{}", html_escape(&o.corner.label)); }
        s.push_str("</thead>\n");
        for row in rows {
            let spec = specs.find(&row.spec_name);
            let unit_s = row.unit.as_deref();
            let min_s = spec.and_then(|s| s.min).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let typ_s = spec.and_then(|s| s.typ).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let max_s = spec.and_then(|s| s.max).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let _ = write!(s, "<tr><td><strong>{}</strong><td>{}<td>{min_s}<td>{typ_s}<td>{max_s}",
                html_escape(&row.spec_name), unit_s.unwrap_or(""));
            for (i, _) in &non_mc {
                let check = row.per_corner.get(*i).map(|(_, c)| *c).unwrap_or(SpecCheck::Skipped);
                let (txt, cls) = check_cell(check, unit_s);
                let delta_html = match (check, spec) {
                    (SpecCheck::Pass { measured }, Some(sp)) | (SpecCheck::Fail { measured, .. }, Some(sp)) => {
                        match delta_from_typ(measured, sp) {
                            Some((pct, dcls)) => format!("<span class={dcls}>{pct:+.1}%</span>"),
                            None => "".into(),
                        }
                    }
                    _ => "".into(),
                };
                let _ = write!(s, "<td class={cls}><div>{txt}</div><div class=delta>{delta_html}</div>");
            }
            s.push('\n');
        }
        s.push_str("</table>");
    }

    // Monte Carlo histograms.
    if mc_corners > 0 {
        let (low_h, hi_h) = match mc_style {
            McSummaryStyle::ThreeStd => ("µ−3σ", "µ+3σ"),
            McSummaryStyle::MinMax => ("min", "max"),
        };
        let _ = writeln!(s, "<h2>Monte Carlo ({mc_corners} draws, summary <code>{}</code>)</h2>", mc_style.as_str());
        let _ = writeln!(s, "<table class=mc><thead><tr><th>spec<th>distribution<th>{low_h}<th>µ<th>{hi_h}<th>σ<th>raw min<th>raw max<th>gate</thead>");
        for row in rows {
            let spec = specs.find(&row.spec_name);
            let unit_s = row.unit.as_deref();
            let vs = mc_values(outcomes, &row.spec_name);
            if vs.is_empty() {
                let _ = writeln!(s, "<tr><td>{}<td colspan=8 class=skip>no MC draws", html_escape(&row.spec_name));
                continue;
            }
            let n = vs.len() as f64;
            let mean = vs.iter().sum::<f64>() / n;
            let sigma = if vs.len() >= 2 {
                (vs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
            } else { 0.0 };
            let mn = vs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let mx = vs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let (lo, _, hi) = mc_style.summarize(&vs).unwrap_or((mean, mean, mean));
            let (gate_text, gate_cls) = match spec.map(|sp| sp.check_mc(&vs, mc_style)) {
                Some(SpecCheck::Pass { .. }) => ("✓ pass", "pass"),
                Some(SpecCheck::Fail { .. }) => ("✗ FAIL", "fail"),
                _ => ("—", "skip"),
            };
            let hist = match spec {
                Some(sp) => mc_histogram_svg(&vs, sp).unwrap_or_default(),
                None => String::new(),
            };
            let _ = writeln!(s, "<tr><td><strong>{}</strong><td>{hist}<td>{lo_s}<td>{mu_s}<td>{hi_s}<td>{sigma_s}<td>{mn_s}<td>{mx_s}<td class={gate_cls}>{gate_text}",
                html_escape(&row.spec_name),
                lo_s = fmt_value(lo, unit_s),
                mu_s = fmt_value(mean, unit_s),
                hi_s = fmt_value(hi, unit_s),
                sigma_s = fmt_value(sigma, unit_s),
                mn_s = fmt_value(mn, unit_s),
                mx_s = fmt_value(mx, unit_s));
        }
        s.push_str("</table>");
    }

    // Per-corner detail with thumbnail waveforms.
    s.push_str("<h2>Per-corner detail</h2><div class=corner-grid>");
    for o in outcomes {
        let lab = &o.corner.label;
        let kind_tag = match o.corner.kind {
            CornerKind::Typical => "typical",
            CornerKind::Etc => "etc",
            CornerKind::Mc => "mc",
        };
        let pass_count = o.spec_checks.iter().filter(|(_, c)| c.is_pass()).count();
        let total_count = o.spec_checks.len();
        let pf_cls = if pass_count == total_count { "pass" } else { "fail" };
        let png_link = if o.waveform.is_some() {
            format!("<a href=\"{tb_name}_{lab}.html\"><img class=thumb src=\"{tb_name}_{lab}.png\" alt=\"{lab}\"></a>")
        } else { String::new() };
        let _ = writeln!(s,
            "<div class=corner-card><div class=corner-head><a href=\"{tb_name}_{lab}.html\"><strong>{lab}</strong></a> <span class=tag-{kind_tag}>{kind_tag}</span><span class=pf-{pf_cls}>{pass_count}/{total_count}</span></div>{png_link}<div class=meta-mini>vdd={vdd:.3} V · {temp:.0} °C · sha={sha}</div></div>",
            vdd = o.corner.vdd, temp = o.corner.temp_c, sha = &o.sha[..8]);
    }
    s.push_str("</div>");
    s
}

#[cfg(test)]
mod csv_tests {
    use super::*;
    use crate::corner::{Corner, CornerKind, View};
    use crate::measure::MeasureLog;
    use std::collections::BTreeMap;
    use std::time::SystemTime;

    fn synth_outcome(label: &str, kind: CornerKind, view: View, ibn: f64, ok: bool) -> RunOutcome {
        let mut values = BTreeMap::new();
        values.insert("ibn".into(), MeasurementValue::Number(ibn));
        values.insert("vgs_m1".into(), MeasurementValue::Number(0.62));
        let corner = Corner {
            kind, label: label.into(), lib_section: "tt".into(),
            vdd: 1.8, temp_c: 27.0, seed: None, view,
        };
        let spec_check = if ok {
            crate::spec::SpecCheck::Pass { measured: ibn }
        } else {
            crate::spec::SpecCheck::Fail { measured: ibn, reason: crate::spec::SpecFail::AboveMax(20e-6) }
        };
        RunOutcome {
            corner, deck: "* test\n".into(), sha: "abc".into(),
            from_cache: false, stdout: "".into(),
            measures: MeasureLog { values },
            waveform: None,
            spec_checks: vec![("ibn".into(), spec_check)],
            verify: crate::VerifyReport::empty(),
            ran_at: SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000),
        }
    }

    #[test]
    fn csv_header_matches_cicsim_shape() {
        let outcomes = vec![
            synth_outcome("typical", CornerKind::Typical, View::Layout, 21.76e-6, true),
        ];
        let refs: Vec<&RunOutcome> = outcomes.iter().collect();
        let csv = render_one_csv(&refs);
        let first = csv.lines().next().unwrap();
        // verify columns sit between measurements and the cicsim
        // `name,type,time,OK` tail.
        assert_eq!(first, ",ibn,vgs_m1,drc,lvs,em,name,type,time,OK");
    }

    #[test]
    fn csv_verify_columns_dash_when_no_verifier_ran() {
        let outcomes = vec![
            synth_outcome("typical", CornerKind::Typical, View::Layout, 21.76e-6, true),
        ];
        let refs: Vec<&RunOutcome> = outcomes.iter().collect();
        let csv = render_one_csv(&refs);
        let row = csv.lines().nth(1).unwrap();
        assert!(row.contains(",-,-,-,typical,"), "row = {row}");
    }

    #[test]
    fn dirty_verify_flips_ok_to_false_in_csv() {
        let mut o = synth_outcome("lay", CornerKind::Typical, View::Layout, 21.76e-6, true);
        o.verify = crate::VerifyReport::empty()
            .set_drc(crate::VerifierResult::clean())
            .set_em(crate::VerifierResult::from_count(2, "MET1.W"));
        let refs = vec![&o];
        let csv = render_one_csv(&refs);
        let row = csv.lines().nth(1).unwrap();
        assert!(row.contains(",0,-,2,"), "verify counts wrong: {row}");
        // measurement-clean + verify-dirty ⇒ OK = False.
        assert!(row.ends_with(",False"), "OK should flip to False when verify dirty: {row}");
    }

    #[test]
    fn csv_row_carries_values_and_pass_boolean() {
        let outcomes = vec![
            synth_outcome("typical", CornerKind::Typical, View::Layout, 21.76e-6, true),
        ];
        let refs: Vec<&RunOutcome> = outcomes.iter().collect();
        let csv = render_one_csv(&refs);
        let row = csv.lines().nth(1).unwrap();
        assert!(row.starts_with("0,"), "row should start with index 0: {row}");
        assert!(row.contains(",typical,"));
        assert!(row.contains(",Lay,"));
        assert!(row.ends_with(",True"));
    }

    #[test]
    fn csv_row_marks_fail_as_false() {
        let outcomes = vec![
            synth_outcome("ss", CornerKind::Etc, View::Schematic, 30e-6, false),
        ];
        let refs: Vec<&RunOutcome> = outcomes.iter().collect();
        let csv = render_one_csv(&refs);
        let row = csv.lines().nth(1).unwrap();
        assert!(row.ends_with(",False"));
        assert!(row.contains(",Sch,"));
    }

    #[test]
    fn csv_buckets_split_by_view_and_kind() {
        let outcomes = vec![
            synth_outcome("typical", CornerKind::Typical, View::Schematic, 21.76e-6, true),
            synth_outcome("typical", CornerKind::Typical, View::Layout,    21.59e-6, true),
            synth_outcome("ff",      CornerKind::Etc,     View::Schematic, 22.77e-6, true),
            synth_outcome("mc_000",  CornerKind::Mc,      View::Schematic, 23.10e-6, true),
            synth_outcome("mc_001",  CornerKind::Mc,      View::Schematic, 24.50e-6, true),
        ];
        let dir = tempfile::tempdir().unwrap();
        let written = render_csv_per_bucket("lelo_ex", &outcomes, dir.path()).unwrap();
        let names: Vec<String> = written.iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        for required in [
            "lelo_ex_Sch_typical.csv",
            "lelo_ex_Lay_typical.csv",
            "lelo_ex_Sch_etc.csv",
            "lelo_ex_Sch_mc.csv",
        ] {
            assert!(names.iter().any(|n| n == required),
                "missing {required} in CSV bucket output: {names:?}");
        }
        let mc_csv = std::fs::read_to_string(dir.path().join("lelo_ex_Sch_mc.csv")).unwrap();
        assert_eq!(mc_csv.lines().count(), 3, "mc bucket should have header + 2 rows");
    }
}

/// Render a multi-page Letter-sized PDF: page 1 is the spec summary
/// table, then one page per (non-MC) corner with that corner's
/// waveform PNG embedded. Uses printpdf's built-in Helvetica so the
/// font is bundled.
fn render_summary_pdf(
    tb_name: &str,
    rows: &[SummaryRow],
    outcomes: &[crate::harness::RunOutcome],
    specs: &SpecBundle,
    mc_style: McSummaryStyle,
    output_dir: &Path,
    path: &Path,
) -> Result<(), String> {
    use printpdf::{BuiltinFont, ImageTransform, ImageXObject, Mm, PdfDocument, Px, ColorSpace, ColorBits};
    use std::io::BufWriter;

    let (doc, page1, layer1) = PdfDocument::new(
        format!("{tb_name} – simulation summary"),
        Mm(215.9), Mm(279.4), "layer1",
    );
    let font = doc.add_builtin_font(BuiltinFont::Helvetica).map_err(|e| e.to_string())?;
    let bold = doc.add_builtin_font(BuiltinFont::HelveticaBold).map_err(|e| e.to_string())?;

    // -------- page 1: title + spec table --------
    let layer = doc.get_page(page1).get_layer(layer1);
    let left = Mm(15.0);
    let mut y = Mm(265.0);

    layer.use_text(format!("{tb_name} – simulation summary"), 18.0, left, y, &bold);
    y = Mm(y.0 - 8.0);

    let mc_count = outcomes.iter().filter(|o| o.corner.kind == CornerKind::Mc).count();
    layer.use_text(
        format!("{} corners ({} typical/etc, {} MC)    {} specs",
            outcomes.len(), outcomes.len() - mc_count, mc_count, rows.len()),
        10.0, left, y, &font,
    );
    y = Mm(y.0 - 6.0);
    let pass_overall = rows.iter().all(|r| r.per_corner.iter().all(|(_, c)| c.is_pass() || matches!(c, SpecCheck::Skipped)));
    let banner = if pass_overall { "ALL CORNERS PASS" } else { "SPEC FAILURES PRESENT" };
    layer.use_text(banner, 12.0, left, y, &bold);
    y = Mm(y.0 - 9.0);

    if outcomes.is_empty() {
        layer.use_text("no corners ran", 11.0, left, y, &font);
    } else {
        let non_mc: Vec<&crate::harness::RunOutcome> = outcomes.iter()
            .filter(|o| o.corner.kind != CornerKind::Mc).collect();

        // Header.
        layer.use_text("Per-corner", 12.0, left, y, &bold);
        y = Mm(y.0 - 6.0);
        let mut header = format!("{:<12} {:<6} {:<10} {:<10} {:<10}", "spec", "unit", "min", "typ", "max");
        for o in &non_mc { header.push_str(&format!(" {:>12}", o.corner.label)); }
        layer.use_text(header, 8.0, left, y, &bold);
        y = Mm(y.0 - 5.0);

        for row in rows {
            let spec = specs.find(&row.spec_name);
            let unit_s = row.unit.as_deref();
            let mn = spec.and_then(|s| s.min).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let tp = spec.and_then(|s| s.typ).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let mx = spec.and_then(|s| s.max).map(|v| fmt_value(v, unit_s)).unwrap_or_else(|| "—".into());
            let mut line = format!("{:<12} {:<6} {:<10.10} {:<10.10} {:<10.10}",
                row.spec_name, unit_s.unwrap_or(""), mn, tp, mx);
            for o in &non_mc {
                let check = o.spec_checks.iter().find(|(n, _)| n == &row.spec_name).map(|(_, c)| *c).unwrap_or(SpecCheck::Skipped);
                let (txt, _) = check_cell(check, unit_s);
                let mark = match check { SpecCheck::Pass { .. } => "✓", SpecCheck::Fail { .. } => "✗", _ => "·" };
                line.push_str(&format!(" {:>11.11}{}", txt, mark));
            }
            layer.use_text(line, 8.0, left, y, &font);
            y = Mm(y.0 - 4.5);
            if y.0 < 30.0 { break; }
        }

        if mc_count > 0 && y.0 > 50.0 {
            y = Mm(y.0 - 4.0);
            layer.use_text(
                format!("Monte Carlo ({mc_count} draws, summary={})", mc_style.as_str()),
                12.0, left, y, &bold,
            );
            y = Mm(y.0 - 6.0);
            let (lo_h, hi_h) = match mc_style {
                McSummaryStyle::ThreeStd => ("u-3s", "u+3s"),
                McSummaryStyle::MinMax => ("min", "max"),
            };
            layer.use_text(format!("{:<12} {:<6} {:>12} {:>12} {:>12} {:>10} {:>8}",
                "spec", "unit", lo_h, "mean", hi_h, "sigma", "gate"), 8.0, left, y, &bold);
            y = Mm(y.0 - 5.0);
            for row in rows {
                let unit_s = row.unit.as_deref();
                let vs = mc_values(outcomes, &row.spec_name);
                if vs.is_empty() { continue; }
                let n = vs.len() as f64;
                let mean = vs.iter().sum::<f64>() / n;
                let sigma = if vs.len() >= 2 {
                    (vs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
                } else { 0.0 };
                let (lo, _, hi) = mc_style.summarize(&vs).unwrap_or((mean, mean, mean));
                let gate = match specs.find(&row.spec_name).map(|sp| sp.check_mc(&vs, mc_style)) {
                    Some(SpecCheck::Pass { .. }) => "pass",
                    Some(SpecCheck::Fail { .. }) => "FAIL",
                    _ => "-",
                };
                let line = format!("{:<12} {:<6} {:>12.12} {:>12.12} {:>12.12} {:>10.10} {:>8}",
                    row.spec_name, unit_s.unwrap_or(""),
                    fmt_value(lo, unit_s), fmt_value(mean, unit_s),
                    fmt_value(hi, unit_s), fmt_value(sigma, unit_s),
                    gate);
                layer.use_text(line, 8.0, left, y, &font);
                y = Mm(y.0 - 4.5);
                if y.0 < 25.0 { break; }
            }
        }
    }

    // -------- one page per non-MC corner with PNG embed --------
    for o in outcomes.iter().filter(|o| o.corner.kind != CornerKind::Mc) {
        let png_path = output_dir.join(format!("{tb_name}_{}.png", o.corner.label));
        if !png_path.exists() { continue; }
        let png_bytes = match std::fs::read(&png_path) { Ok(b) => b, Err(_) => continue };
        let img = match image::load_from_memory(&png_bytes) { Ok(i) => i, Err(_) => continue };
        let rgb = img.to_rgb8();
        let (w_px, h_px) = rgb.dimensions();
        let xobj = ImageXObject {
            width: Px(w_px as usize),
            height: Px(h_px as usize),
            color_space: ColorSpace::Rgb,
            bits_per_component: ColorBits::Bit8,
            interpolate: false,
            image_data: rgb.into_raw(),
            image_filter: None,
            clipping_bbox: None,
            smask: None,
        };
        let (page, layer_ref) = doc.add_page(Mm(215.9), Mm(279.4), format!("{}_layer", o.corner.label));
        let layer = doc.get_page(page).get_layer(layer_ref);
        layer.use_text(format!("{tb_name} – {}", o.corner.label), 14.0, Mm(15.0), Mm(265.0), &bold);
        layer.use_text(
            format!("kind={:?}  vdd={} V  temp={} °C  lib={}",
                o.corner.kind, o.corner.vdd, o.corner.temp_c, o.corner.lib_section),
            9.0, Mm(15.0), Mm(258.0), &font,
        );
        // Scale the image so width fits 180 mm (Letter minus margins).
        let target_w_mm: f32 = 180.0;
        let target_h_mm: f32 = target_w_mm * (h_px as f32) / (w_px as f32);
        let scale_x: f32 = target_w_mm / (w_px as f32 * 25.4 / 300.0);
        let scale_y: f32 = target_h_mm / (h_px as f32 * 25.4 / 300.0);
        printpdf::Image::from(xobj).add_to_layer(layer.clone(), ImageTransform {
            translate_x: Some(Mm(15.0)),
            translate_y: Some(Mm(258.0 - target_h_mm - 4.0)),
            scale_x: Some(scale_x),
            scale_y: Some(scale_y),
            dpi: Some(300.0),
            ..Default::default()
        });

        // Measurements text below the image.
        let mut yy: f32 = 258.0 - target_h_mm - 12.0;
        for (n, val) in &o.measures.values {
            let unit_s = specs.find(n).and_then(|s| s.unit.as_deref());
            let v_s = match val { MeasurementValue::Number(v) => fmt_value(*v, unit_s), MeasurementValue::Failed => "failed".into() };
            layer.use_text(format!("  {n} = {v_s}"), 9.0, Mm(15.0), Mm(yy), &font);
            yy -= 4.5;
            if yy < 15.0 { break; }
        }
    }

    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    doc.save(&mut BufWriter::new(file)).map_err(|e| e.to_string())?;
    Ok(())
}

const STYLE: &str = "<style>
:root{--ok:#22863a;--bad:#b31d28;--warn:#9a6b00;--mute:#586069}
body{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1.5em;color:#24292e}
h1{font-size:1.6em;margin:.4em 0}
h2{font-size:1.15em;margin:1.6em 0 .5em;border-bottom:1px solid #eee;padding-bottom:.2em}
.corner-tag{display:inline-block;font-size:.55em;padding:.15em .55em;border-radius:1em;background:#f1f8ff;color:#0366d6;vertical-align:middle;margin-left:.5em}
table{border-collapse:collapse;margin:1em 0;width:100%}
th,td{border:1px solid #e1e4e8;padding:.4em .7em;text-align:left;vertical-align:middle}
th{background:#f6f8fa;font-weight:600;font-size:90%}
.pass{background:#e6ffed;color:var(--ok)}
.fail{background:#ffeef0;color:var(--bad);font-weight:600}
.skip{background:#f6f8fa;color:var(--mute)}
.meta{color:var(--mute);font-size:90%;margin:.4em 0 1em}
.banner{display:inline-block;padding:.45em 1em;border-radius:.4em;font-weight:700;letter-spacing:.05em;font-size:.95em;margin:.5em 0}
.banner-ok{background:#dcffe4;color:var(--ok)}
.banner-bad{background:#ffd6d8;color:var(--bad)}
.card{background:#fafbfc;border:1px solid #e1e4e8;border-radius:.4em;padding:.7em 1em;margin:.5em 0 1.5em}
.meta-tbl{width:auto}
.meta-tbl th{background:none;color:var(--mute);font-weight:500;border:0;padding:.15em .6em;text-align:right}
.meta-tbl td{border:0;padding:.15em .6em}
table.summary td>div+div.delta{margin-top:.15em;font-size:80%}
.delta-ok{color:var(--ok)}
.delta-warn{color:var(--warn)}
.delta-bad{color:var(--bad);font-weight:600}
svg.range,svg.hist{display:block}
.range circle.mark-ok{fill:#22863a;stroke:#fff;stroke-width:1.5}
.range circle.mark-bad{fill:#b31d28;stroke:#fff;stroke-width:1.5}
.tag-typical{display:inline-block;font-size:.7em;padding:.05em .5em;border-radius:.8em;background:#e7f3ff;color:#0366d6;margin-left:.4em}
.tag-etc{display:inline-block;font-size:.7em;padding:.05em .5em;border-radius:.8em;background:#fff5d6;color:#735c00;margin-left:.4em}
.tag-mc{display:inline-block;font-size:.7em;padding:.05em .5em;border-radius:.8em;background:#f4d7ff;color:#6e2c91;margin-left:.4em}
.pf-pass{float:right;color:var(--ok);font-weight:600}
.pf-fail{float:right;color:var(--bad);font-weight:700}
.corner-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.7em}
.corner-card{border:1px solid #e1e4e8;border-radius:.35em;padding:.55em .7em;background:#fff}
.corner-head{margin-bottom:.4em}
.corner-head a{text-decoration:none;color:#0366d6}
img.thumb{width:100%;height:auto;border:1px solid #eee;border-radius:.25em}
img.wave{max-width:100%;border:1px solid #e1e4e8;border-radius:.3em}
.meta-mini{color:var(--mute);font-size:78%;margin-top:.4em}
details{margin:1em 0;border:1px solid #e1e4e8;border-radius:.3em;padding:.5em .8em;background:#fafbfc}
details summary{cursor:pointer;font-weight:600;color:#0366d6}
pre.src{font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;background:#f6f8fa;padding:.6em .8em;border-radius:.25em;overflow-x:auto;max-height:480px}
code{font:.92em ui-monospace,Menlo,Consolas,monospace;background:#f6f8fa;padding:.1em .35em;border-radius:.25em}
.hist text{font-family:-apple-system,sans-serif}
</style>";

/// Render the waveform as a stacked-subplot PNG. Series are bucketed
/// into voltage-like (`v(...)`, dimensionless), current-like (`i(...)`
/// or names ending in `#branch`), and "other" — one subplot per
/// non-empty bucket. Time-axis units (s/ms/µs/ns) auto-detected from
/// the axis range and applied as the x label suffix.
fn render_waveform_png(w: &Waveform, path: &Path, title: &str) -> Result<(), String> {
    use plotters::prelude::*;

    // Project complex AC waveforms to real magnitude in dB before plotting.
    let (axis_name, axis, series): (String, Vec<f64>, Vec<(String, Vec<f64>)>) = match w {
        Waveform::Real { axis_name, axis, signals } => {
            let s: Vec<_> = signals.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            (axis_name.clone(), axis.clone(), s)
        }
        Waveform::Complex { axis_name, axis, signals } => {
            let s: Vec<_> = signals.iter().map(|(k, v)| {
                let mag_db: Vec<f64> = v.iter()
                    .map(|(re, im)| 20.0 * ((re * re + im * im).sqrt().max(1e-30)).log10())
                    .collect();
                (format!("|{k}| dB"), mag_db)
            }).collect();
            (axis_name.clone(), axis.clone(), s)
        }
    };
    if axis.is_empty() || series.is_empty() {
        return Err("empty waveform".into());
    }

    // Bucket signals by SI kind so they don't share a y-axis.
    let mut volts: Vec<&(String, Vec<f64>)> = Vec::new();
    let mut currents: Vec<&(String, Vec<f64>)> = Vec::new();
    let mut other: Vec<&(String, Vec<f64>)> = Vec::new();
    for sig in &series {
        let lname = sig.0.to_lowercase();
        if lname.starts_with("v(") || lname.starts_with("|v(") { volts.push(sig); }
        else if lname.starts_with("i(") || lname.ends_with("#branch") { currents.push(sig); }
        else { other.push(sig); }
    }
    let panes: Vec<(&'static str, &'static str, Vec<&(String, Vec<f64>)>)> = [
        ("voltages", "V", volts),
        ("currents", "A", currents),
        ("other", "", other),
    ].into_iter().filter(|(_, _, v)| !v.is_empty()).collect();

    // Time axis label + scale.
    let (xmin_raw, xmax_raw) = (axis[0], *axis.last().unwrap());
    let span = (xmax_raw - xmin_raw).abs().max(f64::MIN_POSITIVE);
    let (x_scale, x_unit) =
        if axis_name == "frequency" { (1.0_f64, "Hz") }
        else if span >= 1.0 { (1.0, "s") }
        else if span >= 1e-3 { (1e-3, "ms") }
        else if span >= 1e-6 { (1e-6, "µs") }
        else if span >= 1e-9 { (1e-9, "ns") }
        else { (1e-12, "ps") };
    let x_label = if axis_name == "time" { format!("time ({x_unit})") }
        else if axis_name == "frequency" { format!("frequency ({x_unit})") }
        else { format!("{axis_name} ({x_unit})") };

    let height = 280 + 220 * (panes.len().saturating_sub(1)) as u32;
    let root = BitMapBackend::new(path, (1000, height)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| e.to_string())?;
    root.titled(title, ("sans-serif", 22)).map_err(|e| e.to_string())?;
    let panels = root.split_evenly((panes.len(), 1));

    let orange = RGBColor(255, 87, 34);
    let teal = RGBColor(0, 150, 136);
    let palette = [&BLUE, &RED, &GREEN, &MAGENTA, &CYAN, &BLACK, &orange, &teal];
    for (idx, ((kind, base_unit, sigs), area)) in panes.iter().zip(panels.iter()).enumerate() {
        // Y range across this pane's signals only.
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for (_, vs) in sigs {
            for v in vs.iter() {
                if v.is_finite() {
                    if *v < ymin { ymin = *v; }
                    if *v > ymax { ymax = *v; }
                }
            }
        }
        if !ymin.is_finite() || !ymax.is_finite() {
            ymin = 0.0; ymax = 1.0;
        }
        if (ymax - ymin).abs() < 1e-30 { ymax = ymin + 1.0; }
        let pad = (ymax - ymin) * 0.08;
        let (yl, yh) = (ymin - pad, ymax + pad);

        // Derive a unit prefix for the y axis from |max|.
        let av = ymax.abs().max(ymin.abs());
        let (y_scale, y_prefix) =
            if av >= 1e3 { (1e3, "k") }
            else if av >= 1.0 { (1.0, "") }
            else if av >= 1e-3 { (1e-3, "m") }
            else if av >= 1e-6 { (1e-6, "µ") }
            else if av >= 1e-9 { (1e-9, "n") }
            else if av >= 1e-12 { (1e-12, "p") }
            else { (1.0, "") };
        let y_label = format!("{kind} ({y_prefix}{base_unit})");

        let mut chart = ChartBuilder::on(area)
            .margin(8)
            .x_label_area_size(if idx + 1 == panes.len() { 38 } else { 18 })
            .y_label_area_size(70)
            .build_cartesian_2d(xmin_raw / x_scale .. xmax_raw / x_scale, yl / y_scale .. yh / y_scale)
            .map_err(|e| e.to_string())?;
        chart.configure_mesh()
            .x_desc(if idx + 1 == panes.len() { x_label.as_str() } else { "" })
            .y_desc(&y_label)
            .light_line_style(RGBColor(245, 245, 245))
            .bold_line_style(RGBColor(220, 220, 220))
            .draw()
            .map_err(|e| e.to_string())?;
        for (i, (name, vs)) in sigs.iter().enumerate() {
            let color = palette[i % palette.len()];
            chart.draw_series(LineSeries::new(
                axis.iter().zip(vs.iter()).map(|(x, y)| (*x / x_scale, *y / y_scale)),
                color.clone().stroke_width(2),
            ))
            .map_err(|e| e.to_string())?
            .label(name.clone())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color.clone().stroke_width(2)));
        }
        chart.configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.85))
            .border_style(RGBColor(200, 200, 200))
            .label_font(("sans-serif", 11))
            .draw()
            .map_err(|e| e.to_string())?;
    }
    root.present().map_err(|e| e.to_string())?;
    Ok(())
}
