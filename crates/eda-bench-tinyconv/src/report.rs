//! Bench report — markdown rendering. Backend × metric arm.
//!
//! v1 emits markdown only. PNG charts + baseline-anchored comparison
//! tables (PLAN.md "External baselines anchored from day one") arrive
//! once there are real numbers to plot.

use std::fmt::Write;
use std::io;
use std::path::Path;

use crate::bisect::{bisect, Divergence};
use crate::bundle::BundleEntry;
use crate::inference::InferenceMetrics;
use crate::manifest::Manifest;
use crate::metrics::functional::YieldGate;
use crate::metrics::{Functional, Physical};

pub struct Report {
    pub manifest: Manifest,
    pub physical: Vec<(&'static str, Physical)>,
    pub functional: Vec<(&'static str, Functional)>,
    /// L5 PVT × MC runs for the yield gate. Empty until L5 lands.
    pub l5_runs: Vec<Functional>,
    /// Per-backend inference latency / throughput. Same shape as
    /// `physical` and `functional` — backends populate independently.
    pub inference: Vec<(&'static str, InferenceMetrics)>,
    /// Optional bundle entries (when `bundle.merge_weights = true`
    /// in the config).
    pub bundle: Vec<BundleEntry>,
}

impl Report {
    /// New empty report tied to a captured manifest. Caller pushes
    /// physical/functional/L5/inference/bundle entries as backends
    /// produce them.
    pub fn new(manifest: Manifest) -> Self {
        Self {
            manifest,
            physical: Vec::new(),
            functional: Vec::new(),
            l5_runs: Vec::new(),
            inference: Vec::new(),
            bundle: Vec::new(),
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# TinyConv-MNIST bench report\n");

        render_manifest(&mut s, &self.manifest);
        render_physical(&mut s, &self.physical);
        render_functional(&mut s, &self.functional);
        render_inference(&mut s, &self.inference);
        render_yield(&mut s, &self.l5_runs);
        render_bisection(&mut s, &bisect(&self.functional));
        render_bundle(&mut s, &self.bundle);

        s
    }

    /// Render to markdown and write to `path`. Creates parent
    /// directories if missing. Use case: bench-driven CI writes to
    /// `target/bench/<git-sha>/report.md` so contributors can read
    /// across runs and diff against history.
    pub fn write_markdown(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_markdown())
    }
}

fn render_manifest(s: &mut String, m: &Manifest) {
    let _ = writeln!(s, "## Reproducibility manifest\n");
    let _ = writeln!(s, "| field | value |");
    let _ = writeln!(s, "|---|---|");
    let _ = writeln!(s, "| sky130 commit | `{}` |", m.sky130_commit);
    let _ = writeln!(s, "| ORFS image | `{}` |", m.orfs_image);
    let _ = writeln!(s, "| ngspice version | `{}` |", m.ngspice_version);
    let _ = writeln!(s, "| weights sha256 | `{}` |", m.weights_sha256);
    let _ = writeln!(s, "| Cargo.lock sha256 | `{}` |", m.cargo_lock_sha256);
    let _ = writeln!(s, "| optimizer seed | {} |", m.optimizer_seed);
    let _ = writeln!(s);
}

fn render_physical(s: &mut String, rows: &[(&'static str, Physical)]) {
    let _ = writeln!(s, "## Physical metrics\n");
    if rows.is_empty() {
        let _ = writeln!(s, "_no physical measurements yet_\n");
        return;
    }
    let _ = writeln!(
        s,
        "| backend | area µm² | Fmax MHz | WNS ns | Pdyn mW | Pleak mW | Cpar fF | Tpeak °C | E pJ/img |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|");
    for (name, p) in rows {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            name,
            opt(p.area_um2),
            opt(p.max_freq_mhz),
            opt(p.wns_ns),
            opt(p.dynamic_power_mw),
            opt(p.leakage_power_mw),
            opt(p.parasitic_cap_ff),
            opt(p.peak_temp_c),
            opt(p.energy_pj_per_inference),
        );
    }
    let _ = writeln!(s);
}

fn render_functional(s: &mut String, rows: &[(&'static str, Functional)]) {
    let _ = writeln!(s, "## Functional metrics\n");
    if rows.is_empty() {
        let _ = writeln!(s, "_no functional measurements yet_\n");
        return;
    }
    let _ = writeln!(s, "| backend | level | top-1 | n_images | first divergent layer |");
    let _ = writeln!(s, "|---|---|---|---|---|");
    for (name, f) in rows {
        let _ = writeln!(
            s,
            "| {} | {:?} | {:.4} | {} | {} |",
            name,
            f.level,
            f.top1_acc,
            f.n_images,
            f.divergence_first_layer
                .map(|n| n.to_string())
                .unwrap_or_else(|| "—".to_string()),
        );
    }
    let _ = writeln!(s);
}

fn render_yield(s: &mut String, l5: &[Functional]) {
    let _ = writeln!(s, "## Yield gate (release condition)\n");
    let g = YieldGate::RELEASE;
    if l5.is_empty() {
        let _ = writeln!(
            s,
            "_no L5 PVT × MC runs yet — release gate cannot be evaluated_\n",
        );
        return;
    }
    let pass = g.evaluate(l5);
    let _ = writeln!(
        s,
        "- target: `P(top-1 ≥ {:.2}) ≥ {:.2}`",
        g.min_top1,
        g.min_pvt_mc_pass_rate,
    );
    let _ = writeln!(s, "- runs: {}", l5.len());
    let _ = writeln!(s, "- pass rate: {:.4}", g.pass_rate(l5));
    let _ = writeln!(
        s,
        "- gate: **{}**\n",
        if pass { "PASS" } else { "FAIL" },
    );
}

fn render_bisection(s: &mut String, divs: &[Divergence]) {
    let _ = writeln!(s, "## Functional divergence (bisection)\n");
    if divs.is_empty() {
        let _ = writeln!(s, "_no divergences detected_\n");
        return;
    }
    let _ = writeln!(s, "| backend | level | layer | image | tile |");
    let _ = writeln!(s, "|---|---|---|---|---|");
    for d in divs {
        let _ = writeln!(
            s,
            "| {} | {:?} | {} | {} | {} |",
            d.backend,
            d.level,
            d.layer,
            d.image.map(|i| i.to_string()).unwrap_or_else(|| "—".to_string()),
            d.tile
                .map(|(x, y)| format!("({x},{y})"))
                .unwrap_or_else(|| "—".to_string()),
        );
    }
    let _ = writeln!(s);
}

fn render_inference(s: &mut String, rows: &[(&'static str, InferenceMetrics)]) {
    let _ = writeln!(s, "## Inference performance\n");
    if rows.is_empty() {
        let _ = writeln!(s, "_no inference measurements yet_\n");
        return;
    }

    // ── Silicon-time table (the answer that matters) ─────────────
    let _ = writeln!(s, "### Simulated silicon latency\n");
    let _ = writeln!(
        s,
        "_The number that matters: how fast inference runs on the actual silicon._"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "| backend | cycles / inf | period ns | total ns / inf | silicon throughput / s |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|");
    let mut any_simulated = false;
    for (name, m) in rows {
        match &m.simulated {
            Some(sim) => {
                any_simulated = true;
                let _ = writeln!(
                    s,
                    "| {} | {} | {:.3} | {:.3} | {:.0} |",
                    name,
                    sim.cycles_per_inference,
                    sim.period_ns,
                    sim.total_ns,
                    sim.silicon_throughput_per_sec,
                );
            }
            None => {
                let _ = writeln!(s, "| {} | — | — | — | — |", name);
            }
        }
    }
    if !any_simulated {
        let _ = writeln!(
            s,
            "\n_No backend reported simulated silicon time. \
             L1 (pure Rust reference) doesn't have a clock; \
             enable L2 (Verilator RTL) or L3 (gate-level + SDF) \
             to populate these columns._"
        );
    }
    let _ = writeln!(s);

    // ── Wall-clock table (host bench-runner diagnostics) ────────
    let _ = writeln!(s, "### Host wall-clock (diagnostic only)\n");
    let _ = writeln!(
        s,
        "_How long the bench-runner host took to compute the answer. \
         Not a silicon performance number._"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "| backend | n_images | repetitions | mean µs | p50 µs | p99 µs | min µs | max µs | host throughput / s |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|");
    for (name, m) in rows {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.0} |",
            name, m.n_images, m.repetitions, m.mean_us, m.p50_us, m.p99_us, m.min_us, m.max_us,
            m.throughput_per_sec,
        );
    }
    let _ = writeln!(s);
}

fn render_bundle(s: &mut String, entries: &[BundleEntry]) {
    let _ = writeln!(s, "## Bundle (weights + bitstream)\n");
    if entries.is_empty() {
        let _ = writeln!(s, "_bundle disabled (set `[bundle] merge_weights = true` to enable)_\n");
        return;
    }
    let _ = writeln!(s, "| name | bytes | sha256 |");
    let _ = writeln!(s, "|---|---|---|");
    for e in entries {
        let _ = writeln!(
            s,
            "| `{}` | {} | `{}` |",
            e.name, e.byte_len, e.sha256
        );
    }
    let _ = writeln!(s);
}

fn opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.3}"),
        None => "—".to_string(),
    }
}
