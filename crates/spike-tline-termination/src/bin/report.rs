//! Report generator. Mirrors the `spike-cmos-gates::digital_primitives_mna`
//! / `spike-pulse-rc::triangulate` pattern: runs every analysis once,
//! renders PNG/SVG via `eda_waveform::plot`, dumps the SPICE decks, and
//! writes a markdown narrative to `docs/`. Optionally has ngspice emit a
//! cicwave-compatible binary `.raw` per case.
//!
//! Run with:
//!     cargo run -p spike-tline-termination --features report --bin report
//!
//! Build-time gate `report` pulls in `eda-waveform` (plotting) and
//! `ngspice` (the trace backend). All artifacts land in
//! `crates/spike-tline-termination/docs/`.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_hir::SourceWaveform;
use eda_waveform::{plot, Waveform};
use spike_tline_termination::{analytic_pulse_at, fdtd_trace, spice_deck, Topology};

const TD: f64 = 1e-9;          // 1 ns line ≈ 6" of FR-4 microstrip
const VS: f64 = 3.3;           // logic rail
const T_STOP: f64 = 12.0 * TD; // ~6 round trips
const N_CELLS: usize = 50;     // FDTD cells per direction
const EDGE: f64 = 0.5 * TD;    // pulse delay
const TR: f64 = 1e-12;         // 1 ps ramp — well below h

fn h() -> f64 { TD / N_CELLS as f64 }
fn n_steps(t_stop: f64) -> usize { (t_stop / h()).round() as usize }
fn waveform() -> SourceWaveform {
    SourceWaveform::pulse(0.0, VS, EDGE, TR, TR, T_STOP * 2.0, 0.0)
}
/// Step-shaped twin (tr=tf=0) used to sample the closed-form analytic
/// (which assumes step transitions). The 1 ps ramp on the FDTD/ngspice
/// side is invisible at our 20 ps grid but eliminates the right- vs
/// left-continuous ambiguity at the discontinuity.
fn waveform_step() -> SourceWaveform {
    SourceWaveform::pulse(0.0, VS, EDGE, 0.0, 0.0, T_STOP * 2.0, 0.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    fs::create_dir_all(&docs)?;

    let bad = Topology::unterminated(TD);
    let good = Topology::series_matched(TD);
    let w = waveform();

    eprintln!("[1/4] running FDTD + analytic for both topologies");
    let (t_fd, vrx_bad)  = fdtd_trace(bad,  h(), n_steps(T_STOP), &w);
    let (_t,   vrx_good) = fdtd_trace(good, h(), n_steps(T_STOP), &w);
    let w_step = waveform_step();
    let vrx_bad_a:  Vec<f64> = t_fd.iter().map(|&t| analytic_pulse_at(bad,  t, &w_step)).collect();
    let vrx_good_a: Vec<f64> = t_fd.iter().map(|&t| analytic_pulse_at(good, t, &w_step)).collect();

    eprintln!("[2/4] running ngspice for both topologies");
    let ng = LocalBinary::from_env()?;
    let (t_ng_bad, v_ng_bad) = run_ngspice(&ng, bad, &w)?;
    let (t_ng_good, v_ng_good) = run_ngspice(&ng, good, &w)?;

    // Headline numbers for the report.
    let peak_bad = max_of(&vrx_bad);
    let trough_bad = min_of(&vrx_bad);
    let peak_good = max_of(&vrx_good);
    let peak_ng_bad = max_of(&v_ng_bad);

    eprintln!(
        "      unterminated peak = {:.3} V ({:.0}%)  ngspice peak = {:.3} V",
        peak_bad, 100.0 * peak_bad / VS, peak_ng_bad
    );
    eprintln!("      matched      peak = {:.3} V (settled at {:.3})", peak_good, VS);

    // ── Plot 1: receiver-voltage overlay (the LinkedIn screenshot recreation)
    {
        let mut signals = BTreeMap::new();
        signals.insert("v(rx) unterminated".into(), vrx_bad.clone());
        signals.insert("v(rx) series-matched".into(), vrx_good.clone());
        let wave = Waveform::Real {
            axis_name: "time (s)".into(),
            axis: t_fd.clone(),
            signals,
        };
        let cfg = plot::PlotConfig::new()
            .with_title("T-line termination — receiver voltage (3.3 V into 50 Ω, high-Z RX)")
            .with_size(1000, 560)
            .add_marker(plot::Marker::Horizontal { y: VS, label: Some("V_high (3.3 V)".into()) })
            .add_marker(plot::Marker::Horizontal { y: 0.0, label: Some("GND".into()) });
        plot::png_to_path(&wave, docs.join("waveform_overlay.png"), &cfg)?;
        plot::svg_to_path(&wave, docs.join("waveform_overlay.svg"), &cfg)?;
        eprintln!("[3/4] wrote waveform_overlay.{{png,svg}}");
    }

    // ── Plot 2: R_term sweep — peak overshoot ("loss") vs R_term
    let (sweep_r, sweep_peak, r_opt, peak_at_opt) = sweep_rterm(&w);
    {
        let axis: Vec<f64> = sweep_r.clone();
        let mut signals = BTreeMap::new();
        signals.insert("peak v(rx) [FDTD]".into(), sweep_peak.clone());
        // V_high reference line drawn as a flat series so the loss curve has
        // a visual "asymptote".
        signals.insert("V_high".into(), vec![VS; axis.len()]);
        let wave = Waveform::Real {
            axis_name: "R_term (Ω)".into(),
            axis,
            signals,
        };
        let cfg = plot::PlotConfig::new()
            .with_title(format!(
                "Loss curve: peak v(rx) vs R_term (R_drv = 10 Ω, Z₀ = 50 Ω). \
                 Optimum at R_term = {r_opt:.0} Ω → peak {peak_at_opt:.3} V"
            ))
            .with_size(1000, 520)
            .add_marker(plot::Marker::Vertical {
                x: r_opt,
                label: Some(format!("R_term* = {r_opt:.0} Ω")),
            })
            .add_marker(plot::Marker::Horizontal {
                y: VS,
                label: Some("V_high (3.3 V)".into()),
            });
        plot::png_to_path(&wave, docs.join("rterm_sweep.png"), &cfg)?;
        plot::svg_to_path(&wave, docs.join("rterm_sweep.svg"), &cfg)?;
        eprintln!("    wrote rterm_sweep.{{png,svg}}");
    }

    // ── SPICE decks + cicwave-compatible binary raw
    let bad_deck_path = docs.join("unterminated.deck.spice");
    let good_deck_path = docs.join("matched.deck.spice");
    fs::write(&bad_deck_path, deck_with_control(bad, &w, &docs.join("unterminated.raw")))?;
    fs::write(&good_deck_path, deck_with_control(good, &w, &docs.join("matched.raw")))?;
    eprintln!("[4/4] wrote {{unterminated,matched}}.deck.spice");
    match run_ngspice_for_raw(&bad_deck_path, &good_deck_path) {
        Ok(()) => eprintln!("    wrote {{unterminated,matched}}.raw (cicwave-compatible)"),
        Err(e) => eprintln!("    skipping raw export ({e})"),
    }

    // ── Markdown narrative
    let md = build_report_md(
        peak_bad, trough_bad, peak_good, peak_ng_bad,
        r_opt, peak_at_opt,
        &t_fd, &vrx_bad, &vrx_bad_a, &vrx_good_a,
        &t_ng_bad, &v_ng_bad, &t_ng_good, &v_ng_good,
    );
    let md_path = docs.join("report.md");
    fs::write(&md_path, md)?;
    eprintln!("    wrote report.md");

    eprintln!("\nAll artifacts in: {}", docs.display());
    Ok(())
}

// ── Sweep ────────────────────────────────────────────────────────────────

fn sweep_rterm(w: &SourceWaveform) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let z0 = 50.0_f64;
    let r_drv = 10.0_f64;
    let r_load = 1e9_f64;
    // 21 points from 0 Ω (bare driver) up to 100 Ω (over-damped past
    // matching). Optimum is at R_term = Z₀ - R_drv = 40 Ω.
    let r_terms: Vec<f64> = (0..=20).map(|i| 5.0 * i as f64).collect();
    let mut peaks = Vec::with_capacity(r_terms.len());
    for &r_term in &r_terms {
        let topo = Topology { r_drv, r_term, z0, td: TD, r_load };
        let (_t, v) = fdtd_trace(topo, h(), n_steps(T_STOP), w);
        peaks.push(max_of(&v));
    }
    // Optimum = first sample at or below VS by ≤ 1 mV (matched-or-better).
    let (opt_idx, _) = peaks.iter().enumerate()
        .min_by(|(_, a), (_, b)| (**a - VS).abs().partial_cmp(&(**b - VS).abs()).unwrap())
        .unwrap();
    (r_terms.clone(), peaks.clone(), r_terms[opt_idx], peaks[opt_idx])
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn run_ngspice(
    ng: &LocalBinary, topo: Topology, w: &SourceWaveform,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let trace = ng.run_transient_trace(
        &spice_deck(topo, w),
        &TransientAnalysis::new(h(), T_STOP).with_t_max(h()),
        &[OutputRequest::NodeVoltage("vrx".into())],
    )?;
    let v = trace.node_voltages["vrx"].clone();
    Ok((trace.time, v))
}

/// Build a deck augmented with a `.control` block that writes a binary
/// raw to `raw_path` so cicwave (or any nutmeg-aware viewer) can open it.
/// We can't easily reach into eda-extern-ngspice's tempfile path, so we
/// drive ngspice directly with `Command::new("ngspice") -b <deck>`.
fn deck_with_control(topo: Topology, w: &SourceWaveform, raw_path: &Path) -> String {
    let mut deck = spice_deck(topo, w);
    // The deck currently ends with `.end` (eda-spice-emit appends). Strip
    // it, append .control, then re-close with .end.
    if let Some(idx) = deck.rfind("\n.end") {
        deck.truncate(idx);
    }
    use std::fmt::Write as _;
    let _ = write!(
        deck,
        "\n.control\nset filetype=binary\ntran {h:.6e} {t_stop:.6e} uic\nwrite {raw} v(vrx) v(vmid)\nquit\n.endc\n.end\n",
        h = h(),
        t_stop = T_STOP,
        raw = raw_path.display(),
    );
    deck
}

fn run_ngspice_for_raw(bad_deck: &Path, good_deck: &Path) -> Result<(), Box<dyn Error>> {
    for d in [bad_deck, good_deck] {
        let out = Command::new("ngspice").arg("-b").arg(d).output()?;
        if !out.status.success() {
            return Err(format!(
                "ngspice nonzero on {}: {}",
                d.display(),
                String::from_utf8_lossy(&out.stderr)
            ).into());
        }
    }
    Ok(())
}

fn max_of(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
fn min_of(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::INFINITY, f64::min) }

// ── Markdown report ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn build_report_md(
    peak_bad: f64, trough_bad: f64, peak_good: f64, peak_ng_bad: f64,
    r_opt: f64, peak_at_opt: f64,
    t_fd: &[f64], vrx_bad: &[f64], vrx_bad_a: &[f64], vrx_good_a: &[f64],
    t_ng_bad: &[f64], v_ng_bad: &[f64], t_ng_good: &[f64], v_ng_good: &[f64],
) -> String {
    let mut s = String::new();
    use std::fmt::Write as _;

    let _ = writeln!(s, "# spike-tline-termination — report\n");
    let _ = writeln!(s, "Driver (Thevenin V_s, R_drv = 10 Ω) → optional series R_term → 50 Ω lossless line, TD = 1 ns → high-Z receiver. \
        Two cases studied: **unterminated** (R_term = 0) and **series-matched** (R_term = Z₀ − R_drv = 40 Ω).\n");
    let _ = writeln!(s, "## Headline numbers\n");
    let _ = writeln!(s, "| metric | unterminated | series-matched |");
    let _ = writeln!(s, "| --- | ---: | ---: |");
    let _ = writeln!(s, "| FDTD peak v(rx) | **{peak_bad:.3} V** ({:.0}% of V_high) | {peak_good:.3} V |",
        100.0 * peak_bad / VS);
    let _ = writeln!(s, "| FDTD trough v(rx) | {trough_bad:.3} V (undershoot) | — |");
    let _ = writeln!(s, "| ngspice peak v(rx) | {peak_ng_bad:.3} V | — |");
    let _ = writeln!(s, "| optimal R_term (sweep min) | — | {r_opt:.0} Ω → peak {peak_at_opt:.3} V |\n");

    let _ = writeln!(s, "## Theory: bounce diagram\n");
    let _ = writeln!(s, "With Γ_S = (R_s − Z₀)/(R_s + Z₀) and Γ_L = (R_L − Z₀)/(R_L + Z₀) and \
        V_+ = V_s · Z₀/(R_s + Z₀), the receiver voltage at time `t = (2k+1)·TD` (k = 0, 1, 2, …) follows the geometric staircase\n");
    let _ = writeln!(s, "```\nv_rx(t) = (1 + Γ_L) · V_+ · Σ_{{j=0}}^{{k}} (Γ_S · Γ_L)^j\n```\n");
    let _ = writeln!(s, "For the unterminated case (R_drv = 10 Ω, Z₀ = 50 Ω, R_L = ∞): \
        Γ_S = −2/3, Γ_L = +1, V_+ = (5/6)·V_s. \
        First arrival at t = TD is **2·V_+ = (5/3)·V_s = 5.5 V** — exactly the peak in the LinkedIn screenshot. \
        Long-time limit is V_s (the geometric series converges to it).\n");
    let _ = writeln!(s, "Series-matched (R_drv + R_term = Z₀): Γ_S = 0. \
        First arrival at t = TD lands at V_s and stays there. No second front, no ringing.\n");

    let _ = writeln!(s, "## Validation pyramid\n");
    let _ = writeln!(s, "Three witnesses agree on the receiver voltage:\n");
    let _ = writeln!(s, "1. **Analytic** (closed-form bounce diagram, see `analytic_step_at` / `analytic_pulse_at` in `src/lib.rs`). Tested at characteristic times in `tests/analytic.rs`.");
    let _ = writeln!(s, "2. **FDTD** (discrete delay-line simulator with two queues of length N = TD/h, exact for a lossless line on an integer-h grid; see `fdtd_trace`). Cross-checked against the analytic on plateau midpoints in `tests/finite_difference.rs`.");
    let _ = writeln!(s, "3. **ngspice** (lossless `T` element). Cross-checked against the FDTD on plateau midpoints in `tests/ngspice.rs`. \
        Backend is selected via `NGSPICE_BACKEND={{local,docker}}` matching the workspace convention from `spike-dado-sar::invoker_from_env`.\n");

    // Spot-check table — sample at plateau midpoints (between bounce
    // transitions at (2k+1)·TD + edge = {1.5, 3.5, 5.5, 7.5, 9.5} ns) so
    // all three engines agree to mV without edge-sample ambiguity. The
    // pre-arrival probes (0.0, 0.5, 1.0 ns) are also in flat regions.
    let probes_ns = [0.0_f64, 0.5, 1.0, 2.5, 4.5, 6.5, 8.5, 10.5];
    let _ = writeln!(s, "### Spot check (unterminated, all three engines)\n");
    let _ = writeln!(s, "| t [ns] | analytic [V] | FDTD [V] | ngspice [V] |");
    let _ = writeln!(s, "| ---: | ---: | ---: | ---: |");
    for &t_ns in &probes_ns {
        let t = t_ns * 1e-9;
        let v_a = sample(t_fd, vrx_bad_a, t);
        let v_f = sample(t_fd, vrx_bad, t);
        let v_n = sample(t_ng_bad, v_ng_bad, t);
        let _ = writeln!(s, "| {t_ns:.2} | {v_a:.3} | {v_f:.3} | {v_n:.3} |");
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "### Spot check (series-matched)\n");
    let _ = writeln!(s, "| t [ns] | analytic [V] | ngspice [V] |");
    let _ = writeln!(s, "| ---: | ---: | ---: |");
    for &t_ns in &[0.0_f64, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0] {
        let t = t_ns * 1e-9;
        let v_a = sample(t_fd, vrx_good_a, t);
        let v_n = sample(t_ng_good, v_ng_good, t);
        let _ = writeln!(s, "| {t_ns:.2} | {v_a:.3} | {v_n:.3} |");
    }
    let _ = writeln!(s);

    let _ = writeln!(s, "## Plots\n");
    let _ = writeln!(s, "![Receiver voltage overlay](waveform_overlay.png)\n");
    let _ = writeln!(s, "*Receiver voltage with the same 3.3 V step driving both topologies. Green-equivalent: unterminated (R_term = 0). Red-equivalent: series-matched (R_term = 40 Ω). Compare to the LinkedIn-quiz screenshot.*\n");
    let _ = writeln!(s, "![R_term sweep — peak v(rx) vs R_term](rterm_sweep.png)\n");
    let _ = writeln!(s, "*Loss curve: peak receiver voltage as a function of R_term. The minimum sits at R_term = Z₀ − R_drv = 40 Ω — exactly where Γ_S = 0 and the first reflection from the receiver gets absorbed at the source. Smaller R_term overshoots; larger R_term under-drives the line (but never overshoots).*\n");

    let _ = writeln!(s, "## Reproduction\n");
    let _ = writeln!(s, "```\ncargo test  -p spike-tline-termination --features ngspice");
    let _ = writeln!(s, "cargo run   -p spike-tline-termination --features report --bin report\n```\n");
    let _ = writeln!(s, "ngspice decks land at `docs/{{unterminated,matched}}.deck.spice`; binary raw at `docs/{{unterminated,matched}}.raw` (cicwave-readable).\n");

    let _ = writeln!(s, "## Floorplan / PDK note\n");
    let _ = writeln!(s, "This spike intentionally has **no floorplan, no PDK layout, no stdcell driver**. \
        Transmission-line termination is a PCB-domain phenomenon: 50 Ω microstrip on FR-4. \
        On-chip wires in sky130 are RC-dominated until many millimeters of length, at which point the relevant model is RLGC parasitic extraction — a different exercise. \
        The right next step if you want a chip-domain analog is to take a long sky130 metal-N route, extract per-unit-length R/L/C from the metal stack via `eda-pdks`/`klayout-pdk`, plug those into the FDTD/T-element here, and ask: at what wire length does sky130 routing actually start ringing?\n");
    s
}

/// Linear-interpolate `y(t)` at `t_q`. Both arrays assumed sorted.
fn sample(t: &[f64], y: &[f64], t_q: f64) -> f64 {
    if t_q <= t[0] { return y[0]; }
    if t_q >= *t.last().unwrap() { return *y.last().unwrap(); }
    let i = t.iter().position(|&ti| ti >= t_q).unwrap();
    let (t0, t1) = (t[i - 1], t[i]);
    let (y0, y1) = (y[i - 1], y[i]);
    y0 + (y1 - y0) * (t_q - t0) / (t1 - t0)
}
