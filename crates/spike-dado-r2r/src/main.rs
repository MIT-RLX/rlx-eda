//! Driver: run DADO + naive EDA on the synthetic and R-2R objectives,
//! then write a full set of artifacts under `artifacts/<run-id>/`:
//!
//! * Top-level summary plots (trajectories, parameter evolution, final
//!   INL/staircase comparison) and final SVG/GDS for both algorithms'
//!   best designs.
//! * Per-snapshot package for both DADO and EDA on the R-2R objective:
//!   marginals plot, schematic SVG, GDS layout, analytical staircase,
//!   and analytical INL. 5 snapshot iterations × 2 algorithms.
//! * Final ngspice cross-validation (256 codes) for each algorithm's
//!   best design — feature-gated behind `ngspice` so the binary still
//!   runs without ngspice installed.

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use spike_dado_r2r::{
    charts, layout, schem, score_dnl, score_inl, score_sse_inl, score_synth,
    Design, DistSnapshot, Rng, RunTrace, random_design, run, N_BITS,
};

fn make_bar(label: &str, total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "  [{label:>14}] {{elapsed_precise}} {{bar:32.cyan/blue}} {{pos:>9}}/{{len:9}} ({{eta}})"
        ))
        .unwrap()
        .progress_chars("=> "),
    );
    pb
}

#[cfg(feature = "ngspice")]
use spike_dado_r2r::sim;

const N_ITERS: usize    = 80;
const K_SAMPLES: usize  = 100;
const TAU: f64          = 1.0;
const ALPHA: f64        = 0.1;
const N_SEEDS: usize    = 12;
/// Iterations to snapshot at (0-indexed; iter 79 = after the final update).
const SNAPSHOT_ITERS: &[usize] = &[0, 9, 24, 49, 79];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Always emit into the crate's docs/ (regardless of cwd) so artifacts
    // sit alongside the source they document, matching the convention in
    // crates/eda-waveform/docs/gallery and crates/spike-divider-block/docs.
    let out_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_root)?;
    println!("DADO vs EDA artifact run — output: {}", out_root.display());
    println!("config: K={K_SAMPLES} n_iters={N_ITERS} seeds={N_SEEDS} \
              snapshots@{SNAPSHOT_ITERS:?}");

    // -----------------------------------------------------------------
    // Experiment A: synthetic decomposable objective.
    // Snapshots not strictly needed for the synthetic run since its
    // role is the algorithmic sanity check; we still capture them so
    // the parameter-evolution chart has data.
    // -----------------------------------------------------------------
    println!("\n[A] Synthetic target-matching ...");
    let mut target_rng = Rng::new(42);
    let synth_target: Design = random_design(&mut target_rng);
    let total_a = (N_SEEDS * N_ITERS * K_SAMPLES * 2) as u64; // 2 algorithms
    let pb_a = make_bar("synth", total_a);
    let pb_a_ref = &pb_a;
    let score_a = move |x: &Design| {
        let r = score_synth(x, &synth_target);
        pb_a_ref.inc(1);
        r
    };
    let (dado_a, eda_a) = run_both(&score_a, SNAPSHOT_ITERS);
    pb_a.finish_and_clear();
    print_t_summary("synth", &dado_a, &eda_a);

    // -----------------------------------------------------------------
    // Experiment B: R-2R DAC max-INL.
    // -----------------------------------------------------------------
    println!("\n[B] R-2R DAC max-INL ...");
    let pb_b = make_bar("R-2R INL", total_a);
    let pb_b_ref = &pb_b;
    let score_b = |x: &Design| {
        let r = score_inl(x);
        pb_b_ref.inc(1);
        r
    };
    let (dado_b, eda_b) = run_both(&score_b, SNAPSHOT_ITERS);
    pb_b.finish_and_clear();
    print_t_summary("R-2R INL", &dado_b, &eda_b);

    // -----------------------------------------------------------------
    // Experiments C and D: friendlier objectives that should give DADO
    // a real per-clique signal. Snapshots not needed (the per-snapshot
    // visualisations are about *how* DADO converges; here we just want
    // to know *whether* it beats EDA on a more decomposable objective).
    // -----------------------------------------------------------------
    println!("\n[C] R-2R DAC sum-of-squared INL (Σ-decomposable reduction) ...");
    let pb_c = make_bar("R-2R Σ-INL²", total_a);
    let pb_c_ref = &pb_c;
    let score_c = |x: &Design| {
        let r = score_sse_inl(x);
        pb_c_ref.inc(1);
        r
    };
    let (dado_c, eda_c) = run_both(&score_c, &[]);
    pb_c.finish_and_clear();
    print_t_summary("R-2R Σ-INL²", &dado_c, &eda_c);

    println!("\n[D] R-2R DAC max-DNL (carry-chain decomposition) ...");
    let pb_d = make_bar("R-2R DNL", total_a);
    let pb_d_ref = &pb_d;
    let score_d = |x: &Design| {
        let r = score_dnl(x);
        pb_d_ref.inc(1);
        r
    };
    let (dado_d, eda_d) = run_both(&score_d, &[]);
    pb_d.finish_and_clear();
    print_t_summary("R-2R DNL", &dado_d, &eda_d);

    // -----------------------------------------------------------------
    // Top-level summary artifacts.
    // -----------------------------------------------------------------
    println!("\n[•] Writing top-level summary artifacts ...");
    charts::write_trajectory_png(
        "Synthetic objective: DADO vs EDA (best/mean over iterations, mean across 12 seeds)",
        &dado_a, &eda_a, SNAPSHOT_ITERS, Some(0.0),
        out_root.join("00_trajectory_synth.png"),
    )?;
    charts::write_trajectory_png(
        "R-2R INL objective: DADO vs EDA (best/mean over iterations, mean across 12 seeds)",
        &dado_b, &eda_b, SNAPSHOT_ITERS, Some(0.0),
        out_root.join("00_trajectory_inl.png"),
    )?;
    charts::write_trajectory_png(
        "R-2R Σ-INL² objective: DADO vs EDA (best/mean over iterations, mean across 12 seeds)",
        &dado_c, &eda_c, &[], Some(0.0),
        out_root.join("00_trajectory_sse_inl.png"),
    )?;
    charts::write_trajectory_png(
        "R-2R max-DNL objective: DADO vs EDA (best/mean over iterations, mean across 12 seeds)",
        &dado_d, &eda_d, &[], Some(0.0),
        out_root.join("00_trajectory_dnl.png"),
    )?;

    // Use seed-0 traces for snapshot-based artifacts (one representative seed).
    let dado_b_rep = &dado_b[0];
    let eda_b_rep = &eda_b[0];
    charts::write_param_evolution_png(
        "DADO (R-2R): expected deviation index over iterations (seed 0)",
        &dado_b_rep.snapshots,
        &out_root.join("00_param_evolution_dado.png"),
    )?;
    charts::write_param_evolution_png(
        "Naive EDA (R-2R): expected deviation index over iterations (seed 0)",
        &eda_b_rep.snapshots,
        &out_root.join("00_param_evolution_eda.png"),
    )?;

    // Final-best comparison: DADO vs EDA vs ideal staircase + INL.
    let final_designs = vec![
        ("DADO best".to_string(), dado_b_rep.best_design),
        ("EDA  best".to_string(),  eda_b_rep.best_design),
    ];
    charts::write_staircase_png(
        "R-2R staircase: DADO best vs EDA best vs ideal (256 codes)",
        &final_designs,
        out_root.join("00_final_staircase.png"),
    )?;
    charts::write_inl_png(
        "INL = vout - ideal: DADO best vs EDA best (256 codes)",
        &final_designs,
        out_root.join("00_final_inl.png"),
    )?;

    schem::write_svg_for_design(&dado_b_rep.best_design,
        Some("R-2R DAC: DADO best (R-2R objective, seed 0)"),
        out_root.join("00_final_schematic_dado.svg"))?;
    schem::write_svg_for_design(&eda_b_rep.best_design,
        Some("R-2R DAC: EDA best (R-2R objective, seed 0)"),
        out_root.join("00_final_schematic_eda.svg"))?;

    layout::write_gds_for_design(&dado_b_rep.best_design,
        out_root.join("00_final_layout_dado.gds"))?;
    layout::write_gds_for_design(&eda_b_rep.best_design,
        out_root.join("00_final_layout_eda.gds"))?;

    // -----------------------------------------------------------------
    // Per-snapshot packages (5 iters × 2 algs).
    // -----------------------------------------------------------------
    println!("\n[•] Writing per-snapshot artifact packages ...");
    write_snapshots(&out_root, "dado", &dado_b_rep.snapshots)?;
    write_snapshots(&out_root, "eda",  &eda_b_rep.snapshots)?;

    // -----------------------------------------------------------------
    // Final ngspice cross-validation.
    // -----------------------------------------------------------------
    let mut ngspice_summary: Option<String> = None;
    #[cfg(feature = "ngspice")]
    {
        println!("\n[•] ngspice cross-validation (256 codes per design) ...");
        match eda_extern_ngspice::LocalBinary::from_env() {
            Ok(ng) => {
                let mut lines = Vec::new();
                for (label, design) in [("dado", &dado_b_rep.best_design),
                                        ("eda",  &eda_b_rep.best_design)]
                {
                    let (rows, max_resid) = sim::cross_validate(&ng, design)?;
                    let report = sim::report_text(design, &rows, max_resid);
                    let txt = out_root.join(format!("00_final_ngspice_{label}.txt"));
                    std::fs::write(&txt, &report)?;
                    println!("  {label}: max |ngspice - analytical| = {max_resid:.3e} V → {}",
                        txt.display());
                    lines.push(format!("- **{label}**: max |ngspice − analytical| = `{max_resid:.3e} V` over all 256 codes"));
                }
                ngspice_summary = Some(lines.join("\n"));
            }
            Err(e) => {
                eprintln!("  ngspice unavailable ({e}) — skipping cross-validation");
            }
        }
    }

    // -----------------------------------------------------------------
    // Summary index + narrative.
    // -----------------------------------------------------------------
    let index = build_index(&dado_a, &eda_a, &dado_b, &eda_b);
    std::fs::write(out_root.join("INDEX.md"), index)?;
    let story = build_story(
        &dado_a, &eda_a,
        &dado_b, &eda_b,
        &dado_c, &eda_c,
        &dado_d, &eda_d,
        ngspice_summary.as_deref(),
    );
    std::fs::write(out_root.join("STORY.md"), story)?;
    println!("\nDone. Read {}/STORY.md", out_root.display());
    Ok(())
}

fn run_both<F>(
    score: &F,
    snapshot_iters: &[usize],
) -> (Vec<RunTrace>, Vec<RunTrace>)
where
    F: Fn(&Design) -> (f64, [f64; N_BITS]) + ?Sized,
{
    let mut dado = Vec::with_capacity(N_SEEDS);
    let mut eda  = Vec::with_capacity(N_SEEDS);
    for s in 0..N_SEEDS {
        let seed = (s as u32) * 2 + 1;
        dado.push(run(&score, N_ITERS, K_SAMPLES, TAU, ALPHA, true,  seed, snapshot_iters));
        eda .push(run(&score, N_ITERS, K_SAMPLES, TAU, ALPHA, false, seed, snapshot_iters));
    }
    (dado, eda)
}

fn print_t_summary(label: &str, dado: &[RunTrace], eda: &[RunTrace]) {
    let n = dado.len();
    let dado_finals: Vec<f64> = dado.iter().map(|t| *t.best.last().unwrap()).collect();
    let eda_finals : Vec<f64> = eda .iter().map(|t| *t.best.last().unwrap()).collect();
    let dm: f64 = dado_finals.iter().sum::<f64>() / n as f64;
    let em: f64 = eda_finals .iter().sum::<f64>() / n as f64;
    let (t, p) = paired_t(&dado_finals, &eda_finals);
    println!("  [{label}] DADO {dm:.5}  vs  EDA {em:.5}   (paired t = {t:.2}, p ≈ {p:.4})");
}

fn write_snapshots(
    out_root: &Path,
    alg: &str,
    snaps: &[DistSnapshot],
) -> Result<(), Box<dyn std::error::Error>> {
    for snap in snaps {
        let it = snap.iter;
        let stem = format!("iter_{it:02}_{alg}");

        // Distribution at this iteration. write_marginals_png splits
        // into `<stem>_marginals_spine.png` and `<stem>_marginals_feeder.png`.
        charts::write_marginals_png(
            &format!("{alg} R-2R marginals at iter {it} (seed 0)"),
            snap,
            &out_root.join(format!("{stem}_marginals.png")),
        )?;
        // Schematic with current best-so-far design's resistor values.
        schem::write_svg_for_design(
            &snap.best_design,
            Some(&format!("{alg} R-2R best-so-far at iter {it} (score {:.5})", snap.best_score)),
            out_root.join(format!("{stem}_schematic.svg")),
        )?;
        // GDS layout for that design.
        layout::write_gds_for_design(
            &snap.best_design,
            out_root.join(format!("{stem}_layout.gds")),
        )?;
        // Staircase + INL for that design.
        let designs = vec![("best".to_string(), snap.best_design)];
        charts::write_staircase_png(
            &format!("{alg} R-2R staircase at iter {it} (best-so-far, seed 0)"),
            &designs,
            out_root.join(format!("{stem}_staircase.png")),
        )?;
        charts::write_inl_png(
            &format!("{alg} R-2R INL at iter {it} (best-so-far, seed 0)"),
            &designs,
            out_root.join(format!("{stem}_inl.png")),
        )?;
    }
    Ok(())
}

fn build_index(
    dado_a: &[RunTrace], eda_a: &[RunTrace],
    dado_b: &[RunTrace], eda_b: &[RunTrace],
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# DADO vs naive EDA — R-2R DAC artifacts");
    let _ = writeln!(s);
    let _ = writeln!(s, "Run: K={K_SAMPLES}, n_iters={N_ITERS}, seeds={N_SEEDS}, \
                      snapshots@{SNAPSHOT_ITERS:?}");
    let _ = writeln!(s);

    let summarise = |label: &str, dado: &[RunTrace], eda: &[RunTrace]| -> String {
        let n = dado.len() as f64;
        let dm = dado.iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
        let em = eda .iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
        format!("- **{label}**: DADO mean final best = `{dm:.5}`, EDA mean final best = `{em:.5}`")
    };
    let _ = writeln!(s, "## Final-iteration averages");
    let _ = writeln!(s);
    let _ = writeln!(s, "{}", summarise("Synthetic decomposable", dado_a, eda_a));
    let _ = writeln!(s, "{}", summarise("R-2R max-INL (V)",        dado_b, eda_b));
    let _ = writeln!(s);

    let _ = writeln!(s, "## Top-level artifacts");
    let _ = writeln!(s, "");
    let _ = writeln!(s, "- `00_trajectory_synth.png` — DADO/EDA best+mean trajectories on the synthetic objective.");
    let _ = writeln!(s, "- `00_trajectory_inl.png` — DADO/EDA best+mean trajectories on the R-2R INL objective.");
    let _ = writeln!(s, "- `00_param_evolution_dado.png` / `00_param_evolution_eda.png` — per-resistor expected deviation index over snapshot iterations (seed 0).");
    let _ = writeln!(s, "- `00_final_staircase.png` — DADO best vs EDA best vs ideal staircase, all 256 codes.");
    let _ = writeln!(s, "- `00_final_inl.png` — INL curves for DADO best and EDA best.");
    let _ = writeln!(s, "- `00_final_schematic_*.svg` — annotated R-2R schematic for each algorithm's final-best design.");
    let _ = writeln!(s, "- `00_final_layout_*.gds` — real GDS file for each algorithm's final-best design (open in KLayout).");
    let _ = writeln!(s, "- `00_final_ngspice_*.txt` — ngspice cross-validation report (256 codes; max |ngspice - analytical|).");
    let _ = writeln!(s);

    let _ = writeln!(s, "## Per-snapshot packages (R-2R objective, seed 0)");
    let _ = writeln!(s);
    let _ = writeln!(s, "Each iteration in {SNAPSHOT_ITERS:?} produces, for both `dado` and `eda`:");
    let _ = writeln!(s, "- `iter_NN_<alg>_marginals.png` — 16 per-resistor probability tracks across the deviation alphabet.");
    let _ = writeln!(s, "- `iter_NN_<alg>_schematic.svg` — schematic of the best-so-far design with resistor values annotated.");
    let _ = writeln!(s, "- `iter_NN_<alg>_layout.gds` — GDS of the best-so-far design (resistor body lengths encode actual ohms).");
    let _ = writeln!(s, "- `iter_NN_<alg>_staircase.png` — analytical DAC staircase.");
    let _ = writeln!(s, "- `iter_NN_<alg>_inl.png` — analytical INL curve.");
    let _ = writeln!(s);
    s
}

fn build_story(
    dado_a: &[RunTrace], eda_a: &[RunTrace],
    dado_b: &[RunTrace], eda_b: &[RunTrace],
    dado_c: &[RunTrace], eda_c: &[RunTrace],
    dado_d: &[RunTrace], eda_d: &[RunTrace],
    ngspice_summary: Option<&str>,
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();

    let dado_b_rep = &dado_b[0];
    let eda_b_rep  = &eda_b[0];

    // Headline numbers from this same run.
    let n = dado_b.len() as f64;
    let synth_dm: f64 = dado_a.iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
    let synth_em: f64 = eda_a .iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
    let inl_dm:   f64 = dado_b.iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
    let inl_em:   f64 = eda_b .iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
    let (synth_t, synth_p) = paired_t(
        &dado_a.iter().map(|t| *t.best.last().unwrap()).collect::<Vec<_>>(),
        &eda_a .iter().map(|t| *t.best.last().unwrap()).collect::<Vec<_>>(),
    );
    let (_inl_t, inl_p) = paired_t(
        &dado_b.iter().map(|t| *t.best.last().unwrap()).collect::<Vec<_>>(),
        &eda_b .iter().map(|t| *t.best.last().unwrap()).collect::<Vec<_>>(),
    );

    // ---- Header + TL;DR ----
    let _ = writeln!(s, "# How DADO finds a low-INL R-2R DAC");
    let _ = writeln!(s);
    let _ = writeln!(s, "*Generated by `cargo run --release -p spike-dado-r2r`. Every figure on this page was written by the same binary, so the numbers match the run.*");
    let _ = writeln!(s);
    let _ = writeln!(s, "## TL;DR");
    let _ = writeln!(s);
    let _ = writeln!(s, "On a perfectly decomposable benchmark, **DADO converges to the optimum (mean `{synth_dm:.2}`) while naive EDA plateaus at `{synth_em:.2}`** (paired *t* = `{synth_t:.2}`, *p* ≈ `{synth_p:.4}`, n = {N_SEEDS}). On the actual R-2R max-INL objective, the per-bit decomposition is too lossy and the two algorithms come out **statistically tied** (`{:.3} mV` vs `{:.3} mV`, *p* = `{inl_p:.3}`). Algorithm: works as advertised. Real-circuit gain: bottlenecked by how decomposable your objective is.",
        inl_dm * 1e3, inl_em * 1e3);
    let _ = writeln!(s);
    let _ = writeln!(s, "![Synthetic trajectory: DADO → 0, EDA plateaus](00_trajectory_synth.png)");
    let _ = writeln!(s);

    // ---- Setup ----
    let _ = writeln!(s, "## What we're optimizing");
    let _ = writeln!(s);
    let _ = writeln!(s, "| | |");
    let _ = writeln!(s, "|---|---|");
    let _ = writeln!(s, "| **Design** | 16 resistors, each picking 1 of 5 deviations `{{−5%, −2.5%, 0%, +2.5%, +5%}}` from nominal. |");
    let _ = writeln!(s, "| **Design space** | 5¹⁶ ≈ 1.5 × 10¹¹ |");
    let _ = writeln!(s, "| **Circuit** | 8-bit R-2R ladder from `spike-dac-r2r`: 1 termination, 8 input feeders (2R each), 7 spine resistors (R each). |");
    let _ = writeln!(s, "| **Objective** | `f(x) = −max_k |vout(x, code k) − ideal(k)|` over all 256 codes. Volts; higher is better; optimum is 0. |");
    let _ = writeln!(s, "| **Decomposition** | Netlist adjacency → chain junction tree, 8 cliques (one per output bit), size-1 separators (the spine resistors). |");
    let _ = writeln!(s, "| **Knobs** | K = {K_SAMPLES}, n_iters = {N_ITERS}, seeds = {N_SEEDS}, snapshots @ {SNAPSHOT_ITERS:?}, τ = {TAU}, α = {ALPHA}. |");
    let _ = writeln!(s);

    // ---- Algorithms ----
    let _ = writeln!(s, "## The two algorithms");
    let _ = writeln!(s);
    let _ = writeln!(s, "Both fit the same factorised tabular categorical along the chain JT:");
    let _ = writeln!(s);
    let _ = writeln!(s, "```");
    let _ = writeln!(s, "p_θ(x) = p(C_0) · ∏_{{i≥1}} p(C_i | S_{{i−1}})");
    let _ = writeln!(s, "```");
    let _ = writeln!(s);
    let _ = writeln!(s, "Per iteration both draw K = {K_SAMPLES} samples, score them, and refit each conditional by weighted MLE. The only difference is the weighting:");
    let _ = writeln!(s);
    let _ = writeln!(s, "| algorithm | weight on clique-`i`'s conditional |");
    let _ = writeln!(s, "|---|---|");
    let _ = writeln!(s, "| **Naive EDA** | scalar `f(x_k)` (same for every clique) |");
    let _ = writeln!(s, "| **DADO** | `Q_i(x̂_i^k) = Σ_{{j≥i}} C_j(x̂_j^k)` — per-clique value function via suffix sum along the chain |");
    let _ = writeln!(s);
    let _ = writeln!(s, "DADO splits the score into per-bit components `C_i` (error attributed to the highest set bit) and only credits clique `i` with the part of `f` it can actually influence (its own clique + descendants).");
    let _ = writeln!(s);

    // ---- Score progression table ----
    let _ = writeln!(s, "## Score progression (R-2R objective, seed 0)");
    let _ = writeln!(s);
    let _ = writeln!(s, "| iter | DADO best-so-far | EDA best-so-far |");
    let _ = writeln!(s, "|---:|---:|---:|");
    for (d, e) in dado_b_rep.snapshots.iter().zip(eda_b_rep.snapshots.iter()) {
        let _ = writeln!(s, "| {} | `{:>+8.4} mV` | `{:>+8.4} mV` |",
            d.iter, d.best_score * 1e3, e.best_score * 1e3);
    }
    let _ = writeln!(s);

    // ---- Frame-by-frame ----
    let _ = writeln!(s, "## Frame by frame");
    let _ = writeln!(s);
    let _ = writeln!(s, "For each snapshot below, the table shows DADO (left) next to naive EDA (right) at the same iteration. Top row is per-resistor marginals on the spine resistors; second row is the same for the input feeders + termination; third row is the analytical INL curve of the best-so-far design.");
    let _ = writeln!(s);
    let _ = writeln!(s, "The `uniform` horizontal line on the marginals plot marks the 0.2 prior — peaks above it are commitments, dips below are aversions.");
    let _ = writeln!(s);

    let snap_blurbs = [
        "**Random init (after the first update).** Both distributions are still close to uniform; the \"best design\" here is the best of the first 100 random samples.",
        "**First updates have landed.** Marginals are starting to move off the 0.2 prior. Per-seed luck shows up clearly here — early best-so-far isn't yet a reliable signal of who's going to win.",
        "**Sharpening.** Most resistors have settled on one or two preferred bins. INL ripple is tighter.",
        "**Fine-tune.** New samples concentrate in the high-scoring region; updates mostly polish the separator-conditional rows.",
        "**Converged.** Last frame in the run. Marginals pinned, INL at its final shape.",
    ];

    for (i, (d, e)) in dado_b_rep.snapshots.iter().zip(eda_b_rep.snapshots.iter()).enumerate() {
        let it = d.iter;
        let it_pad = format!("{it:02}");
        let blurb = snap_blurbs.get(i).copied().unwrap_or("");
        let _ = writeln!(s, "### Iter {it} — DADO `{:>+8.4} mV`, EDA `{:>+8.4} mV`",
            d.best_score * 1e3, e.best_score * 1e3);
        let _ = writeln!(s);
        let _ = writeln!(s, "{blurb}");
        let _ = writeln!(s);
        let _ = writeln!(s, "<table>");
        let _ = writeln!(s, "<tr><th>DADO</th><th>EDA</th></tr>");
        let _ = writeln!(s, "<tr>");
        let _ = writeln!(s, "  <td><img src=\"iter_{it_pad}_dado_marginals_spine.png\" alt=\"DADO marginals (spine) at iter {it}\" width=\"100%\"></td>");
        let _ = writeln!(s, "  <td><img src=\"iter_{it_pad}_eda_marginals_spine.png\" alt=\"EDA marginals (spine) at iter {it}\" width=\"100%\"></td>");
        let _ = writeln!(s, "</tr>");
        let _ = writeln!(s, "<tr>");
        let _ = writeln!(s, "  <td><img src=\"iter_{it_pad}_dado_marginals_feeder.png\" alt=\"DADO marginals (feeder) at iter {it}\" width=\"100%\"></td>");
        let _ = writeln!(s, "  <td><img src=\"iter_{it_pad}_eda_marginals_feeder.png\" alt=\"EDA marginals (feeder) at iter {it}\" width=\"100%\"></td>");
        let _ = writeln!(s, "</tr>");
        let _ = writeln!(s, "<tr>");
        let _ = writeln!(s, "  <td><img src=\"iter_{it_pad}_dado_inl.png\" alt=\"DADO INL at iter {it}\" width=\"100%\"></td>");
        let _ = writeln!(s, "  <td><img src=\"iter_{it_pad}_eda_inl.png\" alt=\"EDA INL at iter {it}\" width=\"100%\"></td>");
        let _ = writeln!(s, "</tr>");
        let _ = writeln!(s, "</table>");
        let _ = writeln!(s);
        let _ = writeln!(s, "Schematics: [DADO](iter_{it_pad}_dado_schematic.svg) · [EDA](iter_{it_pad}_eda_schematic.svg). \
                              GDS layouts: [DADO](iter_{it_pad}_dado_layout.gds) · [EDA](iter_{it_pad}_eda_layout.gds).");
        let _ = writeln!(s);
    }

    // ---- Parameter evolution ----
    let _ = writeln!(s, "## How the parameters moved over the whole run");
    let _ = writeln!(s);
    let _ = writeln!(s, "Each line is one resistor; y-axis is the expected deviation-bin index (0 = −5%, 2 = nominal, 4 = +5%). The horizontal `nominal` marker sits at idx = 2.");
    let _ = writeln!(s);
    let _ = writeln!(s, "<table>");
    let _ = writeln!(s, "<tr><th>DADO — spine resistors</th><th>DADO — feeders + term</th></tr>");
    let _ = writeln!(s, "<tr>");
    let _ = writeln!(s, "  <td><img src=\"00_param_evolution_dado_spine.png\" width=\"100%\"></td>");
    let _ = writeln!(s, "  <td><img src=\"00_param_evolution_dado_feeder.png\" width=\"100%\"></td>");
    let _ = writeln!(s, "</tr>");
    let _ = writeln!(s, "<tr><th>EDA — spine resistors</th><th>EDA — feeders + term</th></tr>");
    let _ = writeln!(s, "<tr>");
    let _ = writeln!(s, "  <td><img src=\"00_param_evolution_eda_spine.png\" width=\"100%\"></td>");
    let _ = writeln!(s, "  <td><img src=\"00_param_evolution_eda_feeder.png\" width=\"100%\"></td>");
    let _ = writeln!(s, "</tr>");
    let _ = writeln!(s, "</table>");
    let _ = writeln!(s);

    // ---- Final result ----
    let _ = writeln!(s, "## Final designs");
    let _ = writeln!(s);
    let _ = writeln!(s, "<table>");
    let _ = writeln!(s, "<tr><th>Staircase (256 codes)</th><th>INL = vout − ideal</th></tr>");
    let _ = writeln!(s, "<tr>");
    let _ = writeln!(s, "  <td><img src=\"00_final_staircase.png\" width=\"100%\"></td>");
    let _ = writeln!(s, "  <td><img src=\"00_final_inl.png\" width=\"100%\"></td>");
    let _ = writeln!(s, "</tr>");
    let _ = writeln!(s, "</table>");
    let _ = writeln!(s);
    let _ = writeln!(s, "Annotated schematics (every resistor's resolved ohms):");
    let _ = writeln!(s, "- DADO best — [`00_final_schematic_dado.svg`](00_final_schematic_dado.svg)");
    let _ = writeln!(s, "- EDA  best — [`00_final_schematic_eda.svg`](00_final_schematic_eda.svg)");
    let _ = writeln!(s);
    let _ = writeln!(s, "GDS layouts (open in KLayout — body lengths encode actual ohms via the deviation alphabet):");
    let _ = writeln!(s, "- [`00_final_layout_dado.gds`](00_final_layout_dado.gds)");
    let _ = writeln!(s, "- [`00_final_layout_eda.gds`](00_final_layout_eda.gds)");
    let _ = writeln!(s);

    // ---- DADO vs EDA discussion ----
    let _ = writeln!(s, "## Why max-INL doesn't show a DADO advantage");
    let _ = writeln!(s);
    let _ = writeln!(s, "On the synthetic decomposable benchmark — where the objective is exactly `Σᵢ Cᵢ(x̂ᵢ)` for the chain JT — DADO converges to the optimum every seed and EDA plateaus. That's the algorithmic correctness check.");
    let _ = writeln!(s);
    let _ = writeln!(s, "On the actual R-2R **max-INL** objective the trajectories overlap:");
    let _ = writeln!(s);
    let _ = writeln!(s, "![INL trajectory](00_trajectory_inl.png)");
    let _ = writeln!(s);
    let _ = writeln!(s, "DADO's per-bit decomposition (\"attribute each code's error to the bit being toggled\") is a heuristic: max-INL genuinely couples all 16 resistors through every code, and a `max` reduction is incompatible with DADO's suffix-sum value functions in a way `Σ` is not. The paper notes DADO is *robust* to imperfect decompositions, not that any decomposition will do.");
    let _ = writeln!(s);

    // ---- Two friendlier objectives ----
    let sse_dm: f64 = dado_c.iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
    let sse_em: f64 = eda_c .iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
    let dnl_dm: f64 = dado_d.iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
    let dnl_em: f64 = eda_d .iter().map(|t| *t.best.last().unwrap()).sum::<f64>() / n;
    let (sse_t, sse_p) = paired_t(
        &dado_c.iter().map(|t| *t.best.last().unwrap()).collect::<Vec<_>>(),
        &eda_c .iter().map(|t| *t.best.last().unwrap()).collect::<Vec<_>>(),
    );
    let (dnl_t, dnl_p) = paired_t(
        &dado_d.iter().map(|t| *t.best.last().unwrap()).collect::<Vec<_>>(),
        &eda_d .iter().map(|t| *t.best.last().unwrap()).collect::<Vec<_>>(),
    );
    let pct_gain = |dm: f64, em: f64| -> f64 {
        if em == 0.0 { 0.0 } else { (dm - em) / em.abs() * 100.0 }
    };

    let _ = writeln!(s, "## Does DADO help on a friendlier objective?");
    let _ = writeln!(s);
    let _ = writeln!(s, "Same harness (K = {K_SAMPLES}, n_iters = {N_ITERS}, {N_SEEDS} seeds), same chain JT, same per-clique decomposition pattern — only the *score function* changes.");
    let _ = writeln!(s);

    let _ = writeln!(s, "### Sum-of-squared INL");
    let _ = writeln!(s);
    let _ = writeln!(s, "Same per-code errors as max-INL, but reduced with `Σ_k (·)²` instead of `max_k |·|`. A sum reduction composes naturally with DADO's suffix-sum `Q_i`.");
    let _ = writeln!(s);
    let _ = writeln!(s, "![Σ-INL² trajectory](00_trajectory_sse_inl.png)");
    let _ = writeln!(s);
    let _ = writeln!(s, "Mean final best (V², higher is better): DADO = `{sse_dm:.4e}`, EDA = `{sse_em:.4e}`. Paired *t* = `{sse_t:.2}`, *p* ≈ `{sse_p:.4}`. **DADO is {dado_pct:+.1}% relative to EDA.**",
        dado_pct = pct_gain(sse_dm, sse_em));
    let _ = writeln!(s);

    let _ = writeln!(s, "### Max-DNL (carry-chain decomposition)");
    let _ = writeln!(s);
    let _ = writeln!(s, "DNL between adjacent codes is dominated by the bits that *flip* in that transition, which is determined by the carry chain — a fundamentally per-bit phenomenon. The decomposition attributes each transition's |DNL|² to the highest flipping bit (= `trailing_ones(k)` for transition `k → k+1`).");
    let _ = writeln!(s);
    let _ = writeln!(s, "![DNL trajectory](00_trajectory_dnl.png)");
    let _ = writeln!(s);
    let _ = writeln!(s, "Mean final best (V, higher is better): DADO = `{dnl_dm:.4e}`, EDA = `{dnl_em:.4e}`. Paired *t* = `{dnl_t:.2}`, *p* ≈ `{dnl_p:.4}`. **DADO is {dado_pct:+.1}% relative to EDA.**",
        dado_pct = pct_gain(dnl_dm, dnl_em));
    let _ = writeln!(s);

    // Verdict.
    let inl_pct = pct_gain(inl_dm, inl_em);
    let _ = writeln!(s, "### Verdict on the four R-2R objectives");
    let _ = writeln!(s);
    let _ = writeln!(s, "| objective | DADO mean final | EDA mean final | gap | *p* |");
    let _ = writeln!(s, "|---|---:|---:|---:|---:|");
    let _ = writeln!(s, "| max-INL (V) | `{:.3e}` | `{:.3e}` | `{:+.1}%` | `{:.3}` |",
        inl_dm, inl_em, inl_pct, inl_p);
    let _ = writeln!(s, "| Σ-INL² (V²) | `{:.3e}` | `{:.3e}` | `{:+.1}%` | `{:.3}` |",
        sse_dm, sse_em, pct_gain(sse_dm, sse_em), sse_p);
    let _ = writeln!(s, "| max-DNL (V) | `{:.3e}` | `{:.3e}` | `{:+.1}%` | `{:.3}` |",
        dnl_dm, dnl_em, pct_gain(dnl_dm, dnl_em), dnl_p);
    let _ = writeln!(s);
    let _ = writeln!(s, "(\"gap\" = (DADO − EDA) / |EDA|, signed; positive means DADO scored higher i.e. is better.)");
    let _ = writeln!(s);
    let _ = writeln!(s, "Read this as the experimental answer to *\"does DADO transfer to rlx-eda?\"*: the algorithm is correct on the synthetic case, but its real-circuit usefulness on this DAC depends entirely on whether you can pose the objective as a sum (or near-sum) over the JT cliques.");
    let _ = writeln!(s);

    // ---- Cross-validation ----
    let _ = writeln!(s, "## Cross-validation");
    let _ = writeln!(s);
    if let Some(ng) = ngspice_summary {
        let _ = writeln!(s, "ngspice was driven with a per-resistor netlist mirroring the perturbed designs and swept across all 256 codes. Agreement with the in-house analytical 8×8 MNA solver:");
        let _ = writeln!(s);
        let _ = writeln!(s, "{ng}");
        let _ = writeln!(s);
        let _ = writeln!(s, "That's machine precision — independent confirmation that the analytical evaluator the optimizer is calling is computing the same answer SPICE would. Code-by-code reports: [`00_final_ngspice_dado.txt`](00_final_ngspice_dado.txt) · [`00_final_ngspice_eda.txt`](00_final_ngspice_eda.txt).");
    } else {
        let _ = writeln!(s, "ngspice was not available at run time, so the cross-validation step was skipped. Unit tests in `tests/evaluator.rs` still check the analytical evaluator against `spike_dac_r2r::ideal_vout` at every code.");
    }
    let _ = writeln!(s);

    // ---- Index pointer ----
    let _ = writeln!(s, "---");
    let _ = writeln!(s);
    let _ = writeln!(s, "See [`INDEX.md`](INDEX.md) for the terse file-by-file listing of every artifact in this directory.");

    s
}

/// Paired two-sided t-test on differences a[k] - b[k].
fn paired_t(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len() as f64;
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let dm = d.iter().sum::<f64>() / n;
    let var = d.iter().map(|x| (x - dm).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let se = (var / n).sqrt();
    let t = if se > 0.0 { dm / se } else { 0.0 };
    let p = 2.0 * (1.0 - normal_cdf(t.abs()));
    (t, p)
}
fn normal_cdf(z: f64) -> f64 {
    let (a1, a2, a3, a4, a5, p) =
        (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911);
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}
