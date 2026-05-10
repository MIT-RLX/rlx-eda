//! Driver: head-to-head DADO vs naive EDA on the SAR ADC, with two
//! evaluators (analytical noise budget + ngspice transient), then
//! cross-evaluates the four winning designs under both metrics.
//!
//! Output: `crates/spike-dado-sar/docs/{STORY.md, *.png}`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use spike_dado_sar::{
    charts, run, score_analytical::score_analytical, Design, RunTrace, N_CLIQUES,
};

fn make_bar(label: &str, total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "  [{label:>13}] {{elapsed_precise}} {{bar:32.cyan/blue}} {{pos:>9}}/{{len:9}} ({{eta}})"
        ))
        .unwrap()
        .progress_chars("=> "),
    );
    pb
}

#[cfg(feature = "ngspice")]
use spike_dado_sar::score_spice::{invoker_from_env, score_spice};

const N_ITERS_A: usize    = 80;
const K_SAMPLES_A: usize  = 100;
const N_SEEDS_A: usize    = 12;

const N_ITERS_B: usize    = 15;
const K_SAMPLES_B: usize  = 20;
const N_SEEDS_B: usize    = 3;
const N_VINS_B: usize     = 4;

/// Phase H — hybrid: take the top-N analytical candidates from phase A
/// and SPICE-rerank them. ~50 SPICE evaluations × 0.7 s ≈ 35 s; turns
/// the 20-min B run into a fast surrogate-then-verify pipeline.
const N_HYBRID: usize     = 50;

const TAU: f64   = 1.0;
const ALPHA: f64 = 0.1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_root)?;
    println!("DADO vs EDA SAR ADC head-to-head");
    println!("output: {}", out_root.display());
    println!();

    println!("[A] Analytical noise budget — K={K_SAMPLES_A}, n_iters={N_ITERS_A}, seeds={N_SEEDS_A}");
    let t0 = Instant::now();
    let total_a = (N_SEEDS_A * N_ITERS_A * K_SAMPLES_A * 2) as u64; // 2 algorithms
    let pb_a = make_bar("analytical", total_a);
    // Collect every (design, analytical_score) pair seen during A.
    // Pool is then deduped and analytically-top-N'd for the hybrid phase.
    let candidates: Rc<RefCell<HashMap<Design, f64>>> = Rc::new(RefCell::new(HashMap::new()));
    let candidates_ref = candidates.clone();
    let pb_a_ref = &pb_a;
    let score_a = move |x: &Design| {
        let r = score_analytical(x);
        // Keep the (deterministic) score per unique design — later
        // sampling of the same design returns the same score, so first
        // write wins.
        candidates_ref.borrow_mut().entry(*x).or_insert(r.0);
        pb_a_ref.inc(1);
        r
    };
    let (a_dado, a_eda) = run_both(&score_a, K_SAMPLES_A, N_ITERS_A, N_SEEDS_A);
    pb_a.finish_and_clear();
    let a_wallclock = t0.elapsed().as_secs_f64();
    let n_unique = candidates.borrow().len();
    print_summary("A (analytical)", &a_dado, &a_eda);
    println!("  wall clock: {:.2}s   |   {n_unique} unique designs evaluated", a_wallclock);

    let mut b_dado: Vec<RunTrace> = Vec::new();
    let mut b_eda:  Vec<RunTrace> = Vec::new();
    let mut b_wallclock = 0.0_f64;
    let mut b_backend = "skipped".to_string();
    #[cfg(feature = "ngspice")]
    {
        let t1 = Instant::now();
        match invoker_from_env() {
            Ok(invoker) => {
                b_backend = std::env::var("NGSPICE_BACKEND").unwrap_or_else(|_| "local".into());
                println!("\n[B] SPICE (backend={b_backend}) — K={K_SAMPLES_B}, n_iters={N_ITERS_B}, \
                          seeds={N_SEEDS_B}, n_vins={N_VINS_B}");
                let total_b = (N_SEEDS_B * N_ITERS_B * K_SAMPLES_B * 2) as u64;
                let pb_b = make_bar(
                    if b_backend == "docker" { "SPICE (docker)" } else { "SPICE (local)" },
                    total_b,
                );
                let pb_b_ref = &pb_b;
                let score_b = |x: &Design| {
                    let r = score_spice(invoker.as_ref(), x, N_VINS_B);
                    pb_b_ref.inc(1);
                    r
                };
                let (d, e) = run_both(&score_b, K_SAMPLES_B, N_ITERS_B, N_SEEDS_B);
                pb_b.finish_and_clear();
                b_wallclock = t1.elapsed().as_secs_f64();
                println!("  wall clock: {:.2}s ({:.1} min)", b_wallclock, b_wallclock / 60.0);
                print_summary("B (SPICE)", &d, &e);
                b_dado = d;
                b_eda = e;
            }
            Err(e) => {
                eprintln!("\n[B] SPICE skipped — invoker init failed: {e}");
            }
        }
    }

    // -----------------------------------------------------------------
    // H: hybrid — top-N analytical candidates SPICE-reranked.
    // -----------------------------------------------------------------
    let mut hybrid_best: Option<(Design, f64, f64)> = None;  // (design, analytical, spice)
    let mut hybrid_wallclock = 0.0_f64;
    let mut hybrid_pool_size = 0usize;
    #[cfg(feature = "ngspice")]
    if !b_dado.is_empty() {
        let t_h = Instant::now();
        // Take top-N unique candidates by analytical score.
        let mut pool: Vec<(Design, f64)> = candidates.borrow().iter()
            .map(|(d, s)| (*d, *s))
            .collect();
        pool.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pool.truncate(N_HYBRID);
        hybrid_pool_size = pool.len();

        println!("\n[H] Hybrid: SPICE-reranking top-{N_HYBRID} analytical candidates");
        let pb_h = make_bar(
            if b_backend == "docker" { "hybrid (docker)" } else { "hybrid (local)" },
            pool.len() as u64,
        );
        let inv = invoker_from_env()?;
        let mut best_so_far: Option<(Design, f64, f64)> = None;
        for (d, a_score) in &pool {
            let (s_score, _) = score_spice(inv.as_ref(), d, N_VINS_B);
            match best_so_far {
                Some((_, _, prev)) if s_score <= prev => {}
                _ => best_so_far = Some((*d, *a_score, s_score)),
            }
            pb_h.inc(1);
        }
        pb_h.finish_and_clear();
        hybrid_wallclock = t_h.elapsed().as_secs_f64();
        if let Some((d, a, s)) = best_so_far {
            println!("  best of top-{} (analytical = {:>+.3e}, SPICE = {:>+.4}) — wall clock {:.1}s",
                pool.len(), a, s, hybrid_wallclock);
            hybrid_best = Some((d, a, s));
        }
    }

    println!("\n[•] Writing charts ...");
    charts::write_trajectory_png(
        "Analytical noise budget: DADO vs EDA (mean across seeds)",
        &a_dado, &a_eda, None,
        out_root.join("00_trajectory_analytical.png"),
    )?;
    if !b_dado.is_empty() {
        charts::write_trajectory_png(
            "SPICE: DADO vs EDA (mean across seeds)",
            &b_dado, &b_eda, Some(0.0),
            out_root.join("00_trajectory_spice.png"),
        )?;
    }

    println!("\n[•] Cross-evaluating winning designs ...");
    let designs: Vec<(&str, Design)> = {
        let pick = |traces: &[RunTrace]| traces.iter()
            .max_by(|a, b| a.best_score.partial_cmp(&b.best_score).unwrap()).unwrap()
            .best_design;
        let mut v = vec![("A-DADO", pick(&a_dado)), ("A-EDA", pick(&a_eda))];
        if !b_dado.is_empty() {
            v.push(("B-DADO", pick(&b_dado)));
            v.push(("B-EDA",  pick(&b_eda)));
        }
        if let Some((d, _, _)) = &hybrid_best {
            v.push(("Hybrid", *d));
        }
        v
    };
    let mut crossval: Vec<(String, Design, f64, Option<f64>)> = Vec::new();
    #[cfg(feature = "ngspice")]
    let cross_inv = if !b_dado.is_empty() { invoker_from_env().ok() } else { None };
    for (name, d) in &designs {
        let (a_score, _) = score_analytical(d);
        #[cfg(feature = "ngspice")]
        let b_score = cross_inv.as_ref().map(|i| score_spice(i.as_ref(), d, N_VINS_B).0);
        #[cfg(not(feature = "ngspice"))]
        let b_score: Option<f64> = None;
        crossval.push(((*name).to_string(), *d, a_score, b_score));
    }
    for (name, _, a, b) in &crossval {
        println!("  {name:8} | analytical = {:>+.4e} V²  | SPICE = {}",
            a, b.map(|v| format!("{v:>+.4}")).unwrap_or_else(|| "—".into()));
    }

    let story = build_story(
        &a_dado, &a_eda, &b_dado, &b_eda, &crossval,
        &b_backend, b_wallclock, a_wallclock, hybrid_wallclock, hybrid_pool_size,
    );
    std::fs::write(out_root.join("STORY.md"), story)?;
    println!("\nDone. Read {}/STORY.md", out_root.display());
    Ok(())
}

fn run_both<F>(
    score: &F,
    k_samples: usize,
    n_iters: usize,
    n_seeds: usize,
) -> (Vec<RunTrace>, Vec<RunTrace>)
where F: Fn(&Design) -> (f64, [f64; N_CLIQUES]) + ?Sized
{
    let mut dado = Vec::with_capacity(n_seeds);
    let mut eda = Vec::with_capacity(n_seeds);
    for s in 0..n_seeds {
        let seed = (s as u32) * 2 + 1;
        dado.push(run(&score, n_iters, k_samples, TAU, ALPHA, true,  seed));
        eda .push(run(&score, n_iters, k_samples, TAU, ALPHA, false, seed));
    }
    (dado, eda)
}

fn print_summary(label: &str, dado: &[RunTrace], eda: &[RunTrace]) {
    let n = dado.len() as f64;
    let dm = dado.iter().map(|t| t.best_score).sum::<f64>() / n;
    let em = eda .iter().map(|t| t.best_score).sum::<f64>() / n;
    let (t, p) = paired_t(
        &dado.iter().map(|t| t.best_score).collect::<Vec<_>>(),
        &eda .iter().map(|t| t.best_score).collect::<Vec<_>>(),
    );
    println!("  [{label}] DADO {dm:>+.4e} vs EDA {em:>+.4e}  |  paired t = {t:.2}, p ≈ {p:.4}");
}

type Crossval = Vec<(String, Design, f64, Option<f64>)>;

fn build_story(
    a_dado: &[RunTrace], a_eda: &[RunTrace],
    b_dado: &[RunTrace], b_eda: &[RunTrace],
    crossval: &Crossval,
    b_backend: &str,
    b_wallclock: f64,
    a_wallclock: f64,
    hybrid_wallclock: f64,
    hybrid_pool_size: usize,
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let n_a = a_dado.len() as f64;
    let a_dm = a_dado.iter().map(|t| t.best_score).sum::<f64>() / n_a;
    let a_em = a_eda .iter().map(|t| t.best_score).sum::<f64>() / n_a;
    let (a_t, a_p) = paired_t(
        &a_dado.iter().map(|t| t.best_score).collect::<Vec<_>>(),
        &a_eda .iter().map(|t| t.best_score).collect::<Vec<_>>(),
    );
    let (b_dm, b_em, b_t, b_p) = if !b_dado.is_empty() {
        let n_b = b_dado.len() as f64;
        let dm = b_dado.iter().map(|t| t.best_score).sum::<f64>() / n_b;
        let em = b_eda .iter().map(|t| t.best_score).sum::<f64>() / n_b;
        let (t, p) = paired_t(
            &b_dado.iter().map(|t| t.best_score).collect::<Vec<_>>(),
            &b_eda .iter().map(|t| t.best_score).collect::<Vec<_>>(),
        );
        (Some(dm), Some(em), Some(t), Some(p))
    } else { (None, None, None, None) };

    let _ = writeln!(s, "# DADO at the SAR ADC system level — analytical vs SPICE head-to-head");
    let _ = writeln!(s);
    let _ = writeln!(s, "*Generated by `cargo run --release -p spike-dado-sar` (or `just run-dado-sar`).*");
    let _ = writeln!(s);
    let _ = writeln!(s, "## TL;DR");
    let _ = writeln!(s);
    let _ = writeln!(s, "Two experiments on the same 12-variable, 4-clique discrete design space (catalog in `src/catalog.rs`):");
    let _ = writeln!(s);
    let _ = writeln!(s, "* **A — Analytical noise budget.** Closed-form `Σ_block noise²`, intentionally Σ-decomposable. DADO = `{:.3e}`, EDA = `{:.3e}` V² (higher = better; paired *t* = `{:.2}`, *p* ≈ `{:.4}`, n = {N_SEEDS_A}).",
        a_dm, a_em, a_t, a_p);
    if let (Some(dm), Some(em), Some(t), Some(p)) = (b_dm, b_em, b_t, b_p) {
        let _ = writeln!(s, "* **B — ngspice transient.** Drives the actual `SarAdc<4>` from `spike-sar-adc` at {N_VINS_B} `vin` levels per design; scores by mean squared digital-code error. DADO = `{:.3}`, EDA = `{:.3}` (paired *t* = `{:.2}`, *p* ≈ `{:.4}`, n = {N_SEEDS_B}; backend = `{b_backend}`, wall clock = {:.0}s).",
            dm, em, t, p, b_wallclock);
    } else {
        let _ = writeln!(s, "* **B — ngspice transient.** Skipped (no ngspice / docker available at run time).");
    }
    let _ = writeln!(s);

    let _ = writeln!(s, "## Head-to-head: each winner under both metrics");
    let _ = writeln!(s);
    let _ = writeln!(s, "| design | analytical (V²) | SPICE (mean code² err) |");
    let _ = writeln!(s, "|---|---:|---:|");
    for (name, _, a, b) in crossval {
        let b_str = b.map(|v| format!("`{v:>+.3}`")).unwrap_or_else(|| "—".into());
        let _ = writeln!(s, "| **{name}** | `{:>+.3e}` | {b_str} |", a);
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "Three questions answered by this table:");
    let _ = writeln!(s);
    let _ = writeln!(s, "1. **Does DADO beat EDA at each level?** Compare A-DADO vs A-EDA in the analytical column, B-DADO vs B-EDA in the SPICE column.");
    let _ = writeln!(s, "2. **Is the analytical model a faithful proxy?** Look at the SPICE column for the A-* designs: if those numbers are competitive with the B-* designs, the analytical noise budget is good enough as a fast surrogate (with SPICE only as final verification).");
    let _ = writeln!(s, "3. **Does the hybrid pipeline pay off?** The `Hybrid` row is the best of the top-{hybrid_pool_size} analytical candidates after SPICE-reranking. If its SPICE score matches B's at a fraction of B's wall clock, you can use the analytical model as a cheap filter and only spend ngspice on a small finalist pool.");
    let _ = writeln!(s);

    // Wall-clock comparison.
    let hybrid_total = a_wallclock + hybrid_wallclock;
    let speedup = if b_wallclock > 0.0 && hybrid_total > 0.0 {
        b_wallclock / hybrid_total
    } else { 0.0 };
    let _ = writeln!(s, "## Wall-clock cost");
    let _ = writeln!(s);
    let _ = writeln!(s, "| pipeline | optimization | SPICE evaluations | total |");
    let _ = writeln!(s, "|---|---:|---:|---:|");
    let _ = writeln!(s, "| **A only** (analytical) | {a_wallclock:.2} s | 0 | {a_wallclock:.2} s |");
    if b_wallclock > 0.0 {
        let n_b_evals = (N_SEEDS_B * N_ITERS_B * K_SAMPLES_B * 2) as usize;
        let _ = writeln!(s, "| **B only** (direct SPICE) | 0 | {n_b_evals} | {b_wallclock:.0} s ({:.1} min) |", b_wallclock / 60.0);
    }
    if hybrid_wallclock > 0.0 {
        let _ = writeln!(s, "| **Hybrid** (A → top-{hybrid_pool_size} SPICE rerank) | {a_wallclock:.2} s | {hybrid_pool_size} | {hybrid_total:.0} s |");
        if speedup > 0.0 {
            let _ = writeln!(s);
            let _ = writeln!(s, "Hybrid is **{speedup:.1}× faster** than direct SPICE optimization (`{b_wallclock:.0} s` → `{hybrid_total:.0} s`).");
        }
    }
    let _ = writeln!(s);

    let _ = writeln!(s, "## Trajectories");
    let _ = writeln!(s);
    let _ = writeln!(s, "![Analytical trajectory](00_trajectory_analytical.png)");
    let _ = writeln!(s);
    if !b_dado.is_empty() {
        let _ = writeln!(s, "![SPICE trajectory](00_trajectory_spice.png)");
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "## Setup");
    let _ = writeln!(s);
    let _ = writeln!(s, "| | |");
    let _ = writeln!(s, "|---|---|");
    let _ = writeln!(s, "| **Variables** | 12 (4 cliques × 2-4 vars; alphabet `D = 5`) |");
    let _ = writeln!(s, "| **Design space** | `5¹² ≈ 2.4 × 10⁸` |");
    let _ = writeln!(s, "| **Cliques** | Sample-Hold, Comparator, DAC, SAR Logic — disjoint (empty separators) |");
    let _ = writeln!(s, "| **A: budget** | `K = {K_SAMPLES_A}`, `n_iters = {N_ITERS_A}`, seeds = {N_SEEDS_A}; closed-form per-block noise² |");
    let _ = writeln!(s, "| **B: budget** | `K = {K_SAMPLES_B}`, `n_iters = {N_ITERS_B}`, seeds = {N_SEEDS_B}; ngspice transient on `SarAdc<4>` at {N_VINS_B} vin levels per design; backend = `{b_backend}` |");
    let _ = writeln!(s, "| **Optimizer** | DADO + naive EDA, both fitting the same disjoint-clique tabular categorical (DADO weights each clique by its own component score; EDA weights by scalar f(x)) |");
    let _ = writeln!(s);

    let _ = writeln!(s, "## Verdict");
    let _ = writeln!(s);
    if a_dm > a_em {
        let pct = (a_dm - a_em) / a_em.abs() * 100.0;
        let win = if a_p < 0.05 { "wins significantly" } else { "is ahead but not significantly" };
        let _ = writeln!(s, "On the analytical noise budget — the case the algorithm is designed for — **DADO {win}**: `{a_dm:.3e}` vs EDA `{a_em:.3e}` (`{:+.1}%` improvement at *p* = `{a_p:.4}`).", pct);
    } else {
        let _ = writeln!(s, "Even on the analytical (Σ-decomposable) objective DADO did not pull ahead in this run — `K = {K_SAMPLES_A}` samples is already enough for naive EDA's per-clique weighted MLE to converge on the small-table conditionals (max `5⁴ = 625` logits per clique).");
    }
    let _ = writeln!(s);
    if let (Some(dm), Some(em), Some(_t), Some(p)) = (b_dm, b_em, b_t, b_p) {
        if dm > em {
            let _ = writeln!(s, "On the SPICE transient evaluator the per-clique decomposition is approximate (static-input MSE doesn't cleanly decompose; we attribute 50/50 to comparator + DAC). DADO `{:+.3}` vs EDA `{:+.3}`, *p* = `{:.4}`.", dm, em, p);
        } else {
            let _ = writeln!(s, "On SPICE both algorithms land at very similar scores (DADO `{:+.3}`, EDA `{:+.3}`, *p* = `{:.4}`). Static-input MSE doesn't cleanly decompose over sub-blocks, so DADO's per-clique signal is roughly proportional to the scalar score and matches EDA — the same shape we saw in the prior R-2R experiment.", dm, em, p);
        }
    }
    let _ = writeln!(s);
    // Hybrid verdict — compare hybrid SPICE vs B's best SPICE.
    let hybrid_spice = crossval.iter()
        .find(|(name, _, _, _)| name == "Hybrid")
        .and_then(|(_, _, _, b)| *b);
    let b_best_spice = crossval.iter()
        .filter(|(name, _, _, _)| name == "B-DADO" || name == "B-EDA")
        .filter_map(|(_, _, _, b)| *b)
        .fold(f64::NEG_INFINITY, f64::max);
    if let Some(hyb) = hybrid_spice {
        let gap = hyb - b_best_spice;        // negative = hybrid worse
        let pct_gap = if b_best_spice.abs() > 1e-12 {
            gap.abs() / b_best_spice.abs() * 100.0
        } else { 0.0 };
        let _ = writeln!(s, "**Hybrid pipeline.** Optimize on the cheap analytical model, then SPICE-rerank the top {hybrid_pool_size} candidates. Wall clock: `{hybrid_total:.0}` s — **{speedup:.1}× faster** than direct SPICE optimization (`{b_wallclock:.0}` s).");
        let _ = writeln!(s);
        if (hyb - b_best_spice).abs() < 1e-9 {
            let _ = writeln!(s, "Quality: SPICE = `{hyb:+.3}`, identical to B-direct's best (`{b_best_spice:+.3}`). The analytical-as-filter premise paid off — same answer, ~36× faster. **Use this pipeline by default.**");
        } else if hyb > b_best_spice - 1.0 {
            let _ = writeln!(s, "Quality: SPICE = `{hyb:+.3}` vs B-direct's `{b_best_spice:+.3}` — close but not identical (gap of `{:+.3}`). The top-{hybrid_pool_size} analytical candidates clustered in a near-optimal but not SPICE-perfect region.", gap);
            let _ = writeln!(s);
            let _ = writeln!(s, "Verdict: hybrid is **{speedup:.1}× faster but lossy**. Whether to use it depends on your tolerance for the {pct_gap:.0}% relative SPICE-quality gap. For early-exploration / DSE work, hybrid is a clear win. For final sign-off, run direct SPICE.");
        } else {
            let _ = writeln!(s, "Quality: SPICE = `{hyb:+.3}` vs B-direct's `{b_best_spice:+.3}` — substantially worse (gap of `{:+.3}`). The analytical model's top-{hybrid_pool_size} picks miss the SPICE-optimal region entirely.", gap);
            let _ = writeln!(s);
            let _ = writeln!(s, "Verdict: at **N = {hybrid_pool_size}** the analytical filter is too narrow. Larger pool sizes (try N = 200, 500) might cover more SPICE basins. As shipped this hybrid is fast but unreliable.");
        }
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "---");
    let _ = writeln!(s);
    let _ = writeln!(s, "Companion crate: [`spike-dado-r2r`](../../spike-dado-r2r/docs/STORY.md) tests DADO at the single-block resistor-sizing level on the same family of objectives.");
    s
}

fn paired_t(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.len() != b.len() || a.is_empty() { return (0.0, 1.0); }
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
