//! Smoke-grade experiment driver. The full-protocol runner (K=10
//! seeds, all 3 ablations, all 5 baselines, paired Wilcoxon) lands
//! in a follow-on PR — see PLAN.md for sequencing.
//!
//! Smoke configuration: hybrid ablation only, K=2 seeds, reduced
//! `N_TRAIN` and `N_STEPS`, `M-default` MNA baseline only. The point
//! is to validate that the full pipeline (sampling → oracle → graph
//! → train → infer → score) compiles and runs end-to-end on a chosen
//! device.

use std::fs;
use std::path::Path;
use std::time::Instant;

use rlx_runtime::{Device, Session};

use crate::baselines::run_mna;
use crate::config::*;
use crate::inference::predict;
use crate::metrics::{accuracy, Accuracy};
use crate::oracle::truth_norm;
use crate::polynomial::Polynomial;
use crate::sampling::{lhs_samples, Band};
use crate::stats::{cliffs_delta, delta_label, holm_bonferroni, summarise, wilcoxon_signed_rank, Summary};
use crate::train::{train, RunKnobs};

#[derive(Clone, Copy, Debug)]
pub struct SeedResult {
    pub seed: u32,
    pub final_train_loss: f32,
    pub pinn_indist: Accuracy,
    pub pinn_ood: Accuracy,
    pub pinn_eval_us: u128,
    pub pinn_train_ms: u128,
}

#[derive(Clone, Debug)]
pub struct MnaResult {
    pub id: &'static str,
    pub acc: Accuracy,
    pub eval_ms: u128,
}

#[derive(Debug)]
pub struct SmokeReport {
    pub device: Device,
    pub ablation: char,
    pub n_train: usize,
    pub n_test: usize,
    pub knobs_n_steps: usize,
    pub seeds: Vec<SeedResult>,
    pub mna: Vec<MnaResult>,
}

/// Smoke-grade run: hybrid only, K=2 seeds, reduced sizes.
pub fn run_smoke(device: Device) -> SmokeReport {
    let knobs = RunKnobs::smoke();

    // Smoke-only sizes (NOT pre-registered).
    let n_train_smoke = 1_500;
    let n_test_smoke  =   500;
    let n_ood_smoke   =   500;
    let smoke_seeds: &[u32] = &[1, 2];

    let train_samples = lhs_samples(n_train_smoke, Band::InDist, SPLIT_SEED_LHS);
    let test_samples  = lhs_samples(n_test_smoke,  Band::InDist, SPLIT_SEED_TEST);
    let ood_samples   = lhs_samples(n_ood_smoke,   Band::Ood,    SPLIT_SEED_OOD);

    // Oracle: precompute physical Vmid for both eval slices once.
    let test_truth_v: Vec<f32> = test_samples.iter()
        .map(|s| truth_norm(s) * V_REF).collect();
    let ood_truth_v: Vec<f32> = ood_samples.iter()
        .map(|s| truth_norm(s) * V_REF).collect();

    // Race vs all three MNA configs once on test (does not depend on PINN seed).
    let mna: Vec<MnaResult> = MNA_BASELINES
        .iter()
        .map(|cfg| {
            let t0 = Instant::now();
            let pred = run_mna(&test_samples, *cfg);
            let eval_ms = t0.elapsed().as_millis();
            MnaResult { id: cfg.id, acc: accuracy(&pred, &test_truth_v), eval_ms }
        })
        .collect();

    // Smoke ablation chooser: env var `RLX_EDA_ABLATION ∈ {A, B, H}`,
    // default H (hybrid). After amendment 2026-05-10b, all three
    // rows are expected to converge — A and H rely on the warmup
    // schedule, B is unaffected (λ_phys=0 throughout).
    let smoke_ablation = match std::env::var("RLX_EDA_ABLATION").ok().as_deref() {
        Some("A") => ABL_PURE_PINN,
        Some("B") => ABL_PURE_SURROGATE,
        _         => ABL_HYBRID,
    };

    let mut seeds = Vec::with_capacity(smoke_seeds.len());
    for &seed in smoke_seeds {
        // Warm rlx on this device once (compile time amortised).
        let _ = Session::new(device);

        let t0 = Instant::now();
        let trained = train(smoke_ablation, &train_samples, seed, knobs, device);
        let pinn_train_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        let pinn_test = predict(&trained, &test_samples, device);
        let pinn_eval_us = t0.elapsed().as_micros();

        let pinn_ood_pred = predict(&trained, &ood_samples, device);

        seeds.push(SeedResult {
            seed,
            final_train_loss: *trained.losses.last().unwrap(),
            pinn_indist: accuracy(&pinn_test, &test_truth_v),
            pinn_ood:    accuracy(&pinn_ood_pred, &ood_truth_v),
            pinn_eval_us,
            pinn_train_ms,
        });
    }

    SmokeReport {
        device,
        ablation: smoke_ablation.row,
        n_train: n_train_smoke,
        n_test: n_test_smoke,
        knobs_n_steps: knobs.n_steps,
        seeds,
        mna,
    }
}

// ────────────────────────────────────────────────────────────────────
// Protocol-grade runner (K=10 seeds × 3 ablations × all baselines).
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum ProtocolDevice { Cpu, Mlx }

impl ProtocolDevice {
    fn as_rlx(self) -> Device { match self { ProtocolDevice::Cpu => Device::Cpu, ProtocolDevice::Mlx => Device::Mlx } }
    fn label(self) -> &'static str { match self { ProtocolDevice::Cpu => "CPU", ProtocolDevice::Mlx => "MLX" } }
}

#[derive(Clone, Debug)]
pub struct PinnSeedResult {
    pub seed: u32,
    pub indist: Accuracy,
    pub ood: Accuracy,
    pub eval_us: u128,
    pub train_ms: u128,
    pub final_loss: f32,
}

#[derive(Debug)]
pub struct AblationResults {
    pub row: char,
    pub seeds: Vec<PinnSeedResult>,
}

#[derive(Clone, Debug)]
pub struct BaselineResult {
    pub name: String,
    pub indist: Accuracy,
    pub ood: Accuracy,
    pub eval_ms_test: u128,
    pub n_params: usize,
}

#[derive(Debug)]
pub struct PairwiseStat {
    pub label: String,
    pub p_value: f64,
    pub cliffs_d: f32,
}

#[derive(Debug)]
pub struct ProtocolReport {
    pub device: ProtocolDevice,
    pub n_train: usize,
    pub n_test: usize,
    pub n_ood: usize,
    pub n_steps: usize,
    pub n_seeds: usize,
    pub ablations: Vec<AblationResults>,
    pub baselines: Vec<BaselineResult>,
    pub pairwise: Vec<PairwiseStat>,
    pub holm_thresholds: Vec<f64>,
    pub c1: (bool, String),
    pub c2: (bool, f32),
    pub c3: (bool, String),
    pub c4: (bool, String),
    pub c5: (bool, f32),
}

pub fn run_protocol(device: ProtocolDevice) -> ProtocolReport {
    let knobs = RunKnobs::protocol();
    let rlx_device = device.as_rlx();

    println!("[protocol] device={} N_TRAIN={} N_TEST={} N_OOD={} N_STEPS={} K_SEEDS={}",
        device.label(), N_TRAIN, N_TEST, N_OOD, knobs.n_steps, N_SEEDS);

    println!("[protocol] sampling splits ...");
    let train_samples = lhs_samples(N_TRAIN, Band::InDist, SPLIT_SEED_LHS);
    let test_samples  = lhs_samples(N_TEST,  Band::InDist, SPLIT_SEED_TEST);
    let ood_samples   = lhs_samples(N_OOD,   Band::Ood,    SPLIT_SEED_OOD);

    println!("[protocol] precomputing oracle on {} samples ...", N_TRAIN + N_TEST + N_OOD);
    let t0 = Instant::now();
    let train_truth_norm: Vec<f32> = train_samples.iter().map(truth_norm).collect();
    let test_truth_norm:  Vec<f32> = test_samples.iter().map(truth_norm).collect();
    let ood_truth_norm:   Vec<f32> = ood_samples.iter().map(truth_norm).collect();
    let test_truth_v: Vec<f32> = test_truth_norm.iter().map(|v| v * V_REF).collect();
    let ood_truth_v:  Vec<f32> = ood_truth_norm.iter().map(|v| v * V_REF).collect();
    println!("[protocol] oracle done in {:.1} s", t0.elapsed().as_secs_f32());

    // ── Baselines (run once, not per seed) ───────────────────────────
    let mut baselines: Vec<BaselineResult> = Vec::new();

    for cfg in MNA_BASELINES {
        let t0 = Instant::now();
        let pred_test = run_mna(&test_samples, *cfg);
        let eval_ms_test = t0.elapsed().as_millis();
        let pred_ood = run_mna(&ood_samples, *cfg);
        baselines.push(BaselineResult {
            name: cfg.id.to_string(),
            indist: accuracy(&pred_test, &test_truth_v),
            ood:    accuracy(&pred_ood,  &ood_truth_v),
            eval_ms_test,
            n_params: 0, // MNA has no learned parameters
        });
        println!("[protocol] {} done", cfg.id);
    }

    {
        let t0 = Instant::now();
        let poly = Polynomial::fit(&train_samples, &train_truth_norm);
        let pred_test = poly.predict(&test_samples);
        let eval_ms_test = t0.elapsed().as_millis();
        let pred_ood = poly.predict(&ood_samples);
        baselines.push(BaselineResult {
            name: format!("Poly-d{}", POLY_DEGREE),
            indist: accuracy(&pred_test, &test_truth_v),
            ood:    accuracy(&pred_ood,  &ood_truth_v),
            eval_ms_test,
            n_params: poly.n_params(),
        });
        println!("[protocol] polynomial fit + eval done");
    }

    // ── PINN ablations × seeds ───────────────────────────────────────
    let mut ablations: Vec<AblationResults> = Vec::new();
    for &abl in ABLATIONS {
        let mut seeds: Vec<PinnSeedResult> = Vec::with_capacity(N_SEEDS);
        for seed in 1..=(N_SEEDS as u32) {
            let _ = Session::new(rlx_device);
            let t0 = Instant::now();
            let trained = train(abl, &train_samples, seed, knobs, rlx_device);
            let train_ms = t0.elapsed().as_millis();

            let t0 = Instant::now();
            let pinn_test = predict(&trained, &test_samples, rlx_device);
            let eval_us = t0.elapsed().as_micros();
            let pinn_ood = predict(&trained, &ood_samples, rlx_device);

            let final_loss = *trained.losses.last().unwrap();
            seeds.push(PinnSeedResult {
                seed,
                indist: accuracy(&pinn_test, &test_truth_v),
                ood:    accuracy(&pinn_ood,  &ood_truth_v),
                eval_us,
                train_ms,
                final_loss,
            });
            println!(
                "[protocol] row {} seed {:>2}: train {:>5} ms | indist {:.3}% FS | OOD {:.3}% FS | loss {:.3e}",
                abl.row, seed, train_ms,
                100.0 * seeds.last().unwrap().indist.max_abs_fs,
                100.0 * seeds.last().unwrap().ood.max_abs_fs,
                final_loss,
            );
        }
        ablations.push(AblationResults { row: abl.row, seeds });
    }

    // ── Statistics ───────────────────────────────────────────────────
    let hybrid = ablations.iter().find(|a| a.row == 'H').expect("hybrid row");
    let pure_surrogate = ablations.iter().find(|a| a.row == 'B').expect("row B");
    let pure_pinn      = ablations.iter().find(|a| a.row == 'A').expect("row A");

    let hybrid_indist: Vec<f32> = hybrid.seeds.iter().map(|s| s.indist.max_abs).collect();
    let hybrid_ood:    Vec<f32> = hybrid.seeds.iter().map(|s| s.ood.max_abs).collect();
    let surrogate_ood: Vec<f32> = pure_surrogate.seeds.iter().map(|s| s.ood.max_abs).collect();
    let pinn_ood:      Vec<f32> = pure_pinn.seeds.iter().map(|s| s.ood.max_abs).collect();

    let mut pairwise: Vec<PairwiseStat> = Vec::new();
    // Family for Holm-Bonferroni: 5 tests (vs A, vs B, vs each MNA baseline + poly).
    // Pairwise vs each baseline: replicate baseline's score K times.
    for b in &baselines {
        let b_replicated = vec![b.indist.max_abs; hybrid_indist.len()];
        let p = wilcoxon_signed_rank(&hybrid_indist, &b_replicated);
        let d = cliffs_delta(&hybrid_indist, &b_replicated);
        pairwise.push(PairwiseStat {
            label: format!("Hybrid vs {}", b.name),
            p_value: p,
            cliffs_d: d,
        });
    }
    // Hybrid vs pure surrogate (paired by seed, on OOD max_abs — the
    // metric the §12 C3 criterion targets).
    {
        let p = wilcoxon_signed_rank(&hybrid_ood, &surrogate_ood);
        let d = cliffs_delta(&hybrid_ood, &surrogate_ood);
        pairwise.push(PairwiseStat {
            label: "Hybrid vs Pure-Surrogate (OOD)".to_string(),
            p_value: p,
            cliffs_d: d,
        });
    }
    {
        let p = wilcoxon_signed_rank(&hybrid_ood, &pinn_ood);
        let d = cliffs_delta(&hybrid_ood, &pinn_ood);
        pairwise.push(PairwiseStat {
            label: "Hybrid vs Pure-PINN (OOD)".to_string(),
            p_value: p,
            cliffs_d: d,
        });
    }

    let p_values: Vec<f64> = pairwise.iter().map(|x| x.p_value).collect();
    let holm_thresholds = holm_bonferroni(&p_values, ALPHA);

    // ── Acceptance criteria ──────────────────────────────────────────
    let hybrid_indist_summary = summarise(&hybrid_indist);
    let hybrid_ood_summary    = summarise(&hybrid_ood);
    let surrogate_ood_summary = summarise(&surrogate_ood);

    // C1: hybrid dominates ≥1 baseline on (max-abs, latency) Pareto, p < α/family.
    let mean_pinn_eval_us: f32 = hybrid.seeds.iter().map(|s| s.eval_us as f32).sum::<f32>()
        / hybrid.seeds.len() as f32;
    let pinn_per_query_ns = mean_pinn_eval_us * 1000.0 / N_TEST as f32;
    let mut c1_pass = false;
    let mut c1_msg = String::from("dominated by all baselines");
    for (i, b) in baselines.iter().enumerate() {
        let b_per_query_ns = b.eval_ms_test as f32 * 1e6 / N_TEST as f32;
        let pinn_faster = pinn_per_query_ns < b_per_query_ns;
        let pinn_more_accurate = hybrid_indist_summary.mean < b.indist.max_abs;
        let dominates = pinn_faster && pinn_more_accurate;
        // P-value index in `pairwise` for this baseline:
        let p_idx = i;
        let stat_sig = pairwise[p_idx].p_value < holm_thresholds[p_idx];
        if dominates && stat_sig {
            c1_pass = true;
            c1_msg = format!(
                "dominates {} (PINN max-abs {:.4} V < {:.4} V; PINN {:.0} ns/q < {:.0} ns/q; p={:.3e}, threshold {:.3e})",
                b.name, hybrid_indist_summary.mean, b.indist.max_abs,
                pinn_per_query_ns, b_per_query_ns,
                pairwise[p_idx].p_value, holm_thresholds[p_idx]
            );
            break;
        }
    }

    // C2: OOD ratio mean ≤ 2.0
    let c2_value = hybrid_ood_summary.mean / hybrid_indist_summary.mean.max(1e-9);
    let c2_pass = c2_value <= C2_OOD_RATIO_MAX;

    // C3: hybrid beats pure-surrogate on OOD by ≥1σ on max-abs.
    let c3_diff = surrogate_ood_summary.mean - hybrid_ood_summary.mean;
    let c3_threshold = surrogate_ood_summary.std.max(hybrid_ood_summary.std)
        * C3_HYBRID_BEATS_DATA_BY;
    let c3_pass = c3_diff >= c3_threshold;
    let c3_msg = format!(
        "hybrid OOD μ={:.4} V, surrogate OOD μ={:.4} V, Δ={:.4} V, σ-threshold={:.4} V",
        hybrid_ood_summary.mean, surrogate_ood_summary.mean, c3_diff, c3_threshold
    );

    // C4 (post-amendment): hybrid beats polynomial despite 70× more
    // params. Test: hybrid in-dist mean < polynomial in-dist max-abs.
    // PINN params 8769; poly params 126; ratio 69.6×.
    let poly = baselines.iter().find(|b| b.name.starts_with("Poly")).unwrap();
    let c4_pass = hybrid_indist_summary.mean < poly.indist.max_abs;
    let c4_msg = format!(
        "hybrid {:.4} V vs polynomial {:.4} V (PINN {} params, poly {} params, ratio {:.1}×)",
        hybrid_indist_summary.mean, poly.indist.max_abs,
        TOTAL_PARAMS, poly.n_params,
        TOTAL_PARAMS as f32 / poly.n_params as f32
    );

    // C5: hybrid OOD max-abs < 10% FS, mean across seeds.
    let c5_value = hybrid_ood_summary.mean / V_REF;
    let c5_pass = c5_value < C5_OOD_MAX_ABS_ERR_FS;

    ProtocolReport {
        device,
        n_train: N_TRAIN,
        n_test: N_TEST,
        n_ood: N_OOD,
        n_steps: knobs.n_steps,
        n_seeds: N_SEEDS,
        ablations,
        baselines,
        pairwise,
        holm_thresholds,
        c1: (c1_pass, c1_msg),
        c2: (c2_pass, c2_value),
        c3: (c3_pass, c3_msg),
        c4: (c4_pass, c4_msg),
        c5: (c5_pass, c5_value),
    }
}

fn fmt_summary(s: &Summary) -> String {
    format!("{:.4} ± {:.4} V (95% CI [{:.4}, {:.4}])", s.mean, s.std, s.ci95_lo, s.ci95_hi)
}

pub fn print_protocol_report(r: &ProtocolReport) {
    println!("\n========================================");
    println!("spike-pinn-diode protocol results");
    println!("========================================");
    println!(
        "device={} n_train={} n_test={} n_ood={} n_steps={} K={}",
        r.device.label(), r.n_train, r.n_test, r.n_ood, r.n_steps, r.n_seeds
    );

    println!("\n--- Baselines (run once on test/OOD) ---");
    for b in &r.baselines {
        println!(
            "{:<12} | test max-abs {:.4} V ({:.3}% FS) | OOD {:.4} V ({:.3}% FS) | {:>6} ms test | {} params",
            b.name, b.indist.max_abs, 100.0 * b.indist.max_abs_fs,
            b.ood.max_abs, 100.0 * b.ood.max_abs_fs,
            b.eval_ms_test, b.n_params
        );
    }

    println!("\n--- PINN ablations (per-seed) ---");
    for ar in &r.ablations {
        let indist: Vec<f32> = ar.seeds.iter().map(|s| s.indist.max_abs).collect();
        let ood:    Vec<f32> = ar.seeds.iter().map(|s| s.ood.max_abs).collect();
        let i_sum = summarise(&indist);
        let o_sum = summarise(&ood);
        let mean_train_ms: f32 = ar.seeds.iter().map(|s| s.train_ms as f32).sum::<f32>() / ar.seeds.len() as f32;
        let mean_eval_us: f32  = ar.seeds.iter().map(|s| s.eval_us as f32).sum::<f32>()  / ar.seeds.len() as f32;
        println!(
            "row {}: in-dist max-abs = {} | OOD max-abs = {} | train {:.0} ms/seed | infer {:.0} µs ({:.0} ns/q)",
            ar.row, fmt_summary(&i_sum), fmt_summary(&o_sum),
            mean_train_ms, mean_eval_us, mean_eval_us * 1000.0 / r.n_test as f32,
        );
    }

    println!("\n--- Pairwise statistics (Wilcoxon + Cliff's δ + Holm-Bonferroni) ---");
    for (ps, &thr) in r.pairwise.iter().zip(&r.holm_thresholds) {
        let sig = if ps.p_value < thr { "✓" } else { "·" };
        println!(
            " {} {:<35} p={:.3e} (Holm threshold {:.3e}) | δ={:+.3} {}",
            sig, ps.label, ps.p_value, thr, ps.cliffs_d, delta_label(ps.cliffs_d)
        );
    }

    println!("\n--- Acceptance criteria (§12) ---");
    let mark = |b: bool| if b { "PASS" } else { "FAIL" };
    println!(" C1 [{}] {}", mark(r.c1.0), r.c1.1);
    println!(" C2 [{}] OOD ratio = {:.2}× (≤ {})",
        mark(r.c2.0), r.c2.1, C2_OOD_RATIO_MAX);
    println!(" C3 [{}] {}", mark(r.c3.0), r.c3.1);
    println!(" C4 [{}] {}", mark(r.c4.0), r.c4.1);
    println!(" C5 [{}] OOD max-abs / V_REF = {:.3} (< {})",
        mark(r.c5.0), r.c5.1, C5_OOD_MAX_ABS_ERR_FS);

    let all_pass = r.c1.0 && r.c2.0 && r.c3.0 && r.c4.0 && r.c5.0;
    println!("\nVerdict: {}", if all_pass {
        "HYPOTHESIS ACCEPTED — all five §12 criteria met"
    } else {
        "HYPOTHESIS NOT ACCEPTED — see failed criteria above (each is a reportable result)"
    });
}

impl ProtocolReport {
    pub fn write_markdown<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut s = String::new();
        s.push_str(&format!(
            "# spike-pinn-diode protocol results\n\n\
             device: {}  | n_train={} | n_test={} | n_ood={} | n_steps={} | K_SEEDS={}\n\n",
            self.device.label(), self.n_train, self.n_test, self.n_ood, self.n_steps, self.n_seeds
        ));

        s.push_str("## Baselines\n\n");
        s.push_str("| name | test max-abs (V) | test % FS | OOD max-abs (V) | OOD % FS | test time (ms) | params |\n");
        s.push_str("|---|---|---|---|---|---|---|\n");
        for b in &self.baselines {
            s.push_str(&format!("| {} | {:.4} | {:.3} | {:.4} | {:.3} | {} | {} |\n",
                b.name, b.indist.max_abs, 100.0 * b.indist.max_abs_fs,
                b.ood.max_abs, 100.0 * b.ood.max_abs_fs,
                b.eval_ms_test, b.n_params));
        }

        s.push_str("\n## PINN ablations (K seeds)\n\n");
        s.push_str("| row | in-dist max-abs (V, μ±σ, 95% CI) | OOD max-abs (V, μ±σ, 95% CI) | mean train (ms) | mean infer (µs) |\n");
        s.push_str("|---|---|---|---|---|\n");
        for ar in &self.ablations {
            let i: Vec<f32> = ar.seeds.iter().map(|s| s.indist.max_abs).collect();
            let o: Vec<f32> = ar.seeds.iter().map(|s| s.ood.max_abs).collect();
            let i_sum = summarise(&i);
            let o_sum = summarise(&o);
            let mtr: f32 = ar.seeds.iter().map(|s| s.train_ms as f32).sum::<f32>() / ar.seeds.len() as f32;
            let mev: f32 = ar.seeds.iter().map(|s| s.eval_us as f32).sum::<f32>() / ar.seeds.len() as f32;
            s.push_str(&format!("| {} | {} | {} | {:.0} | {:.0} |\n",
                ar.row, fmt_summary(&i_sum), fmt_summary(&o_sum), mtr, mev));
        }

        s.push_str("\n## Pairwise (Wilcoxon + Cliff's δ + Holm-Bonferroni @ α=0.05)\n\n");
        s.push_str("| comparison | p-value | Holm threshold | reject? | δ | δ magnitude |\n");
        s.push_str("|---|---|---|---|---|---|\n");
        for (ps, &thr) in self.pairwise.iter().zip(&self.holm_thresholds) {
            let rej = if ps.p_value < thr { "✓" } else { "·" };
            s.push_str(&format!("| {} | {:.3e} | {:.3e} | {} | {:+.3} | {} |\n",
                ps.label, ps.p_value, thr, rej, ps.cliffs_d, delta_label(ps.cliffs_d)));
        }

        s.push_str("\n## Acceptance criteria (§12)\n\n");
        let mark = |b: bool| if b { "PASS" } else { "FAIL" };
        s.push_str(&format!("- **C1 [{}]** {}\n", mark(self.c1.0), self.c1.1));
        s.push_str(&format!("- **C2 [{}]** OOD ratio {:.2}× (≤ {})\n",
            mark(self.c2.0), self.c2.1, C2_OOD_RATIO_MAX));
        s.push_str(&format!("- **C3 [{}]** {}\n", mark(self.c3.0), self.c3.1));
        s.push_str(&format!("- **C4 [{}]** {}\n", mark(self.c4.0), self.c4.1));
        s.push_str(&format!("- **C5 [{}]** OOD max-abs/V_REF = {:.3} (< {})\n",
            mark(self.c5.0), self.c5.1, C5_OOD_MAX_ABS_ERR_FS));
        let all_pass = self.c1.0 && self.c2.0 && self.c3.0 && self.c4.0 && self.c5.0;
        s.push_str(&format!("\n**Verdict:** {}\n",
            if all_pass { "HYPOTHESIS ACCEPTED" } else { "HYPOTHESIS NOT ACCEPTED" }));

        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, s)
    }
}

pub fn print_report(r: &SmokeReport) {
    println!("=== spike-pinn-diode smoke ({}) ===", r.device);
    println!(
        "config: ablation={} n_train={} n_test={} n_steps={}",
        r.ablation, r.n_train, r.n_test, r.knobs_n_steps
    );
    println!("--- MNA baselines (test split, in-dist) ---");
    for m in &r.mna {
        println!(
            "MNA {:<10}      {:>5} ms | max_abs {:.4} V ({:.3}% FS) | RMS {:.4} V",
            m.id, m.eval_ms, m.acc.max_abs,
            100.0 * m.acc.max_abs_fs, m.acc.rms
        );
    }
    println!("--- PINN ({} ablation, {} seeds) ---", r.ablation, r.seeds.len());
    for s in &r.seeds {
        println!(
            "seed {:>3}: train {:>5} ms | infer {:>5} µs | max_abs {:.4} V ({:.3}% FS) | RMS {:.4} V | OOD max_abs {:.4} V ({:.3}% FS) | final loss {:.3e}",
            s.seed, s.pinn_train_ms, s.pinn_eval_us,
            s.pinn_indist.max_abs, 100.0 * s.pinn_indist.max_abs_fs, s.pinn_indist.rms,
            s.pinn_ood.max_abs, 100.0 * s.pinn_ood.max_abs_fs,
            s.final_train_loss,
        );
    }

    if r.seeds.len() >= 2 {
        let mean_indist: f32 = r.seeds.iter().map(|s| s.pinn_indist.max_abs).sum::<f32>() / r.seeds.len() as f32;
        let mean_ood:    f32 = r.seeds.iter().map(|s| s.pinn_ood.max_abs).sum::<f32>() / r.seeds.len() as f32;
        let ratio = mean_ood / mean_indist.max(1e-6);
        let mean_infer: f32 = r.seeds.iter().map(|s| s.pinn_eval_us as f32).sum::<f32>() / r.seeds.len() as f32;
        println!("--- aggregate ---");
        println!(
            "OOD/in-dist max-abs ratio (mean): {:.2}× (C2 ≤ 2.0)",
            ratio
        );
        println!(
            "PINN inference: {:.0} µs/batch ({:.0} ns/sample, n={})",
            mean_infer, mean_infer * 1000.0 / r.n_test as f32, r.n_test
        );
    }
}
