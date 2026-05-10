//! Protocol runner: K=10 PINN seeds + 6 baselines, paired stats.

use std::fs;
use std::path::Path;
use std::time::Instant;

use rlx_runtime::{Device, Session};

use crate::baselines::{Lookup, Polynomial1D};
use crate::config::*;
use crate::inference::predict;
use crate::metrics::{accuracy, Accuracy};
use crate::oracle::truth_norm;
use crate::sampling::uniform_samples;
use crate::stats::{cliffs_delta, delta_label, holm_bonferroni, summarise, wilcoxon_signed_rank, Summary};
use crate::train::train;

#[derive(Clone, Copy, Debug)]
pub enum ProtocolDevice { Cpu, Mlx }

impl ProtocolDevice {
    fn as_rlx(self) -> Device { match self { ProtocolDevice::Cpu => Device::Cpu, ProtocolDevice::Mlx => Device::Mlx } }
    fn label(self) -> &'static str { match self { ProtocolDevice::Cpu => "CPU", ProtocolDevice::Mlx => "MLX" } }
}

#[derive(Clone, Debug)]
pub struct PinnSeedResult {
    pub seed: u32,
    pub test: Accuracy,
    pub eval_us: u128,
    pub train_ms: u128,
    pub final_loss: f32,
}

#[derive(Clone, Debug)]
pub struct BaselineResult {
    pub name: String,
    pub test: Accuracy,
    pub eval_ms: u128,
    pub n_params: usize,
    pub n_bytes: usize,
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
    pub n_steps: usize,
    pub n_seeds: usize,
    pub seeds: Vec<PinnSeedResult>,
    pub baselines: Vec<BaselineResult>,
    pub pairwise: Vec<PairwiseStat>,
    pub holm_thresholds: Vec<f64>,
    pub c1: (bool, String),
    pub c2: (bool, String),
    pub c5: (bool, f32),
}

pub fn run_protocol(device: ProtocolDevice) -> ProtocolReport {
    let rlx_device = device.as_rlx();

    println!("[sar] device={} N_TRAIN={} N_TEST={} N_STEPS={} K_SEEDS={}",
        device.label(), N_TRAIN, N_TEST, N_STEPS, N_SEEDS);

    let train_x = uniform_samples(N_TRAIN, SPLIT_SEED_TRAIN);
    let test_x  = uniform_samples(N_TEST,  SPLIT_SEED_TEST);
    let test_truth: Vec<f32> = test_x.iter().map(|x| truth_norm(*x)).collect();
    println!("[sar] splits + oracle done");

    // ── Baselines ────────────────────────────────────────────────
    let mut baselines: Vec<BaselineResult> = Vec::new();

    for &deg in POLY_DEGREES {
        let t0 = Instant::now();
        let p = Polynomial1D::fit(deg, &train_x);
        let pred = p.predict(&test_x);
        let eval_ms = t0.elapsed().as_millis();
        baselines.push(BaselineResult {
            name: format!("Poly-d{deg}"),
            test: accuracy(&pred, &test_truth),
            eval_ms,
            n_params: p.n_params(),
            n_bytes: p.n_bytes(),
        });
        println!("[sar] Poly-d{deg} done");
    }

    for &n in LOOKUP_SIZES {
        let t0 = Instant::now();
        let lk = Lookup::fit(n);
        let pred = lk.predict(&test_x);
        let eval_ms = t0.elapsed().as_millis();
        baselines.push(BaselineResult {
            name: format!("Lookup-{n}"),
            test: accuracy(&pred, &test_truth),
            eval_ms,
            n_params: lk.n_params(),
            n_bytes: lk.n_bytes(),
        });
        println!("[sar] Lookup-{n} done");
    }

    // ── PINN K seeds ─────────────────────────────────────────────
    let mut seeds: Vec<PinnSeedResult> = Vec::with_capacity(N_SEEDS);
    for seed in 1..=(N_SEEDS as u32) {
        let _ = Session::new(rlx_device);
        let t0 = Instant::now();
        let trained = train(seed, &train_x, rlx_device);
        let train_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        let pred = predict(&trained, &test_x, rlx_device);
        let eval_us = t0.elapsed().as_micros();

        let final_loss = *trained.losses.last().unwrap();
        seeds.push(PinnSeedResult {
            seed,
            test: accuracy(&pred, &test_truth),
            eval_us,
            train_ms,
            final_loss,
        });
        println!(
            "[sar] seed {:>2}: train {:>5} ms | max_abs {:.5} ({:.3} LSB) | loss {:.3e}",
            seed, train_ms,
            seeds.last().unwrap().test.max_abs,
            seeds.last().unwrap().test.max_abs_lsb,
            final_loss,
        );
    }

    // ── Statistics ───────────────────────────────────────────────
    let pinn_max_abs: Vec<f32> = seeds.iter().map(|s| s.test.max_abs).collect();
    let mut pairwise: Vec<PairwiseStat> = Vec::new();
    for b in &baselines {
        let b_rep = vec![b.test.max_abs; pinn_max_abs.len()];
        let p = wilcoxon_signed_rank(&pinn_max_abs, &b_rep);
        let d = cliffs_delta(&pinn_max_abs, &b_rep);
        pairwise.push(PairwiseStat {
            label: format!("PINN vs {}", b.name),
            p_value: p,
            cliffs_d: d,
        });
    }
    let p_values: Vec<f64> = pairwise.iter().map(|x| x.p_value).collect();
    let holm_thresholds = holm_bonferroni(&p_values, ALPHA);

    let pinn_summary = summarise(&pinn_max_abs);

    // ── Acceptance criteria ──────────────────────────────────────
    let poly_d16 = baselines.iter().find(|b| b.name == "Poly-d16").unwrap();
    let lookup64 = baselines.iter().find(|b| b.name == "Lookup-64").unwrap();
    let p_idx_poly = baselines.iter().position(|b| b.name == "Poly-d16").unwrap();
    let p_idx_lookup = baselines.iter().position(|b| b.name == "Lookup-64").unwrap();

    let c1_better  = pinn_summary.mean < poly_d16.test.max_abs;
    let c1_sig     = pairwise[p_idx_poly].p_value < holm_thresholds[p_idx_poly];
    let c1_pass    = c1_better && c1_sig && pairwise[p_idx_poly].cliffs_d > 0.0
        // "PINN better" means smaller error → cliffs_d should be NEGATIVE
        // (PINN values < Poly values). Flip to negative.
        || c1_better && c1_sig && pairwise[p_idx_poly].cliffs_d < 0.0;
    // Cleaner:
    let c1_pass    = c1_better && c1_sig;
    let c1_msg     = format!(
        "PINN max-abs μ={:.5} vs Poly-d16 max-abs={:.5} (p={:.3e} thr {:.3e})",
        pinn_summary.mean, poly_d16.test.max_abs,
        pairwise[p_idx_poly].p_value, holm_thresholds[p_idx_poly]
    );

    let c2_better = pinn_summary.mean < lookup64.test.max_abs;
    let c2_sig    = pairwise[p_idx_lookup].p_value < holm_thresholds[p_idx_lookup];
    let c2_pass   = c2_better && c2_sig;
    let c2_msg    = format!(
        "PINN max-abs μ={:.5} vs Lookup-64 max-abs={:.5} (p={:.3e} thr {:.3e}); PINN {} bytes vs Lookup-64 {} bytes",
        pinn_summary.mean, lookup64.test.max_abs,
        pairwise[p_idx_lookup].p_value, holm_thresholds[p_idx_lookup],
        TOTAL_PARAMS * 4, lookup64.n_bytes,
    );

    let c5_value = pinn_summary.mean;
    let c5_pass  = c5_value < C5_HALF_LSB;

    ProtocolReport {
        device,
        n_train: N_TRAIN,
        n_test: N_TEST,
        n_steps: N_STEPS,
        n_seeds: N_SEEDS,
        seeds,
        baselines,
        pairwise,
        holm_thresholds,
        c1: (c1_pass, c1_msg),
        c2: (c2_pass, c2_msg),
        c5: (c5_pass, c5_value),
    }
}

fn fmt_summary(s: &Summary) -> String {
    format!("{:.5} ± {:.5} (95% CI [{:.5}, {:.5}])",
        s.mean, s.std, s.ci95_lo, s.ci95_hi)
}

pub fn print_report(r: &ProtocolReport) {
    println!("\n========================================");
    println!("spike-pinn-sar protocol results");
    println!("========================================");
    println!("device={} n_train={} n_test={} n_steps={} K={}",
        r.device.label(), r.n_train, r.n_test, r.n_steps, r.n_seeds);

    println!("\n--- Baselines ---");
    for b in &r.baselines {
        println!(
            "{:<12} | max_abs {:.5} ({:.3} LSB) | RMS {:.5} | {:>5} ms | {} params, {} bytes",
            b.name, b.test.max_abs, b.test.max_abs_lsb, b.test.rms,
            b.eval_ms, b.n_params, b.n_bytes
        );
    }

    println!("\n--- PINN ({} seeds) ---", r.seeds.len());
    let pinn_max_abs: Vec<f32> = r.seeds.iter().map(|s| s.test.max_abs).collect();
    let pinn_summary = summarise(&pinn_max_abs);
    let mean_train_ms: f32 = r.seeds.iter().map(|s| s.train_ms as f32).sum::<f32>() / r.seeds.len() as f32;
    let mean_eval_us:  f32 = r.seeds.iter().map(|s| s.eval_us as f32).sum::<f32>()  / r.seeds.len() as f32;
    println!("max_abs (units): {}", fmt_summary(&pinn_summary));
    println!("max_abs (LSB):   mean {:.3}", pinn_summary.mean * LEVELS as f32);
    println!("train: {:.0} ms/seed | infer: {:.0} µs ({:.0} ns/q)",
        mean_train_ms, mean_eval_us, mean_eval_us * 1000.0 / r.n_test as f32);

    println!("\n--- Pairwise (Wilcoxon + Cliff's δ + Holm-Bonferroni) ---");
    for (ps, &thr) in r.pairwise.iter().zip(&r.holm_thresholds) {
        let sig = if ps.p_value < thr { "✓" } else { "·" };
        println!(" {} {:<25} p={:.3e} (Holm thr {:.3e}) | δ={:+.3} {}",
            sig, ps.label, ps.p_value, thr, ps.cliffs_d, delta_label(ps.cliffs_d));
    }

    println!("\n--- Acceptance criteria ---");
    let mark = |b: bool| if b { "PASS" } else { "FAIL" };
    println!(" C1' [{}] {}", mark(r.c1.0), r.c1.1);
    println!(" C2' [{}] {}", mark(r.c2.0), r.c2.1);
    println!(" C5' [{}] PINN max-abs μ = {:.5} (< ½ LSB = {:.5})",
        mark(r.c5.0), r.c5.1, C5_HALF_LSB);

    let all = r.c1.0 && r.c2.0 && r.c5.0;
    println!("\nVerdict: {}", if all {
        "HYPOTHESIS ACCEPTED — all three §12 criteria met"
    } else {
        "HYPOTHESIS NOT ACCEPTED — see failed criteria (each is a reportable result)"
    });
}

impl ProtocolReport {
    pub fn write_markdown<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut s = String::new();
        s.push_str(&format!(
            "# spike-pinn-sar protocol results\n\n\
             device: {}  | n_train={} | n_test={} | n_steps={} | K_SEEDS={}\n\n",
            self.device.label(), self.n_train, self.n_test, self.n_steps, self.n_seeds
        ));

        s.push_str("## Baselines\n\n");
        s.push_str("| name | max-abs | LSB | RMS | time (ms) | params | bytes |\n");
        s.push_str("|---|---|---|---|---|---|---|\n");
        for b in &self.baselines {
            s.push_str(&format!("| {} | {:.5} | {:.3} | {:.5} | {} | {} | {} |\n",
                b.name, b.test.max_abs, b.test.max_abs_lsb, b.test.rms,
                b.eval_ms, b.n_params, b.n_bytes));
        }

        let pinn_max_abs: Vec<f32> = self.seeds.iter().map(|s| s.test.max_abs).collect();
        let pinn_summary = summarise(&pinn_max_abs);
        s.push_str("\n## PINN (K seeds)\n\n");
        s.push_str(&format!("max-abs (units): {}\n\n", fmt_summary(&pinn_summary)));
        s.push_str(&format!("max-abs (LSB): mean {:.3}\n\n", pinn_summary.mean * LEVELS as f32));

        s.push_str("\n## Pairwise (Wilcoxon + Cliff's δ + Holm-Bonferroni @ α=0.05)\n\n");
        s.push_str("| comparison | p-value | Holm threshold | reject? | δ | δ magnitude |\n");
        s.push_str("|---|---|---|---|---|---|\n");
        for (ps, &thr) in self.pairwise.iter().zip(&self.holm_thresholds) {
            let rej = if ps.p_value < thr { "✓" } else { "·" };
            s.push_str(&format!("| {} | {:.3e} | {:.3e} | {} | {:+.3} | {} |\n",
                ps.label, ps.p_value, thr, rej, ps.cliffs_d, delta_label(ps.cliffs_d)));
        }

        s.push_str("\n## Acceptance criteria\n\n");
        let mark = |b: bool| if b { "PASS" } else { "FAIL" };
        s.push_str(&format!("- **C1' [{}]** {}\n", mark(self.c1.0), self.c1.1));
        s.push_str(&format!("- **C2' [{}]** {}\n", mark(self.c2.0), self.c2.1));
        s.push_str(&format!("- **C5' [{}]** PINN max-abs μ = {:.5} (< ½ LSB = {:.5})\n",
            mark(self.c5.0), self.c5.1, C5_HALF_LSB));
        let all = self.c1.0 && self.c2.0 && self.c5.0;
        s.push_str(&format!("\n**Verdict:** {}\n",
            if all { "HYPOTHESIS ACCEPTED" } else { "HYPOTHESIS NOT ACCEPTED" }));

        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, s)
    }
}
