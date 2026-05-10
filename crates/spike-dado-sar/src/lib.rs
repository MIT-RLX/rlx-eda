//! DADO at the SAR ADC system level — companion to `spike-dado-r2r`.
//!
//! The previous spike showed DADO doesn't help on a single-block R-2R
//! sizing problem, because max-INL doesn't decompose over a junction
//! tree. This spike applies DADO at the *system* level — discrete
//! choices for each sub-block (sample-hold, comparator, DAC, SAR
//! logic), with two evaluators:
//!
//! * **Analytical** (`score_analytical`) — closed-form ADC noise budget
//!   that genuinely decomposes as `Σ_block noise_block²`. Designed to be
//!   the case where DADO *should* shine.
//! * **SPICE** (`score_spice`, behind the `ngspice` feature) — drives
//!   the existing `SarAdc<4>` from `spike-sar-adc` via ngspice transient,
//!   converts 16 representative inputs, scores by mean squared digital-
//!   code error.
//!
//! The driver runs DADO + naive EDA under each evaluator and then
//! cross-evaluates the four winning designs `{A-DADO, A-EDA, B-DADO,
//! B-EDA}` under both metrics — answering "does DADO transfer to ADC
//! system-level, and is the analytical model a faithful proxy for SPICE?"

#![allow(clippy::needless_range_loop)]

pub mod catalog;
pub mod score_analytical;
#[cfg(feature = "ngspice")]
pub mod score_spice;
pub mod charts;

pub use catalog::{
    clique_vars, Design, D, L, N_CLIQUES,
};

// ---------------------------------------------------------------------
// Score function alias
// ---------------------------------------------------------------------

/// `(total_score, per_clique_components)`. Higher is better; components
/// sum (approximately) to total for additively decomposable objectives.
pub type ScoreFn<'a> = dyn Fn(&Design) -> (f64, [f64; N_CLIQUES]) + 'a;

// ---------------------------------------------------------------------
// Factorised search distribution
//
// Disjoint cliques → empty separators → conditionals collapse to
// unconditional per-clique categoricals. Each clique stores logits over
// `D^|clique_vars|` joint outcomes.
// ---------------------------------------------------------------------

#[inline] fn d_pow(k: usize) -> usize {
    let mut r = 1usize;
    for _ in 0..k { r *= D; }
    r
}

fn encode(vars: &[u8]) -> usize {
    let mut idx = 0usize;
    let mut mult = 1usize;
    for &v in vars { idx += (v as usize) * mult; mult *= D; }
    idx
}

fn decode(mut idx: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(k);
    for _ in 0..k { out.push((idx % D) as u8); idx /= D; }
    out
}

/// Tabular factorised categorical, one independent table per clique.
#[derive(Clone)]
pub struct CliqueDist {
    /// `logits[c]` length `D^|clique_vars(c)|`.
    logits: Vec<Vec<f64>>,
    /// Cached `D^|clique_vars(c)|`.
    n_outcomes: Vec<usize>,
}

impl CliqueDist {
    pub fn uniform() -> Self {
        let mut logits = Vec::with_capacity(N_CLIQUES);
        let mut n_outcomes = Vec::with_capacity(N_CLIQUES);
        for c in 0..N_CLIQUES {
            let n = d_pow(clique_vars(c).len());
            logits.push(vec![0.0; n]);
            n_outcomes.push(n);
        }
        Self { logits, n_outcomes }
    }

    pub fn sample(&self, rng: &mut Rng) -> Design {
        let mut x = [0u8; L];
        for c in 0..N_CLIQUES {
            let pick = sample_softmax(&self.logits[c], rng);
            let cv = clique_vars(c);
            let vals = decode(pick, cv.len());
            for (k, &var) in cv.iter().enumerate() { x[var] = vals[k]; }
        }
        x
    }

    /// Replace logits by smoothed log of weighted counts. `weights[c][k]`
    /// is the weight on sample `k` when fitting clique `c`'s table.
    pub fn fit_weighted(&mut self, samples: &[Design], weights: &[Vec<f64>], alpha: f64) {
        debug_assert_eq!(weights.len(), N_CLIQUES);
        for c in 0..N_CLIQUES {
            let n = self.n_outcomes[c];
            let cv = clique_vars(c);
            let mut counts = vec![alpha; n];
            for (k, x) in samples.iter().enumerate() {
                let w = weights[c][k];
                if w == 0.0 { continue; }
                let vals: Vec<u8> = cv.iter().map(|&v| x[v]).collect();
                counts[encode(&vals)] += w;
            }
            let z: f64 = counts.iter().sum();
            if z > 0.0 {
                for c_v in counts.iter_mut() { *c_v = (*c_v / z).ln(); }
            }
            self.logits[c] = counts;
        }
    }
}

fn sample_softmax(logits: &[f64], rng: &mut Rng) -> usize {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut probs = vec![0.0; logits.len()];
    let mut z = 0.0;
    for (i, &l) in logits.iter().enumerate() {
        let p = (l - m).exp();
        probs[i] = p;
        z += p;
    }
    let u = rng.next_unit() * z;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u <= acc { return i; }
    }
    probs.len() - 1
}

/// Boltzmann sample weights, unnormalised so the best sample gets
/// weight 1 and counts have the right scale relative to `alpha`.
fn softmax_weights(scores: &[f64], tau: f64) -> Vec<f64> {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    scores.iter().map(|s| ((s - m) / tau).exp()).collect()
}

// ---------------------------------------------------------------------
// DADO + naive EDA
//
// Q_c for our chain JT (rooted at clique 0) = suffix sum of components
// from c through the last clique. With disjoint cliques each Q_c is the
// reward this conditional + later conditionals can affect.
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RunTrace {
    pub best: Vec<f64>,
    pub mean: Vec<f64>,
    pub best_design: Design,
    pub best_score: f64,
}

pub fn run(
    score: &ScoreFn,
    n_iters: usize,
    k_samples: usize,
    tau: f64,
    alpha: f64,
    use_decomposition: bool,
    seed: u32,
) -> RunTrace {
    let mut rng = Rng::new(seed);
    let mut dist = CliqueDist::uniform();
    let mut best_so_far = f64::NEG_INFINITY;
    let mut best_design_so_far: Design = [(D / 2) as u8; L];
    let mut best_history = Vec::with_capacity(n_iters);
    let mut mean_history = Vec::with_capacity(n_iters);

    for _it in 0..n_iters {
        let mut samples = Vec::with_capacity(k_samples);
        let mut totals  = Vec::with_capacity(k_samples);
        let mut comps   = Vec::with_capacity(k_samples);
        for _ in 0..k_samples {
            let x = dist.sample(&mut rng);
            let (t, c) = score(&x);
            samples.push(x);
            totals.push(t);
            comps.push(c);
        }
        // Per-clique sample weights.
        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(N_CLIQUES);
        for c in 0..N_CLIQUES {
            let raw: Vec<f64> = if use_decomposition {
                // Q_c = suffix sum of components from c onwards.
                (0..k_samples).map(|k| {
                    let mut s = 0.0;
                    for j in c..N_CLIQUES { s += comps[k][j]; }
                    s
                }).collect()
            } else {
                totals.clone()
            };
            weights.push(softmax_weights(&raw, tau));
        }
        dist.fit_weighted(&samples, &weights, alpha);

        let (best_idx, &batch_best) = totals
            .iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        if batch_best > best_so_far {
            best_so_far = batch_best;
            best_design_so_far = samples[best_idx];
        }
        best_history.push(best_so_far);
        mean_history.push(totals.iter().sum::<f64>() / k_samples as f64);
    }
    RunTrace {
        best: best_history, mean: mean_history,
        best_design: best_design_so_far, best_score: best_so_far,
    }
}

// ---------------------------------------------------------------------
// PRNG (xorshift32, matches spike-dado-r2r style).
// ---------------------------------------------------------------------

#[derive(Clone)]
pub struct Rng(u32);
impl Rng {
    pub fn new(seed: u32) -> Self { Self(seed.max(1)) }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x;
        x
    }
    pub fn next_unit(&mut self) -> f64 { self.next_u32() as f64 / u32::MAX as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_is_uniform_on_init() {
        let mut rng = Rng::new(7);
        let d = CliqueDist::uniform();
        // Sample many times — every variable should hit every value.
        let mut hits = vec![[false; D]; L];
        for _ in 0..2000 {
            let x = d.sample(&mut rng);
            for v in 0..L { hits[v][x[v] as usize] = true; }
        }
        for v in 0..L {
            for c in 0..D {
                assert!(hits[v][c], "var {v} value {c} never sampled under uniform prior");
            }
        }
    }
}
