//! Paired Wilcoxon signed-rank, Cliff's δ, Holm-Bonferroni
//! correction, and bootstrap CI. Just enough statistics to evaluate
//! the §11 protocol; nothing more.
//!
//! No `statrs` dependency — this is small and audit-friendly hand-
//! rolled code, and the methodology benefits from being inspectable
//! without chasing through a third-party library.

/// Mean ± std + 95% bootstrap CI on a vector of K seed-level scores.
pub struct Summary {
    pub mean: f32,
    pub std: f32,
    pub ci95_lo: f32,
    pub ci95_hi: f32,
}

pub fn summarise(xs: &[f32]) -> Summary {
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
    let std = var.sqrt();

    // Percentile bootstrap with 1000 resamples. Deterministic seed
    // so reports are reproducible.
    let mut rng = 0xCFB1_2345_u32;
    let mut means = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut s = 0.0_f32;
        for _ in 0..xs.len() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let idx = (rng as usize) % xs.len();
            s += xs[idx];
        }
        means.push(s / n);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = means[(0.025 * 1000.0) as usize];
    let hi = means[(0.975 * 1000.0) as usize];
    Summary { mean, std, ci95_lo: lo, ci95_hi: hi }
}

/// Cliff's δ: probability that a random `a[i]` exceeds a random
/// `b[j]` minus the probability of the reverse. Range `[-1, 1]`.
/// Magnitude bins per §11: <0.147 negligible, <0.33 small,
/// <0.474 medium, ≥0.474 large.
pub fn cliffs_delta(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let m = b.len() as f32;
    let mut greater = 0;
    let mut less = 0;
    for x in a {
        for y in b {
            if x > y      { greater += 1; }
            else if x < y { less    += 1; }
        }
    }
    (greater as f32 - less as f32) / (n * m)
}

pub fn delta_label(d: f32) -> &'static str {
    let m = d.abs();
    if      m < 0.147 { "negligible" }
    else if m < 0.33  { "small" }
    else if m < 0.474 { "medium" }
    else              { "large" }
}

/// Two-sided paired Wilcoxon signed-rank exact test. Returns the
/// p-value computed by exhaustive sign enumeration over the K paired
/// samples (`O(2^K)`; K=10 → 1024 enumerations). Exact null
/// distribution avoids the small-K asymptotic-approximation
/// artefacts that the textbook normal-approximation suffers.
pub fn wilcoxon_signed_rank(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut diffs: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    // Drop zero diffs — they contribute no rank but can inflate K.
    diffs.retain(|d| *d != 0.0);
    let k = diffs.len();
    if k == 0 { return 1.0; }

    // Rank by absolute value, with average-rank ties.
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&i, &j| diffs[i].abs().partial_cmp(&diffs[j].abs()).unwrap());
    let mut ranks = vec![0.0_f32; k];
    let mut i = 0;
    while i < k {
        let mut j = i;
        while j + 1 < k && diffs[idx[j + 1]].abs() == diffs[idx[i]].abs() {
            j += 1;
        }
        let avg = (i + j) as f32 / 2.0 + 1.0;
        for p in i..=j { ranks[idx[p]] = avg; }
        i = j + 1;
    }

    // Observed test statistic: sum of signed ranks where d > 0.
    let observed: f64 = diffs.iter().enumerate()
        .filter(|(_, d)| **d > 0.0)
        .map(|(i, _)| ranks[i] as f64)
        .sum();
    let total: f64 = ranks.iter().map(|r| *r as f64).sum();

    // Null: each rank's sign is i.i.d. uniform on {+1, -1}. Exact
    // two-sided p-value: count permutations with W ≥ observed or
    // W ≤ total − observed (whichever is more extreme).
    let extreme = observed.max(total - observed);
    let mut count = 0_u32;
    for mask in 0..(1u64 << k) {
        let mut w = 0.0_f64;
        for i in 0..k {
            if (mask >> i) & 1 == 1 {
                w += ranks[i] as f64;
            }
        }
        if w >= extreme - 1e-9 || w <= total - extreme + 1e-9 {
            count += 1;
        }
    }
    (count as f64) / (1u64 << k) as f64
}

/// Apply Holm-Bonferroni correction to a family of p-values.
/// Returns the per-test rejection threshold ordered by original
/// position. A test rejects if `p[i] ≤ threshold[i]`.
pub fn holm_bonferroni(p_values: &[f64], alpha: f64) -> Vec<f64> {
    let n = p_values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| p_values[i].partial_cmp(&p_values[j]).unwrap());
    let mut thr = vec![0.0_f64; n];
    for (k, &i) in order.iter().enumerate() {
        thr[i] = alpha / ((n - k) as f64);
    }
    thr
}
