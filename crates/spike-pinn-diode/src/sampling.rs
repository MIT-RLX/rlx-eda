//! Latin-hypercube sampling over the 4-D parameter cube + uniform t.
//!
//! Each parameter axis is divided into `n` strata, one sample drawn
//! per stratum, and strata are independently permuted across axes.
//! This gives marginal stratification on every axis (no clusters in
//! any 1-D projection) without needing the full multi-dimensional
//! discrepancy minimisation that "true" LHS sometimes targets.

use eda_nn::Rng;

use crate::config::*;
use crate::encoding::Sample;

/// Slice configuration for `lhs_samples`: which parameter band to
/// draw from. In-distribution uses §3 ranges; OOD uses the §3 OOD
/// columns.
#[derive(Clone, Copy, Debug)]
pub enum Band {
    InDist,
    Ood,
}

fn band_ranges(b: Band) -> [(f32, f32); 4] {
    match b {
        Band::InDist => [
            (R_LO,   R_HI),
            (IS_LO,  IS_HI),
            (C_LO,   C_HI),
            (VDC_LO, VDC_HI),
        ],
        Band::Ood => [
            (R_OOD_LO,   R_OOD_HI),
            (IS_OOD_LO,  IS_OOD_HI),
            (C_OOD_LO,   C_OOD_HI),
            (VDC_OOD_LO, VDC_OOD_HI),
        ],
    }
}

/// Returns N samples in physical units, drawn LHS-style from the
/// requested band. `t/τ_ref` is sampled uniformly on each per-sample
/// `(R, C)` so the time axis is meaningfully scaled.
pub fn lhs_samples(n: usize, band: Band, seed: u64) -> Vec<Sample> {
    let mut rng = Rng::new((seed as u32) ^ ((seed >> 32) as u32));
    let ranges = band_ranges(band);

    // Stratified samples on `[0, 1)`, one column per parameter axis.
    let mut cols: Vec<Vec<f32>> = (0..5)
        .map(|_| {
            let mut col: Vec<f32> = (0..n).map(|i| (i as f32 + rng.next_unit()) / n as f32).collect();
            // Fisher-Yates shuffle.
            for i in (1..n).rev() {
                let j = (rng.next() as usize) % (i + 1);
                col.swap(i, j);
            }
            col
        })
        .collect();

    let t_col = cols.pop().unwrap();
    let v_col = cols.pop().unwrap();
    let c_col = cols.pop().unwrap();
    let i_col = cols.pop().unwrap();
    let r_col = cols.pop().unwrap();

    (0..n)
        .map(|k| {
            // R, Is, C are log-uniform on the configured range.
            let r = log_uniform(r_col[k], ranges[0]);
            let is_ = log_uniform(i_col[k], ranges[1]);
            let c = log_uniform(c_col[k], ranges[2]);
            // V_dc is linear-uniform.
            let v_dc = ranges[3].0 + (ranges[3].1 - ranges[3].0) * v_col[k];
            // t/τ uniform on [0.01, 5.0] then unscaled by τ = R·C.
            let tau = r * c;
            let t_over_tau =
                T_OVER_TAU_LO + (T_OVER_TAU_HI - T_OVER_TAU_LO) * t_col[k];
            Sample { r, is_, c, v_dc, t: t_over_tau * tau }
        })
        .collect()
}

fn log_uniform(u: f32, range: (f32, f32)) -> f32 {
    let lo = range.0.log10();
    let hi = range.1.log10();
    10f32.powf(lo + (hi - lo) * u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_across_runs() {
        let a = lhs_samples(64, Band::InDist, SPLIT_SEED_LHS);
        let b = lhs_samples(64, Band::InDist, SPLIT_SEED_LHS);
        for (s1, s2) in a.iter().zip(&b) {
            assert_eq!(s1.r,    s2.r);
            assert_eq!(s1.is_,  s2.is_);
            assert_eq!(s1.c,    s2.c);
            assert_eq!(s1.v_dc, s2.v_dc);
            assert_eq!(s1.t,    s2.t);
        }
    }

    #[test]
    fn samples_land_in_band() {
        let samples = lhs_samples(256, Band::InDist, SPLIT_SEED_LHS);
        for s in &samples {
            assert!(s.r >= R_LO && s.r <= R_HI);
            assert!(s.is_ >= IS_LO && s.is_ <= IS_HI);
            assert!(s.c >= C_LO && s.c <= C_HI);
            assert!(s.v_dc >= VDC_LO && s.v_dc <= VDC_HI);
            let t_over_tau = s.t / (s.r * s.c);
            assert!(t_over_tau >= T_OVER_TAU_LO && t_over_tau <= T_OVER_TAU_HI);
        }
    }
}
