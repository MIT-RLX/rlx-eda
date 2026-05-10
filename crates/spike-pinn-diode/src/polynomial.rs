//! Degree-`POLY_DEGREE` polynomial regression baseline (§10 row P).
//!
//! Tensor-product monomials in the 5-D normalised input, total
//! `C(5+4, 4) = 126` monomials at degree 4. Fit by least squares
//! on the training split (normal equations + Cholesky); predict on
//! test / OOD slices.
//!
//! Why this baseline matters: capacity. The polynomial has 126
//! parameters. The PINN MLP has 8,769. If polynomial regression on
//! the same training data sits on or ahead of the PINN's
//! (accuracy, latency) point, the PINN's capacity is unused — i.e.
//! a 70× smaller model is the right answer for this problem. C4
//! (post-amendment) tests this directly.

use crate::config::*;
use crate::encoding::Sample;

/// Enumerate all multi-indices `(α0, ..., α4)` with `Σα ≤ d`.
fn monomials_5(d: usize) -> Vec<[u8; 5]> {
    let mut out = Vec::new();
    for a0 in 0..=d {
        for a1 in 0..=(d - a0) {
            for a2 in 0..=(d - a0 - a1) {
                for a3 in 0..=(d - a0 - a1 - a2) {
                    for a4 in 0..=(d - a0 - a1 - a2 - a3) {
                        out.push([a0 as u8, a1 as u8, a2 as u8, a3 as u8, a4 as u8]);
                    }
                }
            }
        }
    }
    out
}

#[inline]
fn eval_monomial(m: &[u8; 5], x: &[f32; 5]) -> f32 {
    let mut v = 1.0;
    for k in 0..5 {
        for _ in 0..m[k] {
            v *= x[k];
        }
    }
    v
}

/// In-place Cholesky factorisation of an `n×n` symmetric positive
/// definite matrix, then forward + backward solve `A x = b`. `a` is
/// overwritten with its Cholesky factor; `b` with the solution.
fn solve_spd(a: &mut [f32], b: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                a[i * n + j] = sum.max(0.0).sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * n + k] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= a[k * n + i] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
}

pub struct Polynomial {
    pub monomials: Vec<[u8; 5]>,
    pub coeffs: Vec<f32>,
}

impl Polynomial {
    /// Fit on training samples + their `Vmid/V_REF` ground truth.
    pub fn fit(samples: &[Sample], y_truth_norm: &[f32]) -> Self {
        assert_eq!(samples.len(), y_truth_norm.len());
        let monomials = monomials_5(POLY_DEGREE);
        let m = monomials.len();
        let n = samples.len();

        let mut xtx = vec![0.0_f32; m * m];
        let mut xty = vec![0.0_f32; m];

        // Streaming accumulation: avoid materialising the full N×m
        // design matrix (saves ~6 MiB at N=12k).
        let mut xrow = vec![0.0_f32; m];
        for i in 0..n {
            let enc = samples[i].encode();
            for (j, mono) in monomials.iter().enumerate() {
                xrow[j] = eval_monomial(mono, &enc);
            }
            let yi = y_truth_norm[i];
            for j in 0..m {
                xty[j] += xrow[j] * yi;
                for k in 0..=j {
                    xtx[j * m + k] += xrow[j] * xrow[k];
                }
            }
        }
        // Mirror the lower triangle.
        for j in 0..m {
            for k in 0..j {
                xtx[k * m + j] = xtx[j * m + k];
            }
        }
        // Tiny ridge for numerical stability; with 12k samples the
        // problem is well-posed but the f32 accumulation can lose
        // significance on the diagonal of the high-degree monomials.
        for i in 0..m {
            xtx[i * m + i] += 1e-6;
        }
        solve_spd(&mut xtx, &mut xty, m);

        Self { monomials, coeffs: xty }
    }

    /// Predict `Vmid` (volts) for each sample.
    pub fn predict(&self, samples: &[Sample]) -> Vec<f32> {
        samples
            .iter()
            .map(|s| {
                let enc = s.encode();
                let mut sum = 0.0_f32;
                for (j, mono) in self.monomials.iter().enumerate() {
                    sum += self.coeffs[j] * eval_monomial(mono, &enc);
                }
                sum * V_REF
            })
            .collect()
    }

    pub fn n_params(&self) -> usize { self.coeffs.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monomial_count_is_126_at_degree_4() {
        // C(5+4, 4) = C(9, 4) = 126
        assert_eq!(monomials_5(4).len(), 126);
    }
}
