//! Multi-D polynomial regression baselines.

use crate::config::*;
use crate::oracle::truth_norm;
use crate::sample::McSample;

/// Enumerate multi-indices `(α_0, ..., α_{D-1})` with `Σα ≤ d`.
fn monomials(d: usize, dim: usize) -> Vec<Vec<u8>> {
    fn rec(prefix: &mut Vec<u8>, remaining_vars: usize, remaining_deg: usize, out: &mut Vec<Vec<u8>>) {
        if remaining_vars == 0 {
            out.push(prefix.clone());
            return;
        }
        for da in 0..=remaining_deg {
            prefix.push(da as u8);
            rec(prefix, remaining_vars - 1, remaining_deg - da, out);
            prefix.pop();
        }
    }
    let mut out = Vec::new();
    let mut prefix = Vec::with_capacity(dim);
    rec(&mut prefix, dim, d, &mut out);
    out
}

#[inline]
fn eval_monomial(m: &[u8], x: &[f32]) -> f64 {
    let mut v = 1.0_f64;
    for k in 0..x.len() {
        for _ in 0..m[k] { v *= x[k] as f64; }
    }
    v
}

fn solve_spd(a: &mut [f64], b: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j { a[i * n + j] = sum.max(0.0).sqrt(); }
            else      { a[i * n + j] = sum / a[j * n + j]; }
        }
    }
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i { sum -= a[i * n + k] * b[k]; }
        b[i] = sum / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n { sum -= a[k * n + i] * b[k]; }
        b[i] = sum / a[i * n + i];
    }
}

pub struct PolynomialND {
    pub degree: usize,
    pub monomials: Vec<Vec<u8>>,
    pub coeffs: Vec<f64>,
}

impl PolynomialND {
    pub fn fit(degree: usize, train_samples: &[McSample]) -> Self {
        let monomials = monomials(degree, INPUT_DIM);
        let m = monomials.len();
        let mut xtx = vec![0.0_f64; m * m];
        let mut xty = vec![0.0_f64; m];
        let mut row = vec![0.0_f64; m];

        for s in train_samples {
            let enc = s.encode();
            for (j, mono) in monomials.iter().enumerate() {
                row[j] = eval_monomial(mono, &enc);
            }
            let yi = truth_norm(s) as f64;
            for j in 0..m {
                xty[j] += row[j] * yi;
                for k in 0..=j {
                    xtx[j * m + k] += row[j] * row[k];
                }
            }
        }
        for j in 0..m {
            for k in 0..j { xtx[k * m + j] = xtx[j * m + k]; }
        }
        for i in 0..m { xtx[i * m + i] += 1e-8; }
        solve_spd(&mut xtx, &mut xty, m);

        Self { degree, monomials, coeffs: xty }
    }

    pub fn predict(&self, samples: &[McSample]) -> Vec<f32> {
        samples.iter().map(|s| {
            let enc = s.encode();
            let mut sum = 0.0_f64;
            for (j, mono) in self.monomials.iter().enumerate() {
                sum += self.coeffs[j] * eval_monomial(mono, &enc);
            }
            sum as f32
        }).collect()
    }

    pub fn n_params(&self) -> usize { self.coeffs.len() }
    pub fn n_bytes(&self) -> usize { self.coeffs.len() * 4 }
}
