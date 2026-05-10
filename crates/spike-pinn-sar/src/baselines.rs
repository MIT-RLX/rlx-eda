//! Polynomial regression (1-D, varying degree) and lookup-table
//! interpolation baselines. Both are deterministic given the
//! training samples + oracle, so each runs once (no per-seed
//! variance to report).

use crate::config::*;
use crate::oracle::truth_norm;

// ── 1-D polynomial regression via normal equations + Cholesky ────

fn solve_spd(a: &mut [f64], b: &mut [f64], n: usize) {
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

pub struct Polynomial1D {
    pub degree: usize,
    pub coeffs: Vec<f64>, // c[0] + c[1]·x + ... + c[d]·x^d
}

impl Polynomial1D {
    pub fn fit(degree: usize, train_x: &[f32]) -> Self {
        let m = degree + 1;
        let mut xtx = vec![0.0_f64; m * m];
        let mut xty = vec![0.0_f64; m];
        let mut row = vec![0.0_f64; m];
        for &x in train_x {
            let xf = x as f64;
            row[0] = 1.0;
            for k in 1..m { row[k] = row[k - 1] * xf; }
            let y = truth_norm(x) as f64;
            for j in 0..m {
                xty[j] += row[j] * y;
                for k in 0..=j {
                    xtx[j * m + k] += row[j] * row[k];
                }
            }
        }
        for j in 0..m {
            for k in 0..j { xtx[k * m + j] = xtx[j * m + k]; }
        }
        for i in 0..m { xtx[i * m + i] += 1e-9; }
        solve_spd(&mut xtx, &mut xty, m);
        Self { degree, coeffs: xty }
    }

    pub fn predict(&self, xs: &[f32]) -> Vec<f32> {
        xs.iter().map(|&x| {
            let mut p = 1.0_f64;
            let xf = x as f64;
            let mut sum = 0.0_f64;
            for &c in &self.coeffs {
                sum += c * p;
                p *= xf;
            }
            sum as f32
        }).collect()
    }

    pub fn n_params(&self) -> usize { self.coeffs.len() }
    pub fn n_bytes(&self) -> usize { self.coeffs.len() * 4 }
}

// ── Lookup table with linear interpolation ───────────────────────

pub struct Lookup {
    pub n_nodes: usize,
    pub values:  Vec<f32>,  // values at uniformly spaced grid in [0, 1]
}

impl Lookup {
    pub fn fit(n_nodes: usize) -> Self {
        // Grid points at i / (n_nodes - 1) for i ∈ {0, ..., n_nodes-1}.
        // Note: the LAST grid point is at x=1.0, which the SAR oracle
        // clamps to code=255 (since x=1.0 means vin = vref → top code).
        let denom = (n_nodes - 1) as f32;
        let values: Vec<f32> = (0..n_nodes)
            .map(|i| truth_norm(i as f32 / denom))
            .collect();
        Self { n_nodes, values }
    }

    pub fn predict(&self, xs: &[f32]) -> Vec<f32> {
        let last = self.n_nodes - 1;
        let denom = last as f32;
        xs.iter().map(|&x| {
            // Map x ∈ [0, 1] to grid coordinate.
            let g = x.clamp(0.0, 1.0) * denom;
            let lo = (g.floor() as usize).min(last - 1);
            let hi = lo + 1;
            let t = g - lo as f32;
            self.values[lo] * (1.0 - t) + self.values[hi] * t
        }).collect()
    }

    pub fn n_params(&self) -> usize { self.n_nodes }
    pub fn n_bytes(&self) -> usize { self.n_nodes * 4 }
}
