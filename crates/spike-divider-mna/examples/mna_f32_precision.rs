//! f32-vs-f64 dense LU drift on representative MNA matrices.
//!
//! Run release-mode for stable timings; numerics are mode-independent:
//!   cargo run --example mna_f32_precision -p spike-divider-mna --release
//!
//! ## What this answers
//!
//! Before investing upstream rlx-mlx work to add a GPU `DenseSolve`,
//! we need to know: would f32 LU on Apple GPU produce usable analog
//! operating points, or would precision loss kill the workload?
//!
//! ## What it does NOT use
//!
//! Deliberately no rlx graph here — `rlx-mlx` has no `DenseSolve`
//! lowering and `rlx-cpu` has no f32 path. The probe builds matrices
//! in pure Rust + hand-rolled LU in both precisions to isolate the
//! numerical question from the missing op work.
//!
//! ## The matrix
//!
//! N-stage resistor ladder: nodes 0..N driven by V at node 0. Each
//! stage has a series R_i and a shunt R_g to ground. Conductance
//! spread (`decades`) controls the dynamic range of R values within
//! the matrix — the realistic stress test is mixing 1Ω power-ground
//! return paths with 1MΩ bias resistors in the same system.

use std::time::Instant;

// ── Tiny LU solver, generic over float ──────────────────────────────

trait Float: Copy + std::fmt::Debug + std::ops::Neg<Output = Self> {
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn lt(self, other: Self) -> bool;
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_f64(x: f64) -> Self { x }
    fn to_f64(self) -> f64 { self }
    fn abs(self) -> Self { f64::abs(self) }
    fn add(self, o: Self) -> Self { self + o }
    fn sub(self, o: Self) -> Self { self - o }
    fn mul(self, o: Self) -> Self { self * o }
    fn div(self, o: Self) -> Self { self / o }
    fn lt(self, o: Self) -> bool { self < o }
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_f64(x: f64) -> Self { x as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn abs(self) -> Self { f32::abs(self) }
    fn add(self, o: Self) -> Self { self + o }
    fn sub(self, o: Self) -> Self { self - o }
    fn mul(self, o: Self) -> Self { self * o }
    fn div(self, o: Self) -> Self { self / o }
    fn lt(self, o: Self) -> bool { self < o }
}

/// Gaussian elimination with partial pivoting. Solves A·x = b in place
/// of A and b; returns x. Panics on near-singular pivot. Plain
/// textbook implementation — same algorithm in both precisions so
/// drift is purely the float type, not implementation differences.
fn lu_solve<F: Float>(a: &mut [Vec<F>], b: &mut [F]) -> Vec<F> {
    let n = b.len();
    for k in 0..n {
        // Partial pivot on column k.
        let mut piv = k;
        let mut piv_abs = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if piv_abs.lt(v) { piv = i; piv_abs = v; }
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }
        let akk = a[k][k];
        if akk.abs().lt(F::from_f64(1e-30)) {
            panic!("singular pivot at row {k}");
        }
        for i in (k + 1)..n {
            let factor = a[i][k].div(akk);
            for j in k..n {
                let v = a[i][j].sub(factor.mul(a[k][j]));
                a[i][j] = v;
            }
            b[i] = b[i].sub(factor.mul(b[k]));
        }
    }
    // Back-substitute.
    let mut x = vec![F::ZERO; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s = s.sub(a[i][j].mul(x[j]));
        }
        x[i] = s.div(a[i][i]);
    }
    x
}

// ── Representative MNA matrix builder ───────────────────────────────

/// N-stage resistor ladder driven by V at the input node.
/// Returns `(A, b)` for `A·x = b` where x is node voltages 1..=N.
///
/// `decades` controls the conductance spread inside the matrix:
/// stage k uses series R = R0·10^((k mod decades) × log10_step) and
/// shunt R = R0·10^(((k+1) mod decades) × log10_step). decades=0 →
/// uniform R; decades=6 → 6-OOM spread, the realistic worst-case.
fn build_ladder<F: Float>(n_stages: usize, v: f64, decades: usize) -> (Vec<Vec<F>>, Vec<F>) {
    let r0 = 1e3_f64;
    let log10_step = if decades == 0 { 0.0 } else { decades as f64 / (n_stages as f64).max(1.0) };

    let mut conductances_series = Vec::with_capacity(n_stages);
    let mut conductances_shunt  = Vec::with_capacity(n_stages);
    for k in 0..n_stages {
        let rs = r0 * 10f64.powf((k as f64) * log10_step);
        let rg = r0 * 10f64.powf(((k + 1) as f64) * log10_step);
        conductances_series.push(1.0 / rs);
        conductances_shunt.push(1.0 / rg);
    }

    let n = n_stages;
    let mut a = vec![vec![F::ZERO; n]; n];
    let mut b = vec![F::ZERO; n];

    // Node 0 (boundary V) is folded into b. Node k>0 has KCL:
    //   g_series[k-1]·(v_k - v_{k-1}) + g_series[k]·(v_k - v_{k+1}) + g_shunt[k]·v_k = 0
    // with v_0 = V (boundary) and v_N = floating (terminate the
    // series with a shunt only).
    for k in 0..n {
        let gs_in  = conductances_series[k];
        let gs_out = if k + 1 < n { conductances_series[k + 1] } else { 0.0 };
        let gg     = conductances_shunt[k];
        // diagonal: gs_in + gs_out + gg
        a[k][k] = F::from_f64(gs_in + gs_out + gg);
        // upper neighbor: -gs_out
        if k + 1 < n {
            a[k][k + 1] = F::from_f64(-gs_out);
        }
        // lower neighbor: -gs_in
        if k > 0 {
            a[k][k - 1] = F::from_f64(-gs_in);
        }
        // RHS: at k=0 the boundary contributes gs_in·V
        if k == 0 {
            b[k] = F::from_f64(gs_in * v);
        }
    }
    (a, b)
}

// ── Probe ───────────────────────────────────────────────────────────

/// Power iteration on |A| to estimate the spectral norm; then on |A^-1|
/// (via inverse power iteration using the f64 LU) for κ_2. Cheap and
/// good enough for an order-of-magnitude reading.
fn cond_estimate(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    fn pow_iter<F>(matvec: F, n: usize, iters: usize) -> f64
    where F: Fn(&[f64]) -> Vec<f64> {
        let mut v = vec![1.0_f64 / (n as f64).sqrt(); n];
        let mut lambda = 0.0_f64;
        for _ in 0..iters {
            let w = matvec(&v);
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
            lambda = norm;
            for i in 0..n { v[i] = w[i] / norm; }
        }
        lambda
    }
    let s_max = pow_iter(|x| {
        let mut y = vec![0.0; n];
        for i in 0..n { for j in 0..n { y[i] += a[i][j] * x[j]; } }
        y
    }, n, 50);
    // smallest singular ≈ 1 / largest singular of A^-1, approximated
    // via inverse iteration: solve A·y = x then y becomes the new x.
    let mut v = vec![1.0_f64 / (n as f64).sqrt(); n];
    let mut s_min_inv = 0.0_f64;
    for _ in 0..50 {
        let mut a_copy: Vec<Vec<f64>> = a.iter().cloned().collect();
        let mut b_copy = v.clone();
        let y = lu_solve(&mut a_copy, &mut b_copy);
        let norm = y.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        s_min_inv = norm;
        for i in 0..n { v[i] = y[i] / norm; }
    }
    s_max * s_min_inv
}

fn max_rel_drift(x_ref: &[f64], x: &[f64]) -> f64 {
    x_ref.iter().zip(x.iter())
        .map(|(r, v)| {
            let denom = r.abs().max(1e-30);
            ((r - v).abs()) / denom
        })
        .fold(0.0_f64, f64::max)
}

/// One step of mixed-precision iterative refinement: f32 LU gives x0,
/// recompute residual r = b - A·x0 in f64, solve A·dx = r in f32,
/// add. This is the standard cure when f32 LU on its own is too noisy.
fn refined_solve(a64: &[Vec<f64>], b64: &[f64]) -> Vec<f64> {
    let n = b64.len();
    // Initial f32 solve.
    let mut a32: Vec<Vec<f32>> = a64.iter().map(|row| row.iter().map(|&v| v as f32).collect()).collect();
    let mut b32: Vec<f32> = b64.iter().map(|&v| v as f32).collect();
    let x32 = lu_solve(&mut a32, &mut b32);
    let mut x: Vec<f64> = x32.iter().map(|&v| v as f64).collect();

    // Residual in f64.
    let mut r = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b64[i];
        for j in 0..n { s -= a64[i][j] * x[j]; }
        r[i] = s;
    }
    // f32 correction.
    let mut a32: Vec<Vec<f32>> = a64.iter().map(|row| row.iter().map(|&v| v as f32).collect()).collect();
    let mut r32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
    let dx = lu_solve(&mut a32, &mut r32);
    for i in 0..n { x[i] += dx[i] as f64; }
    x
}

fn main() {
    println!("MNA dense-LU precision probe — f32 vs f64 vs mixed-precision refinement");
    println!("Matrix: N-stage resistor ladder, conductance spread = `decades` (10^decades range)");
    println!();
    println!("{:>5}  {:>9}  {:>12}  {:>13}  {:>13}  {:>13}",
             "N", "decades", "cond κ_2", "f32 max rel", "refined max rel", "f32 time/solve");

    for &n in &[8usize, 16, 32, 64, 128] {
        for &decades in &[0usize, 2, 4, 6] {
            let v = 1.0;

            // Reference: f64 LU.
            let (mut a64_solve, mut b64_solve) = build_ladder::<f64>(n, v, decades);
            let a64_orig: Vec<Vec<f64>> = a64_solve.clone();
            let b64_orig: Vec<f64> = b64_solve.clone();
            let x_ref = lu_solve(&mut a64_solve, &mut b64_solve);

            // f32 LU.
            let (mut a32, mut b32) = build_ladder::<f32>(n, v, decades);
            let t0 = Instant::now();
            let x32 = lu_solve(&mut a32, &mut b32);
            let t_f32 = t0.elapsed().as_secs_f64() * 1e6;
            let x32_64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();

            // Mixed-precision refined.
            let x_ref_mixed = refined_solve(&a64_orig, &b64_orig);

            let drift_f32  = max_rel_drift(&x_ref, &x32_64);
            let drift_ref  = max_rel_drift(&x_ref, &x_ref_mixed);
            let kappa = cond_estimate(&a64_orig);

            println!("{:>5}  {:>9}  {:>12.3e}  {:>13.3e}  {:>13.3e}  {:>11.2} µs",
                     n, decades, kappa, drift_f32, drift_ref, t_f32);
        }
    }

    println!();
    println!("Reading: f32 max rel ≈ κ_2 × ε_f32 (ε_f32 ≈ 1.2e-7) is the textbook bound.");
    println!("Refined column shows whether one mixed-precision iteration recovers");
    println!("near-f64 accuracy — the cheap cure if naked f32 isn't enough.");
}
