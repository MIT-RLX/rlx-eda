//! Latin-hypercube on the 10-D parameter cube. vin uniform on [0,1];
//! bit-weight errors and comparator offset Gaussian (Box-Muller from
//! stratified uniforms — keeps the LHS structure on the underlying
//! quantile axis).

use eda_nn::Rng;

use crate::config::*;
use crate::sample::McSample;

pub fn lhs_samples(n: usize, seed: u64) -> Vec<McSample> {
    let mut rng = Rng::new((seed as u32) ^ ((seed >> 32) as u32));
    let dim = 1 + N_BITS + 1;

    // Stratified+permuted columns per axis.
    let mut cols: Vec<Vec<f32>> = (0..dim).map(|_| {
        let mut col: Vec<f32> = (0..n).map(|i| (i as f32 + rng.next_unit()) / n as f32).collect();
        for i in (1..n).rev() {
            let j = (rng.next() as usize) % (i + 1);
            col.swap(i, j);
        }
        col
    }).collect();

    let off_col = cols.pop().unwrap();
    let bit_cols: Vec<Vec<f32>> = (0..N_BITS).map(|_| cols.pop().unwrap()).collect();
    let vin_col = cols.pop().unwrap();

    let sigma_eff = (SIGMA_R as f32) * (2.0_f32).sqrt();
    let sigma_off = SIGMA_OFFSET as f32;

    (0..n).map(|k| {
        // vin: uniform on [0, 1).
        let vin_norm = vin_col[k];

        // Per-bit weight errors via Box-Muller on the LHS uniforms.
        // Use pairs (cols[i], cols[i+1 mod N]) — consumes the
        // stratified structure without duplicating samples.
        let bit_weight_err: Vec<f32> = (0..N_BITS).map(|i| {
            let u1 = bit_cols[i][k].max(1e-9);
            let u2 = bit_cols[(i + 1) % N_BITS][k];
            let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            1.0 + sigma_eff * g
        }).collect();

        // Comparator offset.
        let u1 = off_col[k].max(1e-9);
        let u2 = vin_col[k]; // borrow vin's uniform for the second BM stream
        let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let comp_offset = sigma_off * g;

        McSample { vin_norm, bit_weight_err, comp_offset }
    }).collect()
}
