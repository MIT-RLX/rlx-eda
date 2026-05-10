//! Per-sample (vin + realization) record + 10-D normalised encoding.

use crate::config::*;

/// Physical-units sample: `vin/vref ∈ [0, 1]`, per-bit weight errors
/// (1.0 = no error), single comparator offset in volts.
#[derive(Clone, Debug)]
pub struct McSample {
    pub vin_norm:        f32,           // vin / vref
    pub bit_weight_err:  Vec<f32>,      // length = N_BITS
    pub comp_offset:     f32,           // volts
}

impl McSample {
    /// Encode to the network's 10-D input (each dim ≈ [-1, 1]).
    pub fn encode(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(INPUT_DIM);
        // vin: [0,1] → [-1, 1]
        v.push(2.0 * self.vin_norm - 1.0);
        // bit-weight errors: scale by 3σ so |z| < 1 99.7% of the time.
        let sigma_eff = 3.0 * (SIGMA_R as f32) * (2.0_f32).sqrt();
        for &e in &self.bit_weight_err {
            v.push((e - 1.0) / sigma_eff);
        }
        // comparator offset: scale by 3σ.
        let sigma_off = 3.0 * SIGMA_OFFSET as f32;
        v.push(self.comp_offset / sigma_off);
        v
    }
}
