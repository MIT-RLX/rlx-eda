//! Behavioral oracle: `BehavioralSar::convert` with the per-sample
//! Realization. `comp_noise_sigma=0` (deterministic), so the rng is a
//! no-op pass-through.

use spike_sar_adc::behavioral::{BehavioralSar, Lcg64, Realization};

use crate::config::*;
use crate::sample::McSample;

fn spec() -> BehavioralSar {
    BehavioralSar {
        n_bits: N_BITS,
        vref:   VREF as f64,
        r_mismatch_sigma:   SIGMA_R,
        comp_offset_sigma:  SIGMA_OFFSET,
        comp_noise_sigma:   0.0,
        sh_droop_tau:       None,
        conversion_time:    1e-6,
        comp_decision_time: 1e-9,
        comp_latch_tau:     50e-12,
    }
}

pub fn truth_norm(s: &McSample) -> f32 {
    let realization = Realization {
        bit_weight_err: s.bit_weight_err.iter().map(|&e| e as f64).collect(),
        comp_offset:    s.comp_offset as f64,
    };
    let mut rng = Lcg64::new(0xC0DE_C0DE);
    let code = spec().convert(s.vin_norm as f64, &realization, &mut rng);
    code as f32 / LEVELS as f32
}
