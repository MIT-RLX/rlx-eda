//! MNA baselines from §10. M-default is the canonical comparison; the
//! coarse and fine variants vary `n_newton_step` and `steps_per_tau`
//! to draw the (accuracy, latency) tradeoff curve.
//!
//! Each call inlines the same BE+Newton math `spike_diode::ref_transient`
//! uses but with the `vmid_initial = 0` step-from-zero setup the
//! oracle uses too. Skipping the rlx graph compile keeps per-call
//! cost honest — what we're racing the PINN against is the *solver
//! work*, not graph compilation overhead.

use crate::config::*;
use crate::encoding::Sample;

/// Run the BE+Newton MNA solver at the given config and return
/// `Vmid(t)` per sample, in volts.
pub fn run_mna(samples: &[Sample], cfg: MnaConfig) -> Vec<f32> {
    samples
        .iter()
        .map(|s| {
            let n_steps = cfg.steps_per_tau.max(1);
            let h = s.t / n_steps as f32;
            let mut vmid = 0.0_f32;
            for _ in 0..n_steps {
                let mut x = vmid;
                for _ in 0..cfg.n_newton_step {
                    let exp_v = (x / VT).exp();
                    let f  = (s.v_dc - x) / s.r
                           - s.is_ * (exp_v - 1.0)
                           - s.c * (x - vmid) / h;
                    let fp = -1.0 / s.r
                           - (s.is_ / VT) * exp_v
                           - s.c / h;
                    x -= f / fp;
                }
                vmid = x;
            }
            vmid
        })
        .collect()
}
