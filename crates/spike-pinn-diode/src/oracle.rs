//! Data-anchor oracle: `spike_diode::ref_transient` evaluated at the
//! per-sample (R, Is, C, V_dc, t).
//!
//! Setup per §7 amendment: step input from 0 V to V_dc at t=0+. The
//! ref_transient call uses `v_dc_seed=0.0` (so the BE loop's initial
//! Vmid is the DC OP at 0, which is 0) and a constant `v_per_step =
//! [V_dc; N_STEPS_ORACLE]`. The result is `Vmid(t)` in physical
//! units; the network predicts `Vmid/V_REF`, so the trainer divides
//! by `V_REF` before feeding `y_truth` into the loss graph.

use crate::config::*;
use crate::encoding::Sample;

/// Number of BE steps used in the oracle. Higher than the M-default
/// MNA baseline so the data-anchor's accuracy is not bounded by the
/// baseline's discretisation error.
pub const N_STEPS_ORACLE: usize = 500;
pub const N_NEWTON_DC_ORACLE: usize = 16;
pub const N_NEWTON_STEP_ORACLE: usize = 8;

/// Compute `Vmid(t)/V_REF` for one sample using the pure-Rust BE +
/// Newton reference. Pure-Rust path keeps the oracle device-agnostic
/// — calling spike-diode's rlx graph here would force a graph
/// compile per sample, which dominates wall-clock at N=12k.
pub fn truth_norm(s: &Sample) -> f32 {
    // We could call `spike_diode::ref_transient`, but it always
    // seeds vmid via `ref_dc_op(v_dc, ...)` at the *final* voltage,
    // which means a constant `v_per_step` produces a no-op transient
    // (vmid stays at the DC OP). For the step-from-zero setup we
    // need vmid_initial = 0, so we inline the same BE+Newton math
    // ref_transient runs and seed it ourselves.
    let h = s.t / N_STEPS_ORACLE as f32;
    let mut vmid = 0.0_f32;
    for _ in 0..N_STEPS_ORACLE {
        let mut x = vmid; // Newton seed: previous step's solution.
        for _ in 0..N_NEWTON_STEP_ORACLE {
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
    vmid / V_REF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vmid_starts_near_zero_and_rises() {
        // At t very small, Vmid should be near 0 (capacitor not
        // charged yet). At t large, Vmid should approach the DC OP
        // (some forward-diode voltage drop near 0.6 V).
        let early = Sample { r: 1e4, is_: 1e-13, c: 1e-9, v_dc: 1.0, t: 1e-9 };
        let late  = Sample { r: 1e4, is_: 1e-13, c: 1e-9, v_dc: 1.0, t: 1e-3 };
        let v_early = truth_norm(&early);
        let v_late  = truth_norm(&late);
        assert!(v_early < 0.05, "expected small Vmid at small t, got {v_early}");
        assert!(v_late > 0.4 && v_late < 0.8, "expected diode forward drop at large t, got {v_late}");
        assert!(v_late > v_early);
    }
}
