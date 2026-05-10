//! Tier 2: finite-difference probe of the rlx forward across pulse params.
//!
//! Per-step BE gradients (∂vout/∂R, ∂vout/∂C) are already validated by
//! `spike-rc-transient`. The new gradient surface this spike adds is the
//! response to **the source value itself**: how does `vout(T)` change when
//! we shift the pulsed level `v2`? We don't have AD on `v2` (it enters
//! through the host loop, not the graph parameters), so this is a
//! *forward-only* FD smoke test — perturb v2, verify the response moves
//! by approximately (1 − exp(−pw/τ)) when the probe lands inside the
//! charging plateau.

use eda_hir::SourceWaveform;
use spike_pulse_rc::run_transient_final;

const R: f64 = 1_000.0;
const C: f64 = 1e-9;
const TAU: f64 = R * C;

#[test]
fn finite_difference_dvout_dv2_matches_analytic() {
    let td = 0.0;
    let pw = TAU;          // 1 RC long pulse
    let h = TAU / 200.0;   // fine timestep
    let t_eval = 0.5 * TAU; // mid-plateau
    let n_steps = (t_eval / h).round() as usize;

    let v2_base = 1.0;
    let eps = 1e-3;
    let make = |v2| SourceWaveform::pulse(0.0, v2, td, 0.0, 0.0, pw, 0.0);

    let v_plus  = run_transient_final(n_steps, h, R, C, 0.0, &make(v2_base + eps));
    let v_minus = run_transient_final(n_steps, h, R, C, 0.0, &make(v2_base - eps));
    let dv_dv2_fd = (v_plus - v_minus) / (2.0 * eps);

    // Analytic: vout(t) = v2 · (1 − exp(−t/τ)) for v1=0, td=0; so
    // ∂vout/∂v2 = 1 − exp(−t/τ).
    let dv_dv2_analytic = 1.0 - (-(t_eval / TAU)).exp();

    let env = 5e-3 + 5e-3 * dv_dv2_analytic.abs();
    assert!(
        (dv_dv2_fd - dv_dv2_analytic).abs() < env,
        "FD dv/dv2 = {dv_dv2_fd:.4} vs analytic {dv_dv2_analytic:.4}",
    );
}
