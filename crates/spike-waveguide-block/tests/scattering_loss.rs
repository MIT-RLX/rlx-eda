//! Loss-only `OpticalScattering` tier — same `Waveguide` instance
//! drives both Layout AND simulation, with the loss param keyed by
//! `Block::name()`. The architectural payoff mirrors `spike-divider`:
//! one Rust value, two flows, gradient-correct.
//!
//! Tier 1 (this file): forward `|T| = exp(-α·L·ln(10)/20)` matches
//! analytic; reverse-mode AD `∂(loss)/∂α` matches both analytic and
//! a finite-difference probe; SGD on the loss converges toward a
//! target `|T|`.
//!
//! Phase / dispersion / full S-matrix differentiation is gated on
//! Sin/Cos landing in rlx-ir — see the follow-up.

use eda_validate::gradcheck_scalar;
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_waveguide_block::Waveguide;

const LN10_OVER_20: f32 = 2.302_585_093 / 20.0;
const DBU_TO_CM: f32 = 1.0e-7;

fn analytic_t_mag(alpha_db_per_cm: f32, length_dbu: i64) -> f32 {
    let length_cm = (length_dbu as f32) * DBU_TO_CM;
    (-alpha_db_per_cm * length_cm * LN10_OVER_20).exp()
}

#[test]
fn waveguide_t_mag_matches_analytic_forward() {
    let wg = Waveguide { width: 500, length: 1_000_000, id: "wg1".into() }; // 100 µm
    let (fwd, loss_id) = wg.build_loss_graph();
    // grad_with_loss treats inputs to grad as "wrt" — we use it here
    // just to materialize the forward path; we ignore the gradient.
    let bwd = grad_with_loss(&fwd, &[loss_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    let alpha = 1.0_f32; // 1 dB/cm — typical SOI strip
    sess.set_param(&wg.loss_param_name(), &[alpha]);
    // |T|² target irrelevant for forward — pick any. We read out [0] = loss.
    let target = 0.5_f32;
    let outs = sess.run(&[("target", &[target]), ("d_output", &[1.0_f32])]);
    let loss = outs[0][0];

    // |T| = analytic; loss = (|T| - target)²
    let t_expect = analytic_t_mag(alpha, wg.length);
    let loss_expect = (t_expect - target).powi(2);
    assert!(
        (loss - loss_expect).abs() < 1e-5,
        "loss = {loss}, expected {loss_expect}",
    );
}

#[test]
fn waveguide_t_mag_grad_matches_analytic_and_fd() {
    // 100 µm waveguide, α = 0.8 dB/cm, target |T| = 0.95.
    let wg = Waveguide { width: 500, length: 1_000_000, id: "wg_grad".into() };
    let (fwd, loss_id) = wg.build_loss_graph();
    let bwd = grad_with_loss(&fwd, &[loss_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    let alpha_init = 0.8_f32;
    let target = 0.95_f32;

    // ── AD value ─────────────────────────────────────────────────────
    sess.set_param(&wg.loss_param_name(), &[alpha_init]);
    let outs = sess.run(&[("target", &[target]), ("d_output", &[1.0_f32])]);
    let loss = outs[0][0];
    let d_alpha_ad = outs[1][0];

    // ── Analytic ────────────────────────────────────────────────────
    // d/dα of (T(α) - target)² = 2 (T - target) · dT/dα
    // T(α) = exp(-α · L_cm · ln(10)/20)
    // dT/dα = T · (-L_cm · ln(10)/20)
    let l_cm = (wg.length as f32) * DBU_TO_CM;
    let t = analytic_t_mag(alpha_init, wg.length);
    let dt_dalpha = t * (-l_cm * LN10_OVER_20);
    let d_alpha_analytic = 2.0 * (t - target) * dt_dalpha;

    let rel = (d_alpha_ad - d_alpha_analytic).abs() / d_alpha_analytic.abs().max(1e-9);
    assert!(
        rel < 1e-3,
        "∂loss/∂α: AD = {d_alpha_ad}, analytic = {d_alpha_analytic}, rel = {rel:.3e}",
    );

    // Finite-difference cross-check via eda_validate. Re-runs the loss
    // forward with α perturbed; never touches the gradient graph.
    let mut probe_loss = |params: &[f32]| -> f32 {
        sess.set_param(&wg.loss_param_name(), &[params[0]]);
        let o = sess.run(&[("target", &[target]), ("d_output", &[1.0_f32])]);
        o[0][0]
    };
    gradcheck_scalar(&mut probe_loss, &[alpha_init], &[d_alpha_ad], 1e-3, 1e-3, 1e-7).unwrap();

    // Sanity tag: the loss matches the analytic forward too.
    let loss_expect = (t - target).powi(2);
    assert!((loss - loss_expect).abs() < 1e-5);
}

#[test]
fn adam_on_loss_converges_to_target_t_mag() {
    // Optimize α such that |T| hits 0.7 on a 100 µm waveguide.
    // Analytic optimum: |T| = exp(-α·L·ln(10)/20) = 0.7
    //                  → α* = -20·ln(0.7) / (ln(10) · L_cm)
    let wg = Waveguide { width: 500, length: 1_000_000, id: "wg_opt".into() };
    let l_cm = (wg.length as f32) * DBU_TO_CM;
    let target = 0.7_f32;
    let alpha_star = -20.0 * target.ln() / (10f32.ln() * l_cm);

    let (fwd, loss_id) = wg.build_loss_graph();
    let bwd = grad_with_loss(&fwd, &[loss_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    // Adam — vanilla SGD with constant lr struggles here because the
    // gradient w.r.t. α is small in absolute units (it carries the
    // L·ln(10)/20 ≈ 0.012 prefactor). Adam's per-param adaptive scaling
    // sidesteps the issue and converges in well under 1000 steps.
    let (mut alpha, lr) = (0.1_f32, 0.05_f32);
    let (b1, b2, eps) = (0.9_f32, 0.999_f32, 1e-8_f32);
    let (mut m, mut v) = (0.0_f32, 0.0_f32);
    let mut last_loss = f32::INFINITY;
    for t in 1..=2000 {
        sess.set_param(&wg.loss_param_name(), &[alpha]);
        let o = sess.run(&[("target", &[target]), ("d_output", &[1.0_f32])]);
        let loss = o[0][0];
        let g = o[1][0];
        m = b1 * m + (1.0 - b1) * g;
        v = b2 * v + (1.0 - b2) * g * g;
        let m_hat = m / (1.0 - b1.powi(t));
        let v_hat = v / (1.0 - b2.powi(t));
        alpha -= lr * m_hat / (v_hat.sqrt() + eps);
        last_loss = loss;
    }

    // Final loss should be tiny; α should be near α*.
    assert!(last_loss < 1e-6, "loss didn't converge: {last_loss}");
    let rel = (alpha - alpha_star).abs() / alpha_star.abs();
    assert!(
        rel < 5e-3,
        "α did not reach α*: got {alpha}, want {alpha_star} (rel {rel:.3e})",
    );
}
