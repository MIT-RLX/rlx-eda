//! Full complex `S₂₁` tier — same `Waveguide` instance produces both a
//! wavelength-dependent `(Re, Im)` S-parameter pair (via `OpticalScattering::s21`)
//! and an autodiff-able phase-loss graph (`build_phase_loss_graph`).
//!
//! Scope:
//!
//! 1. Forward `S₂₁` matches the analytic model
//!    `T_mag · (cos φ − i·sin φ)` with `φ = 2π·n_eff·L / λ`.
//! 2. AD gradient `∂(∠S₂₁ − φ*)² / ∂n_eff` matches both analytic and
//!    finite differences.
//! 3. Adam-on-phase-loss converges `n_eff` to a target phase at fixed λ.
//!
//! These exercise the new `Activation::Sin` / `Activation::Cos` ops in
//! rlx-ir end-to-end through the rlx CPU executor and reverse-mode
//! autodiff.

use eda_validate::gradcheck_scalar;
use rlx_ir::{op::BinaryOp, DType, Graph, Shape as TensorShape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_waveguide_block::{OpticalScattering, Waveguide};

const NEPER_PER_DB: f32 = 2.302_585_093 / 20.0;
const DBU_TO_CM: f32 = 1.0e-7;
const TAU: f32 = std::f32::consts::TAU;

fn analytic_s21(alpha_db_per_cm: f32, neff: f32, length_dbu: i64, wavelength_nm: f32) -> (f32, f32) {
    let length_cm = (length_dbu as f32) * DBU_TO_CM;
    let t_mag = (-alpha_db_per_cm * length_cm * NEPER_PER_DB).exp();
    let length_nm = length_dbu as f32; // 1 DBU = 1 nm at the standard PDK DBU.
    let phi = TAU * neff * length_nm / wavelength_nm;
    (t_mag * phi.cos(), -t_mag * phi.sin())
}

#[test]
fn s21_forward_matches_analytic_at_multiple_wavelengths() {
    // 100 µm waveguide; sweep λ across the C-band.
    let wg = Waveguide { width: 500, length: 100_000, id: "wg_s21".into() };

    // Build a graph that returns `[Re S21, Im S21]`.
    let mut g = Graph::new("s21_forward");
    let s = TensorShape::new(&[1], DType::F32);
    let lambda = g.input("wavelength_nm", s.clone());
    let (re, im) = wg.s21(lambda, &mut g);
    g.set_outputs(vec![re, im]);

    let mut sess = Session::new(Device::Cpu).compile(g);
    let alpha = 0.5_f32;
    let neff = 2.4_f32;
    sess.set_param(&wg.loss_param_name(), &[alpha]);
    sess.set_param(&wg.neff_param_name(), &[neff]);

    for &wl in &[1300_f32, 1310.0, 1500.0, 1550.0, 1600.0] {
        let outs = sess.run(&[("wavelength_nm", &[wl])]);
        let (re_got, im_got) = (outs[0][0], outs[1][0]);
        let (re_exp, im_exp) = analytic_s21(alpha, neff, wg.length, wl);

        // Magnitude tolerance: ~1e-5 absolute. Phase wraps every 2π so
        // we compare `Re` and `Im` directly rather than the angle.
        assert!(
            (re_got - re_exp).abs() < 1e-4 && (im_got - im_exp).abs() < 1e-4,
            "λ={wl}: got ({re_got:.6}, {im_got:.6}), expected ({re_exp:.6}, {im_exp:.6})",
        );
    }
}

#[test]
fn s21_magnitude_squared_recovers_t_mag_squared() {
    // Sanity: |S₂₁|² = T_mag² regardless of λ or n_eff.
    let wg = Waveguide { width: 500, length: 50_000, id: "wg_mag".into() };

    let mut g = Graph::new("s21_mag_sq");
    let s = TensorShape::new(&[1], DType::F32);
    let lambda = g.input("wavelength_nm", s.clone());
    let (re, im) = wg.s21(lambda, &mut g);
    let re2 = g.binary(BinaryOp::Mul, re, re, s.clone());
    let im2 = g.binary(BinaryOp::Mul, im, im, s.clone());
    let mag2 = g.binary(BinaryOp::Add, re2, im2, s);
    g.set_outputs(vec![mag2]);

    let mut sess = Session::new(Device::Cpu).compile(g);
    let alpha = 1.2_f32;
    let neff = 2.4_f32;
    sess.set_param(&wg.loss_param_name(), &[alpha]);
    sess.set_param(&wg.neff_param_name(), &[neff]);

    let length_cm = (wg.length as f32) * DBU_TO_CM;
    let t_mag = (-alpha * length_cm * NEPER_PER_DB).exp();
    let expected = t_mag * t_mag;

    // |S₂₁|² is independent of λ; check at a few.
    for &wl in &[1310_f32, 1550.0] {
        let outs = sess.run(&[("wavelength_nm", &[wl])]);
        let got = outs[0][0];
        assert!(
            (got - expected).abs() < 1e-5,
            "|S₂₁|² at λ={wl}: got {got}, expected {expected}",
        );
    }
}

#[test]
fn phase_loss_grad_wrt_neff_matches_analytic_and_fd() {
    // Phase loss = (∠S₂₁ − target)² = (-2π·neff·L/λ − target)².
    // d/d(neff) = 2·(-2π·neff·L/λ − target) · (-2π·L/λ).
    let wg = Waveguide { width: 500, length: 50_000, id: "wg_phase".into() };
    let (fwd, neff_id) = wg.build_phase_loss_graph();
    let bwd = grad_with_loss(&fwd, &[neff_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    let neff_init = 2.0_f32;
    let alpha = 0.5_f32;
    let lambda = 1550_f32;
    // Target phase: pick a realistic non-zero value.
    let target_phase = -1.0_f32; // radians
    sess.set_param(&wg.loss_param_name(), &[alpha]);
    sess.set_param(&wg.neff_param_name(), &[neff_init]);

    let outs = sess.run(&[
        ("wavelength_nm", &[lambda]),
        ("target_phase", &[target_phase]),
        ("d_output", &[1.0_f32]),
    ]);
    let _loss = outs[0][0];
    let d_neff_ad = outs[1][0];

    // Analytic.
    let length_nm = wg.length as f32;
    let phi = -TAU * neff_init * length_nm / lambda;
    let dphi_dneff = -TAU * length_nm / lambda;
    let d_neff_analytic = 2.0 * (phi - target_phase) * dphi_dneff;

    let rel = (d_neff_ad - d_neff_analytic).abs() / d_neff_analytic.abs().max(1e-9);
    assert!(
        rel < 1e-3,
        "∂L/∂neff: AD = {d_neff_ad}, analytic = {d_neff_analytic}, rel = {rel:.3e}",
    );

    // FD cross-check.
    let mut probe = |params: &[f32]| -> f32 {
        sess.set_param(&wg.neff_param_name(), &[params[0]]);
        let o = sess.run(&[
            ("wavelength_nm", &[lambda]),
            ("target_phase", &[target_phase]),
            ("d_output", &[1.0_f32]),
        ]);
        o[0][0]
    };
    gradcheck_scalar(&mut probe, &[neff_init], &[d_neff_ad], 1e-3, 1e-3, 1e-7).unwrap();
}

#[test]
fn adam_on_phase_loss_converges_to_target_phase() {
    // Optimize n_eff such that ∠S₂₁(λ=1550) hits a target phase.
    let wg = Waveguide { width: 500, length: 50_000, id: "wg_phase_opt".into() };
    let (fwd, neff_id) = wg.build_phase_loss_graph();
    let bwd = grad_with_loss(&fwd, &[neff_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    let lambda = 1550_f32;
    let length_nm = wg.length as f32;
    // Pick a target phase that's a wrap-aware analytic optimum:
    // we want -2π·neff*·L/λ = target → neff* = -target·λ/(2π·L).
    let target_phase = -1.5_f32;
    let neff_star = -target_phase * lambda / (TAU * length_nm);

    sess.set_param(&wg.loss_param_name(), &[0.5_f32]); // unused by the phase loss but compiled in

    // Adam.
    let (mut neff, lr) = (1.0_f32, 0.01_f32);
    let (b1, b2, eps) = (0.9_f32, 0.999_f32, 1e-8_f32);
    let (mut m, mut v) = (0.0_f32, 0.0_f32);
    let mut last_loss = f32::INFINITY;
    for t in 1..=2000 {
        sess.set_param(&wg.neff_param_name(), &[neff]);
        let o = sess.run(&[
            ("wavelength_nm", &[lambda]),
            ("target_phase", &[target_phase]),
            ("d_output", &[1.0_f32]),
        ]);
        let loss = o[0][0];
        let g = o[1][0];
        m = b1 * m + (1.0 - b1) * g;
        v = b2 * v + (1.0 - b2) * g * g;
        let m_hat = m / (1.0 - b1.powi(t));
        let v_hat = v / (1.0 - b2.powi(t));
        neff -= lr * m_hat / (v_hat.sqrt() + eps);
        last_loss = loss;
    }

    assert!(last_loss < 1e-6, "loss didn't converge: {last_loss}");
    let rel = (neff - neff_star).abs() / neff_star.abs();
    assert!(
        rel < 5e-3,
        "n_eff did not reach n_eff*: got {neff}, want {neff_star} (rel {rel:.3e})",
    );
}
