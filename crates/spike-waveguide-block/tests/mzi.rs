//! Mach-Zehnder forward / autodiff / inverse-design.
//!
//! Three rungs of the photonic validation pyramid (no ngspice equivalent
//! for optics, so analytic + FD + AD-converges stand in for the SPICE
//! witness rung):
//!
//! 1. Forward `|T_through|²` matches the closed-form `cos²(Δφ/2)` at
//!    multiple wavelengths, and `|T_through|² + |T_cross|²` conserves
//!    energy in the lossless balanced case.
//! 2. AD gradient `∂|T_through|² / ∂n_eff_A` at fixed λ matches finite
//!    differences (small FD step needed — sin²(Δφ/2) is highly
//!    oscillatory in `n_eff` for L/λ ≫ 1).
//! 3. Adam on `n_eff_A` lands a transmission notch (`|T_through|² ≈ 0`)
//!    at λ = 1550 nm — the photonic analog of the existing CMOS
//!    inverse-design tests.

use eda_validate::gradcheck_scalar;
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{CompiledGraph, Device, Session};
use spike_waveguide_block::{Mzi, Waveguide};

const TAU: f32 = std::f32::consts::TAU;

/// Lossless-balanced through intensity: `cos²(Δφ/2)` with
/// `Δφ = 2π·(n_A·L_A − n_B·L_B)/λ`.
fn analytic_through_lossless(
    neff_a: f32,
    neff_b: f32,
    length_a: i64,
    length_b: i64,
    wavelength_nm: f32,
) -> f32 {
    let delta = TAU * (neff_a * length_a as f32 - neff_b * length_b as f32) / wavelength_nm;
    (delta * 0.5).cos().powi(2)
}

fn find_param(g: &Graph, name: &str) -> NodeId {
    g.nodes()
        .iter()
        .enumerate()
        .find_map(|(i, n)| match &n.op {
            Op::Param { name: pn, .. } if pn == name => Some(NodeId(i as u32)),
            _ => None,
        })
        .expect("param missing")
}

fn set_lossless(sess: &mut CompiledGraph, mzi: &Mzi, neff_a: f32, neff_b: f32) {
    sess.set_param(&mzi.arm_a.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_b.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_a.neff_param_name(), &[neff_a]);
    sess.set_param(&mzi.arm_b.neff_param_name(), &[neff_b]);
}

#[test]
fn through_intensity_matches_cos_squared_lossless() {
    // 100 µm vs 110 µm arms — visible FSR over the C-band.
    let mzi = Mzi::new(500, 100_000, 110_000, "fsr");
    let g = mzi.build_intensity_graph();
    let mut sess = Session::new(Device::Cpu).compile(g);

    let (neff_a, neff_b) = (2.4_f32, 2.4_f32);
    set_lossless(&mut sess, &mzi, neff_a, neff_b);

    for &wl in &[1500.0_f32, 1525.0, 1550.0, 1575.0, 1600.0] {
        let outs = sess.run(&[("wavelength_nm", &[wl])]);
        let through = outs[0][0];
        let cross = outs[1][0];
        let expected =
            analytic_through_lossless(neff_a, neff_b, mzi.arm_a.length, mzi.arm_b.length, wl);
        assert!(
            (through - expected).abs() < 1e-4,
            "λ={wl}: through={through}, expected={expected}",
        );
        assert!(
            (through + cross - 1.0).abs() < 1e-4,
            "λ={wl}: T+C={} (energy conservation)",
            through + cross,
        );
    }
}

#[test]
fn balanced_arms_are_fully_through_at_all_wavelengths() {
    // L_A = L_B, n_A = n_B → Δφ = 0 → all light to the through port.
    let mzi = Mzi::new(500, 50_000, 50_000, "balanced");
    let g = mzi.build_intensity_graph();
    let mut sess = Session::new(Device::Cpu).compile(g);
    set_lossless(&mut sess, &mzi, 2.4, 2.4);
    for &wl in &[1310_f32, 1550.0, 1600.0] {
        let outs = sess.run(&[("wavelength_nm", &[wl])]);
        assert!(
            (outs[0][0] - 1.0).abs() < 1e-5,
            "balanced through ≠ 1 at λ={wl}: {}",
            outs[0][0]
        );
        assert!(outs[1][0].abs() < 1e-5, "balanced cross ≠ 0 at λ={wl}: {}", outs[1][0]);
    }
}

#[test]
fn notch_loss_grad_wrt_neff_a_matches_fd() {
    // Use short arms (5 µm vs 5.5 µm) so cos²(Δφ/2) varies slowly enough
    // in n_eff that f32 FD with eps=1e-5 captures the slope cleanly.
    let mzi = Mzi::new(500, 5_000, 5_500, "grad");
    let fwd = mzi.build_notch_loss_graph();
    let neff_a_id = find_param(&fwd, &mzi.arm_a.neff_param_name());
    let bwd = grad_with_loss(&fwd, &[neff_a_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    sess.set_param(&mzi.arm_a.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_b.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_b.neff_param_name(), &[2.4]);
    let neff_a = 2.35_f32;
    sess.set_param(&mzi.arm_a.neff_param_name(), &[neff_a]);

    let lambda = 1550_f32;
    let outs = sess.run(&[("wavelength_nm", &[lambda]), ("d_output", &[1.0])]);
    let g_ad = outs[1][0];

    let mut probe = |params: &[f32]| -> f32 {
        sess.set_param(&mzi.arm_a.neff_param_name(), &[params[0]]);
        let o = sess.run(&[("wavelength_nm", &[lambda]), ("d_output", &[1.0])]);
        o[0][0]
    };
    // Args: (probe, params, ad_grad, eps, rtol, atol).
    gradcheck_scalar(&mut probe, &[neff_a], &[g_ad], 1e-5, 1e-2, 1e-3).unwrap();
}

#[test]
fn adam_places_notch_at_target_wavelength() {
    // Fix arm B; tune arm A's n_eff so that |T_through(λ=1550)|² → 0.
    let mzi = Mzi::new(500, 100_000, 110_000, "notch");
    let fwd = mzi.build_notch_loss_graph();
    let neff_a_id = find_param(&fwd, &mzi.arm_a.neff_param_name());
    let bwd = grad_with_loss(&fwd, &[neff_a_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    sess.set_param(&mzi.arm_a.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_b.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_b.neff_param_name(), &[2.4]);

    let lambda = 1550_f32;
    let (mut neff_a, lr) = (2.35_f32, 1e-3_f32);
    let (b1, b2, eps) = (0.9_f32, 0.999_f32, 1e-8_f32);
    let (mut m, mut v) = (0.0_f32, 0.0_f32);
    let mut last = f32::INFINITY;
    for t in 1..=4000 {
        sess.set_param(&mzi.arm_a.neff_param_name(), &[neff_a]);
        let o = sess.run(&[("wavelength_nm", &[lambda]), ("d_output", &[1.0])]);
        let loss = o[0][0];
        let g = o[1][0];
        m = b1 * m + (1.0 - b1) * g;
        v = b2 * v + (1.0 - b2) * g * g;
        let m_hat = m / (1.0 - b1.powi(t));
        let v_hat = v / (1.0 - b2.powi(t));
        neff_a -= lr * m_hat / (v_hat.sqrt() + eps);
        last = loss;
    }
    // 1e-3 ≈ -30 dB extinction on unit-power input — plenty for the demo.
    // The function is sin²(Δφ/2)-shaped near the notch, so further
    // tightening would just need more Adam iters or a smaller LR.
    assert!(last < 1e-3, "notch not reached: |T_through|² = {last}");

    // Sanity: solution satisfies cos²(Δφ/2) ≈ 0 — i.e. Δφ ≈ (2k+1)π.
    let delta =
        TAU * (neff_a * mzi.arm_a.length as f32 - 2.4 * mzi.arm_b.length as f32) / lambda;
    let residual = (delta * 0.5).cos().powi(2);
    assert!(residual < 1e-3, "Δφ/2 not at odd multiple of π/2: residual {residual}");

    // Witness on a freshly-compiled forward graph: through ≈ 0, cross ≈ 1.
    let _wg_a = Waveguide { width: 500, length: 100_000, id: "notch_armA".into() }; // names match below
    let mzi2 = Mzi::new(500, 100_000, 110_000, "notch_check");
    let fwd2 = mzi2.build_intensity_graph();
    let mut sess2 = Session::new(Device::Cpu).compile(fwd2);
    sess2.set_param(&mzi2.arm_a.loss_param_name(), &[0.0]);
    sess2.set_param(&mzi2.arm_b.loss_param_name(), &[0.0]);
    sess2.set_param(&mzi2.arm_a.neff_param_name(), &[neff_a]);
    sess2.set_param(&mzi2.arm_b.neff_param_name(), &[2.4]);
    let outs = sess2.run(&[("wavelength_nm", &[lambda])]);
    assert!(outs[0][0] < 1e-3, "through ≠ 0 at notch: {}", outs[0][0]);
    assert!((outs[1][0] - 1.0).abs() < 1e-3, "cross ≠ 1 at notch: {}", outs[1][0]);
}

#[test]
fn lossy_arms_dim_both_outputs_consistently() {
    // Identical loss on both arms, balanced lengths → all light still
    // through, but attenuated by T².
    let mzi = Mzi::new(500, 75_000, 75_000, "lossy");
    let g = mzi.build_intensity_graph();
    let mut sess = Session::new(Device::Cpu).compile(g);
    let alpha = 30.0_f32;
    sess.set_param(&mzi.arm_a.loss_param_name(), &[alpha]);
    sess.set_param(&mzi.arm_b.loss_param_name(), &[alpha]);
    sess.set_param(&mzi.arm_a.neff_param_name(), &[2.4]);
    sess.set_param(&mzi.arm_b.neff_param_name(), &[2.4]);

    let outs = sess.run(&[("wavelength_nm", &[1550_f32])]);
    let through = outs[0][0];
    let cross = outs[1][0];
    // L = 75 µm = 7.5e-3 cm; total dB = 30 · 7.5e-3 = 0.225 dB.
    // T = 10^(-0.225/20), expected through = T².
    let length_cm = (mzi.arm_a.length as f32) * 1.0e-7;
    let t = (-alpha * length_cm * 2.302_585_093 / 20.0).exp();
    let expected_through = t * t;
    assert!(
        (through - expected_through).abs() < 1e-4,
        "lossy through: got {through}, expected {expected_through}"
    );
    assert!(cross.abs() < 1e-5, "balanced cross should be 0, got {cross}");
}
