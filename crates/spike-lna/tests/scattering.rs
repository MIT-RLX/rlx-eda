//! Validation pyramid for the LNA's S-parameter model.
//!
//! Tier 1 — analytic forward: closed-form `S₁₁(ω)` from Razavi §5.3.3
//! evaluated against the rlx graph at multiple frequencies and at the
//! match condition.
//!
//! Tier 2 — finite-difference: `∂|S₁₁|²/∂Lg` reverse-mode AD vs central
//! differences on the same compiled session.
//!
//! Tier 3 — inverse design: Adam on `Lg` drives `|S₁₁(2.4 GHz)|²` to
//! the Razavi closed-form optimum `Lg* = 1/(ω₀²·Cgs) − Ls`.

use eda_validate::gradcheck_scalar;
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_ir::{Op, NodeId};
use spike_lna::Lna;

const F0_HZ: f32 = 2.4e9;
const TAU: f32 = std::f32::consts::TAU;

/// Bit-for-bit closed-form S₁₁ — what the rlx graph should match.
fn analytic_s11(gm: f32, cgs: f32, lg: f32, ls: f32, f_hz: f32, z0: f32) -> (f32, f32) {
    let omega = TAU * f_hz;
    let r_in = gm * ls / cgs;
    let x_in = omega * (lg + ls) - 1.0 / (omega * cgs);
    let a = r_in - z0;
    let b = x_in;
    let c = r_in + z0;
    let d = x_in;
    let denom = c * c + d * d;
    ((a * c + b * d) / denom, (b * c - a * d) / denom)
}

fn set_default_params(sess: &mut CompiledGraph, lna: &Lna, lg_h: f32) {
    sess.set_param(&lna.gm_param_name(),  &[50.0e-3]);  // 50 mS
    sess.set_param(&lna.cgs_param_name(), &[250.0e-15]); // 250 fF
    sess.set_param(&lna.lg_param_name(),  &[lg_h]);
    sess.set_param(&lna.ls_param_name(),  &[250.0e-12]); // 250 pH (matches Razavi §5.3.3)
    sess.set_param(&lna.ld_param_name(),  &[10.0e-9]);   // 10 nH
    sess.set_param(&lna.rl_param_name(),  &[500.0]);     // 500 Ω
}

#[test]
fn s11_forward_matches_analytic_across_band() {
    let lna = Lna::lna_24ghz("test_s11");
    let g = lna.build_forward_graph();
    let mut sess = Session::new(Device::Cpu).compile(g);

    // Use the canonical Razavi sizing: gm = 50 mS, Cgs = 250 fF,
    // Ls = 250 pH (so gm·Ls/Cgs = 50 Ω → Re Z_in = Z₀); Lg ≈ 17.34 nH
    // so the imaginary part nulls at 2.4 GHz.
    let cgs = 250e-15;
    let ls = 250e-12;
    let omega0 = TAU * F0_HZ;
    let lg = 1.0 / (omega0 * omega0 * cgs) - ls;
    set_default_params(&mut sess, &lna, lg);

    for &f in &[1.0e9, 1.8e9, 2.4e9, 3.0e9, 5.0e9] {
        let outs = sess.run(&[("freq_hz", &[f])]);
        let (re, im) = (outs[0][0], outs[1][0]);
        let (re_a, im_a) = analytic_s11(50e-3, cgs, lg, ls, f, 50.0);
        assert!(
            (re - re_a).abs() < 1e-4 && (im - im_a).abs() < 1e-4,
            "f = {f:.2e}: graph ({re:.6}, {im:.6}) vs analytic ({re_a:.6}, {im_a:.6})",
        );
    }
}

#[test]
fn s11_at_match_condition_is_zero() {
    // Razavi §5.3.3: when gm·Ls/Cgs = Z₀ and ω₀²(Lg+Ls)Cgs = 1,
    // S₁₁ = 0 (perfect match). The graph should reproduce this.
    let lna = Lna::lna_24ghz("test_match");
    let g = lna.build_forward_graph();
    let mut sess = Session::new(Device::Cpu).compile(g);

    let cgs = 250e-15;
    let ls = 250e-12;
    let omega0 = TAU * F0_HZ;
    let lg_star = 1.0 / (omega0 * omega0 * cgs) - ls;
    set_default_params(&mut sess, &lna, lg_star);

    let outs = sess.run(&[("freq_hz", &[F0_HZ])]);
    let mag2 = outs[0][0].powi(2) + outs[1][0].powi(2);
    assert!(
        mag2 < 1e-8,
        "|S₁₁|² at canonical match: {mag2:.3e} (should be ~0); re={}, im={}",
        outs[0][0], outs[1][0],
    );
}

#[test]
fn s21_matched_gain_matches_analytic() {
    let lna = Lna::lna_24ghz("test_s21");
    let g = lna.build_forward_graph();
    let mut sess = Session::new(Device::Cpu).compile(g);

    let cgs = 250e-15;
    let ls = 250e-12;
    let omega0 = TAU * F0_HZ;
    let lg_star = 1.0 / (omega0 * omega0 * cgs) - ls;
    set_default_params(&mut sess, &lna, lg_star);

    let outs = sess.run(&[("freq_hz", &[F0_HZ])]);
    let s21_mag = outs[2][0];
    // Razavi 5.79: |S₂₁| = gm·R_L / (2·ω·Cgs·Z₀)
    let expected = 50e-3 * 500.0 / (2.0 * omega0 * cgs * 50.0);
    let rel = (s21_mag - expected).abs() / expected;
    assert!(rel < 1e-5, "|S₂₁|: got {s21_mag}, expected {expected} (rel {rel:.3e})");
    // sanity: gain at 2.4 GHz should be > unity for these numbers.
    assert!(s21_mag > 1.0, "|S₂₁| = {s21_mag} should be > 1 for matched LNA");
}

fn find_param(g: &rlx_ir::Graph, name: &str) -> NodeId {
    g.nodes()
        .iter()
        .enumerate()
        .find_map(|(i, n)| match &n.op {
            Op::Param { name: pn, .. } if pn == name => Some(NodeId(i as u32)),
            _ => None,
        })
        .expect("param missing")
}

#[test]
fn match_loss_grad_wrt_lg_matches_finite_difference() {
    let lna = Lna::lna_24ghz("test_grad");
    let fwd = lna.build_match_loss_graph();
    let lg_id = find_param(&fwd, &lna.lg_param_name());
    let bwd = grad_with_loss(&fwd, &[lg_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    // Start a bit off the optimum so the gradient is nontrivial.
    let cgs = 250e-15;
    let ls = 250e-12;
    let omega0 = TAU * F0_HZ;
    let lg_star = 1.0 / (omega0 * omega0 * cgs) - ls;
    let lg_init = lg_star * 1.5;
    set_default_params(&mut sess, &lna, lg_init);

    let outs = sess.run(&[("freq_hz", &[F0_HZ]), ("d_output", &[1.0])]);
    let _loss = outs[0][0];
    let d_lg_ad = outs[1][0];

    let mut probe = |params: &[f32]| -> f32 {
        sess.set_param(&lna.lg_param_name(), &[params[0]]);
        let o = sess.run(&[("freq_hz", &[F0_HZ]), ("d_output", &[1.0])]);
        o[0][0]
    };
    // Lg is in henries (~2.6e-8). FD step `eps` must be absolute
    // and scale with the param magnitude — a 5 %-of-Lg perturbation
    // (~1.3 nH) leaves enough significance in the squared-magnitude
    // subtraction to recover a clean central difference, while
    // staying small enough that the O(h²·L''') truncation error
    // matches `rtol`.
    gradcheck_scalar(&mut probe, &[lg_init], &[d_lg_ad], lg_init * 0.05, 5e-2, 5e-2).unwrap();
}

#[test]
fn adam_on_lg_converges_to_match_optimum() {
    let lna = Lna::lna_24ghz("test_adam");
    let fwd = lna.build_match_loss_graph();
    let lg_id = find_param(&fwd, &lna.lg_param_name());
    let bwd = grad_with_loss(&fwd, &[lg_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    let cgs = 250e-15;
    let ls = 250e-12;
    let omega0 = TAU * F0_HZ;
    let lg_star = 1.0 / (omega0 * omega0 * cgs) - ls;

    // Set non-Lg params to canonical values (only Lg is the knob).
    sess.set_param(&lna.gm_param_name(),  &[50e-3]);
    sess.set_param(&lna.cgs_param_name(), &[cgs]);
    sess.set_param(&lna.ls_param_name(),  &[ls]);
    sess.set_param(&lna.ld_param_name(),  &[10e-9]);
    sess.set_param(&lna.rl_param_name(),  &[500.0]);

    // Adam — Lg is at the nano-henries scale, so use lr matched to it.
    let mut lg = lg_star * 0.4;       // Start far below optimum
    let lr = lg_star * 1e-2;
    let (b1, b2, eps) = (0.9_f32, 0.999_f32, 1e-12_f32);
    let (mut m, mut v) = (0.0_f32, 0.0_f32);
    let mut last_loss = f32::INFINITY;
    for t in 1..=4000 {
        sess.set_param(&lna.lg_param_name(), &[lg]);
        let o = sess.run(&[("freq_hz", &[F0_HZ]), ("d_output", &[1.0_f32])]);
        let loss = o[0][0];
        let grad = o[1][0];
        m = b1 * m + (1.0 - b1) * grad;
        v = b2 * v + (1.0 - b2) * grad * grad;
        let m_hat = m / (1.0 - b1.powi(t));
        let v_hat = v / (1.0 - b2.powi(t));
        lg -= lr * m_hat / (v_hat.sqrt() + eps);
        last_loss = loss;
    }

    assert!(last_loss < 1e-6, "|S₁₁|² did not converge: {last_loss:.3e}");
    let rel = (lg - lg_star).abs() / lg_star;
    assert!(
        rel < 1e-2,
        "Lg did not reach optimum: got {lg:.3e}, want {lg_star:.3e} (rel {rel:.3e})",
    );
}
