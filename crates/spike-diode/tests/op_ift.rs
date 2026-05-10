//! IFT (custom_vjp / custom_jvp) variant of the diode-DC AD path.
//! Validates that the closed-form IFT gradient — emitted via
//! Op::CustomFn override AD bodies — matches:
//!   1. the existing differentiate-through-the-loop AD,
//!   2. the analytic IFT formula evaluated on the converged Vmid,
//!   3. centered finite differences on the rlx forward.
//! Forward parity vs the existing `run_forward` is also checked.

use eda_validate::assert_close;
use spike_diode::*;

const N: usize = 30;

#[test]
fn ift_forward_matches_through_the_loop() {
    let cases = &[
        (1.0_f32, 1_000.0_f32, 1e-15_f32),
        (3.3,     2_200.0,     1e-12),
        (5.0,     10_000.0,    1e-15),
        (0.8,     1_000.0,     1e-15),
    ];
    for &(v, r, is_) in cases {
        let through_loop = run_forward(v, r, is_, VT, N);
        let ift          = run_forward_ift(v, r, is_, VT, N);
        // Same Newton math; outputs should be bitwise close (the only
        // difference is which compile path lowered the body).
        assert_close(ift, through_loop, 1e-6, 1e-9,
            &format!("forward parity @ V={v}, R={r}, Is={is_:.0e}"));
    }
}

#[test]
fn ift_grad_matches_through_the_loop() {
    // Compare reverse-mode AD via custom_vjp (IFT formula) vs reverse-mode
    // AD via differentiating-through-the-loop. Both should converge to the
    // same answer at well-converged Newton; difference is only the
    // residual-from-Newton effect on the linearization point.
    let cases = &[
        (1.0_f32, 1_000.0_f32, 1e-15_f32),
        (3.3,     2_200.0,     1e-12),
        (5.0,     10_000.0,    1e-15),
    ];
    for &(v, r, is_) in cases {
        let (_, dr_loop, dis_loop) = run_forward_and_grad    (v, r, is_, VT, N);
        let (_, dr_ift,  dis_ift ) = run_forward_and_grad_ift(v, r, is_, VT, N);
        assert_close(dr_ift,  dr_loop,  1e-3, 1e-9,
            &format!("∂Vmid/∂R: IFT vs through-the-loop @ V={v}"));
        assert_close(dis_ift, dis_loop, 1e-3, 1e-9,
            &format!("∂Vmid/∂Is: IFT vs through-the-loop @ V={v}"));
    }
}

#[test]
fn ift_grad_matches_analytic() {
    let v   = 1.0_f32;
    let r   = 1_000.0_f32;
    let is_ = 1e-15_f32;

    let (vmid, dr, dis) = run_forward_and_grad_ift(v, r, is_, VT, N);
    let dr_an  = analytic_dvmid_dr (v, r, is_, VT, vmid);
    let dis_an = analytic_dvmid_dis(v, r, is_, VT, vmid);
    // The IFT body emits exactly this formula, so they should agree to
    // f32 ulps (modulo the order-of-operations dance through rlx ops).
    assert_close(dr,  dr_an,  1e-4, 1e-9, "∂Vmid/∂R: IFT-AD vs analytic");
    assert_close(dis, dis_an, 1e-4, 1e-9, "∂Vmid/∂Is: IFT-AD vs analytic");
}

#[test]
fn ift_grad_matches_finite_differences() {
    let v   = 1.0_f32;
    let r   = 1_000.0_f32;
    let is_ = 1e-15_f32;

    let (_, dr_ad, _) = run_forward_and_grad_ift(v, r, is_, VT, N);
    let h_r = 1e-3 * r;
    let dr_fd = (run_forward_ift(v, r + h_r, is_, VT, N)
              -  run_forward_ift(v, r - h_r, is_, VT, N)) / (2.0 * h_r);
    assert_close(dr_ad, dr_fd, 1e-2, 1e-9, "∂Vmid/∂R: IFT-AD vs FD");
}

#[test]
fn ift_jvp_matches_reverse_mode_dot_product() {
    // For scalar output, JVP and VJP carry identical information when
    // the upstream is 1.0: t_y = ∂y/∂x · t_x = (∂y/∂x) · t_x. With four
    // perturbed primals and a one-hot tangent on each, the resulting
    // tangents must equal the corresponding reverse-mode gradients.
    let v   = 1.0_f32;
    let r   = 1_000.0_f32;
    let is_ = 1e-15_f32;

    let (_, dr_rev, dis_rev) = run_forward_and_grad_ift(v, r, is_, VT, N);

    // Tangent (0, 0, 1, 0) ⇒ t_Vmid should equal ∂Vmid/∂R.
    let (_, t_dr) = run_jvp_ift(v, r, is_, VT,  0.0, 0.0, 1.0, 0.0, N);
    // Tangent (0, 0, 0, 1) ⇒ t_Vmid should equal ∂Vmid/∂Is.
    let (_, t_dis) = run_jvp_ift(v, r, is_, VT,  0.0, 0.0, 0.0, 1.0, N);

    assert_close(t_dr,  dr_rev,  1e-3, 1e-9, "JVP one-hot R vs reverse ∂R");
    assert_close(t_dis, dis_rev, 1e-3, 1e-9, "JVP one-hot Is vs reverse ∂Is");
}
