//! Tier 2: AD `∂Id/∂{Vth, kp, λ}` against centered FD on the rlx
//! forward AND against the analytic gradient of the strict L1 formula
//! (in saturation; the FD path also exercises triode, where the smooth
//! gradient differs from strict but FD-on-rlx still catches AD bugs).

use spike_mosfet_dc::*;

const VTH: f64 = 0.5;
const KP: f64 = 100e-6;
const LAM: f64 = 0.02;

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] AD={a:+.6e} ref={b:+.6e} |Δ|={:.3e} env={env:.3e}",
            (a - b).abs());
    }
}

#[test]
fn ad_grads_match_analytic_in_saturation() {
    // Deep saturation: Vov=1.5, Vds=2.0. Smooth ≈ strict to a few
    // hundred ppm here (smooth_min residual at this point ~4e-4 rel).
    let vgs = 2.0;
    let vds = 2.0;
    let (_id, ad_dvth, ad_dkp, ad_dlam) = run_id_grad(vgs, vds, VTH, KP, LAM);

    let an_dvth = analytic_did_dvth_saturation(vgs, vds, VTH, KP, LAM);
    let an_dkp  = analytic_did_dkp_saturation(vgs, vds, VTH, KP, LAM);
    let an_dlam = analytic_did_dlam_saturation(vgs, vds, VTH, KP, LAM);

    assert_close(ad_dvth, an_dvth, 1e-3, 1e-12, "∂Id/∂Vth AD vs analytic");
    assert_close(ad_dkp,  an_dkp,  1e-3, 1e-12, "∂Id/∂kp AD vs analytic");
    assert_close(ad_dlam, an_dlam, 1e-3, 1e-12, "∂Id/∂λ  AD vs analytic");
}

#[track_caller]
fn check_ad_vs_fd_at(vgs: f64, vds: f64, label: &str) {
    let (_id, ad_dvth, ad_dkp, ad_dlam) = run_id_grad(vgs, vds, VTH, KP, LAM);

    let eps_v = 1e-5;
    let eps_kp = KP * 1e-5;
    let eps_l = 1e-7;

    let fd_dvth = (run_id(vgs, vds, VTH + eps_v, KP, LAM)
                 - run_id(vgs, vds, VTH - eps_v, KP, LAM)) / (2.0 * eps_v);
    let fd_dkp = (run_id(vgs, vds, VTH, KP + eps_kp, LAM)
                - run_id(vgs, vds, VTH, KP - eps_kp, LAM)) / (2.0 * eps_kp);
    let fd_dlam = (run_id(vgs, vds, VTH, KP, LAM + eps_l)
                 - run_id(vgs, vds, VTH, KP, LAM - eps_l)) / (2.0 * eps_l);

    assert_close(ad_dvth, fd_dvth, 1e-4, 1e-10, &format!("∂Id/∂Vth AD vs FD ({label})"));
    assert_close(ad_dkp,  fd_dkp,  1e-4, 1e-10, &format!("∂Id/∂kp AD vs FD ({label})"));
    assert_close(ad_dlam, fd_dlam, 1e-4, 1e-10, &format!("∂Id/∂λ  AD vs FD ({label})"));
}

#[test]
fn ad_grads_match_finite_difference_in_saturation() {
    check_ad_vs_fd_at(2.0, 2.0, "saturation Vov=1.5, Vds=2.0");
}

#[test]
fn ad_grads_match_finite_difference_in_triode() {
    // FD on the rlx forward in triode — exercises both the softplus
    // chain (Vov_s) and the smooth_min path (Vds_eff = Vds since
    // Vds < Vov). Bug-catcher independent of the analytic formula.
    check_ad_vs_fd_at(2.0, 0.5, "triode Vov=1.5, Vds=0.5");
}

#[test]
fn ad_grads_match_finite_difference_at_vov_boundary() {
    // Vds = Vov = 1.5: smack on the saturation/triode knee where
    // smooth_min is at its softest. AD and FD should still agree.
    check_ad_vs_fd_at(2.0, 1.5, "knee Vov=Vds=1.5");
}
