//! Tier 2: AD through the unrolled Newton graph matches the
//! implicit-function-theorem analytic gradient at the converged point,
//! and matches centered finite differences on the rlx forward.

use eda_validate::assert_close;
use spike_diode::*;

const N: usize = 30; // plenty of Newton iters so the IFT linearization is exact

#[test]
fn ad_matches_ift_analytic() {
    let v   = 1.0_f32;
    let r   = 1_000.0_f32;
    let is_ = 1e-15_f32;

    let (vmid, d_r_ad, d_is_ad) = run_forward_and_grad(v, r, is_, VT, N);
    let d_r_an  = analytic_dvmid_dr(v, r, is_, VT, vmid);
    let d_is_an = analytic_dvmid_dis(v, r, is_, VT, vmid);

    // ∂Vmid/∂R is on the order of 1e-5 here. AD should match IFT to f32
    // ulps modulo the residual Newton error after 30 iters (~1e-8 ulp).
    assert_close(d_r_ad,  d_r_an,  1e-3, 1e-9, "∂Vmid/∂R: AD vs IFT");
    assert_close(d_is_ad, d_is_an, 1e-3, 1e-9, "∂Vmid/∂Is: AD vs IFT");
}

#[test]
fn ad_matches_finite_differences() {
    let v   = 1.0_f32;
    let r   = 1_000.0_f32;
    let is_ = 1e-15_f32;

    let (_, d_r_ad, _d_is_ad) = run_forward_and_grad(v, r, is_, VT, N);

    // Relative perturbation. For Is at 1e-15 the FD numerator is at
    // f32's noise floor — skip ∂Is FD; the analytic comparison covers it.
    let h_r = 1e-3 * r;
    let d_r_fd = (run_forward(v, r + h_r, is_, VT, N)
               -  run_forward(v, r - h_r, is_, VT, N)) / (2.0 * h_r);

    assert_close(d_r_ad, d_r_fd, 1e-2, 1e-9, "∂Vmid/∂R: AD vs FD");
}

#[test]
fn ad_signs_are_physically_correct() {
    // ∂Vmid/∂R < 0: increasing R reduces the current driving the diode,
    // so the diode forward voltage (= Vmid) drops slightly. Counter to
    // first intuition — but exactly what KCL says.
    // ∂Vmid/∂Is < 0: increasing Is (the diode "leaks more easily") means
    // less Vmid is required to pass the same current.
    let (_, d_r, d_is) = run_forward_and_grad(1.0, 1_000.0, 1e-15, VT, N);
    assert!(d_r  < 0.0, "∂Vmid/∂R should be < 0 (more R → lower Vmid), got {d_r}");
    assert!(d_is < 0.0, "∂Vmid/∂Is should be < 0 (more Is → lower Vmid), got {d_is}");
}
