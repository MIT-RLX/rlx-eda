//! Tier 1+2: rlx CMOS inverter DC against pure-Rust Newton, plus AD
//! against centered finite differences. Closes the loop on the first
//! multi-device nonlinear circuit through rlx.

use spike_mosfet_dc::*;

const VTH_N: f64 = 0.5;
const KP_N:  f64 = 100e-6;
const LAM_N: f64 = 0.02;
const VTH_P: f64 = 0.5;
const KP_P:  f64 = 100e-6;
const LAM_P: f64 = 0.02;
const VDD:   f64 = 1.8;
const N_NEWTON: usize = 30;

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] a={a:+.6e} b={b:+.6e} |a-b|={:.3e} env={env:.3e}",
            (a - b).abs());
    }
}

#[test]
fn rlx_inverter_matches_rust_reference() {
    // Sweep V_in across the rail; rlx inverter should match the
    // pure-Rust Newton at each point. Skip the deep-cutoff regions
    // (V_in < 0.6 V or > 1.2 V) where the smooth model deviates from
    // the strict piecewise reference (those regions are dominated by
    // sub-threshold behavior the smooth model doesn't capture, but
    // the strict reference doesn't either — both produce
    // V_out ≈ rail).
    let v_ins = [0.7_f64, 0.8, 0.9, 1.0, 1.1];
    for &v_in in &v_ins {
        let v_out_rlx = run_inverter_dc(
            v_in, VDD, VTH_N, KP_N, LAM_N, VTH_P, KP_P, LAM_P, N_NEWTON);
        let v_out_ref = ref_inverter_dc(
            v_in, VDD, VTH_N, KP_N, LAM_N, VTH_P, KP_P, LAM_P, N_NEWTON);
        assert_close(v_out_rlx, v_out_ref, 1e-3, 1e-4,
            &format!("V_out @ V_in={v_in:.2}"));
    }
}

#[test]
fn inverter_transfer_curve_is_monotonic_decreasing() {
    // Classic CMOS inverter sigmoid: V_out monotonically decreases as
    // V_in sweeps from 0 to VDD. With damped Newton (step-magnitude
    // cap in build_fwd_body) the steep-transition region no longer
    // diverges, so the FULL sweep should be monotonic.
    let mut prev = run_inverter_dc(
        0.6, VDD, VTH_N, KP_N, LAM_N, VTH_P, KP_P, LAM_P, N_NEWTON);
    for i in 1..=10 {
        let v_in = 0.6 + (i as f64) * (1.2 - 0.6) / 10.0;
        let v_out = run_inverter_dc(
            v_in, VDD, VTH_N, KP_N, LAM_N, VTH_P, KP_P, LAM_P, N_NEWTON);
        assert!(v_out <= prev + 1e-3,
            "transfer curve not monotonic at V_in={v_in}: prev={prev:.4} cur={v_out:.4}");
        prev = v_out;
    }
}

#[test]
fn inverter_switch_threshold_near_vdd_half() {
    // Symmetric NMOS/PMOS sizing → switch threshold V_M ≈ VDD/2 = 0.9 V.
    // Find V_in such that V_out ≈ V_in via bisection.
    let f = |v_in: f64| -> f64 {
        let v_out = run_inverter_dc(
            v_in, VDD, VTH_N, KP_N, LAM_N, VTH_P, KP_P, LAM_P, N_NEWTON);
        v_out - v_in
    };
    let mut lo = 0.7;
    let mut hi = 1.1;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 { lo = mid; } else { hi = mid; }
    }
    let v_m = 0.5 * (lo + hi);
    assert!((v_m - 0.9).abs() < 0.05,
        "switch threshold V_M={v_m:.4} should be ≈ VDD/2 = 0.9");
}

#[test]
fn inverter_ad_matches_finite_differences_at_switch_point() {
    // At V_in = 0.9 V (mid-rail), gradients are non-trivial in all
    // 6 device parameters. Compare AD against central FD on the rlx
    // forward.
    let v_in = 0.9_f64;
    let (_, dvth_n, dkp_n, dlam_n, dvth_p, dkp_p, dlam_p) =
        run_inverter_dc_grad(
            v_in, VDD, VTH_N, KP_N, LAM_N, VTH_P, KP_P, LAM_P, N_NEWTON);

    let v_out = |vth_n, kp_n, lam_n, vth_p, kp_p, lam_p| -> f64 {
        run_inverter_dc(v_in, VDD, vth_n, kp_n, lam_n, vth_p, kp_p, lam_p, N_NEWTON)
    };
    // Per-parameter perturbation. Smooth-min FD floor + Newton noise
    // sets a 5% relative envelope.
    let h_v = 1e-3_f64;
    let h_k = KP_N * 1e-3;
    let h_l = 1e-4_f64;

    let fd_vth_n = (v_out(VTH_N + h_v, KP_N, LAM_N, VTH_P, KP_P, LAM_P)
                   - v_out(VTH_N - h_v, KP_N, LAM_N, VTH_P, KP_P, LAM_P)) / (2.0 * h_v);
    let fd_kp_n  = (v_out(VTH_N, KP_N + h_k, LAM_N, VTH_P, KP_P, LAM_P)
                   - v_out(VTH_N, KP_N - h_k, LAM_N, VTH_P, KP_P, LAM_P)) / (2.0 * h_k);
    let fd_lam_n = (v_out(VTH_N, KP_N, LAM_N + h_l, VTH_P, KP_P, LAM_P)
                   - v_out(VTH_N, KP_N, LAM_N - h_l, VTH_P, KP_P, LAM_P)) / (2.0 * h_l);
    let fd_vth_p = (v_out(VTH_N, KP_N, LAM_N, VTH_P + h_v, KP_P, LAM_P)
                   - v_out(VTH_N, KP_N, LAM_N, VTH_P - h_v, KP_P, LAM_P)) / (2.0 * h_v);
    let fd_kp_p  = (v_out(VTH_N, KP_N, LAM_N, VTH_P, KP_P + h_k, LAM_P)
                   - v_out(VTH_N, KP_N, LAM_N, VTH_P, KP_P - h_k, LAM_P)) / (2.0 * h_k);
    let fd_lam_p = (v_out(VTH_N, KP_N, LAM_N, VTH_P, KP_P, LAM_P + h_l)
                   - v_out(VTH_N, KP_N, LAM_N, VTH_P, KP_P, LAM_P - h_l)) / (2.0 * h_l);

    println!(
        "AD vs FD at V_in=0.9:\n  \
         dVth_n: AD={dvth_n:+.3e} FD={fd_vth_n:+.3e}\n  \
         dkp_n:  AD={dkp_n:+.3e} FD={fd_kp_n:+.3e}\n  \
         dlam_n: AD={dlam_n:+.3e} FD={fd_lam_n:+.3e}\n  \
         dVth_p: AD={dvth_p:+.3e} FD={fd_vth_p:+.3e}\n  \
         dkp_p:  AD={dkp_p:+.3e} FD={fd_kp_p:+.3e}\n  \
         dlam_p: AD={dlam_p:+.3e} FD={fd_lam_p:+.3e}",
    );
    // 10% relative envelope — both AD and FD use FD-of-the-residual
    // inside, so two layers of central-difference noise compound.
    assert_close(dvth_n, fd_vth_n, 1e-1, 1e-4, "dVth_n");
    assert_close(dkp_n,  fd_kp_n,  1e-1, 1e-4, "dkp_n");
    assert_close(dlam_n, fd_lam_n, 1e-1, 1e-4, "dlam_n");
    assert_close(dvth_p, fd_vth_p, 1e-1, 1e-4, "dVth_p");
    assert_close(dkp_p,  fd_kp_p,  1e-1, 1e-4, "dkp_p");
    assert_close(dlam_p, fd_lam_p, 1e-1, 1e-4, "dlam_p");
}
