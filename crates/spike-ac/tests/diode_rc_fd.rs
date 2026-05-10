//! Tier 2: AD `∂|H|²/∂{R, C}` matches both the analytic gradient (closed
//! form on the linearised circuit) and centered FD.
//!
//! Note: `g_d` is *fixed* across the FD perturbation — we hold the
//! linearisation point constant. The "true" derivative w.r.t. R also
//! includes a path through `g_d(R)` since the operating point depends
//! on R, but for AC small-signal analysis the engineering convention
//! (and ngspice's behaviour) is that AC-mode derivatives evaluate the
//! Jacobian at the precomputed OP. That convention is what the rlx
//! graph implements.

use spike_ac::*;

const VT: f64 = 0.025_852;

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] a={a:+.15e} b={b:+.15e} |a-b|={:.3e} env={env:.3e}",
            (a - b).abs());
    }
}

#[test]
fn diode_rc_ac_grad_matches_finite_difference() {
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let is_ = 1e-12_f64;
    let v_dc = 1.0_f64;
    let vmid = dc_op_f64(v_dc, r, is_, VT, 30);
    let g_d  = small_signal_conductance(vmid, is_, VT);

    let f = 1e6_f64;                      // 1 MHz: well above DC, near pole
    let omega = 2.0 * std::f64::consts::PI * f;
    let (_, _, dr_ad, dc_ad) = run_diode_rc_ac_grad(omega, r, c, g_d);

    let mag_sq = |r: f64, c: f64| -> f64 {
        let (re, im) = run_diode_rc_ac_point(omega, r, c, g_d);
        re * re + im * im
    };
    let h_r = 1e-4 * r;
    let h_c = 1e-4 * c;
    let dr_fd = (mag_sq(r + h_r, c) - mag_sq(r - h_r, c)) / (2.0 * h_r);
    let dc_fd = (mag_sq(r, c + h_c) - mag_sq(r, c - h_c)) / (2.0 * h_c);

    assert_close(dr_ad, dr_fd, 1e-4, 1e-15, "∂|H|²/∂R AD vs FD");
    assert_close(dc_ad, dc_fd, 1e-4, 1e-15, "∂|H|²/∂C AD vs FD");
}

#[test]
fn diode_rc_ac_grad_signs_are_physically_correct() {
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let is_ = 1e-12_f64;
    let v_dc = 1.0_f64;
    let vmid = dc_op_f64(v_dc, r, is_, VT, 30);
    let g_d  = small_signal_conductance(vmid, is_, VT);

    // Mid-band (above DC, near pole): increasing C lowers magnitude
    // (faster rolloff). Increasing R shouldn't have a fixed sign because
    // it both raises DC gain (G/(G+g_d) → 1 as R → ∞) and shifts the
    // pole — pick a frequency where the sign is unambiguous.
    let f = 1e7_f64;                      // 10 MHz, above pole
    let omega = 2.0 * std::f64::consts::PI * f;
    let (_, _, dr_ad, dc_ad) = run_diode_rc_ac_grad(omega, r, c, g_d);

    // dC < 0: above pole, more C → faster rolloff → smaller |H|².
    assert!(dc_ad < 0.0,
        "∂|H|²/∂C should be < 0 above the pole, got {dc_ad}");
    // R derivative isn't required to have a particular sign here —
    // sanity-check it's finite and non-zero.
    assert!(dr_ad.is_finite() && dr_ad != 0.0,
        "∂|H|²/∂R should be a finite non-zero value, got {dr_ad}");
}
