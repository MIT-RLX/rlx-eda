//! Tier 1: rlx Bode response of the linearised diode-RC matches the
//! closed-form 1-pole analytic expression at well-spaced frequencies.

use spike_ac::*;

const VT: f64 = 0.025_852;       // thermal voltage at 300 K

#[track_caller]
fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] a={a:+.15e} b={b:+.15e} |a-b|={:.3e} env={env:.3e}",
            (a - b).abs());
    }
}

#[test]
fn diode_rc_h_at_dc_matches_analytic() {
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let is_ = 1e-12_f64;
    let v_dc = 1.0_f64;
    let vmid = dc_op_f64(v_dc, r, is_, VT, 30);
    let g_d  = small_signal_conductance(vmid, is_, VT);

    // ω → 0: |H| → G/(G + g_d) = R_eff/R where R_eff = R ∥ 1/g_d.
    let omega = 0.0;
    let (re, im) = run_diode_rc_ac_point(omega, r, c, g_d);
    let mag = (re * re + im * im).sqrt();
    let mag_an = diode_rc_analytic_mag(omega, r, c, g_d);
    assert_close(mag, mag_an, 1e-9, 1e-12,
        "|H(0)| vs analytic G/(G+g_d)");
}

#[test]
fn diode_rc_h_at_3db_pole_matches_analytic() {
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let is_ = 1e-12_f64;
    let v_dc = 1.0_f64;
    let vmid = dc_op_f64(v_dc, r, is_, VT, 30);
    let g_d  = small_signal_conductance(vmid, is_, VT);

    let f3db  = diode_rc_analytic_f3db(r, c, g_d);
    let omega = 2.0 * std::f64::consts::PI * f3db;
    let (re, im) = run_diode_rc_ac_point(omega, r, c, g_d);
    let mag = (re * re + im * im).sqrt();
    let mag_an = diode_rc_analytic_mag(omega, r, c, g_d);

    // At f₃dB the magnitude should be DC_gain / √2.
    let dc_gain = diode_rc_analytic_mag(0.0, r, c, g_d);
    assert_close(mag, dc_gain / 2.0_f64.sqrt(), 1e-9, 1e-12,
        "|H(f₃dB)| ≈ DC_gain / √2");
    assert_close(mag, mag_an, 1e-9, 1e-12,
        "|H(f₃dB)| vs analytic");
}

#[test]
fn diode_rc_log_sweep_matches_analytic() {
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let is_ = 1e-12_f64;
    let v_dc = 1.0_f64;

    let (freq, re, im) = run_diode_rc_ac_sweep(
        1e3, 1e9, 4, r, c, is_, VT, v_dc, 30);

    let vmid = dc_op_f64(v_dc, r, is_, VT, 30);
    let g_d  = small_signal_conductance(vmid, is_, VT);
    for (i, (&f, (&re, &im))) in freq.iter().zip(re.iter().zip(im.iter())).enumerate() {
        let omega = 2.0 * std::f64::consts::PI * f;
        let mag = (re * re + im * im).sqrt();
        let mag_an = diode_rc_analytic_mag(omega, r, c, g_d);
        assert_close(mag, mag_an, 1e-9, 1e-12,
            &format!("|H| at f={f:.3e} (point {i})"));
    }
}
