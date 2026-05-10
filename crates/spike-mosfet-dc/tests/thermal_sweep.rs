//! Thermal corner sweep: analytic + FD + ngspice witnesses across
//! T ∈ {-40, 27, 125} °C.
//!
//! ## Pyramid at each T
//!
//! 1. **Analytic consistency.** `run_id_at_temp` (rlx smooth graph after
//!    parameter remap) vs `id_strict_at_temp` (closed-form piecewise),
//!    deep-saturation operating points where the two agree to ~ppm.
//! 2. **FD-of-analytic.** Central FD on `run_id_at_temp` w.r.t. (Vth0,
//!    kp0, λ) at each T matches AD via `run_id_grad` evaluated at the
//!    T-shifted parameters. This proves the rlx graph still differentiates
//!    correctly when its inputs are scaled by our T-remap.
//! 3. **Cross-engine vs ngspice.** ngspice runs the same NMOS at the
//!    same biases with `.temp T_celsius` — its built-in LEVEL 1 scaling
//!    (μ ∝ T^-1.5 + bandgap-based VTO shift) is independent of our
//!    analytic remap, so the two engines crossing inside tolerance is a
//!    real corroboration, not a tautology.
//!
//! ## Tolerance schedule
//!
//! - **Tnom (27 °C):** 1 % rel — same envelope as the existing
//!   `tests/ngspice.rs` nominal test.
//! - **Corners (-40, 125 °C):** 7 % rel — accounts for the difference
//!   between our linear KT1 Vth shift and ngspice's bandgap-based shift
//!   (see `KT1` doc in `lib.rs`). Tighter would force us to mirror
//!   ngspice's exact PHI/Eg formulas, which buys nothing physically.
//!
//! ## Sanity guards
//!
//! Two cheap checks rule out "feature did nothing":
//! - `Id(125 °C) < Id(-40 °C)` in saturation — μ falls faster than Vth
//!   shifts help, so cold device sources more current.
//! - `|Id(125 °C) − Id(27 °C)| / Id(27 °C) > 5 %` — rules out a no-op
//!   where T failed to thread anywhere.

#![cfg(feature = "ngspice")]

use spike_mosfet_dc::*;

const VTH0: f64 = 0.5;
const KP0:  f64 = 100e-6;
const LAM:  f64 = 0.02;
const W:    f64 = 10e-6;
const L:    f64 = 2e-6;

/// (Vgs, Vds) — deep saturation so the smooth-vs-strict deviation is
/// dominated by O(1/β) cutoff smoothing, well below 1 ppm here.
const BIAS: (f64, f64) = (2.0, 2.0);

/// Tnom + two corners. Sky130 standard PVT corners use these exact
/// temperatures (see sky130 PDK `lib/sky130_fd_sc_hd__*` corner names).
const T_CORNERS: [f64; 3] = [-40.0, 27.0, 125.0];

#[test]
fn analytic_smooth_matches_strict_at_each_corner() {
    let (vgs, vds) = BIAS;
    for &t in &T_CORNERS {
        let id_smooth = run_id_at_temp(vgs, vds, VTH0, KP0, LAM, t);
        let id_strict = id_strict_at_temp(vgs, vds, VTH0, KP0, LAM, t);
        let rel = (id_smooth - id_strict).abs() / id_strict.abs().max(1e-15);
        assert!(
            rel < 1e-4,
            "T={t}°C: smooth Id={id_smooth:.6e}, strict Id={id_strict:.6e}, rel={rel:.3e}",
        );
    }
}

#[test]
fn fd_matches_ad_at_each_corner() {
    // Central FD on `run_id_at_temp` w.r.t. (Vth0, kp0, λ). The graph
    // itself is built at the T-shifted parameters; we re-use
    // `run_id_grad` evaluated at those shifted scalars.
    let (vgs, vds) = BIAS;
    for &t in &T_CORNERS {
        let vth_t = vth_at_temp(VTH0, t);
        let kp_t  = kp_at_temp(KP0, t);
        let (_id, ad_dvth, ad_dkp, ad_dlam) =
            run_id_grad(vgs, vds, vth_t, kp_t, LAM);

        let eps_v  = 1e-5;
        let eps_kp = KP0 * 1e-3;
        let eps_l  = 1e-5;

        let fd_dvth = (run_id_at_temp(vgs, vds, VTH0 + eps_v, KP0, LAM, t)
                     - run_id_at_temp(vgs, vds, VTH0 - eps_v, KP0, LAM, t))
                     / (2.0 * eps_v);
        let fd_dkp  = (run_id_at_temp(vgs, vds, VTH0, KP0 + eps_kp, LAM, t)
                     - run_id_at_temp(vgs, vds, VTH0, KP0 - eps_kp, LAM, t))
                     / (2.0 * eps_kp);
        let fd_dlam = (run_id_at_temp(vgs, vds, VTH0, KP0, LAM + eps_l, t)
                     - run_id_at_temp(vgs, vds, VTH0, KP0, LAM - eps_l, t))
                     / (2.0 * eps_l);

        // ∂Id/∂Vth0 = ∂Id/∂Vth(T) (chain rule has factor 1 for KT1's
        // additive shift). ∂Id/∂kp0 = ∂Id/∂kp(T) · (T_K/Tnom_K)^UTE.
        let ratio = (celsius_to_kelvin(t) / celsius_to_kelvin(T_NOM_C)).powf(UTE);
        let ad_dkp0 = ad_dkp * ratio;

        let tol = 5e-3;
        assert!(
            (fd_dvth - ad_dvth).abs() / ad_dvth.abs().max(1e-9) < tol,
            "T={t}°C: ∂Id/∂Vth0 fd={fd_dvth:.6e} ad={ad_dvth:.6e}",
        );
        assert!(
            (fd_dkp - ad_dkp0).abs() / ad_dkp0.abs().max(1e-9) < tol,
            "T={t}°C: ∂Id/∂kp0 fd={fd_dkp:.6e} ad={ad_dkp0:.6e}",
        );
        assert!(
            (fd_dlam - ad_dlam).abs() / ad_dlam.abs().max(1e-9) < tol,
            "T={t}°C: ∂Id/∂λ fd={fd_dlam:.6e} ad={ad_dlam:.6e}",
        );
    }
}

#[test]
fn ngspice_matches_analytic_at_each_corner() {
    use eda_extern_ngspice::LocalBinary;
    if LocalBinary::from_env().is_err() {
        eprintln!("skipping: ngspice not on PATH");
        return;
    }

    let (vgs, vds) = BIAS;

    // Hold per-T analytic Ids so the sanity guards below can use them.
    let mut id_analytic = [0.0_f64; 3];
    let mut id_ngspice  = [0.0_f64; 3];

    for (i, &t) in T_CORNERS.iter().enumerate() {
        let id_a = run_id_at_temp(vgs, vds, VTH0, KP0, LAM, t);
        let id_n = run_ngspice_id_at_temp(vgs, vds, t);

        let rel = (id_a - id_n).abs() / id_n.abs().max(1e-15);
        let tol = if (t - T_NOM_C).abs() < 1e-9 { 1e-2 } else { 7e-2 };
        assert!(
            rel < tol,
            "T={t}°C: analytic Id={id_a:.6e}, ngspice Id={id_n:.6e}, rel={rel:.3e}, tol={tol}",
        );

        id_analytic[i] = id_a;
        id_ngspice[i]  = id_n;
    }

    // Sanity 1: mobility wins over Vth-shift in saturation, so the cold
    // device sources more current than the hot one. Both engines should
    // agree on the sign.
    let i_cold = 0; // -40 °C
    let i_hot  = 2; // 125 °C
    assert!(id_analytic[i_cold] > id_analytic[i_hot],
        "analytic Id should drop with T: cold={}, hot={}",
        id_analytic[i_cold], id_analytic[i_hot]);
    assert!(id_ngspice[i_cold] > id_ngspice[i_hot],
        "ngspice Id should drop with T: cold={}, hot={}",
        id_ngspice[i_cold], id_ngspice[i_hot]);

    // Sanity 2: T actually moved the needle (rules out "feature did
    // nothing"). At Vov=1.5 V, hot vs nominal should differ by >5 %.
    let nominal = id_analytic[1];
    let hot_shift = (id_analytic[i_hot] - nominal).abs() / nominal;
    assert!(hot_shift > 0.05,
        "T-shift was suspiciously small ({hot_shift:.3} rel) — did T thread through?");
}

/// Build a `.temp <T>` deck via `spice_deck_at_temp`, run ngspice, scrape
/// drain current. Mirrors `tests/ngspice.rs`'s `run_ngspice_id` but at a
/// chosen corner temperature.
fn run_ngspice_id_at_temp(vgs: f64, vds: f64, t_celsius: f64) -> f64 {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let deck = spice_deck_at_temp(vgs, vds, VTH0, KP0, LAM, W, L, t_celsius);
    let bin  = std::env::var("NGSPICE_BIN").unwrap_or_else(|_| "ngspice".into());
    let full = format!(
        "* rlx-eda thermal sweep probe\n{deck}\
         .control\nop\nprint i(Vd)\n.endc\n.end\n",
    );

    let mut child = Command::new(bin)
        .args(["-b", "-n"])
        .arg("/dev/stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ngspice spawn failed");
    child.stdin.as_mut().unwrap().write_all(full.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);

    let raw = parse_i_vd(&stdout).unwrap_or_else(|| {
        panic!("could not parse i(Vd) from ngspice stdout at T={t_celsius}°C:\n{stdout}");
    });
    // ngspice reports current flowing INTO the Vd source, which is -Id.
    -raw
}

fn parse_i_vd(stdout: &str) -> Option<f64> {
    let needle = "i(vd)";
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        if let Some(idx) = lower.find(needle) {
            let tail = &line[idx + needle.len()..];
            let tail = tail.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
            if let Some(tok) = tail.split_whitespace().next() {
                if let Ok(v) = tok.parse::<f64>() {
                    return Some(v);
                }
            }
        }
    }
    None
}
