//! CMOS inverter through `eda-mna` end-to-end — no SPICE oracle, no
//! spike-cmos-gates dependency. Two `Mosfet` instances composed
//! directly into a `Circuit`:
//!
//! ```text
//!         vdd
//!          │
//!          │ Mp.S = Mp.B = vdd
//!         Mp        Mp.D = vout, Mp.G = vin
//!          │
//!          ├── vout
//!          │
//!         Mn        Mn.D = vout, Mn.G = vin
//!          │ Mn.S = Mn.B = gnd
//!          ▼
//!         gnd
//! ```
//!
//! Three properties under test:
//!
//! 1. **VTC** (DC sweep): Vout(Vin) is monotone-falling, hits the
//!    rails at the input extremes, and crosses Vdd/2 near Vin = Vdd/2
//!    (matched NMOS/PMOS sizes + matched |Vth|).
//!
//! 2. **Falling-edge transient** (Vin held high, Vout IC = Vdd):
//!    Vout falls monotonically from Vdd to its NMOS-pull-down steady
//!    state, with the first step *not* collapsing instantly to
//!    steady state — i.e. Cgd/Cgs are stamping companion currents.
//!
//! 3. **Rising-edge transient** (Vin held low, Vout IC = 0):
//!    Vout rises monotonically toward Vdd via the PMOS pull-up.
//!
//! The propagation-delay tests don't drive a *time-varying* PULSE on
//! Vin — `transient_from`'s boundary nets are fixed across the run.
//! Instead each direction is its own transient with the input held
//! at the post-edge value and the output IC set to its pre-edge
//! value. That captures `tphl` / `tplh` cleanly without needing
//! piecewise-linear boundary support yet.

use std::collections::HashMap;
use eda_mna::{
    pulse_boundary, solve_dc, transient_from, transient_pwl,
    Circuit, NetId, NewtonOptions,
};
use spike_divider_block::{attach_mosfet_with_caps, MosModel, Mosfet};

const V_DD: f32 = 1.0;
const V_TH: f32 = 0.5;
const KP:   f32 = 100e-6;

/// Build a fresh CMOS inverter circuit. Returns the circuit, its
/// boundary / unknown net handles, and the two MOSFETs (so the test
/// can grab `default_params` from the same instances that were
/// attached).
fn build_inverter() -> (Circuit, NetId, NetId, NetId, Mosfet, Mosfet) {
    let mut c = Circuit::new();
    let v_dd  = c.alloc_boundary_net();
    let v_in  = c.alloc_boundary_net();
    let v_out = c.alloc_unknown_net();

    // Matched sizing: same W/L for NMOS and PMOS so the inverter's
    // switching threshold lands near Vdd/2 with |Vthn| = |Vthp|.
    //
    // EKV-lite, not square-law: smooth weak/strong-inversion interp
    // makes the transfer function differentiable everywhere, which
    // matters for Newton convergence at the switching point. Square-
    // law has a flat saturation region (∂I_D/∂V_DS = 0 with λ=0),
    // making KCL's Jacobian rank-deficient at corner operating points
    // — Newton oscillates or stalls.
    let nmos = Mosfet::nmos(1_000, 1_000, "Mn").with_model(MosModel::EkvLite);
    let pmos = Mosfet::pmos(1_000, 1_000, "Mp").with_model(MosModel::EkvLite);

    // NMOS: D=vout, G=vin, S=B=gnd
    attach_mosfet_with_caps(&mut c, nmos.clone(),
        [v_out, v_in, NetId::GND, NetId::GND]);
    // PMOS: D=vout, G=vin, S=B=vdd
    attach_mosfet_with_caps(&mut c, pmos.clone(),
        [v_out, v_in, v_dd, v_dd]);

    (c, v_dd, v_in, v_out, nmos, pmos)
}

/// Seed both MOSFETs' default Param values into one map. Caller
/// can override individual keys (e.g., bump Cgs/Cgd) before
/// passing to solve_dc / transient_from.
///
/// Critical: bumps `Lambda` from the 0 default to 0.05 V⁻¹. With
/// λ=0 the square-law model is flat in saturation (∂I_D/∂V_DS = 0),
/// which makes KCL's Jacobian rank-deficient at any operating point
/// where one transistor is in saturation and the other is off — i.e.
/// at every Vin = 0 / Vin = Vdd corner of the inverter. Finite λ
/// gives Newton a non-zero diagonal and lets it converge cleanly.
fn merged_params(nmos: &Mosfet, pmos: &Mosfet) -> HashMap<String, f32> {
    let mut p = nmos.default_params();
    for (k, v) in pmos.default_params() {
        p.insert(k, v);
    }
    let n_name = <Mosfet as eda_hir::Block>::name(nmos);
    let p_name = <Mosfet as eda_hir::Block>::name(pmos);
    p.insert(format!("{n_name}_Lambda"), 0.05);
    p.insert(format!("{p_name}_Lambda"), 0.05);
    p
}

// ── VTC (DC sweep) ─────────────────────────────────────────────────────

#[test]
fn inverter_vtc_is_monotone_falling_with_midrail_crossing() {
    let (c, v_dd, v_in, v_out, nmos, pmos) = build_inverter();
    let params = merged_params(&nmos, &pmos);

    // 11-point sweep across [0, Vdd] with a single fixed init = Vdd/2.
    // Newton's Armijo backtracking handles the EKV-lite softplus
    // overflow that pure Newton would otherwise hit — first step would
    // try to land at the steady-state Vout, but if it overshoots into
    // f32 inf territory, the line search halves α until the residual
    // stops being NaN. No continuation tricks required.
    let n_pts = 11;
    let mut samples: Vec<(f32, f32)> = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let v_in_val = i as f32 * V_DD / (n_pts - 1) as f32;
        let mut boundary = HashMap::new();
        boundary.insert(v_dd, V_DD);
        boundary.insert(v_in, v_in_val);
        let opt = NewtonOptions {
            tol: 1e-8, vntol: 1e-6, max_iters: 100, init: V_DD / 2.0, max_backtracks: 20,
        };
        let dc = solve_dc(&c, &params, &boundary, opt);
        assert!(dc.converged,
            "DC failed at Vin = {:.3}: residual = {:.3e}",
            v_in_val, dc.final_residual_max);
        let vout_v = dc.voltages[&v_out];
        assert!(vout_v.is_finite(),
            "DC at Vin = {:.3} returned non-finite Vout = {}",
            v_in_val, vout_v);
        samples.push((v_in_val, vout_v));
    }

    // Rails: Vin=0 → Vout ≈ Vdd (NMOS off, PMOS on).
    let v_out_lo_in = samples[0].1;
    assert!((v_out_lo_in - V_DD).abs() < 0.05,
        "VTC at Vin=0: Vout = {}, expected ≈ {}", v_out_lo_in, V_DD);
    // Vin=Vdd → Vout ≈ 0 (NMOS on, PMOS off).
    let v_out_hi_in = samples[n_pts - 1].1;
    assert!(v_out_hi_in < 0.05,
        "VTC at Vin=Vdd: Vout = {}, expected ≈ 0", v_out_hi_in);

    // Monotone falling (small slack for f32 noise).
    for w in samples.windows(2) {
        assert!(w[1].1 <= w[0].1 + 1e-3,
            "VTC not monotone: ({:.3}, {:.4}) → ({:.3}, {:.4})",
            w[0].0, w[0].1, w[1].0, w[1].1);
    }

    // Switching threshold: with matched W/L and matched |Vth|, the
    // inverter switches near Vdd/2. We bracket the Vdd/2 crossing
    // and assert it lands within ±0.15 V of Vdd/2.
    let mid_v = V_DD / 2.0;
    let crossing = samples.windows(2).find_map(|w| {
        if (w[0].1 - mid_v).signum() != (w[1].1 - mid_v).signum() {
            // Linear interp to find the V_in where Vout crosses Vdd/2.
            let frac = (w[0].1 - mid_v) / (w[0].1 - w[1].1);
            Some(w[0].0 + frac * (w[1].0 - w[0].0))
        } else { None }
    }).expect("VTC must cross Vdd/2 somewhere in the sweep");
    assert!((crossing - mid_v).abs() < 0.15,
        "switching threshold = {:.3} V, expected within 0.15 V of Vdd/2 = {}",
        crossing, mid_v);
}

// ── Transient: falling edge (NMOS pulls vout down) ────────────────────

#[test]
fn inverter_falling_edge_has_finite_tphl() {
    let (c, v_dd, v_in, v_out, nmos, pmos) = build_inverter();
    let mut params = merged_params(&nmos, &pmos);
    // Bump gate caps to 100 fF so τ is much larger than dt.
    let n_name = <Mosfet as eda_hir::Block>::name(&nmos);
    let p_name = <Mosfet as eda_hir::Block>::name(&pmos);
    for k in [
        format!("{n_name}_Cgs"), format!("{n_name}_Cgd"),
        format!("{p_name}_Cgs"), format!("{p_name}_Cgd"),
    ] {
        params.insert(k, 100e-15);
    }

    // Vin held high (NMOS on, PMOS off); Vout starts pre-charged.
    let mut boundary = HashMap::new();
    boundary.insert(v_dd, V_DD);
    boundary.insert(v_in, V_DD);
    let mut ic = HashMap::new();
    ic.insert(v_out, V_DD);

    let dt = 0.5e-9_f32;     // 0.5 ns/step
    let n_steps = 200;       // 100 ns total
    let waveform = transient_from(&c, &params, &boundary, &ic, dt, n_steps,
                                   NewtonOptions::default());

    for (k, step) in waveform.iter().enumerate() {
        assert!(step.converged, "step {k} failed: residual = {:.3e}",
            step.final_residual_max);
    }

    // Initial: Vout = Vdd.
    assert!((waveform[0].voltages[&v_out] - V_DD).abs() < 1e-6);

    // Monotone falling.
    for w in waveform.windows(2) {
        let a = w[0].voltages[&v_out];
        let b = w[1].voltages[&v_out];
        assert!(b <= a + 1e-3, "falling not monotone: {} → {}", a, b);
    }

    // First-step sanity: Vout must NOT have collapsed to steady state
    // in one step. With Cgs+Cgd ≈ 200 fF on Vout and saturation
    // current ~12 µA at Vov=0.5, dV/dt ≈ 60 MV/s → ΔV in 0.5 ns ≈
    // 30 mV. Vout should still be > 0.95 V after the first step.
    let v_step1 = waveform[1].voltages[&v_out];
    assert!(v_step1 > 0.95,
        "Vout fell {:.3} V in one step ({} ns) — gate caps not stamping",
        V_DD - v_step1, dt * 1e9);

    // Find tphl: first crossing through Vdd/2 (linear-interp).
    let half = V_DD / 2.0;
    let tphl = waveform.windows(2).find_map(|w| {
        let a = w[0].voltages[&v_out];
        let b = w[1].voltages[&v_out];
        (a >= half && b < half).then(|| {
            let frac = (a - half) / (a - b);
            w[0].t + frac * (w[1].t - w[0].t)
        })
    });
    let tphl = tphl.expect("Vout never reached Vdd/2 — bump n_steps");
    // Sanity: tphl must be > dt (not instantaneous) and < total run time.
    assert!(tphl > dt, "tphl = {:.2e} s ≤ dt — too fast", tphl);
    assert!(tphl < n_steps as f32 * dt,
        "tphl = {:.2e} s ≥ run time", tphl);
}

// ── Transient: rising edge (PMOS pulls vout up) ───────────────────────

#[test]
fn inverter_rising_edge_has_finite_tplh() {
    let (c, v_dd, v_in, v_out, nmos, pmos) = build_inverter();
    let mut params = merged_params(&nmos, &pmos);
    let n_name = <Mosfet as eda_hir::Block>::name(&nmos);
    let p_name = <Mosfet as eda_hir::Block>::name(&pmos);
    for k in [
        format!("{n_name}_Cgs"), format!("{n_name}_Cgd"),
        format!("{p_name}_Cgs"), format!("{p_name}_Cgd"),
    ] {
        params.insert(k, 100e-15);
    }

    // Vin held low (NMOS off, PMOS on); Vout starts at 0.
    let mut boundary = HashMap::new();
    boundary.insert(v_dd, V_DD);
    boundary.insert(v_in, 0.0);
    let mut ic = HashMap::new();
    ic.insert(v_out, 0.0);

    let dt = 0.5e-9_f32;
    let n_steps = 200;
    let waveform = transient_from(&c, &params, &boundary, &ic, dt, n_steps,
                                   NewtonOptions::default());

    for (k, step) in waveform.iter().enumerate() {
        assert!(step.converged, "step {k} failed: residual = {:.3e}",
            step.final_residual_max);
    }

    assert!(waveform[0].voltages[&v_out].abs() < 1e-6, "IC not honored");

    // Monotone rising.
    for w in waveform.windows(2) {
        let a = w[0].voltages[&v_out];
        let b = w[1].voltages[&v_out];
        assert!(b >= a - 1e-3, "rising not monotone: {} → {}", a, b);
    }

    // First-step: not instantaneous to rail.
    let v_step1 = waveform[1].voltages[&v_out];
    assert!(v_step1 < 0.05,
        "Vout rose {:.3} V in one step — gate caps not stamping",
        v_step1);

    // tplh: crossing through Vdd/2 from below.
    let half = V_DD / 2.0;
    let tplh = waveform.windows(2).find_map(|w| {
        let a = w[0].voltages[&v_out];
        let b = w[1].voltages[&v_out];
        (a < half && b >= half).then(|| {
            let frac = (half - a) / (b - a);
            w[0].t + frac * (w[1].t - w[0].t)
        })
    });
    let tplh = tplh.expect("Vout never reached Vdd/2 — bump n_steps");
    assert!(tplh > dt, "tplh = {:.2e} s ≤ dt — too fast", tplh);
    assert!(tplh < n_steps as f32 * dt,
        "tplh = {:.2e} s ≥ run time", tplh);
}

// ── Cross-check: tphl and tplh are within ~3× of each other ──────────

#[test]
fn inverter_falling_and_rising_delays_are_comparable() {
    // Re-runs the same builds as the two delay tests above (small
    // duplication is the price of independent test isolation), then
    // asserts the asymmetry isn't extreme. With matched W/L and the
    // PMOS having half the Kp of NMOS by default (Pmos::DEFAULT_KP =
    // 10e-6 vs Nmos = 20e-6 in eda-spice-emit primitives), tplh ≈
    // 2·tphl is expected. We assert ratio ≤ 5 to leave slack for the
    // smooth-min / square-law interplay near the rails.
    let extract_delay = |is_falling: bool| -> f32 {
        let (c, v_dd, v_in, v_out, nmos, pmos) = build_inverter();
        let mut params = merged_params(&nmos, &pmos);
        let n_name = <Mosfet as eda_hir::Block>::name(&nmos);
        let p_name = <Mosfet as eda_hir::Block>::name(&pmos);
        for k in [
            format!("{n_name}_Cgs"), format!("{n_name}_Cgd"),
            format!("{p_name}_Cgs"), format!("{p_name}_Cgd"),
        ] {
            params.insert(k, 100e-15);
        }
        let mut boundary = HashMap::new();
        boundary.insert(v_dd, V_DD);
        boundary.insert(v_in, if is_falling { V_DD } else { 0.0 });
        let mut ic = HashMap::new();
        ic.insert(v_out, if is_falling { V_DD } else { 0.0 });

        let dt = 0.5e-9_f32;
        let n_steps = 400;
        let wf = transient_from(&c, &params, &boundary, &ic, dt, n_steps,
                                 NewtonOptions::default());

        let half = V_DD / 2.0;
        wf.windows(2).find_map(|w| {
            let a = w[0].voltages[&v_out];
            let b = w[1].voltages[&v_out];
            let crossed = if is_falling {
                a >= half && b < half
            } else {
                a < half && b >= half
            };
            crossed.then(|| {
                let frac = if is_falling { (a - half) / (a - b) }
                           else          { (half - a) / (b - a) };
                w[0].t + frac * (w[1].t - w[0].t)
            })
        }).expect("Vout never crossed Vdd/2")
    };

    let tphl = extract_delay(true);
    let tplh = extract_delay(false);
    let ratio = if tphl > tplh { tphl / tplh } else { tplh / tphl };
    assert!(ratio < 5.0,
        "tphl / tplh asymmetry = {:.2} (tphl = {:.2e}, tplh = {:.2e}) — \
         expected ≤ 5 with default-sized matched MOSFETs",
        ratio, tphl, tplh);
}

// Suppress an unused-constant warning when V_TH/KP land in test-config
// land but don't appear in every test.
#[allow(dead_code)]
const _UNUSED_GUARD: (f32, f32) = (V_TH, KP);

// ── Transient with PULSE (time-varying boundary) ──────────────────────

/// Drive the inverter with a single rectangular Vin pulse and observe
/// **both** edges (rising input → falling output, falling input →
/// rising output) inside one transient run. Uses `transient_pwl` +
/// `pulse_boundary` — no two-stage workaround.
#[test]
fn inverter_pulse_input_produces_inverted_pulse_output() {
    let (c, v_dd, v_in, v_out, nmos, pmos) = build_inverter();
    let mut params = merged_params(&nmos, &pmos);
    let n_name = <Mosfet as eda_hir::Block>::name(&nmos);
    let p_name = <Mosfet as eda_hir::Block>::name(&pmos);
    for k in [
        format!("{n_name}_Cgs"), format!("{n_name}_Cgd"),
        format!("{p_name}_Cgs"), format!("{p_name}_Cgd"),
    ] {
        params.insert(k, 100e-15);
    }

    // Vdd held constant; Vin pulses high between t = 30 ns and 70 ns.
    let mut static_bnd = HashMap::new();
    static_bnd.insert(v_dd, V_DD);
    let boundary = pulse_boundary(
        static_bnd,
        v_in,
        /*v_lo*/ 0.0,
        /*v_hi*/ V_DD,
        /*t_rise*/ 30e-9,
        /*t_fall*/ 70e-9,
    );

    // IC: Vin starts low → steady-state Vout ≈ Vdd. Set the IC there.
    let mut ic = HashMap::new();
    ic.insert(v_out, V_DD);

    let dt = 0.5e-9_f32;
    let n_steps = 200;          // 100 ns total — covers low → high → low
    let opt = NewtonOptions {
        tol: 1e-8, vntol: 1e-6, max_iters: 100, init: V_DD / 2.0, max_backtracks: 20,
    };
    let waveform = transient_pwl(&c, &params, &boundary, &ic, dt, n_steps, opt);

    for (k, step) in waveform.iter().enumerate() {
        assert!(step.converged, "step {k} (t={:.2e}) failed: residual = {:.3e}",
            step.t, step.final_residual_max);
    }

    // Sample-time helper: closest waveform index to a target t.
    let at = |t_target: f32| -> f32 {
        waveform.iter()
            .min_by(|a, b| (a.t - t_target).abs()
                .partial_cmp(&(b.t - t_target).abs()).unwrap())
            .map(|s| s.voltages[&v_out]).unwrap()
    };

    // Pre-pulse: Vout near Vdd.
    let v_pre  = at(20e-9);
    assert!(v_pre > 0.9 * V_DD,
        "Vout before pulse = {}, expected near Vdd", v_pre);

    // Mid-pulse (after settling): Vout pulled low.
    let v_mid  = at(60e-9);
    assert!(v_mid < 0.5 * V_DD,
        "Vout mid-pulse = {}, expected pulled below mid-rail", v_mid);

    // Post-pulse: Vout recovered toward Vdd.
    let v_post = at(95e-9);
    assert!(v_post > 0.5 * V_DD,
        "Vout post-pulse = {}, expected recovered toward Vdd", v_post);

    // Across the rising-Vin / falling-Vout edge: monotone fall on the
    // [t_rise, t_rise + 30 ns] window.
    let i_rise = (30e-9_f32 / dt).round() as usize;
    let i_mid  = ((30e-9_f32 + 30e-9) / dt).round() as usize;
    for w in waveform[i_rise..=i_mid].windows(2) {
        let a = w[0].voltages[&v_out];
        let b = w[1].voltages[&v_out];
        assert!(b <= a + 5e-3,
            "Vout not monotone-fall during rising-Vin window: {} → {}", a, b);
    }
    // Across the falling-Vin / rising-Vout edge: monotone rise.
    let i_fall = (70e-9_f32 / dt).round() as usize;
    let i_end  = ((70e-9_f32 + 25e-9) / dt).round() as usize;
    for w in waveform[i_fall..=i_end].windows(2) {
        let a = w[0].voltages[&v_out];
        let b = w[1].voltages[&v_out];
        assert!(b >= a - 5e-3,
            "Vout not monotone-rise during falling-Vin window: {} → {}", a, b);
    }
}
