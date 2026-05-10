//! End-to-end transient through `Mosfet` parasitic caps. With the
//! gate caps attached, an NMOS-as-resistor inverter exhibits real
//! switching delay through the BE-step solver — not just the
//! instantaneous DC OP at every step.
//!
//! Topology (CMOS-ish, NMOS pull-down + resistor pull-up):
//!
//! ```text
//!     V_DD ──[R_load]── vout ──[NMOS, D=vout, G=vin, S=B=gnd]── gnd
//!                                            (Cgs across G-S)
//!                                            (Cgd across G-D = G-vout)
//! ```
//!
//! Driving `vin` low → high makes the NMOS conduct → vout falls.
//! With C_load on Cgd and on the NMOS's gate, vout falls with a
//! finite τ. We don't try to match the analytic τ to high precision
//! (the system is nonlinear; τ depends on the operating point); just
//! that the response is monotonic and delayed past the input edge.

use std::collections::HashMap;
use eda_mna::{transient_from, Circuit, NetId, NewtonOptions};
use spike_divider_block::{attach_mosfet_with_caps, Mosfet, Resistor};

#[test]
fn nmos_inverter_transient_has_finite_fall_time() {
    let mut c = Circuit::new();
    let v_dd = c.alloc_boundary_net();
    let vin  = c.alloc_boundary_net();
    let vout = c.alloc_unknown_net();

    let r_load = Resistor { length: 10_000, id: "R1".into() };
    let nmos   = Mosfet::nmos(2_000, 1_000, "M1");

    c.add_device(r_load.clone(), &[v_dd, vout]);
    attach_mosfet_with_caps(&mut c, nmos.clone(),
        [vout, vin, NetId::GND, NetId::GND]);

    // Attached: 1 NonlinearDcBehavioral (Mosfet) + 2 storage (Cgs, Cgd).
    assert_eq!(c.n_storage(), 2);

    let r_ohms = 10_000.0_f32;     // 10 kΩ pull-up
    let mut params = nmos.default_params();
    params.insert(eda_hir::Block::name(&r_load), r_ohms);
    // Make τ comfortably larger than dt. With Cgd = 1 pF and the
    // effective discharge path (Rload || NMOS_g_on ≈ Rload at strong
    // overdrive), τ_RC ≈ 10 ns. We pick dt = 1 ns so each step
    // resolves a small fraction of τ.
    let m_name = <Mosfet as eda_hir::Block>::name(&nmos);
    params.insert(format!("{m_name}_Cgs"), 1e-12);
    params.insert(format!("{m_name}_Cgd"), 1e-12);

    let mut boundary = HashMap::new();
    boundary.insert(v_dd, 1.0_f32);
    boundary.insert(vin,  1.0_f32);    // input held high → NMOS on

    // Initial condition: vout starts pre-charged to V_DD (NMOS off
    // before t=0 — i.e., the input was low until just now). We
    // step into the new boundary (vin=1 V) and watch vout fall.
    let mut ic = HashMap::new();
    ic.insert(vout, 1.0_f32);

    let dt = 1e-9_f32;          // 1 ns per step
    let n_steps = 200;          // 200 ns ≈ 20τ
    let waveform = transient_from(&c, &params, &boundary, &ic, dt, n_steps,
                                   NewtonOptions::default());

    // Every step converged.
    for (k, step) in waveform.iter().enumerate() {
        assert!(step.converged, "step {k} (t={:.2e}) did not converge: residual = {:.3e}",
            step.t, step.final_residual_max);
    }

    // Initial value: vout = V_DD = 1.0 (from IC).
    let v0 = waveform[0].voltages[&vout];
    assert!((v0 - 1.0).abs() < 1e-6, "vout(0) = {}, expected 1.0", v0);

    // Monotonic fall: vout(t) ≤ vout(t-dt) for every t (small slack
    // for f32 rounding).
    for w in waveform.windows(2) {
        let a = w[0].voltages[&vout];
        let b = w[1].voltages[&vout];
        assert!(b <= a + 1e-4,
            "vout not monotone fall: t={:.2e}: {} → t={:.2e}: {}",
            w[0].t, a, w[1].t, b);
    }

    // Finite delay: at t = dt (one step in) vout must NOT yet have
    // collapsed to the steady-state value. If gate caps weren't
    // working, every BE step would just return the new DC OP — vout
    // would jump in step 1. With τ ≈ 10 ns, after 1 ns we expect
    // vout to have fallen at most a few percent.
    let v_step1 = waveform[1].voltages[&vout];
    assert!(v_step1 > 0.97,
        "vout fell too far in one step (got {} after {} ns) — \
         gate caps may not be wired", v_step1, dt * 1e9);

    // Late-time settling: by t=200 ns ≈ 20τ, vout should be at its
    // saturation-region steady state. With Kp=100 µA/V², W/L=2,
    // V_ov=0.5 → I_D_sat = ½·Kp·(W/L)·V_ov² = 25 µA. Through 10 kΩ
    // pull-up: V_drop = 0.25 V → vout ≈ 0.75 V.
    let v_late = waveform[n_steps].voltages[&vout];
    assert!((v_late - 0.75).abs() < 0.05,
        "vout(200 ns) = {}, expected ≈ 0.75 V (saturation steady state)",
        v_late);
}

#[test]
fn dc_solve_unaffected_by_gate_caps() {
    // attach_mosfet_with_caps must produce the same DC OP as plain
    // add_device — caps are open at DC.
    let mut c1 = Circuit::new();
    let v_dd = c1.alloc_boundary_net();
    let vmid = c1.alloc_unknown_net();
    let r = Resistor { length: 10_000, id: "R1".into() };
    let nmos = Mosfet::nmos(1_000, 1_000, "M1");
    c1.add_device(r.clone(), &[v_dd, vmid]);
    c1.add_device(nmos.clone(), &[vmid, vmid, NetId::GND, NetId::GND]);

    let mut c2 = Circuit::new();
    let v_dd2 = c2.alloc_boundary_net();
    let vmid2 = c2.alloc_unknown_net();
    c2.add_device(r.clone(), &[v_dd2, vmid2]);
    attach_mosfet_with_caps(&mut c2, nmos.clone(),
        [vmid2, vmid2, NetId::GND, NetId::GND]);

    let mut params = nmos.default_params();
    params.insert(eda_hir::Block::name(&r), 10_000.0_f32);

    let mut bnd1 = HashMap::new();
    bnd1.insert(v_dd, 2.0_f32);
    let mut bnd2 = HashMap::new();
    bnd2.insert(v_dd2, 2.0_f32);

    let dc1 = eda_mna::solve_dc(&c1, &params, &bnd1, NewtonOptions::default());
    let dc2 = eda_mna::solve_dc(&c2, &params, &bnd2, NewtonOptions::default());
    assert!((dc1.voltages[&vmid] - dc2.voltages[&vmid2]).abs() < 1e-6,
        "DC differs: without_caps = {}, with_caps = {}",
        dc1.voltages[&vmid], dc2.voltages[&vmid2]);
}
