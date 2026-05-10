//! Transient (Backward-Euler) end-to-end tests.
//!
//! Each test exercises a different capacitor topology and matches the
//! framework's BE-step output against a closed-form RC analytic. BE
//! has O(h) error in dt, so we pick `dt ≈ τ/50` and tolerate ~3% in
//! the worst case.

use std::collections::HashMap;
use eda_mna::{transient_from, Circuit, NetId, NewtonOptions};
use spike_divider_block::{Capacitor, Resistor, VoltageSource};

/// **RC charge** through a voltage source.
///
/// ```text
///     V_DD ──[VS=1V]── vplus ──[R]── vmid ──[C]── gnd
/// ```
///
/// Initial state: cap fully discharged → `vmid(0) = 0`.
/// At t > 0: `vmid(t) = V_DD · (1 − exp(−t/τ))`, `τ = R·C`.
#[test]
fn rc_charge_through_voltage_source_matches_analytic() {
    let mut c = Circuit::new();
    let vplus = c.alloc_unknown_net();
    let vmid  = c.alloc_unknown_net();

    let r   = Resistor { length: 10_000, id: "R1".into() };
    let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
    let vs  = VoltageSource::from_volts(1.0, "VS");

    c.add_mna_device(vs.clone(), &[vplus, NetId::GND]);
    c.add_device(r.clone(),      &[vplus, vmid]);
    c.add_storage(cap.clone(),   [vmid,  NetId::GND]);

    assert_eq!(c.n_storage(), 1);

    let r_ohms = 1_000.0_f32;
    let c_farads = 1e-9_f32;     // 1 nF
    let tau = r_ohms * c_farads; // 1 µs

    let mut params = HashMap::new();
    params.insert(eda_hir::Block::name(&r), r_ohms);
    params.insert(format!("{}_C", eda_hir::Block::name(&cap)), c_farads);

    let boundary: HashMap<NetId, f32> = HashMap::new();

    // Explicit IC: cap starts at 0 V (vmid=0). vplus's IC is irrelevant —
    // the solver finds it from KCL each step. Use transient_from so we
    // don't get the DC steady-state (which has the cap already charged).
    let mut ic = HashMap::new();
    ic.insert(vmid,  0.0_f32);
    ic.insert(vplus, 0.0_f32);

    let dt = tau / 50.0;
    let n_steps = 250;
    let waveform = transient_from(&c, &params, &boundary, &ic, dt, n_steps,
                                  NewtonOptions::default());

    // Initial state: cap discharged → vmid(0) ≈ 0.
    let v0 = waveform[0].voltages[&vmid];
    assert!(v0.abs() < 5e-3, "initial vmid = {}, expected ≈0", v0);

    // Mid-charge sample at t = τ: should be 1·(1 − e⁻¹) ≈ 0.6321.
    let idx_at_tau = 50;
    let v_tau = waveform[idx_at_tau].voltages[&vmid];
    let expected_tau = 1.0 - (-1.0_f32).exp();
    assert!((v_tau - expected_tau).abs() < 0.03,
        "vmid(τ) = {}, expected {} (BE drifts low at this dt)", v_tau, expected_tau);

    // Late-time settling: at t = 5τ, vmid should be > 0.99·V_DD.
    let v_late = waveform[n_steps].voltages[&vmid];
    assert!(v_late > 0.99,
        "vmid(5τ) = {}, expected > 0.99 (cap should be fully charged)", v_late);

    // Every step converged.
    for (k, step) in waveform.iter().enumerate() {
        assert!(step.converged, "step {k} did not converge: residual = {:.3e}",
            step.final_residual_max);
    }
}

/// **RC discharge** — cap pre-charged via the boundary, then released.
///
/// ```text
///     V_init (boundary) ──[R]── vmid ──[C]── gnd
/// ```
///
/// At t=0, V_init is held at 1.0 V → DC OP gives `vmid = 1.0` (no
/// current flowing, R has no drop). Then we step V_init down to 0 V
/// and let the cap discharge:  `vmid(t) = V_init · exp(−t/τ)`.
#[test]
fn rc_discharge_into_grounded_source_matches_analytic() {
    let mut c = Circuit::new();
    let v_in = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();

    let r   = Resistor { length: 10_000, id: "R1".into() };
    let cap = Capacitor { plate_size: 2_000, id: "C1".into() };

    c.add_device(r.clone(),    &[v_in, vmid]);
    c.add_storage(cap.clone(), [vmid, NetId::GND]);

    let r_ohms = 1_000.0_f32;
    let c_farads = 1e-9_f32;
    let tau = r_ohms * c_farads;

    let mut params = HashMap::new();
    params.insert(eda_hir::Block::name(&r), r_ohms);
    params.insert(format!("{}_C", eda_hir::Block::name(&cap)), c_farads);

    // Stage 1: pre-charge the cap. v_in held at 1.0 V long enough to
    // settle. We just take the DC OP — that's `vmid = 1.0`.
    let mut boundary = HashMap::new();
    boundary.insert(v_in, 1.0_f32);

    // Confirm DC steady state.
    let dc = eda_mna::solve_dc(&c, &params, &boundary, NewtonOptions::default());
    assert!((dc.voltages[&vmid] - 1.0).abs() < 1e-5);

    // Stage 2: drop v_in to 0 V and run transient. Initial guess for
    // the transient driver is the boundary's *new* DC OP, but `transient`
    // computes it itself — so we just call it with the new boundary.
    boundary.insert(v_in, 0.0_f32);

    // We need the cap to start charged at 1.0 V even though the new DC
    // OP would say 0. Trick: hand-thread the prev_voltages by stepping
    // `solve_be_step` directly.
    let mut prev: HashMap<NetId, f32> = dc.voltages.clone();
    let dt = tau / 50.0;
    let n_steps = 250;
    let mut samples: Vec<(f32, f32)> = Vec::with_capacity(n_steps + 1);
    samples.push((0.0, prev[&vmid]));
    for k in 1..=n_steps {
        let op = eda_mna::solve_be_step(&c, &params, &boundary, &prev,
                                         &[], dt,
                                         NewtonOptions::default());
        assert!(op.converged, "step {k} did not converge: residual = {:.3e}",
            op.final_residual_max);
        let t = k as f32 * dt;
        samples.push((t, op.voltages[&vmid]));
        prev = op.voltages;
    }

    // Sanity: monotone decreasing.
    for w in samples.windows(2) {
        assert!(w[1].1 <= w[0].1 + 1e-6,
            "discharge not monotone: {:?} → {:?}", w[0], w[1]);
    }

    // At t = τ, expect vmid ≈ exp(-1) = 0.3679. BE has order-h error;
    // at dt=τ/50 we expect ~3% absolute slack.
    let (_t_tau, v_tau) = samples[50];
    let expected_tau = (-1.0_f32).exp();
    assert!((v_tau - expected_tau).abs() < 0.03,
        "vmid(τ) = {}, expected {} (RC discharge analytic)", v_tau, expected_tau);

    // At t = 5τ, expect < 0.01.
    let (_t5, v5) = samples[n_steps];
    assert!(v5 < 0.01, "vmid(5τ) = {}, should be ≪ 1 V", v5);
}

#[test]
fn dc_solve_ignores_storage_devices() {
    // Adding a capacitor must not change solve_dc's answer — at DC,
    // caps are open circuits. We verify by solving an RC divider with
    // and without the cap; both should give vmid = V·R2/(R1+R2).
    let r1 = Resistor { length: 10_000, id: "R1".into() };
    let r2 = Resistor { length: 30_000, id: "R2".into() };
    let cap = Capacitor { plate_size: 2_000, id: "C1".into() };

    let mut params = HashMap::new();
    params.insert(eda_hir::Block::name(&r1), 1_000.0_f32);
    params.insert(eda_hir::Block::name(&r2), 3_000.0_f32);
    params.insert(format!("{}_C", eda_hir::Block::name(&cap)), 1e-9_f32);

    let mut without_cap = Circuit::new();
    let v_in = without_cap.alloc_boundary_net();
    let vmid = without_cap.alloc_unknown_net();
    without_cap.add_device(r1.clone(), &[v_in, vmid]);
    without_cap.add_device(r2.clone(), &[vmid, NetId::GND]);

    let mut with_cap = Circuit::new();
    let v_in2 = with_cap.alloc_boundary_net();
    let vmid2 = with_cap.alloc_unknown_net();
    with_cap.add_device(r1.clone(), &[v_in2, vmid2]);
    with_cap.add_device(r2.clone(), &[vmid2, NetId::GND]);
    with_cap.add_storage(cap.clone(), [vmid2, NetId::GND]);

    let mut boundary1 = HashMap::new();
    boundary1.insert(v_in, 1.0_f32);
    let mut boundary2 = HashMap::new();
    boundary2.insert(v_in2, 1.0_f32);

    let dc1 = eda_mna::solve_dc(&without_cap, &params, &boundary1, NewtonOptions::default());
    let dc2 = eda_mna::solve_dc(&with_cap,    &params, &boundary2, NewtonOptions::default());

    assert!((dc1.voltages[&vmid] - 0.75).abs() < 1e-5);
    assert!((dc2.voltages[&vmid2] - 0.75).abs() < 1e-5,
        "cap shifted DC answer: with_cap vmid = {}, without = {}",
        dc2.voltages[&vmid2], dc1.voltages[&vmid]);
}
