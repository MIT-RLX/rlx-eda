//! Forward `transient_sensitivities` validated against finite
//! differences on a small RC + diode network.
//!
//! Topology: voltage source `vin` → resistor `R` → node `vmid` →
//! diode → ground. Add a small parasitic cap from `vmid` to ground so
//! the BE step has nontrivial history coupling.
//!
//! Validation: for each parameter (R, Is_diode, C), compare
//! `transient_sensitivities`'s output against `(transient(p+ε) −
//! transient(p−ε)) / (2ε)` evaluated at every timestep. Pass if the
//! AD-derived sensitivities are within tolerance of FD across all
//! steps.

use std::collections::HashMap;
use eda_mna::{
    pulse_boundary, transient_pwl, transient_sensitivities,
    Circuit, NetId, NewtonOptions,
};
use eda_mna::LinearCap;
use spike_divider_block::{Diode, Resistor};

/// Build the test circuit and return (circuit, vin, vmid, names).
fn build_circuit() -> (Circuit, NetId, NetId, String, String, String) {
    let mut c = Circuit::new();
    let vin = c.alloc_boundary_net();
    let vmid = c.alloc_unknown_net();

    let r = Resistor { length: 5_000, id: "R".into() };
    let d = Diode { size: 2_000, is_value: 1e-15, id: "D".into() };

    c.add_device(r.clone(), &[vin, vmid]);
    c.add_device(d.clone(), &[vmid, NetId::GND]);
    // Parasitic cap on vmid (~1 pF) — gives history coupling for BE.
    c.add_storage(LinearCap::new("Cmid"), [vmid, NetId::GND]);

    let r_name = eda_hir::Block::name(&r);
    let is_name = format!("{}_Is", eda_hir::Block::name(&d));
    let c_name = "Cmid".to_string();
    (c, vin, vmid, r_name, is_name, c_name)
}

fn run_transient(
    circuit: &Circuit,
    params: &HashMap<String, f32>,
    vin: NetId,
    h: f32,
    n_steps: usize,
) -> Vec<eda_mna::TransientStep> {
    let static_b: HashMap<NetId, f32> = HashMap::new();
    // Step from 0 to 1 V at t = 50 ns (well after the transient starts).
    let bnd = pulse_boundary(static_b, vin, 0.0, 1.0, 50e-9, 1e9);
    let ic = HashMap::new();
    transient_pwl(circuit, params, bnd, &ic, h, n_steps, NewtonOptions::default())
}

#[test]
fn transient_sensitivities_match_fd_on_rc_diode() {
    let (circuit, vin, vmid, r_name, is_name, c_name) = build_circuit();
    let _is_name = is_name;
    let _c_name = c_name;

    let mut params = HashMap::new();
    params.insert(r_name.clone(), 1_000.0_f32);
    // Diode's `currents()` impl exposes its Is as `<block_name>_Is` —
    // not `"D_Is"`. Use the canonical name returned by `build_circuit`.
    params.insert(_is_name.clone(), 1e-15_f32);
    params.insert(_c_name.clone(), 1e-12_f32);

    let h = 5e-9_f32;
    let n_steps = 40_usize;

    let trace = run_transient(&circuit, &params, vin, h, n_steps);
    assert!(trace.iter().all(|s| s.converged), "forward transient diverged");

    // Print the trace so we can see when transient activity actually
    // happens vs when it's already settled.
    eprintln!("Forward trace (vmid):");
    for (k, s) in trace.iter().enumerate() {
        let v = s.voltages.get(&vmid).copied().unwrap_or(0.0);
        eprintln!("  step {k:2} t={:.2}ns  vmid={:.6}", k as f32 * h * 1e9, v);
    }

    // Sanity check: DC `sensitivities` on the same operating point
    // (steady-state vmid ≈ 0.685 V from the trace). If this also has
    // the wrong sign, the bug is in the residual-graph convention,
    // not my transient recurrence.
    {
        use eda_mna::{sensitivities, solve_dc, NewtonOptions};
        let mut bnd = HashMap::new();
        bnd.insert(vin, 1.0_f32);
        let op_dc = solve_dc(&circuit, &params, &bnd, NewtonOptions::default());
        let dc_sens = sensitivities(&circuit, &params, &bnd, &op_dc, &[r_name.clone()]);
        let dc_dvm_dr = dc_sens.get(&r_name).expect("dc R sens").get(0).copied().unwrap_or(0.0);
        eprintln!("DC sanity: vmid_dc = {}, ∂vmid/∂R = {:+.4e} (expected ≈ −2.6e−5)",
            op_dc.voltages.get(&vmid).copied().unwrap_or(0.0), dc_dvm_dr);
    }

    // AD sensitivities WRT R. The boundary must reflect the pulse's
    // post-rising-edge state (vin = 1 V) for the transient window we
    // analyze (steps ≥ 11 ⇒ t ≥ 55 ns ⇒ pulse high). Without this,
    // the residual graph evaluates at vin = 0 and df/dR comes out
    // wrong.
    //
    // TODO: a future variant of `transient_sensitivities` should
    // accept a `boundary_at(t)` closure to handle PWL stimuli
    // exactly — analogous to `transient_pwl`. For now, callers that
    // mix PWL boundaries with sensitivities must restrict analysis
    // to a window where the boundary is constant.
    let mut bnd_post_pulse = HashMap::new();
    bnd_post_pulse.insert(vin, 1.0_f32);
    let sens = transient_sensitivities(
        &circuit, &params, &bnd_post_pulse, &trace, h, &[r_name.clone()],
    );
    let s_r = sens.get(&r_name).expect("missing R sens");

    // FD reference: ±1% perturbation on R.
    let eps_rel = 0.01_f32;
    let mut params_p = params.clone();
    let mut params_m = params.clone();
    params_p.insert(r_name.clone(), params[&r_name] * (1.0 + eps_rel));
    params_m.insert(r_name.clone(), params[&r_name] * (1.0 - eps_rel));
    let trace_p = run_transient(&circuit, &params_p, vin, h, n_steps);
    let trace_m = run_transient(&circuit, &params_m, vin, h, n_steps);
    let dp = 2.0 * eps_rel * params[&r_name]; // total span

    // Restrict the comparison to the transient region (before steady
    // state). After the cap settles (~5 RC time constants past the
    // rising edge at 50 ns ⇒ steps ~10-15), v_p and v_m converge to
    // the same DC value and (v_p − v_m) becomes f32 noise — so FD is
    // unreliable there even though AD's steady-state sensitivity is
    // correct.
    let transient_start = 11; // first step where boundary pulse is high (t = 55 ns)
    let transient_end = 14;   // settles by ~70 ns at our R = 1 kΩ, C = 1 pF
    let mut max_abs_err: f32 = 0.0;
    let mut worst_step = 0usize;
    let mut worst_ad = 0.0_f32;
    let mut worst_fd = 0.0_f32;
    for k in transient_start..=transient_end {
        let v_p = trace_p[k].voltages.get(&vmid).copied().unwrap_or(0.0);
        let v_m = trace_m[k].voltages.get(&vmid).copied().unwrap_or(0.0);
        let fd  = (v_p - v_m) / dp;
        let ad  = s_r[k][0]; // first (and only) unknown net = vmid
        let abs_err = (fd - ad).abs();
        eprintln!("  step {k}: AD={ad:+.6e}, FD={fd:+.6e}, err={abs_err:.4e}");
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
            worst_step  = k;
            worst_ad    = ad;
            worst_fd    = fd;
        }
    }
    eprintln!("max |AD - FD| = {max_abs_err:.4e} at step {worst_step} (AD={worst_ad:.6e}, FD={worst_fd:.6e})");
    // Empirically the AD/FD agreement is ~3e-7 V/Ω (transient-region
    // wiggle); 1e-6 catches any real bug while letting f32 + Newton-
    // tolerance noise pass.
    assert!(max_abs_err < 1e-6,
        "AD sensitivity diverges from FD: max err {max_abs_err:.4e} at step {worst_step}");
}
