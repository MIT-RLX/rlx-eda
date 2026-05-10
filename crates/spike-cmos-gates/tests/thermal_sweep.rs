//! Thermal-corner sweep on an inverter built with `eda_mna::Circuit` +
//! `add_inverter`, validating the **circuit-level** path (not just the
//! single-device `spike-mosfet-dc` witness).
//!
//! The remap entry point under test is
//! `spike_divider_block::thermal::remap_mosfet_params_for_temp`. It
//! walks the params HashMap built by the gate constructors and shifts
//! every `*_Vth` / `*_Kp` entry through the same `vth_at_temp` /
//! `kp_at_temp` formulas the leaf model uses, so a circuit-level
//! corner sweep is "build-once, remap-many" with no graph rebuild.
//!
//! ## Why transient, not solve_dc
//!
//! `solve_dc` on this inverter is degenerate at extreme inputs (NMOS
//! off + PMOS off-rail = near-singular Jacobian, Newton blows up). The
//! validated path that all the digital_primitives_mna gates use is
//! `transient_pwl` with cap-stabilized BE steps — same path here, just
//! with a constant-in-time boundary so the trace settles to steady
//! state. Read Vout at the final step.
//!
//! ## Signal we use
//!
//! Drive `Vin ≈ V_M_nominal` (the inverter's switching threshold at
//! Tnom). With β_p = 2·β_n from the 4 µm:2 µm PMOS:NMOS sizing,
//! V_M_nom ≈ 1.054 - 0.172·|Vth| ≈ 0.97 V. At Vin = V_M, Vout sits
//! mid-rail and the small-signal gain `dVout/dVin` peaks (~5–10 in
//! this model). KT1 = -1 mV/°C shifts both Vth_n and |Vth_p| by ±67 mV
//! across {−40, 27, 125} °C, which moves V_M by ~30 mV (the beta-
//! asymmetry partially cancels), and that 30 mV gets multiplied by
//! the gain into a Vout swing of hundreds of mV.
//!
//! Direction: hotter T shrinks |Vth| symmetrically, but PMOS β
//! dominates (2× wider) so the PMOS overdrive grows faster in
//! relative terms — Vout rises with T. We don't lock the sign in the
//! assertion, just monotonicity.

use std::collections::HashMap;

use eda_mna::{transient_pwl, Circuit, NetId, NewtonOptions};
use spike_cmos_gates::mna::add_inverter;
use spike_divider_block::thermal::remap_mosfet_params_for_temp;

const VDD: f32 = 1.8;
/// Nominal switching threshold for the 4 µm:2 µm PMOS:NMOS inverter:
/// `V_M ≈ (r·Vdd + Vt·(1−r)) / (1+r)` with `r = √(β_p/β_n) = √2` and
/// `Vt = 0.5` → V_M ≈ 0.97 V. Small offsets shift Vout by 5–10× the
/// V_M shift, so this bias maximizes corner leverage.
const VIN_AT_VM: f32 = 0.97;

const H_NS:        f32   = 1e-9; // 1 ns BE step
const N_SETTLE:    usize = 50;   // 50 ns total — slew is ~200 ps, plenty of margin

/// Build a fresh inverter circuit + nominal-T params. Returned in shape
/// `(circuit, params, in_net, out_net, vdd_net)` so each corner test
/// can clone-and-remap params instead of rebuilding the circuit.
fn build_inverter() -> (Circuit, HashMap<String, f32>, NetId, NetId, NetId) {
    let mut c = Circuit::new();
    let mut params = HashMap::new();
    let vdd = c.alloc_boundary_net();
    let in_ = c.alloc_boundary_net();
    let out = c.alloc_unknown_net();
    add_inverter(&mut c, [in_, out, vdd, NetId::GND], "iv", &mut params);
    (c, params, in_, out, vdd)
}

/// Settle the inverter under constant `(vdd, vin)` and return Vout at
/// steady state. Uses transient_pwl + cap-stabilized BE so we don't
/// hit `solve_dc`'s near-singular Jacobian on near-rail inputs.
fn settled_vout(
    c: &Circuit, params: &HashMap<String, f32>,
    in_net: NetId, vdd_net: NetId, out_net: NetId,
    vin: f32,
) -> f32 {
    let bnd = move |_t: f32| {
        let mut m = HashMap::new();
        m.insert(vdd_net, VDD);
        m.insert(in_net, vin);
        m
    };
    let mut ic = HashMap::new();
    ic.insert(out_net, VDD * 0.5);  // mid-rail seed
    let trace = transient_pwl(c, params, bnd, &ic, H_NS, N_SETTLE, NewtonOptions::default());
    *trace.last().unwrap().voltages.get(&out_net)
        .expect("output net missing from final trace step")
}

#[test]
fn rails_hold_at_every_corner() {
    let (c, params0, in_net, out_net, vdd_net) = build_inverter();
    for &t in &[-40.0_f64, 27.0, 125.0] {
        let mut params = params0.clone();
        remap_mosfet_params_for_temp(&mut params, t);

        // Vin = 0 → NMOS off, PMOS on → Vout = Vdd
        let v_hi = settled_vout(&c, &params, in_net, vdd_net, out_net, 0.0);
        assert!((v_hi - VDD).abs() < 0.10,
            "T={t}°C Vin=0: Vout={v_hi:.4}, expected ≈ {VDD}");

        // Vin = Vdd → PMOS off, NMOS on → Vout ≈ 0
        let v_lo = settled_vout(&c, &params, in_net, vdd_net, out_net, VDD);
        assert!(v_lo.abs() < 0.10,
            "T={t}°C Vin=Vdd: Vout={v_lo:.4}, expected ≈ 0");
    }
}

#[test]
fn switching_threshold_shifts_with_temperature() {
    // At Vin ≈ V_M_nominal, the inverter sits in its high-gain region
    // and a small T-induced V_M shift produces a large Vout shift.
    // We assert:
    //   1. monotonicity in T (sign agnostic — beta-ratio decides
    //      whether Vout climbs or drops),
    //   2. magnitude > 50 mV between corners (rules out no-op),
    //   3. all corners stay inside the rails.

    let (c, params0, in_net, out_net, vdd_net) = build_inverter();
    let mut vouts = Vec::new();
    for &t in &[-40.0_f64, 27.0, 125.0] {
        let mut params = params0.clone();
        remap_mosfet_params_for_temp(&mut params, t);
        let v = settled_vout(&c, &params, in_net, vdd_net, out_net, VIN_AT_VM);
        eprintln!("T={t:>5.1}°C  Vin={VIN_AT_VM}  Vout={v:.4}");
        vouts.push(v);
    }
    let (cold, nominal, hot) = (vouts[0], vouts[1], vouts[2]);

    let increasing = cold < nominal && nominal < hot;
    let decreasing = cold > nominal && nominal > hot;
    assert!(increasing || decreasing,
        "Vout should be monotonic in T: cold={cold:.4}, nominal={nominal:.4}, hot={hot:.4}");

    assert!((cold - hot).abs() > 0.05,
        "T-shift was suspiciously small: cold={cold:.4} V, hot={hot:.4} V");

    for v in &vouts {
        assert!(*v >= -0.05 && *v <= VDD + 0.05,
            "Vout out of rails: {v}");
    }
}
