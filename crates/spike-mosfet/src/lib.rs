//! CMOS inverter as the first MOSFET circuit, mirroring Fig 1.1 of the
//! LTspice SAR ADC paper.
//!
//! Why this circuit, why first:
//!
//! - The inverter is the **simplest non-trivial CMOS circuit** — one
//!   PMOS pull-up, one NMOS pull-down, both gates tied to Vin, both
//!   drains tied to Vout. If this works, NAND, NOR, AND, and the DFFs
//!   built from them in the SAR paper all compose correctly.
//! - Its **voltage transfer characteristic** (VTC) is a textbook
//!   curve: Vout = Vdd for Vin < Vth, ramps through a sharp transition
//!   centered near Vdd/2, then Vout = 0 for Vin > Vdd-Vth. Easy to
//!   validate visually and numerically.
//! - It exercises the LEVEL=1 model in **all three operating regions**
//!   per device — cutoff, saturation, triode — across one Vin sweep.
//!   A single deck triangulated against ngspice and LTspice gives high
//!   confidence that the MOSFET primitives emit correctly.
//!
//! ## Why LEVEL=1, not BSIM
//!
//! The validation goal here is *cross-simulator agreement*, not
//! accuracy against a real PDK. ngspice and LTspice both implement
//! LEVEL=1 to spec; BSIM has version skew. Once the harness is solid,
//! BSIM models drop in with a model-card swap.

use eda_spice_emit::{Netlist, Nmos, Pmos, SpiceEmit};

/// One CMOS inverter spec.
///
/// Defaults: 1.8V supply, NMOS W=10µ L=2µ, PMOS W=20µ L=2µ (the 2:1
/// W ratio compensates for the ~2:1 mobility difference between
/// electrons and holes, putting the switching threshold near Vdd/2).
#[derive(Debug, Clone, Copy)]
pub struct InverterSpec {
    pub vdd: f64,
    pub nmos: Nmos,
    pub pmos: Pmos,
}

impl Default for InverterSpec {
    fn default() -> Self {
        Self {
            vdd: 1.8,
            nmos: Nmos::default(),
            pmos: Pmos::default(),
        }
    }
}

impl InverterSpec {
    /// Build a SPICE deck for one operating point: Vdd held at `vdd`,
    /// Vin held at `vin`, output node `vout` left for `.op` to solve.
    ///
    /// Net layout:
    /// - `vdd`  : supply rail
    /// - `gnd`  : ground (0)
    /// - `vin`  : input gate (both transistors)
    /// - `vout` : output (both drains)
    pub fn deck_at(&self, vin: f64) -> Netlist {
        let mut net = Netlist::new("CMOS inverter (LEVEL=1)");
        net.add_dc_source("dd", "vdd", "0", self.vdd);
        net.add_dc_source("in", "vin", "0", vin);
        // PMOS: drain=vout, gate=vin, source=vdd, bulk=vdd
        self.pmos
            .emit_spice(&mut net, &["vout", "vin", "vdd", "vdd"], "p")
            .unwrap();
        // NMOS: drain=vout, gate=vin, source=0, bulk=0
        self.nmos
            .emit_spice(&mut net, &["vout", "vin", "0", "0"], "n")
            .unwrap();
        net
    }
}

/// A linear sweep of Vin from `0` to `spec.vdd` with `n_points` samples.
/// Returns the `Vin` axis used; intended to feed [`InverterSpec::deck_at`]
/// per-point.
pub fn vin_sweep(spec: &InverterSpec, n_points: usize) -> Vec<f64> {
    (0..n_points)
        .map(|i| spec.vdd * (i as f64) / ((n_points - 1) as f64))
        .collect()
}

/// Closed-form *approximation* of the inverter's switching threshold
/// `Vm` — the point where Vin = Vout. For matched-Vth devices this is
/// where both transistors are saturated; setting Idn = Idp and solving
/// gives:
///
/// `Vm = (Vdd + Vtp + Vtn·sqrt(βn/βp)) / (1 + sqrt(βn/βp))`
///
/// where `βn = Kp_n · Wn / Ln`, `βp = Kp_p · Wp / Lp`, and `Vtp` is
/// negative (PMOS Vto). Useful as the analytic third witness when the
/// SPICE backends agree on the VTC.
pub fn switching_threshold(spec: &InverterSpec) -> f64 {
    let beta_n = spec.nmos.params.kp * spec.nmos.w / spec.nmos.l;
    let beta_p = spec.pmos.params.kp * spec.pmos.w / spec.pmos.l;
    let r = (beta_n / beta_p).sqrt();
    let vtn = spec.nmos.params.vto;
    let vtp = spec.pmos.params.vto; // already negative
    // |Vtp| = -vtp. The closed form uses Vdd + Vtp (which subtracts |Vtp|).
    (spec.vdd + vtp + vtn * r) / (1.0 + r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_contains_both_transistors_and_supplies() {
        let s = InverterSpec::default().deck_at(0.9).deck();
        assert!(s.contains("Vdd vdd 0"));
        assert!(s.contains("Vin vin 0"));
        assert!(s.contains("Mp vout vin vdd vdd"));
        assert!(s.contains("Mn vout vin 0 0"));
    }

    #[test]
    fn switching_threshold_near_half_vdd_for_balanced_inverter() {
        // With NMOS Kp=20µ Wn=10u and PMOS Kp=10µ Wp=20u, βn/βp = 1
        // (KpN·WN = KpP·WP) and |Vtn|=|Vtp|, so Vm ≈ Vdd/2 = 0.9V.
        let vm = switching_threshold(&InverterSpec::default());
        assert!((vm - 0.9).abs() < 1e-9, "Vm = {vm}, expected ≈ 0.9");
    }

    #[test]
    fn vin_sweep_endpoints() {
        let v = vin_sweep(&InverterSpec::default(), 5);
        assert_eq!(v.len(), 5);
        assert!((v[0] - 0.0).abs() < 1e-12);
        assert!((v[4] - 1.8).abs() < 1e-12);
    }
}
