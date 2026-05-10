//! Phase-1 validation pipe demo: voltage divider through two SPICE
//! backends.
//!
//! This crate is the end-to-end proof that the Phase-1 infrastructure
//! composes:
//!
//! ```text
//!   eda-spice-emit  →  netlist  →  ngspice  →  HashMap<node, V>
//!                              \→  LTspice  →  HashMap<node, V>
//!                                                    │
//!                                          eda-validate::compare_dc_voltages
//!                                                    │
//!                                            DcDiffReport (asserted)
//! ```
//!
//! It exposes a single `divider_deck()` helper so the integration tests
//! and other future spikes can re-use the same circuit.

use eda_spice_emit::{Netlist, R, SpiceEmit};

/// Simple resistive divider: Vin → R1 → mid → R2 → 0.
///
/// Vin is held at `vin_volts` via a DC source. Resistor values default
/// to a 1:3 ratio (R1=10kΩ top, R2=30kΩ bottom → mid = 0.75 · vin).
pub struct Divider {
    pub vin_volts: f64,
    pub r1_ohms: f64,
    pub r2_ohms: f64,
}

impl Default for Divider {
    fn default() -> Self {
        Self { vin_volts: 1.0, r1_ohms: 10e3, r2_ohms: 30e3 }
    }
}

impl Divider {
    /// Closed-form mid-node voltage. Useful as the analytic third witness.
    pub fn mid_voltage(&self) -> f64 {
        self.vin_volts * self.r2_ohms / (self.r1_ohms + self.r2_ohms)
    }

    /// Build the SPICE deck. Output node names: `vin`, `mid`, `0` (ground).
    pub fn deck(&self) -> Netlist {
        let mut net = Netlist::new("rlx-eda phase-1 divider triangulation");
        net.add_dc_source("in", "vin", "0", self.vin_volts);
        R { ohms: self.r1_ohms }.emit_spice(&mut net, &["vin", "mid"], "1").unwrap();
        R { ohms: self.r2_ohms }.emit_spice(&mut net, &["mid", "0"], "2").unwrap();
        net
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_contains_both_resistors_and_source() {
        let d = Divider::default().deck().deck();
        assert!(d.contains("Vin vin 0 DC"));
        assert!(d.contains("R1 vin mid"));
        assert!(d.contains("R2 mid 0"));
    }

    #[test]
    fn mid_voltage_matches_closed_form() {
        let d = Divider { vin_volts: 1.0, r1_ohms: 10e3, r2_ohms: 30e3 };
        assert!((d.mid_voltage() - 0.75).abs() < 1e-12);
    }
}
