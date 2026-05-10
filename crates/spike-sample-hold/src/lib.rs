//! `SampleHold` — CMOS transmission-gate sample-and-hold for the SAR ADC.
//!
//! ## Topology
//!
//! ```text
//!     vin ──┬─[NMOS  drain]
//!           │   gate ──── clk_sh
//!           │   src   ───┐
//!           │             │
//!           ├─[PMOS  drain]
//!           │   gate ──── clk_sh_n  (= !clk_sh)
//!           │   src   ───┤
//!           │             │
//!           └─────────────┴──vhold──[C_hold]──gnd
//! ```
//!
//! When `clk_sh` is high, both transistors conduct (NMOS turns on for
//! the low half of the rail, PMOS for the high half), so `vhold` tracks
//! `vin` through `Ron · C_hold`. When `clk_sh` is low, both transistors
//! are off and the capacitor holds the last sampled value (decay only
//! through SPICE-default subthreshold leakage and the comparator's
//! infinite input impedance — i.e. essentially zero in this model).
//!
//! The Inverter on `clk_sh` produces `clk_sh_n` internally, so the
//! caller only routes the single-rail clock.
//!
//! ## Why a transmission gate, not a single NMOS pass switch
//!
//! A naked NMOS switch suffers a `Vth` drop on the high side: when
//! `vin > vdd − Vth_n`, the NMOS's `Vgs = vdd − vin < Vth_n` and the
//! switch turns off. Output stops tracking ~500 mV below the rail at
//! Vth=0.5 V. The complementary PMOS in a TG covers that high-side
//! gap (and symmetrically the NMOS covers PMOS's low-side `|Vth_p|`
//! drop). Together they pass the full swing.
//!
//! ## Net order
//!
//! `[vin, vhold, clk_sh, vdd, gnd]` — 5 terminals.

use eda_spice_emit::{C, EmitError, Netlist, Nmos, Pmos, SpiceEmit};
use spike_cmos_gates::Inverter;

pub mod mna;

/// Hold capacitor in farads. 100 fF is a textbook SAR-ADC value:
/// big enough that the comparator's input load doesn't disturb the
/// held voltage, small enough that the sample-phase RC time constant
/// (≈ Ron × C ≈ 1 kΩ × 100 fF = 100 ps) is well below the typical
/// 500 ns sample window.
pub const DEFAULT_C_HOLD: f64 = 100e-15;

#[derive(Debug, Clone, Copy)]
pub struct SampleHold {
    pub nmos: Nmos,
    pub pmos: Pmos,
    pub inv:  Inverter,
    pub c_hold: f64,
}

impl Default for SampleHold {
    fn default() -> Self {
        Self {
            nmos: Nmos::default(),
            pmos: Pmos::default(),
            inv:  Inverter::default(),
            c_hold: DEFAULT_C_HOLD,
        }
    }
}

impl SpiceEmit for SampleHold {
    fn n_terminals(&self) -> usize { 5 }

    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != 5 {
            return Err(EmitError::ArityMismatch {
                block: "SampleHold".into(), expected: 5, got: nets.len(),
            });
        }
        let (vin, vhold, clk_sh, vdd, gnd) =
            (nets[0], nets[1], nets[2], nets[3], nets[4]);

        // Inverter on clk_sh → clk_sh_n.
        let clk_sh_n = format!("{id}_clkn");
        self.inv.emit_spice(
            n,
            &[clk_sh, &clk_sh_n, vdd, gnd],
            &format!("{id}_iv"),
        )?;

        // NMOS pass: drain=vin, gate=clk_sh, source=vhold, bulk=gnd.
        // Conducts when clk_sh is high (and vin > Vth_n above source).
        self.nmos.emit_spice(
            n,
            &[vin, clk_sh, vhold, gnd],
            &format!("{id}_n"),
        )?;

        // PMOS pass: drain=vin, gate=clk_sh_n, source=vhold, bulk=vdd.
        // Conducts when clk_sh_n is LOW (i.e. clk_sh is high).
        self.pmos.emit_spice(
            n,
            &[vin, &clk_sh_n, vhold, vdd],
            &format!("{id}_p"),
        )?;

        // Hold cap on vhold.
        C { farads: self.c_hold }.emit_spice(
            n,
            &[vhold, gnd],
            &format!("{id}_hold"),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_is_5() {
        let sh = SampleHold::default();
        assert_eq!(sh.n_terminals(), 5);
    }

    #[test]
    fn arity_mismatch_caught() {
        let mut net = Netlist::new("t");
        let sh = SampleHold::default();
        let err = sh.emit_spice(&mut net, &["vin"], "u1").unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 5, got: 1, .. }));
    }

    #[test]
    fn emit_includes_all_four_devices() {
        let mut net = Netlist::new("t");
        let sh = SampleHold::default();
        sh.emit_spice(
            &mut net,
            &["vin", "vhold", "clk_sh", "vdd", "0"],
            "u1",
        ).unwrap();
        let body = net.body.join("\n");
        // Inverter (2 transistors) + NMOS (1) + PMOS (1) + Capacitor (1) = 5 elements.
        // The Inverter primitive emits `M…_p` and `M…_n` for its devices,
        // so we look for its instance prefix (`u1_iv`).
        assert!(body.contains("u1_iv"),    "missing inverter prefix");
        assert!(body.contains("Mu1_n"),    "missing NMOS pass-gate");
        assert!(body.contains("Mu1_p"),    "missing PMOS pass-gate");
        assert!(body.contains("Cu1_hold"), "missing hold capacitor");
        // clk_sh_n is the internal complement node; should be referenced
        // by the inverter output AND the PMOS gate.
        assert!(body.lines().filter(|l| l.contains("u1_clkn")).count() >= 2,
            "clk_sh_n net should appear in both inverter and PMOS lines");
    }
}
