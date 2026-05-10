//! Transistor-level CMOS comparator: NMOS differential pair + PMOS
//! current-mirror load + 2-inverter output buffer.
//!
//! Simplified Baker-style two-stage comparator that mirrors LTspice
//! paper Fig 10.1 in spirit but skips the explicit biasing block — we
//! drive the tail current source's gate from an external `vbias` net
//! that the parent block sets to a fixed DC voltage. The full Baker
//! topology with cascode mirrors and a dedicated bias generator is a
//! follow-up; this version validates the diff-pair + mirror pattern
//! end-to-end through ngspice.
//!
//! ## Topology (~9 transistors)
//!
//! ```text
//!                 vdd
//!                  │
//!         ┌────────┴────────┐
//!         │                 │
//!       (M3) PMOS         (M4) PMOS
//!     mirror diode      mirror output
//!     drain=gate        drain → diff_out
//!         │                 │
//!         d1                d2 ─→ INV ─→ INV ─→ vout
//!         │                 │
//!     ┌───┴───┐         ┌───┴───┐
//!   (M1) NMOS        (M2) NMOS
//!   gate=vp          gate=vm
//!     │                 │
//!     └────┬────────────┘
//!          │  (sources tied)
//!         (M_tail) NMOS
//!         gate=vbias
//!          │
//!         gnd
//! ```
//!
//! Stage 1 (M1/M2/M3/M4/M_tail): differential transconductance + active
//! load. Output `d2` swings as `gm·R_out·(vp − vm)` until M2 cuts off
//! or M4 saturates.
//!
//! Stage 2 (two cascaded inverters from `d2` to `vout`): converts the
//! analog stage-1 output to clean rail-to-rail digital levels suitable
//! for the SAR Logic to consume.
//!
//! ## Net order
//!
//! `[vp, vm, vout, vbias, vdd, gnd]` — 6 terminals.
//!
//! `vbias` should be set to ~0.5–0.8 V (above NMOS Vto = 0.5 V) so the
//! tail transistor operates in saturation. ~0.8 V is a good default
//! for our LEVEL=1 model at 1.8 V Vdd.

use eda_spice_emit::{EmitError, Netlist, Nmos, Pmos, SpiceEmit};
use spike_cmos_gates::Inverter;

#[derive(Debug, Clone, Copy)]
pub struct CmosComparator {
    /// Differential-pair NMOS sizing. Wider = more transconductance.
    pub diff_pair: Nmos,
    /// Tail current source. Wider = more bias current.
    pub tail: Nmos,
    /// PMOS current-mirror load.
    pub mirror: Pmos,
    /// Output buffer (2 cascaded inverters).
    pub buffer: Inverter,
}

impl Default for CmosComparator {
    fn default() -> Self {
        // Wider diff pair than the default Nmos for higher gm.
        let diff_pair = Nmos { w: 4.0 * Nmos::default().w, ..Nmos::default() };
        // Tail at 2× default — sets the bias current.
        let tail = Nmos { w: 2.0 * Nmos::default().w, ..Nmos::default() };
        Self {
            diff_pair,
            tail,
            mirror: Pmos::default(),
            buffer: Inverter::default(),
        }
    }
}

impl SpiceEmit for CmosComparator {
    fn n_terminals(&self) -> usize { 6 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != 6 {
            return Err(EmitError::ArityMismatch {
                block: "CmosComparator".into(),
                expected: 6,
                got: nets.len(),
            });
        }
        let (vp, vm, vout, vbias, vdd, gnd) =
            (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5]);

        // Internal nodes:
        //   tail_s : sources of M1/M2 + drain of M_tail
        //   d1     : drain of M1 + drain/gate of M3 (mirror diode)
        //   d2     : drain of M2 + drain of M4 (mirror output, stage 1 out)
        //   int1   : output of first inverter
        let tail_s = format!("{id}_tails");
        let d1 = format!("{id}_d1");
        let d2 = format!("{id}_d2");
        let int1 = format!("{id}_int1");

        // M_tail: drain=tail_s, gate=vbias, source=gnd, bulk=gnd
        self.tail.emit_spice(n, &[&tail_s, vbias, gnd, gnd], &format!("{id}_mt"))?;
        // M1: drain=d1, gate=vp, source=tail_s, bulk=gnd
        self.diff_pair.emit_spice(n, &[&d1, vp, &tail_s, gnd], &format!("{id}_m1"))?;
        // M2: drain=d2, gate=vm, source=tail_s, bulk=gnd
        self.diff_pair.emit_spice(n, &[&d2, vm, &tail_s, gnd], &format!("{id}_m2"))?;
        // M3 (mirror diode): drain=d1, gate=d1, source=vdd, bulk=vdd
        self.mirror.emit_spice(n, &[&d1, &d1, vdd, vdd], &format!("{id}_m3"))?;
        // M4 (mirror output): drain=d2, gate=d1, source=vdd, bulk=vdd
        self.mirror.emit_spice(n, &[&d2, &d1, vdd, vdd], &format!("{id}_m4"))?;

        // Output buffer: 2 cascaded inverters d2 → int1 → vout.
        self.buffer.emit_spice(n, &[&d2, &int1, vdd, gnd], &format!("{id}_iv1"))?;
        self.buffer.emit_spice(n, &[&int1, vout, vdd, gnd], &format!("{id}_iv2"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_is_6() {
        assert_eq!(CmosComparator::default().n_terminals(), 6);
    }

    #[test]
    fn emit_produces_nine_devices() {
        let mut net = Netlist::new("t");
        CmosComparator::default()
            .emit_spice(&mut net, &["vp", "vm", "vout", "vbias", "vdd", "0"], "u1")
            .unwrap();
        // 5 stage-1 transistors (M_tail, M1, M2, M3, M4) + 2 inverters
        // (2 transistors each) = 9.
        assert_eq!(net.body.len(), 9);
        let body = net.body.join("\n");
        for n in ["u1_tails", "u1_d1", "u1_d2", "u1_int1"] {
            assert!(body.contains(n), "missing internal node {n}");
        }
    }

    #[test]
    fn arity_mismatch_caught() {
        let mut net = Netlist::new("t");
        let err = CmosComparator::default()
            .emit_spice(&mut net, &["vp"], "u1")
            .unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 6, got: 1, .. }));
    }
}
