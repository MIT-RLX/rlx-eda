//! `SarAdc<N>` — closed-loop top-level N-bit SAR ADC.
//!
//! Composes the analog and digital halves of the SAR ADC into a
//! single `SpiceEmit` block. The clock-generation network (ripple
//! counter + decoder) is **out of scope** for this crate — those are
//! validated independently in `spike-ripple-counter` and
//! `spike-clock-decoder`. Here, the caller drives the per-bit phase
//! signals + sample-and-hold strobe + reset directly (PWL in tests,
//! eventually wired up to the clock network in a higher integration).
//!
//! ## Block diagram
//!
//! ```text
//!     vin ──► SampleHold ──► vhold ──┬───────────►  Comparator(v+, v−) ──► cmp
//!              ▲                     │              ▲
//!              clk_sh                │              │
//!                                    │              │
//!     phase[0..N-1] ──► SarRegister  │              │
//!     phase_done    ─►   bit[0..N-1] ┴───► R2RDac ──┘
//!     reset_b       ─►                       (v_dac)
//! ```
//!
//! `bit[0..N-1]` are exposed as outputs so the caller can read the
//! converted code at the end of the conversion (or wire them into a
//! `spike-output-door` for a synchronous capture).
//!
//! ## Comparator inline
//!
//! Rather than instantiating `spike-comparator::spice_deck` (which
//! includes its own voltage sources for `v+`/`v−`), we emit a single
//! `B` element directly: `B_cmp cmp 0 V=<closed-form>`. That keeps
//! the SAR ADC's net interface clean — `vhold` and `v_dac` flow into
//! the comparator without an extra source layer.
//!
//! ## Net order
//!
//! `[vin,
//!   phase_0,   phase_1,   ..., phase_{N-1},
//!   capture_0, capture_1, ..., capture_{N-1},
//!   clk_sh, reset_b,
//!   bit_0,     bit_1,     ..., bit_{N-1},
//!   vdd, gnd]`
//!
//! Total: `3N + 5` terminals. Phases and captures are paired per-bit
//! (caller drives `phase[i]` first to set the trial bit, then a clean
//! `capture[i]` pulse after the comparator settles to latch the
//! decision into `bit[i]`).

use eda_spice_emit::{EmitError, Netlist, SpiceEmit};
use spike_comparator::Comparator;
use spike_dac_r2r::R2RDac;
use spike_sample_hold::SampleHold;
use spike_sar_register::SarRegister;

#[derive(Debug, Clone, Copy)]
pub struct SarAdc<const N: usize = 4> {
    pub sh:   SampleHold,
    pub sar:  SarRegister<N>,
    pub dac:  R2RDac<N>,
    pub comp: Comparator,
}

impl<const N: usize> Default for SarAdc<N> {
    fn default() -> Self {
        Self {
            sh:   SampleHold::default(),
            sar:  SarRegister::<N>::default(),
            dac:  R2RDac::<N>::default(),
            comp: Comparator::default(),
        }
    }
}

impl<const N: usize> SarAdc<N> {
    pub const N_TERMINALS: usize = 3 * N + 5;
}

impl<const N: usize> SpiceEmit for SarAdc<N> {
    fn n_terminals(&self) -> usize { Self::N_TERMINALS }

    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != Self::N_TERMINALS {
            return Err(EmitError::ArityMismatch {
                block: format!("SarAdc<{N}>"),
                expected: Self::N_TERMINALS,
                got: nets.len(),
            });
        }

        let vin      = nets[0];
        let phases   = &nets[1..1 + N];
        let captures = &nets[1 + N..1 + 2 * N];
        let clk_sh   = nets[1 + 2 * N];
        let reset_b  = nets[2 + 2 * N];
        let bits     = &nets[3 + 2 * N..3 + 3 * N];
        let vdd      = nets[3 + 3 * N];
        let gnd      = nets[4 + 3 * N];

        // Internal nets.
        let vhold = format!("{id}_vhold");
        let v_dac = format!("{id}_vdac");
        let cmp   = format!("{id}_cmp");

        // S/H: [vin, vhold, clk_sh, vdd, gnd]
        self.sh.emit_spice(
            n,
            &[vin, &vhold, clk_sh, vdd, gnd],
            &format!("{id}_sh"),
        )?;

        // SAR register: phase_0..N-1, capture_0..N-1, cmp, reset_b,
        //                bit_0..N-1, vdd, gnd  — 3N + 4 nets.
        let mut sar_nets: Vec<&str> = Vec::with_capacity(3 * N + 4);
        for p in phases   { sar_nets.push(*p); }
        for c in captures { sar_nets.push(*c); }
        sar_nets.push(&cmp);
        sar_nets.push(reset_b);
        for b in bits { sar_nets.push(*b); }
        sar_nets.push(vdd);
        sar_nets.push(gnd);
        self.sar.emit_spice(n, &sar_nets, &format!("{id}_sar"))?;

        // R-2R DAC: [bit_0..bit_{N-1}, vlow, vout].
        // vlow tied to gnd; vref is implicitly the rail the bit
        // inputs swing on (vdd, since SAR bits drive vdd or 0).
        let mut dac_nets: Vec<&str> = Vec::with_capacity(N + 2);
        for b in bits { dac_nets.push(*b); }
        dac_nets.push(gnd);    // vlow
        dac_nets.push(&v_dac); // vout
        self.dac.emit_spice(n, &dac_nets, &format!("{id}_dac"))?;

        // Behavioral comparator as a single B-source. v+ = vhold,
        // v− = v_dac. cmp lands on the SAR's `d` input wired above.
        n.add_element(format!(
            "B{id}_cmp {cmp} 0 V={vol} + ({voh}-{vol})*0.5*(1+tanh({k}*(v({vhold})-v({v_dac}))))",
            vol = self.comp.vol, voh = self.comp.voh, k = self.comp.k,
        ));

        Ok(())
    }
}

/// Pure-Rust algorithm reference. Re-exported from `spike-sar-register`.
pub fn ideal_sar_code(vin: f64, vref: f64, n_bits: usize) -> u32 {
    spike_sar_register::ideal_sar_code(vin, vref, n_bits)
}

pub mod behavioral;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_matches() {
        let a4: SarAdc<4> = SarAdc::default();
        assert_eq!(a4.n_terminals(), 17);  // 3·4 + 5
        let a8: SarAdc<8> = SarAdc::default();
        assert_eq!(a8.n_terminals(), 29);
    }

    #[test]
    fn arity_mismatch_caught() {
        let mut net = Netlist::new("t");
        let a: SarAdc<4> = SarAdc::default();
        let err = a.emit_spice(&mut net, &["vin"], "u1").unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 17, got: 1, .. }));
    }

    #[test]
    fn emit_includes_all_subblocks() {
        let mut net = Netlist::new("t");
        let a: SarAdc<4> = SarAdc::default();
        let nets = vec![
            "vin",
            "p0", "p1", "p2", "p3",
            "c0", "c1", "c2", "c3",
            "clk_sh", "rb",
            "b0", "b1", "b2", "b3",
            "vdd", "0",
        ];
        a.emit_spice(&mut net, &nets, "u1").unwrap();
        let body = net.body.join("\n");
        assert!(body.contains("u1_sh"),   "missing SampleHold prefix");
        assert!(body.contains("u1_sar"),  "missing SarRegister prefix");
        assert!(body.contains("u1_dac"),  "missing R2RDac prefix");
        assert!(body.contains("Bu1_cmp"), "missing comparator B-source");
    }

    #[test]
    fn ideal_sar_code_4bit_quantization() {
        let vref = 1.8_f64;
        let cases = &[
            (0.0,    0u32),
            (0.05,   0),
            (0.20,   1),
            (0.95,   8),
            (1.40,  12),
            (1.799, 15),
        ];
        for &(vin, expected) in cases {
            let code = ideal_sar_code(vin, vref, 4);
            assert_eq!(code, expected,
                "vin = {vin} V → got code {code}, expected {expected}");
        }
    }
}
