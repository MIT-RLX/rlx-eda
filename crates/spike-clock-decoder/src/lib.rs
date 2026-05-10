//! `ClockDecoder` — combinational decode of a ripple-counter state to
//! per-cycle SAR-ADC control strobes.
//!
//! ## Where this fits
//!
//! The LTspice paper's "Clocks" block (Fig 12.1) is two halves: a
//! ripple counter (`spike-ripple-counter`) that divides the input
//! clock into binary state bits, and a decoder (this crate) that
//! pattern-matches the counter state to fire individual strobes:
//!
//!   - `clk_sh`    — sample-and-hold capture pulse, asserted at the
//!     start of a new conversion (state `s_sh`).
//!   - `clk_door`  — output-door latch enable, asserted after the
//!     SAR has settled on its final code (state `s_door`).
//!   - `clk_reset` — SAR-register and ripple-counter reset, asserted
//!     to start the next conversion cycle (state `s_reset`).
//!
//! ## Decode topology
//!
//! For an N-bit counter and a target state `s`, we want
//! `strobe = ∏_{i=0..N} (q_i if (s>>i)&1 else !q_i)`. We build that as:
//!
//! 1. For each bit `i` where `(s >> i) & 1 == 0`, an `Inverter` on
//!    `q_i` produces a local complement net. Other bits feed `q_i`
//!    directly.
//! 2. The N selected nets pass through a binary tree of `And2`s to
//!    produce the final strobe.
//!
//! N is a const generic so the decoder is monomorphized per counter
//! width. Tested at N=4 (the "8-bit SAR + 1 sample + 1 latch" layout
//! the paper uses); other widths should work but aren't validated.
//!
//! ## Net order
//!
//! `[q_0, q_1, ..., q_{N-1}, clk_sh, clk_door, clk_reset, vdd, gnd]`
//! — `N + 5` terminals. The first N are the ripple counter's `q`
//! outputs (LSB first); the next three are the strobe outputs; last
//! two are supplies.

use eda_spice_emit::{EmitError, Netlist, SpiceEmit};
use spike_cmos_gates::{And2, Inverter};

/// Counter states (0..2^N) at which each strobe fires. Defaults sized
/// for an 8-bit SAR ADC clocked by a 4-bit ripple counter (16 states,
/// of which 10 are used per conversion: 1 sample + 8 SAR bits + 1
/// latch).
#[derive(Debug, Clone, Copy)]
pub struct DecoderStates {
    pub s_sh:    u32,
    pub s_door:  u32,
    pub s_reset: u32,
}

impl Default for DecoderStates {
    fn default() -> Self { Self { s_sh: 0, s_door: 9, s_reset: 10 } }
}

#[derive(Debug, Clone, Copy)]
pub struct ClockDecoder<const N: usize = 4> {
    pub states: DecoderStates,
    pub inv:    Inverter,
    pub and2:   And2,
}

impl<const N: usize> Default for ClockDecoder<N> {
    fn default() -> Self {
        Self {
            states: DecoderStates::default(),
            inv:    Inverter::default(),
            and2:   And2::default(),
        }
    }
}

impl<const N: usize> ClockDecoder<N> {
    /// `N + 5`: counter `q[0..N-1]`, then `clk_sh`, `clk_door`,
    /// `clk_reset`, then `vdd`, `gnd`.
    pub const N_TERMINALS: usize = N + 5;

    /// Build the AND-tree for a single strobe. May insert intermediate
    /// `Inverter` and `And2` instances into `n`. The final tree output
    /// lands on `out`.
    fn emit_strobe(
        &self,
        n: &mut Netlist,
        qs: &[&str],
        target: u32,
        out: &str,
        vdd: &str,
        gnd: &str,
        id: &str,
    ) -> Result<(), EmitError> {
        assert_eq!(qs.len(), N, "qs slice must have N entries");
        assert!(N >= 1, "ClockDecoder requires N >= 1");

        // Per-bit selected net: q_i (when target bit = 1) or !q_i.
        let mut selected: Vec<String> = Vec::with_capacity(N);
        for i in 0..N {
            if (target >> i) & 1 == 1 {
                selected.push(qs[i].to_string());
            } else {
                let inv_net = format!("{id}_b{i}n");
                self.inv.emit_spice(
                    n,
                    &[qs[i], &inv_net, vdd, gnd],
                    &format!("{id}_inv{i}"),
                )?;
                selected.push(inv_net);
            }
        }

        if N == 1 {
            // Single bit — buffer the selected net (two inverters) so
            // there's a SPICE element on `out` and the polarity matches.
            let mid = format!("{id}_buf_mid");
            self.inv.emit_spice(n, &[&selected[0], &mid, vdd, gnd], &format!("{id}_buf_a"))?;
            self.inv.emit_spice(n, &[&mid, out, vdd, gnd],          &format!("{id}_buf_b"))?;
            return Ok(());
        }

        // Binary And2 tree. Track each layer; the LAST And2 in the
        // last layer writes directly to `out`.
        let mut layer: Vec<String> = selected;
        let mut stage = 0usize;
        while layer.len() > 1 {
            let mut next: Vec<String> = Vec::with_capacity(layer.len().div_ceil(2));
            let mut i = 0usize;
            while i + 1 < layer.len() {
                // Final reduce → write to `out` directly.
                let writes_to_out = layer.len() == 2 && next.is_empty();
                let pair_out: String = if writes_to_out {
                    out.to_string()
                } else {
                    format!("{id}_a{stage}_{i}")
                };
                self.and2.emit_spice(
                    n,
                    &[&layer[i], &layer[i + 1], &pair_out, vdd, gnd],
                    &format!("{id}_and{stage}_{i}"),
                )?;
                next.push(pair_out);
                i += 2;
            }
            if i < layer.len() {
                // Odd one out — carries to next stage.
                next.push(layer[i].clone());
            }
            layer = next;
            stage += 1;
        }
        if layer[0] != out {
            // Tree collapsed to a wire — buffer to `out`.
            let mid = format!("{id}_buf_mid");
            self.inv.emit_spice(n, &[&layer[0], &mid, vdd, gnd], &format!("{id}_buf_a"))?;
            self.inv.emit_spice(n, &[&mid, out, vdd, gnd],        &format!("{id}_buf_b"))?;
        }
        Ok(())
    }
}

impl<const N: usize> SpiceEmit for ClockDecoder<N> {
    fn n_terminals(&self) -> usize { Self::N_TERMINALS }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != Self::N_TERMINALS {
            return Err(EmitError::ArityMismatch {
                block: format!("ClockDecoder<{N}>"),
                expected: Self::N_TERMINALS,
                got: nets.len(),
            });
        }
        let qs   = &nets[0..N];
        let sh   = nets[N];
        let door = nets[N + 1];
        let rst  = nets[N + 2];
        let vdd  = nets[N + 3];
        let gnd  = nets[N + 4];

        self.emit_strobe(n, qs, self.states.s_sh,    sh,   vdd, gnd, &format!("{id}_sh"))?;
        self.emit_strobe(n, qs, self.states.s_door,  door, vdd, gnd, &format!("{id}_dr"))?;
        self.emit_strobe(n, qs, self.states.s_reset, rst,  vdd, gnd, &format!("{id}_rs"))?;
        Ok(())
    }
}

/// Pure-Rust evaluation of one strobe at a given counter state.
/// Useful as the analytic witness alongside ngspice transient tests.
pub fn strobe_active(target: u32, counter_state: u32, n_bits: u32) -> bool {
    let mask = (1u32 << n_bits) - 1;
    (counter_state & mask) == (target & mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_matches() {
        let d4: ClockDecoder<4> = ClockDecoder::default();
        assert_eq!(d4.n_terminals(), 9);  // 4 q + sh + door + rst + vdd + gnd
    }

    #[test]
    fn arity_mismatch_caught() {
        let mut net = Netlist::new("t");
        let d: ClockDecoder<4> = ClockDecoder::default();
        let err = d.emit_spice(&mut net, &["q0", "q1"], "u1").unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 9, got: 2, .. }));
    }

    #[test]
    fn emit_4bit_includes_inverters_and_ands_per_strobe() {
        let mut net = Netlist::new("t");
        let d: ClockDecoder<4> = ClockDecoder::default();
        d.emit_spice(
            &mut net,
            &["q0", "q1", "q2", "q3", "sh", "door", "rst", "vdd", "0"],
            "u1",
        ).unwrap();
        let body = net.body.join("\n");
        for prefix in ["u1_sh", "u1_dr", "u1_rs"] {
            assert!(body.contains(prefix),
                "expected nets with prefix {prefix}, got body:\n{body}");
        }
    }

    #[test]
    fn strobe_active_truth_table() {
        // Default: s_sh=0, s_door=9, s_reset=10.
        assert!( strobe_active(0,  0b0000, 4));
        assert!(!strobe_active(0,  0b0001, 4));
        assert!( strobe_active(9,  0b1001, 4));
        assert!(!strobe_active(9,  0b1000, 4));
        assert!( strobe_active(10, 0b1010, 4));
    }
}
