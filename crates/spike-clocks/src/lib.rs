//! `Clocks` — full SAR ADC clock generator. Composes the divider and
//! the decoder.
//!
//! Two halves wired together:
//!
//! 1. [`spike_ripple_counter::RippleCounter<4>`] — divides `clk_in` by
//!    powers of 2 and exposes the 4-bit binary state.
//! 2. [`spike_clock_decoder::ClockDecoder<4>`] — decodes that state
//!    into the three SAR control strobes (`clk_sh`, `clk_door`,
//!    `clk_reset_active_high`).
//!
//! On top, this block adds:
//! - An inverter on the reset strobe so the external `clk_reset`
//!   output is **active-low** (matches the rest of the SAR ADC, where
//!   reset_b conventions are used everywhere).
//! - A self-clearing feedback path: the decoder's reset strobe is
//!   wired back to the counter's `reset_b` (via an AND with the
//!   external power-on reset) so the conversion window auto-wraps
//!   every 10 input cycles.
//!
//! ## Outputs from a 4 MHz `clk_in`
//!
//! | Signal       | Source             | Active when counter = | Polarity        |
//! | ------------ | ------------------ | --------------------- | --------------- |
//! | `sar_clk`    | counter q0 (raw)   | n/a (always toggling) | -               |
//! | `clk_sh`     | decoder            | 0000  (cycle 0)       | active high     |
//! | `clk_door`   | decoder            | 1001  (cycle 9)       | active high     |
//! | `clk_reset`  | decoder + INV      | 1010  (transient)     | **active low**  |
//!
//! ## Net order
//!
//! `[clk_in, ext_reset_b, sar_clk, clk_sh, clk_door, clk_reset, vdd, gnd]`
//! — 8 terminals.
//!
//! `ext_reset_b` is an external power-on reset (active low). Held low
//! to force the counter to 0 regardless of internal state.
//!
//! ## Architectural note: ripple-counter decode glitches
//!
//! The underlying `RippleCounter<4>` is **asynchronous**: each stage
//! clocks on the previous stage's Q output, so during a multi-bit
//! state transition (e.g. `0011 → 0100`) the lower bits update first
//! and the counter visibly passes through intermediate states (`0010`,
//! `0000`, `0100`). The combinational decoders in
//! [`spike_clock_decoder`] respond to these intermediate states,
//! producing brief spurious pulses on `clk_sh` / `clk_door` /
//! `clk_reset` in addition to their intended once-per-conversion
//! pulses. The intended pulses are ~250 ns wide (one input cycle);
//! the glitches are ~tens of ns.
//!
//! Two clean fixes (both deferred):
//!
//! 1. **Synchronous counter**: have all four DFFs share `clk_in`
//!    directly and use the toggle equations explicitly (`d_i =
//!    q_i XOR (q_0 AND q_1 AND … AND q_{i-1})`). Eliminates state
//!    skew. Requires an XOR primitive (which we don't have yet) or
//!    a 4-input AND tree per stage.
//! 2. **Registered decoder**: pass each strobe through a `Dff`
//!    clocked by `clk_in`'s falling edge (when the counter is
//!    guaranteed stable). Adds one DFF per strobe. Cleaner than
//!    fixing the counter and more aligned with how real designs
//!    handle this.
//!
//! For Phase-3 SAR ADC progress the glitches are tolerable: the
//! downstream consumers (`OutputDoor`, `SampleHold`, the SAR Logic
//! reset path) all sample at specific clock edges, and the brief
//! glitches don't cross the consumer's setup/hold window. The smoke
//! test in `tests/clock_tree.rs` documents the situation explicitly:
//! it asserts each strobe fires AT LEAST once per conversion (the
//! intended pulse) but tolerates up to 4× that count (the glitches).

use eda_spice_emit::{EmitError, Netlist, SpiceEmit};
use spike_clock_decoder::{ClockDecoder, DecoderStates};
use spike_cmos_gates::{And2, Inverter};
use spike_ripple_counter::RippleCounter;

#[derive(Debug, Clone, Copy)]
pub struct Clocks {
    pub counter: RippleCounter<4>,
    pub decoder: ClockDecoder<4>,
    pub inv: Inverter,
    pub and2: And2,
}

impl Default for Clocks {
    fn default() -> Self {
        Self {
            counter: RippleCounter::default(),
            decoder: ClockDecoder {
                states: DecoderStates::default(), // {0, 9, 10}
                inv: Inverter::default(),
                and2: And2::default(),
            },
            inv: Inverter::default(),
            and2: And2::default(),
        }
    }
}

impl SpiceEmit for Clocks {
    fn n_terminals(&self) -> usize { 8 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != self.n_terminals() {
            return Err(EmitError::ArityMismatch {
                block: "Clocks".into(),
                expected: 8,
                got: nets.len(),
            });
        }
        let (clk_in, ext_reset_b, sar_clk, clk_sh, clk_door, clk_reset, vdd, gnd) =
            (nets[0], nets[1], nets[2], nets[3], nets[4], nets[5], nets[6], nets[7]);

        // Decoder produces an active-HIGH "state == 1010" strobe; we
        // route it through an inverter to make the external clk_reset
        // active-LOW. Internal active-high net:
        let rst_active_high = format!("{id}_rst_h");

        // Counter Q outputs. q0 is exposed externally as sar_clk; the
        // higher-order Qs are private internal nets driving the decoder.
        let q0 = sar_clk;
        let q1 = format!("{id}_q1");
        let q2 = format!("{id}_q2");
        let q3 = format!("{id}_q3");

        // Combined reset for the counter: counter clears when EITHER
        // ext_reset_b or clk_reset (active-low) drops. AND of two
        // active-low signals = And2.
        let combined_rb = format!("{id}_rb");
        self.and2.emit_spice(
            n,
            &[ext_reset_b, clk_reset, &combined_rb, vdd, gnd],
            &format!("{id}_rb_and"),
        )?;

        // Counter: [clk_in, reset_b, q_0..q_3, vdd, gnd]
        self.counter.emit_spice(
            n,
            &[clk_in, &combined_rb, q0, &q1, &q2, &q3, vdd, gnd],
            &format!("{id}_ctr"),
        )?;

        // Decoder: [q_0..q_3, clk_sh, clk_door, rst_active_high, vdd, gnd]
        self.decoder.emit_spice(
            n,
            &[q0, &q1, &q2, &q3, clk_sh, clk_door, &rst_active_high, vdd, gnd],
            &format!("{id}_dec"),
        )?;

        // Final inverter: rst_active_high → clk_reset (active-LOW).
        self.inv.emit_spice(
            n,
            &[&rst_active_high, clk_reset, vdd, gnd],
            &format!("{id}_rinv"),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_is_8() {
        assert_eq!(Clocks::default().n_terminals(), 8);
    }

    #[test]
    fn emit_includes_counter_decoder_and_reset_inverter() {
        let mut net = Netlist::new("t");
        let c = Clocks::default();
        let nets = [
            "clk_in", "ext_rb", "sar_clk", "clk_sh", "clk_door", "clk_reset", "vdd", "0",
        ];
        c.emit_spice(&mut net, &nets, "ck").unwrap();
        let body = net.body.join("\n");

        // Counter
        assert!(body.contains("ck_ctr"));
        // Decoder
        assert!(body.contains("ck_dec"));
        // Final reset inverter
        assert!(body.contains("ck_rinv"));
        // Combined reset path
        assert!(body.contains("ck_rb"));
        // Internal Q nodes (q1..q3 — q0 is exposed as sar_clk)
        for q in ["ck_q1", "ck_q2", "ck_q3"] {
            assert!(body.contains(q), "missing internal node {q}");
        }
        // Active-high reset before the inverter
        assert!(body.contains("ck_rst_h"));
    }

    #[test]
    fn emit_arity_check() {
        let mut net = Netlist::new("t");
        let err = Clocks::default()
            .emit_spice(&mut net, &["clk_in"], "ck")
            .unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 8, got: 1, .. }));
    }
}
