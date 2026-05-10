//! `OutputDoor` — N-bit parallel-in / parallel-out latch.
//!
//! Mirrors Fig 11.1 of the LTspice SAR ADC paper: N independent
//! [`spike_cmos_gates::Dff`] flip-flops sharing one clock. On each
//! rising clock edge, every input bit is captured into its
//! corresponding output bit, and the captured value holds until the
//! next rising edge.
//!
//! ## Where this fits in the SAR ADC
//!
//! The Output Door sits between the SAR Logic block (whose internal
//! state churns during a conversion) and the outside world. The
//! Clocks circuit produces a once-per-conversion `clk_door` pulse
//! that's high only at the end of a conversion cycle, so the Output
//! Door samples the result *after* it's stable and *before* the SAR
//! Logic resets for the next cycle. Without this latch, downstream
//! consumers would see the SAR's internal binary-search trajectory.
//!
//! ## Net order
//!
//! `[in_0, ..., in_{N-1}, clk, out_0, ..., out_{N-1}, vdd, gnd]`
//! — `2N + 3` terminals. Inputs and outputs are listed in
//! same-index order so a parent block's `for i in 0..N { … }` loop
//! can wire `in_i ↔ out_i` symmetrically.
//!
//! ## qb fan-out
//!
//! Each underlying [`Dff`] has both `q` and `qb`. We expose only `q`
//! at the OutputDoor level (the typical use case); the per-bit `qb`
//! lands on a synthesized internal node `<id>_qb_{i}` and is left
//! floating. That keeps the public terminal count at `2N + 3` instead
//! of `3N + 3`.

use eda_spice_emit::{EmitError, Netlist, SpiceEmit};
use spike_cmos_gates::Dff;

/// Pure-Rust behavioral reference: ideal positive-edge-triggered
/// `N`-bit latch. Returns the new output state given the current
/// inputs, the previous outputs, and whether this tick was a rising
/// clock edge.
///
/// Independent of the SPICE deck, so cross-witness tests can compare
/// ngspice's transient against the truth table without circularity.
/// `inputs` and `prev` are bit-vectors `<N` bits wide; bit `i` of
/// each corresponds to input/output index `i`, matching the net order
/// `[in_0, …, in_{N-1}, …, out_0, …, out_{N-1}, …]`.
pub fn behavioral_capture(inputs: u64, prev: u64, clk_rising: bool) -> u64 {
    if clk_rising { inputs } else { prev }
}

#[derive(Debug, Clone, Copy)]
pub struct OutputDoor<const N: usize = 8> {
    pub dff: Dff,
}

impl<const N: usize> Default for OutputDoor<N> {
    fn default() -> Self {
        Self { dff: Dff::default() }
    }
}

impl<const N: usize> SpiceEmit for OutputDoor<N> {
    fn n_terminals(&self) -> usize { 2 * N + 3 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != self.n_terminals() {
            return Err(EmitError::ArityMismatch {
                block: format!("OutputDoor<{N}>"),
                expected: self.n_terminals(),
                got: nets.len(),
            });
        }
        let inputs  = &nets[..N];
        let clk     = nets[N];
        let outputs = &nets[N + 1..2 * N + 1];
        let vdd     = nets[2 * N + 1];
        let gnd     = nets[2 * N + 2];

        for i in 0..N {
            let qb = format!("{id}_qb_{i}");
            self.dff.emit_spice(
                n,
                &[inputs[i], clk, outputs[i], &qb, vdd, gnd],
                &format!("{id}_b{i}"),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_8_bit_is_19() {
        let d: OutputDoor<8> = OutputDoor::default();
        assert_eq!(d.n_terminals(), 19);
    }

    #[test]
    fn n_terminals_4_bit_is_11() {
        let d: OutputDoor<4> = OutputDoor::default();
        assert_eq!(d.n_terminals(), 11);
    }

    #[test]
    fn emit_produces_n_dff_blocks() {
        let mut net = Netlist::new("t");
        let d: OutputDoor<8> = OutputDoor::default();
        let in_names: Vec<String> = (0..8).map(|i| format!("in{i}")).collect();
        let out_names: Vec<String> = (0..8).map(|i| format!("out{i}")).collect();
        let mut nets: Vec<&str> = in_names.iter().map(String::as_str).collect();
        nets.push("clk");
        nets.extend(out_names.iter().map(String::as_str));
        nets.push("vdd");
        nets.push("0");
        d.emit_spice(&mut net, &nets, "od").unwrap();
        // 8 Dff × 42 transistors per Dff = 336 element lines.
        assert_eq!(net.body.len(), 336);
        let body = net.body.join("\n");
        for i in 0..8 {
            assert!(body.contains(&format!("od_qb_{i}")), "missing od_qb_{i}");
        }
    }

    #[test]
    fn behavioral_capture_holds_when_clock_is_low() {
        let prev = 0xA5u64;
        // Clock low: inputs change, outputs hold.
        assert_eq!(behavioral_capture(0xFF, prev, false), prev);
        assert_eq!(behavioral_capture(0x00, prev, false), prev);
        // Rising edge: outputs follow inputs.
        assert_eq!(behavioral_capture(0xFF, prev, true), 0xFF);
        assert_eq!(behavioral_capture(0x5A, prev, true), 0x5A);
    }

    #[test]
    fn emit_arity_check() {
        let mut net = Netlist::new("t");
        let d: OutputDoor<8> = OutputDoor::default();
        let err = d.emit_spice(&mut net, &["only_one_net"], "od").unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 19, got: 1, .. }));
    }
}
