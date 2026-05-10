//! `RippleCounter<N>` — N-bit asynchronous ripple counter.
//!
//! Mirrors the counter portion of LTspice paper Fig 12.1. Each stage
//! is a [`spike_cmos_gates::DffSR`] in **toggle mode**: `d` is wired
//! to `qb`, so on each rising clock edge `q` flips. Stages are
//! cascaded: stage `i+1`'s clock is stage `i`'s `q`. Stage 0's clock
//! is the external `clk_in`.
//!
//! ## Frequency division
//!
//! Stage `i` divides `clk_in` by `2^(i+1)`. For a 4 MHz input and
//! N = 4:
//!
//! | Stage | Output | Frequency |
//! | ----- | ------ | --------- |
//! | 0     | `q[0]` | 2.000 MHz |
//! | 1     | `q[1]` | 1.000 MHz |
//! | 2     | `q[2]` | 500 kHz   |
//! | 3     | `q[3]` | 250 kHz   |
//!
//! ## Reset semantics
//!
//! `reset_b` (active low) drives every stage's reset input. When
//! asserted, all `q` outputs go to 0 — the counter restarts at 0000.
//! This is the SAR ADC's "start a new conversion" signal.
//!
//! `set_b` is held high inside the block (each stage's set_b is tied
//! to vdd). If you ever want to preset specific stages, instantiate
//! the underlying `DffSR`s directly.
//!
//! ## Net order
//!
//! `[clk_in, reset_b, q_0, q_1, ..., q_{N-1}, vdd, gnd]`
//! — `N + 4` terminals. The `qb` outputs are exposed as
//! synthesized internal nodes (`<id>_qb_<i>`) so the toggle-mode
//! feedback works without leaking them to the parent.

use eda_spice_emit::{EmitError, Netlist, SpiceEmit};
use spike_cmos_gates::DffSR;

#[derive(Debug, Clone, Copy)]
pub struct RippleCounter<const N: usize = 4> {
    pub stage: DffSR,
}

impl<const N: usize> Default for RippleCounter<N> {
    fn default() -> Self {
        Self { stage: DffSR::default() }
    }
}

impl<const N: usize> SpiceEmit for RippleCounter<N> {
    fn n_terminals(&self) -> usize { N + 4 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != self.n_terminals() {
            return Err(EmitError::ArityMismatch {
                block: format!("RippleCounter<{N}>"),
                expected: self.n_terminals(),
                got: nets.len(),
            });
        }
        let clk_in   = nets[0];
        let reset_b  = nets[1];
        let qs       = &nets[2..2 + N];
        let vdd      = nets[2 + N];
        let gnd      = nets[3 + N];

        // Each stage's qb is private (used as the d input via toggle
        // feedback). set_b for every stage is hardwired to vdd
        // (inactive). reset_b is the shared async clear.
        let qb_nets: Vec<String> = (0..N).map(|i| format!("{id}_qb_{i}")).collect();

        for i in 0..N {
            let stage_clk = if i == 0 { clk_in } else { qs[i - 1] };
            // DffSR net order: [d, clk, set_b, reset_b, q, qb, vdd, gnd]
            // Toggle: d = qb (this stage's own).
            self.stage.emit_spice(
                n,
                &[
                    qb_nets[i].as_str(), // d = qb (toggle feedback)
                    stage_clk,
                    vdd,                 // set_b inactive
                    reset_b,
                    qs[i],
                    qb_nets[i].as_str(),
                    vdd,
                    gnd,
                ],
                &format!("{id}_s{i}"),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_matches() {
        let c4: RippleCounter<4> = RippleCounter::default();
        assert_eq!(c4.n_terminals(), 8);
        let c8: RippleCounter<8> = RippleCounter::default();
        assert_eq!(c8.n_terminals(), 12);
    }

    #[test]
    fn emit_chains_n_dffsr_blocks() {
        let mut net = Netlist::new("t");
        let c: RippleCounter<4> = RippleCounter::default();
        let qs: Vec<String> = (0..4).map(|i| format!("q{i}")).collect();
        let mut nets: Vec<&str> = vec!["clk_in", "rb"];
        nets.extend(qs.iter().map(String::as_str));
        nets.push("vdd");
        nets.push("0");
        c.emit_spice(&mut net, &nets, "rc").unwrap();
        // 4 DffSR × 50 transistors per DffSR = 200 element lines.
        assert_eq!(net.body.len(), 200);
        let body = net.body.join("\n");
        for i in 0..4 {
            assert!(body.contains(&format!("rc_qb_{i}")), "missing rc_qb_{i}");
        }
    }

    #[test]
    fn emit_arity_check() {
        let mut net = Netlist::new("t");
        let c: RippleCounter<4> = RippleCounter::default();
        let err = c.emit_spice(&mut net, &["clk_in"], "rc").unwrap_err();
        assert!(matches!(err, EmitError::ArityMismatch { expected: 8, got: 1, .. }));
    }
}
