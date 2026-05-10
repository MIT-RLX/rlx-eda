//! 8-bit R-2R DAC, mirroring Fig 8.1 of the LTspice SAR ADC paper.
//!
//! ## Topology
//!
//! ```text
//!                                                      vout (= n_{N-1})
//!   in_0 ─2R─┐                                           │
//!            │                                          MSB
//!            n_0 ──R── n_1 ──R── n_2 ── ... ──R── n_{N-1}
//!            │         │         │                  │
//!   VLOW ─2R─┘  in_1─2R┘  in_2─2R┘   ...   in_{N-1}─2R┘
//!            ↑                                       ↑
//!           LSB end                              MSB end
//! ```
//!
//! Each input `in_i` (0 = LSB, N-1 = MSB) feeds its node `n_i`
//! through a 2R resistor (default 20 kΩ). Adjacent nodes are joined
//! by an R resistor (default 10 kΩ). The LSB end (`n_0`) is also
//! tied to `vlow` through a 2R termination so the network looks the
//! same from every bit's perspective. The MSB-end node `n_{N-1}` is
//! the DAC output.
//!
//! ## Output formula
//!
//! For an *unloaded* output (high-impedance probe):
//!
//! ```text
//!   vout = (code / 2^N) · (vref - vlow) + vlow
//! ```
//!
//! where `code = sum(in_i · 2^i)` taking each input as `0` (= vlow)
//! or `1` (= vref). The paper's Fig 8.1 example: code=10100100b=164,
//! vref=1V, vlow=0V → vout = 164/256 = 0.6406 V. Reproduced exactly
//! by the validation tests.
//!
//! ## Net order
//!
//! `[in_0, in_1, ..., in_{N-1}, vlow, vout]` — N+2 terminals.

use eda_spice_emit::{EmitError, Netlist, SpiceEmit, R};

pub mod mna;

/// 8-bit R-2R ladder. The const-generic `N` makes a 6-bit or 12-bit
/// variant a one-line change; defaults to 8 to match the SAR paper.
#[derive(Debug, Clone, Copy)]
pub struct R2RDac<const N: usize = 8> {
    /// "R" value (the 1× resistor in the spine). Default 10 kΩ.
    pub r_ohms: f64,
}

impl<const N: usize> Default for R2RDac<N> {
    fn default() -> Self {
        Self { r_ohms: 10e3 }
    }
}

impl<const N: usize> SpiceEmit for R2RDac<N> {
    fn n_terminals(&self) -> usize { N + 2 }
    fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
        if nets.len() != N + 2 {
            return Err(EmitError::ArityMismatch {
                block: format!("R2RDac<{N}>"),
                expected: N + 2,
                got: nets.len(),
            });
        }
        let inputs = &nets[..N];
        let vlow   = nets[N];
        let vout   = nets[N + 1];

        let r1 = R { ohms: self.r_ohms };       // R   = 10 kΩ
        let r2 = R { ohms: 2.0 * self.r_ohms }; // 2R  = 20 kΩ

        // Per-stage internal node names. n_{N-1} == vout, so we re-use
        // the caller's vout net for the MSB-end node and synthesize
        // names for the others.
        let internal_owned: Vec<String> = (0..N - 1).map(|i| format!("{id}_n{i}")).collect();
        let node = |i: usize| -> &str {
            if i == N - 1 { vout } else { internal_owned[i].as_str() }
        };

        // 2R termination at the LSB end: vlow -- 2R -- n_0
        r2.emit_spice(n, &[vlow, node(0)], &format!("{id}_term"))?;

        // Per-bit input feeders: in_i -- 2R -- n_i
        for i in 0..N {
            r2.emit_spice(n, &[inputs[i], node(i)], &format!("{id}_in{i}"))?;
        }

        // Spine: n_i -- R -- n_{i+1} for i = 0..N-1 (i.e. N-1 spans)
        for i in 0..N - 1 {
            r1.emit_spice(n, &[node(i), node(i + 1)], &format!("{id}_sp{i}"))?;
        }
        Ok(())
    }
}

/// Closed-form ideal DAC output. Useful as the analytic third witness
/// when SPICE backends agree.
///
/// `code` is the unsigned binary value formed by the input bits with
/// `in_i` weighted at `2^i`. `vref` is the input "1" voltage; `vlow`
/// is the input "0" voltage and the ladder's bottom termination.
pub fn ideal_vout(code: u32, n_bits: u32, vref: f64, vlow: f64) -> f64 {
    let max = (1u64 << n_bits) as f64;
    vlow + (code as f64 / max) * (vref - vlow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terminals_matches_const_n() {
        let d8: R2RDac<8> = R2RDac::default();
        assert_eq!(d8.n_terminals(), 10);
        let d6: R2RDac<6> = R2RDac::default();
        assert_eq!(d6.n_terminals(), 8);
    }

    #[test]
    fn ideal_matches_paper_example() {
        // Fig 8.1 caption: 10100100(2) = 164(10), 164/2^8 = 0.6406.
        let v = ideal_vout(164, 8, 1.0, 0.0);
        assert!((v - 0.640625).abs() < 1e-9);
    }

    #[test]
    fn emit_produces_expected_resistor_count_and_internal_nodes() {
        let mut net = Netlist::new("t");
        let dac: R2RDac<8> = R2RDac::default();
        let nets: Vec<String> = (0..8).map(|i| format!("in{i}")).collect();
        let mut nets_refs: Vec<&str> = nets.iter().map(String::as_str).collect();
        nets_refs.push("vlow");
        nets_refs.push("vout");
        dac.emit_spice(&mut net, &nets_refs, "u1").unwrap();
        // 1 termination + 8 input feeders + 7 spine = 16 resistors.
        assert_eq!(net.body.len(), 16);
        let body = net.body.join("\n");
        // Internal nodes for n_0..n_6 (n_7 reuses vout).
        for i in 0..7 {
            assert!(body.contains(&format!("u1_n{i}")), "missing u1_n{i}");
        }
        // No spurious n_7 node — n_7 is collapsed to vout.
        assert!(!body.contains("u1_n7"), "n_7 should be vout, not a fresh node");
    }

    #[test]
    fn emit_arity_check() {
        let mut net = Netlist::new("t");
        let dac: R2RDac<8> = R2RDac::default();
        let err = dac.emit_spice(&mut net, &["in0"], "u1").unwrap_err();
        match err {
            EmitError::ArityMismatch { expected: 10, got: 1, .. } => {}
            other => panic!("wrong err: {other:?}"),
        }
    }
}
