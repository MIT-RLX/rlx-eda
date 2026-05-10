//! Value-bearing SPICE primitives: `R`, `C`, `D`, `Nmos`, `Pmos`.
//!
//! These are deliberately tiny — they own the *numeric values* needed to
//! emit a SPICE element line and nothing else. They're complementary to
//! the layout-side `Resistor` / `Capacitor` / `Diode` types in spike
//! crates: layout types own geometry; these own the SPICE numbers.
//!
//! The validation harness uses these to build cross-simulator decks
//! without coupling SPICE emission to klayout-rs geometry. When a layout
//! block needs to emit SPICE, it constructs the matching primitive with
//! its current parameter value and delegates.

use crate::{EmitError, Netlist, SpiceEmit};

/// Linear resistor — `Rname p n <ohms>`.
#[derive(Debug, Clone, Copy)]
pub struct R {
    pub ohms: f64,
}

impl SpiceEmit for R {
    fn n_terminals(&self) -> usize { 2 }
    fn emit_spice(
        &self,
        n: &mut Netlist,
        nets: &[&str],
        id: &str,
    ) -> Result<(), EmitError> {
        if nets.len() != 2 {
            return Err(EmitError::ArityMismatch {
                block: "R".into(), expected: 2, got: nets.len(),
            });
        }
        n.add_element(format!("R{id} {} {} {:.10e}", nets[0], nets[1], self.ohms));
        Ok(())
    }
}

/// Linear capacitor — `Cname p n <farads>`.
#[derive(Debug, Clone, Copy)]
pub struct C {
    pub farads: f64,
}

impl SpiceEmit for C {
    fn n_terminals(&self) -> usize { 2 }
    fn emit_spice(
        &self,
        n: &mut Netlist,
        nets: &[&str],
        id: &str,
    ) -> Result<(), EmitError> {
        if nets.len() != 2 {
            return Err(EmitError::ArityMismatch {
                block: "C".into(), expected: 2, got: nets.len(),
            });
        }
        n.add_element(format!("C{id} {} {} {:.10e}", nets[0], nets[1], self.farads));
        Ok(())
    }
}

/// Diode — emits a `.model` card and a `Dname anode cathode <model>` line.
///
/// The model name is derived from `is_amps`'s bit pattern so that two
/// diodes with different saturation currents get distinct model cards
/// (and matching name). Same trick the spike-divider-block `Diode` block
/// uses for graph parameter slot keying.
#[derive(Debug, Clone, Copy)]
pub struct D {
    pub is_amps: f64,
}

impl SpiceEmit for D {
    fn n_terminals(&self) -> usize { 2 }
    fn emit_spice(
        &self,
        n: &mut Netlist,
        nets: &[&str],
        id: &str,
    ) -> Result<(), EmitError> {
        if nets.len() != 2 {
            return Err(EmitError::ArityMismatch {
                block: "D".into(), expected: 2, got: nets.len(),
            });
        }
        let model = format!("Drlx_Is{:016x}", self.is_amps.to_bits());
        n.add_model(format!(".model {model} D(Is={:.10e})", self.is_amps));
        // anode = nets[0], cathode = nets[1] — matches the
        // `NonlinearDcBehavioral` sign convention (current flows
        // a → b inside the device when v_a > v_b).
        n.add_element(format!("D{id} {} {} {model}", nets[0], nets[1]));
        Ok(())
    }
}

// ── MOSFET (level-1 Shichman–Hodges) ─────────────────────────────────
//
// SPICE `LEVEL=1` is the simplest, oldest MOSFET model — long-channel
// only, no DIBL, no body effect when GAMMA=0, finite ro via LAMBDA.
// We pick it as Phase 2's first transistor model because both ngspice
// and LTspice implement it identically — exactly what we want for the
// cross-simulator validation harness. BSIM-class models would diverge
// version-to-version.
//
// Real PDK matching is not the goal here. The goal is "if you put two
// transistors in a CMOS inverter, the gate switches at the right point
// and both simulators agree on Vout(Vin)". Once that works, BSIM
// drop-in replacement is mechanical.

/// Shared parameters for the LEVEL=1 model.
#[derive(Debug, Clone, Copy)]
pub struct MosL1Params {
    /// Threshold voltage. Sign convention: positive for NMOS, negative
    /// for PMOS (matches SPICE convention).
    pub vto: f64,
    /// Transconductance parameter `μ·Cox` (A/V²). Typical: 20e-6 NMOS,
    /// 10e-6 PMOS for textbook long-channel CMOS.
    pub kp: f64,
    /// Channel-length modulation (1/V). 0 ⇒ ideal current source in
    /// saturation; ~0.02 gives a finite output resistance.
    pub lambda: f64,
}

impl MosL1Params {
    /// Textbook NMOS defaults: Vto=+0.5V, KP=20µA/V², λ=0.02.
    pub const NMOS_DEFAULT: Self = Self { vto:  0.5, kp: 20e-6, lambda: 0.02 };
    /// Textbook PMOS defaults: Vto=–0.5V, KP=10µA/V², λ=0.02.
    pub const PMOS_DEFAULT: Self = Self { vto: -0.5, kp: 10e-6, lambda: 0.02 };
}

/// 4-terminal NMOS (drain, gate, source, bulk). Geometry params W and L
/// in metres.
#[derive(Debug, Clone, Copy)]
pub struct Nmos {
    pub w: f64,
    pub l: f64,
    pub params: MosL1Params,
}

impl Default for Nmos {
    fn default() -> Self {
        Self { w: 10e-6, l: 2e-6, params: MosL1Params::NMOS_DEFAULT }
    }
}

impl SpiceEmit for Nmos {
    fn n_terminals(&self) -> usize { 4 }
    fn emit_spice(
        &self,
        n: &mut Netlist,
        nets: &[&str],
        id: &str,
    ) -> Result<(), EmitError> {
        if nets.len() != 4 {
            return Err(EmitError::ArityMismatch {
                block: "Nmos".into(), expected: 4, got: nets.len(),
            });
        }
        let model = mos_model_name("Nmos", &self.params);
        n.add_model(format!(
            ".model {model} NMOS(LEVEL=1 VTO={:.6} KP={:.6e} LAMBDA={:.6} GAMMA=0)",
            self.params.vto, self.params.kp, self.params.lambda,
        ));
        n.add_element(format!(
            "M{id} {} {} {} {} {model} W={:.6e} L={:.6e}",
            nets[0], nets[1], nets[2], nets[3], self.w, self.l,
        ));
        Ok(())
    }
}

/// 4-terminal PMOS (drain, gate, source, bulk).
#[derive(Debug, Clone, Copy)]
pub struct Pmos {
    pub w: f64,
    pub l: f64,
    pub params: MosL1Params,
}

impl Default for Pmos {
    fn default() -> Self {
        Self { w: 20e-6, l: 2e-6, params: MosL1Params::PMOS_DEFAULT }
    }
}

impl SpiceEmit for Pmos {
    fn n_terminals(&self) -> usize { 4 }
    fn emit_spice(
        &self,
        n: &mut Netlist,
        nets: &[&str],
        id: &str,
    ) -> Result<(), EmitError> {
        if nets.len() != 4 {
            return Err(EmitError::ArityMismatch {
                block: "Pmos".into(), expected: 4, got: nets.len(),
            });
        }
        let model = mos_model_name("Pmos", &self.params);
        n.add_model(format!(
            ".model {model} PMOS(LEVEL=1 VTO={:.6} KP={:.6e} LAMBDA={:.6} GAMMA=0)",
            self.params.vto, self.params.kp, self.params.lambda,
        ));
        n.add_element(format!(
            "M{id} {} {} {} {} {model} W={:.6e} L={:.6e}",
            nets[0], nets[1], nets[2], nets[3], self.w, self.l,
        ));
        Ok(())
    }
}

/// Build a deterministic model name from the LEVEL=1 params, so two
/// transistors with identical parameters share one `.model` card via
/// `Netlist::add_model`'s dedup. Same trick as the Diode primitive.
fn mos_model_name(prefix: &str, p: &MosL1Params) -> String {
    format!(
        "{prefix}_L1_Vt{:08x}_Kp{:08x}_La{:08x}",
        p.vto.to_bits() as u32 ^ (p.vto.to_bits() >> 32) as u32,
        p.kp.to_bits() as u32 ^ (p.kp.to_bits() >> 32) as u32,
        p.lambda.to_bits() as u32 ^ (p.lambda.to_bits() >> 32) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r_emits_simple_line() {
        let mut n = Netlist::new("t");
        R { ohms: 1.5e3 }.emit_spice(&mut n, &["a", "b"], "1").unwrap();
        assert_eq!(n.body[0], "R1 a b 1.5000000000e3");
    }

    #[test]
    fn c_emits_simple_line() {
        let mut n = Netlist::new("t");
        C { farads: 1e-12 }.emit_spice(&mut n, &["a", "0"], "load").unwrap();
        assert!(n.body[0].starts_with("Cload a 0"));
    }

    #[test]
    fn d_emits_model_and_instance() {
        let mut n = Netlist::new("t");
        D { is_amps: 1e-15 }.emit_spice(&mut n, &["anode", "cathode"], "1").unwrap();
        assert_eq!(n.models.len(), 1);
        assert!(n.models[0].starts_with(".model Drlx_Is"));
        assert!(n.body[0].starts_with("D1 anode cathode Drlx_Is"));
    }

    #[test]
    fn d_dedupes_identical_models() {
        let mut n = Netlist::new("t");
        D { is_amps: 1e-15 }.emit_spice(&mut n, &["a1", "k1"], "1").unwrap();
        D { is_amps: 1e-15 }.emit_spice(&mut n, &["a2", "k2"], "2").unwrap();
        assert_eq!(n.models.len(), 1, "same Is → one model card");
        assert_eq!(n.body.len(), 2, "but two device instances");
    }

    #[test]
    fn d_distinct_is_distinct_models() {
        let mut n = Netlist::new("t");
        D { is_amps: 1e-15 }.emit_spice(&mut n, &["a1", "k1"], "1").unwrap();
        D { is_amps: 2e-15 }.emit_spice(&mut n, &["a2", "k2"], "2").unwrap();
        assert_eq!(n.models.len(), 2);
    }

    #[test]
    fn nmos_emits_model_and_element() {
        let mut n = Netlist::new("t");
        Nmos::default()
            .emit_spice(&mut n, &["d", "g", "s", "b"], "1")
            .unwrap();
        assert_eq!(n.models.len(), 1);
        assert!(n.models[0].contains("NMOS(LEVEL=1"));
        assert!(n.models[0].contains("VTO=0.500000"));
        assert!(n.body[0].starts_with("M1 d g s b "));
        assert!(n.body[0].contains("W="));
        assert!(n.body[0].contains("L="));
    }

    #[test]
    fn pmos_uses_pmos_card() {
        let mut n = Netlist::new("t");
        Pmos::default()
            .emit_spice(&mut n, &["d", "g", "s", "b"], "p1")
            .unwrap();
        assert!(n.models[0].contains("PMOS(LEVEL=1"));
        assert!(n.models[0].contains("VTO=-0.500000"));
        assert!(n.body[0].starts_with("Mp1 d g s b "));
    }

    #[test]
    fn mos_dedupes_identical_models() {
        let mut n = Netlist::new("t");
        Nmos::default().emit_spice(&mut n, &["d1", "g1", "s1", "0"], "1").unwrap();
        Nmos::default().emit_spice(&mut n, &["d2", "g2", "s2", "0"], "2").unwrap();
        assert_eq!(n.models.len(), 1, "same params → one .model card");
        assert_eq!(n.body.len(), 2);
    }

    #[test]
    fn mos_distinct_params_distinct_models() {
        let mut n = Netlist::new("t");
        Nmos::default().emit_spice(&mut n, &["d1", "g1", "s1", "0"], "1").unwrap();
        let nmos2 = Nmos { params: MosL1Params { vto: 0.7, ..MosL1Params::NMOS_DEFAULT }, ..Nmos::default() };
        nmos2.emit_spice(&mut n, &["d2", "g2", "s2", "0"], "2").unwrap();
        assert_eq!(n.models.len(), 2);
    }

    #[test]
    fn nmos_arity_check() {
        let mut n = Netlist::new("t");
        let err = Nmos::default()
            .emit_spice(&mut n, &["d", "g", "s"], "1") // missing bulk
            .unwrap_err();
        match err {
            EmitError::ArityMismatch { expected: 4, got: 3, .. } => {}
            other => panic!("wrong err: {other:?}"),
        }
    }
}
