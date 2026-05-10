//! SPICE netlist emission.
//!
//! Each circuit primitive in rlx-eda gets two presentations of itself:
//!
//! 1. A graph stamping (MNA / rlx-IR) so the in-house solver can run it
//!    *and differentiate through it*.
//! 2. A SPICE element line so an external simulator (ngspice, LTspice,
//!    Xyce) can run the same circuit for cross-validation.
//!
//! This crate owns the second presentation. The trait [`SpiceEmit`] is
//! deliberately small: a block writes its own line(s) into a [`Netlist`]
//! given the external nets connected to its terminals. Hierarchical
//! blocks (a divider made of two resistors) call into their children's
//! `emit_spice` with the appropriate inner nets.
//!
//! ## What this crate doesn't do
//!
//! - **Subcircuit (`.SUBCKT`) emission.** Phase 1 inlines every
//!   instance into the top-level deck. When we need true hierarchy
//!   (millions of identical SAR slices), `Netlist` grows a
//!   `register_subckt` API. For ~100-element designs inlining is fine
//!   and easier to debug.
//! - **Parameter extraction.** A block emits its parameter values as
//!   literal numbers, not `.param X={…}` references. When we wire
//!   inverse-design loops to SPICE we'll re-emit the deck per iteration
//!   with new numbers, not parameterize. Keeps the deck simple.
//! - **Netname auto-generation.** The caller hands `nets: &[&str]` —
//!   the parent block knows which nets go where. Auto-numbering
//!   internal nets at the top of a hierarchical block is the parent's
//!   job.

use std::fmt::Write as _;

use eda_hir::SourceWaveform;
use thiserror::Error;

pub mod primitives;
pub use primitives::{C, D, MosL1Params, Nmos, Pmos, R};

#[derive(Debug, Error)]
pub enum EmitError {
    /// A block was handed the wrong number of nets for its terminal count.
    #[error("{block}: expected {expected} terminals, got {got}")]
    ArityMismatch { block: String, expected: usize, got: usize },
}

/// A SPICE deck under construction.
///
/// Three logical sections, all kept distinct so we can re-order them at
/// `deck()` time:
///   - **preamble** — `.title`, `.options`, `.include`, `.lib`
///   - **models** — `.model` cards, ordered by insertion but deduplicated
///     by exact-string equality (a block that uses the same model twice
///     only emits the card once)
///   - **body** — element instances, voltage/current sources
///   - **control** — `.control` block contents (bare ngspice syntax;
///     LTspice's `.tran`/`.print` directives go in `body` instead)
#[derive(Debug, Default, Clone)]
pub struct Netlist {
    pub title: String,
    pub preamble: Vec<String>,
    pub models: Vec<String>,
    pub body: Vec<String>,
}

impl Netlist {
    pub fn new(title: impl Into<String>) -> Self {
        Self { title: title.into(), ..Default::default() }
    }

    /// Push a `.include`, `.lib`, or `.option` line. Written verbatim
    /// before the body.
    pub fn add_preamble(&mut self, line: impl Into<String>) {
        self.preamble.push(line.into());
    }

    /// Add a `.model` card. Dedupes by exact-string equality so a Diode
    /// block using the same model twice doesn't emit two cards.
    pub fn add_model(&mut self, line: impl Into<String>) {
        let s = line.into();
        if !self.models.iter().any(|m| m == &s) {
            self.models.push(s);
        }
    }

    /// Add an element instance line (`R1 a b 10k`, `C2 mid 0 1p`, `Vin
    /// in 0 PULSE(...)`, …).
    pub fn add_element(&mut self, line: impl Into<String>) {
        self.body.push(line.into());
    }

    /// `Vname p n DC <volts>` — the standard DC source form, accepted
    /// by ngspice and LTspice alike.
    pub fn add_dc_source(&mut self, name: &str, p: &str, n: &str, volts: f64) {
        self.add_element(format!("V{name} {p} {n} DC {volts:.10e}"));
    }

    /// `Vname p n PULSE(V1 V2 TD TR TF PW PER)`.
    pub fn add_pulse_source(&mut self, name: &str, p: &str, n: &str, pulse: &Pulse) {
        self.add_element(format!(
            "V{name} {p} {n} PULSE({:.10e} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e})",
            pulse.v_initial, pulse.v_pulsed, pulse.t_delay, pulse.t_rise, pulse.t_fall,
            pulse.pulse_width, pulse.period,
        ));
    }

    /// `Vname p n SIN(VO VA FREQ TD THETA)`.
    pub fn add_sine_source(&mut self, name: &str, p: &str, n: &str, sine: &Sine) {
        self.add_element(format!(
            "V{name} {p} {n} SIN({:.10e} {:.10e} {:.10e} {:.10e} {:.10e})",
            sine.v_offset, sine.v_amplitude, sine.frequency, sine.t_delay, sine.damping,
        ));
    }

    /// `Vname p n <…>` — pick the right SPICE card from a HIR
    /// [`SourceWaveform`]. This is the bridge we want most user circuits to
    /// use: a single `SourceWaveform` value drives both the rlx outer-loop
    /// transient (via `value_at(t)`) and the SPICE deck (via this method),
    /// guaranteeing the two simulators see identical stimulus.
    pub fn add_waveform_source(&mut self, name: &str, p: &str, n: &str, w: &SourceWaveform) {
        match *w {
            SourceWaveform::Dc(v) => self.add_dc_source(name, p, n, v),
            SourceWaveform::Pulse { v1, v2, td, tr, tf, pw, per } => {
                // SourceWaveform::Pulse uses `per <= 0` to mean "single
                // pulse, no repetition" — that's how `value_at` reads it.
                // ngspice and LTspice instead default `PER=0` to
                // `pw + tr + tf`, which chatter-collapses the pulse into
                // back-to-back rising/falling edges at the intended fall
                // point. Substitute a very large period so the SPICE
                // engines see the same single-pulse semantics rlx uses.
                // 1e30 s is comfortably beyond any realistic tstop and
                // round-trips cleanly through %e formatting.
                let spice_per = if per > 0.0 { per } else { 1e30 };
                self.add_pulse_source(name, p, n, &Pulse {
                    v_initial: v1, v_pulsed: v2,
                    t_delay: td, t_rise: tr, t_fall: tf,
                    pulse_width: pw, period: spice_per,
                });
            }
            SourceWaveform::Sine { v_off, v_amp, freq, td, theta } => {
                self.add_sine_source(name, p, n, &Sine {
                    v_offset: v_off, v_amplitude: v_amp,
                    frequency: freq, t_delay: td, damping: theta,
                });
            }
        }
    }

    /// `Vname p n PWL(t0 v0 t1 v1 …)`.
    pub fn add_pwl_source(&mut self, name: &str, p: &str, n: &str, pwl: &Pwl) {
        let mut line = format!("V{name} {p} {n} PWL(");
        for (i, (t, v)) in pwl.points.iter().enumerate() {
            if i > 0 { line.push(' '); }
            let _ = write!(line, "{t:.10e} {v:.10e}");
        }
        line.push(')');
        self.add_element(line);
    }

    /// Render the deck as text. Layout:
    /// ```text
    /// * <title>
    /// <preamble lines>
    /// <model cards>
    /// <body>
    /// .end
    /// ```
    /// Caller appends `.op` / `.tran` / `.ac` / `.control` themselves —
    /// `Invoker` implementations stitch their own analysis directives
    /// on top of this deck text.
    pub fn deck(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "* {}", self.title);
        for line in &self.preamble { let _ = writeln!(s, "{line}"); }
        for line in &self.models   { let _ = writeln!(s, "{line}"); }
        for line in &self.body     { let _ = writeln!(s, "{line}"); }
        s
    }
}

/// SPICE-emittable circuit block.
///
/// Implementors are responsible for:
///   1. Validating `nets.len() == n_terminals()` (returning [`EmitError::ArityMismatch`]).
///   2. Inserting any model cards via [`Netlist::add_model`] (deduped).
///   3. Pushing element / source lines via [`Netlist::add_element`] (or the
///      typed source helpers).
///
/// `instance_id` is a unique designator (e.g. `"R1"`, `"D7"`) chosen by
/// the parent block. The implementor combines it with SPICE's
/// element-prefix convention (`R…` for resistors, `D…` for diodes,
/// `M…` for MOSFETs).
pub trait SpiceEmit {
    /// Number of electrical terminals — must match the length of `nets`
    /// passed to `emit_spice`.
    fn n_terminals(&self) -> usize;

    /// Emit this block's contribution to `net`. `nets` are the external
    /// node names connected to terminals 0..n_terminals(), in order.
    fn emit_spice(
        &self,
        net: &mut Netlist,
        nets: &[&str],
        instance_id: &str,
    ) -> Result<(), EmitError>;
}

// ── Source descriptors ─────────────────────────────────────────────────

/// PULSE source — initial / pulsed level + timing.
///
/// Maps to SPICE `PULSE(V1 V2 TD TR TF PW PER)`. The defaults are tuned
/// for a 1V rail-to-rail clock at 1 µs period (matches the LTspice paper).
#[derive(Debug, Clone, Copy)]
pub struct Pulse {
    pub v_initial: f64,
    pub v_pulsed: f64,
    pub t_delay: f64,
    pub t_rise: f64,
    pub t_fall: f64,
    pub pulse_width: f64,
    pub period: f64,
}

impl Pulse {
    /// Symmetric clock: 0V→V_high, t_delay=0, equal rise/fall = 1ps,
    /// 50% duty.
    pub fn clock(v_high: f64, period: f64) -> Self {
        Self {
            v_initial: 0.0,
            v_pulsed: v_high,
            t_delay: 0.0,
            t_rise: 1e-12,
            t_fall: 1e-12,
            pulse_width: period * 0.5 - 1e-12,
            period,
        }
    }
}

/// SIN source — sinusoidal AC stimulus.
///
/// Maps to SPICE `SIN(VO VA FREQ TD THETA)`. `damping` is the decay
/// constant `THETA` (0 for an undamped sine).
#[derive(Debug, Clone, Copy)]
pub struct Sine {
    pub v_offset: f64,
    pub v_amplitude: f64,
    pub frequency: f64,
    pub t_delay: f64,
    pub damping: f64,
}

impl Sine {
    /// Simple sine wave centered on `v_offset` with peak amplitude `v_amp`.
    pub fn ac(v_offset: f64, v_amp: f64, frequency: f64) -> Self {
        Self { v_offset, v_amplitude: v_amp, frequency, t_delay: 0.0, damping: 0.0 }
    }
}

/// Piecewise-linear source.
#[derive(Debug, Clone)]
pub struct Pwl {
    /// `(time, voltage)` pairs in monotonically-increasing time order.
    pub points: Vec<(f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2-terminal resistor — simplest possible SpiceEmit.
    struct ResistorVal { ohms: f64 }
    impl SpiceEmit for ResistorVal {
        fn n_terminals(&self) -> usize { 2 }
        fn emit_spice(&self, n: &mut Netlist, nets: &[&str], id: &str) -> Result<(), EmitError> {
            if nets.len() != 2 {
                return Err(EmitError::ArityMismatch {
                    block: "Resistor".into(), expected: 2, got: nets.len(),
                });
            }
            n.add_element(format!("R{id} {} {} {:.10e}", nets[0], nets[1], self.ohms));
            Ok(())
        }
    }

    #[test]
    fn deck_has_title_preamble_body_no_end() {
        let mut n = Netlist::new("divider");
        n.add_preamble(".include models.lib");
        ResistorVal { ohms: 10e3 }.emit_spice(&mut n, &["in", "mid"], "1").unwrap();
        ResistorVal { ohms: 30e3 }.emit_spice(&mut n, &["mid", "0"], "2").unwrap();
        let d = n.deck();
        assert!(d.starts_with("* divider\n"));
        assert!(d.contains(".include models.lib"));
        assert!(d.contains("R1 in mid"));
        assert!(d.contains("R2 mid 0"));
    }

    #[test]
    fn arity_mismatch_errors() {
        let mut n = Netlist::new("t");
        let r = ResistorVal { ohms: 1.0 };
        let err = r.emit_spice(&mut n, &["a"], "1").unwrap_err();
        match err {
            EmitError::ArityMismatch { expected: 2, got: 1, .. } => {}
            other => panic!("wrong err: {other:?}"),
        }
    }

    #[test]
    fn pulse_source_format_matches_spice() {
        let mut n = Netlist::new("t");
        n.add_pulse_source("clk", "clk", "0", &Pulse::clock(1.0, 1e-6));
        let line = n.body.last().unwrap();
        assert!(line.starts_with("Vclk clk 0 PULSE("));
        // Period = 1e-6 should appear as the 7th argument.
        assert!(line.contains("1.0000000000e-6"));
    }

    #[test]
    fn waveform_dc_dispatches_to_dc_source() {
        let mut n = Netlist::new("t");
        n.add_waveform_source("vdd", "vdd", "0", &SourceWaveform::Dc(1.8));
        let line = n.body.last().unwrap();
        assert!(line.starts_with("Vvdd vdd 0 DC "), "got {line}");
        assert!(line.contains("1.8"));
    }

    #[test]
    fn waveform_pulse_dispatches_to_pulse_source() {
        let mut n = Netlist::new("t");
        let w = SourceWaveform::pulse(0.0, 1.0, 1e-9, 100e-12, 100e-12, 5e-9, 10e-9);
        n.add_waveform_source("in", "in", "0", &w);
        let line = n.body.last().unwrap();
        assert!(line.starts_with("Vin in 0 PULSE("), "got {line}");
    }

    #[test]
    fn waveform_pulse_with_zero_period_emits_huge_period() {
        // Per <= 0 in SourceWaveform means "no repeat"; in SPICE that
        // must be a very large explicit period or ngspice/LTspice
        // chatter-collapse the pulse. Ensure the bridge handles this.
        let mut n = Netlist::new("t");
        let w = SourceWaveform::pulse(0.0, 1.0, 0.0, 0.0, 0.0, 1e-9, 0.0);
        n.add_waveform_source("in", "in", "0", &w);
        let line = n.body.last().unwrap();
        assert!(line.contains("1.0000000000e30"), "expected huge period, got {line}");
    }

    #[test]
    fn waveform_sine_dispatches_to_sine_source() {
        let mut n = Netlist::new("t");
        n.add_waveform_source("in", "in", "0", &SourceWaveform::sine(0.0, 1.0, 1e3, 0.0));
        let line = n.body.last().unwrap();
        assert!(line.starts_with("Vin in 0 SIN("), "got {line}");
    }

    #[test]
    fn model_dedup() {
        let mut n = Netlist::new("t");
        n.add_model(".model Dgen D(Is=1e-15)");
        n.add_model(".model Dgen D(Is=1e-15)");
        n.add_model(".model Dother D(Is=2e-15)");
        assert_eq!(n.models.len(), 2);
    }
}
