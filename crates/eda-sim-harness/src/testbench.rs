//! Testbench trait + analysis spec.
//!
//! A testbench knows how to build a SPICE deck for one corner and what
//! measurements to extract from the result. The harness drives many
//! corners through one testbench; the testbench is stateless.

use eda_spice_emit::Netlist;

use crate::corner::Corner;
use crate::measure::Measurement;

/// One analysis directive emitted into the deck's `.control` block.
#[derive(Debug, Clone)]
pub enum Analysis {
    /// `.op` — operating point only.
    Op,
    /// `.tran tstep tstop [uic]`.
    Tran { t_step: f64, t_stop: f64, uic: bool },
    /// `.ac dec <ppd> <fstart> <fstop>`.
    AcDec { points_per_decade: usize, f_start: f64, f_stop: f64 },
    /// `.dc Vsource start stop step`.
    DcSweep { source: String, start: f64, stop: f64, step: f64 },
}

/// A testbench: build deck + declare measurements + analysis.
pub trait Testbench {
    /// Stable name. Used as the file prefix for `<name>_<corner>.html`,
    /// `<name>_<corner>.png`, and the SHA cache key.
    fn name(&self) -> &str;

    /// Build the deck for one `corner`. The harness wraps the result in a
    /// `.control` block with the chosen [`Analysis`] directive and the
    /// `.meas` lines from [`Self::measurements`]; the testbench just owns
    /// element instances + sources + `.lib` includes.
    fn build_netlist(&self, corner: &Corner) -> Netlist;

    /// What to extract from the run.
    fn measurements(&self) -> Vec<Measurement>;

    /// Which analysis to run.
    fn analysis(&self) -> Analysis;

    /// Explicit signal-name allow-list for waveform plotting.
    /// The harness already filters out hierarchical model internals
    /// (anything containing `.` or `#` in its node identifier); use
    /// this to include otherwise-skipped names like `i(vmeas)` or to
    /// pin the plot to a small curated subset.
    fn plot_signals(&self) -> Vec<String> {
        Vec::new()
    }

    /// Post-process the parsed `.meas` log to emit *derived*
    /// measurements. Mirrors cicsim's `tran.py` pattern where you
    /// compute settling errors, differentials, common-mode rejection,
    /// etc. from the raw `.meas` outputs.
    ///
    /// Returns a list of `(name, value)` pairs that get folded into
    /// the [`crate::MeasureLog`] alongside the simulator's outputs and
    /// participate in spec checks. Names should match a [`crate::Spec`]
    /// entry to be useful.
    ///
    /// Default impl returns an empty list. Implementors typically read
    /// a few existing measures via `measures.get(name)` and arithmetic
    /// them into a derived quantity:
    ///
    /// ```ignore
    /// fn derive(&self, m: &MeasureLog) -> Vec<(String, f64)> {
    ///     let a = m.get("ibns_20u").and_then(|v| v.as_number());
    ///     let b = m.get("ibns_20u_9n").and_then(|v| v.as_number());
    ///     match (a, b) {
    ///         (Some(a), Some(b)) => vec![("ibn_settl_err".into(), a - b)],
    ///         _ => vec![],
    ///     }
    /// }
    /// ```
    fn derive(&self, _measures: &crate::MeasureLog) -> Vec<(String, f64)> {
        Vec::new()
    }

    /// Run verifiers (DRC / LVS / EM) for this corner. Default is
    /// "no verifiers" — `VerifyReport::empty()` is vacuously clean,
    /// so corners that don't override this flow through the harness
    /// exactly as before.
    ///
    /// Receives the parsed `MeasureLog` for this corner so EM checks
    /// can use actual simulated peak currents (rather than analytical
    /// guesses). DRC and LVS typically don't need it; EM almost
    /// always does.
    ///
    /// Layout-aware testbenches typically:
    ///
    /// ```ignore
    /// fn verify(&self, corner: &Corner, m: &MeasureLog) -> VerifyReport {
    ///     if corner.view != View::Layout { return VerifyReport::empty(); }
    ///     let drc_v = eda_drc::check_sky130a(/* lib, top, layers */);
    ///     let i_peak = m.get("i_vbias").and_then(|v| v.as_number()).unwrap_or(0.0).abs();
    ///     let em_v  = eda_em::check(&segments(i_peak), &jmax, &thk).unwrap_or_default();
    ///     VerifyReport::empty()
    ///         .set_drc(VerifierResult::from_count(drc_v.len(), drc_v.first().map_or("", |v| &v.rule)))
    ///         .set_em(VerifierResult::from_count(em_v.len(),  em_v.first().map_or(String::new(), |v| format!("{:?}", v.layer))))
    /// }
    /// ```
    ///
    /// The harness consumes only counts + first-message strings, so
    /// the testbench owns the klayout-side dep graph; harness stays
    /// klayout-free.
    fn verify(&self, _corner: &crate::Corner, _measures: &crate::MeasureLog)
        -> crate::VerifyReport
    {
        crate::VerifyReport::empty()
    }
}
