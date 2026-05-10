//! Trace data model — what one optimization run produces, before
//! charts and reports.
//!
//! Centerpiece: [`TraceRow`] is `step + BTreeMap<String, f64>`. Every
//! per-step number — loss, parameters, gradients, derived quantities
//! — is just a named scalar. Caller picks the names; the harness
//! never inspects them, only renders them.

use std::collections::BTreeMap;

/// One step's worth of named scalars.
#[derive(Clone, Debug, Default)]
pub struct TraceRow {
    pub step: u32,
    pub values: BTreeMap<String, f64>,
}

impl TraceRow {
    pub fn new(step: u32) -> Self {
        Self { step, values: BTreeMap::new() }
    }

    /// Builder-style insertion. Repeated keys overwrite — the harness
    /// doesn't care which of two values "wins"; that's a caller bug
    /// to catch upstream.
    pub fn with(mut self, name: impl Into<String>, value: f64) -> Self {
        self.values.insert(name.into(), value);
        self
    }

    /// Equivalent to `with` but mutating; lets a non-builder caller
    /// add fields conditionally.
    pub fn set(&mut self, name: impl Into<String>, value: f64) {
        self.values.insert(name.into(), value);
    }

    /// Lookup helper used by chart rendering. Returns `0.0` when the
    /// key is missing — keeps charts robust against optional series
    /// that some steps don't emit (e.g. `early_stop` flag).
    pub fn get(&self, name: &str) -> f64 {
        self.values.get(name).copied().unwrap_or(0.0)
    }
}

/// Anything that produces one [`TraceRow`] per step. Implemented as a
/// blanket impl on `FnMut(u32) -> TraceRow` so callers can pass a
/// closure directly.
pub trait OptStep {
    fn step(&mut self, step: u32) -> TraceRow;
}

impl<F> OptStep for F
where
    F: FnMut(u32) -> TraceRow,
{
    fn step(&mut self, step: u32) -> TraceRow {
        self(step)
    }
}

/// How often a row gets retained for plotting / reporting. The
/// in-memory trace always carries one row per logged step (callers
/// that want every step set [`LogSchedule::Every`]); the schedule
/// just decides which steps the harness asks the closure to run.
///
/// Logarithmic (1, 2, 4, 8, …) is what most optimization writeups use
/// because it captures the early "loss falling fast" portion at full
/// resolution and keeps the late asymptote summarized — same shape
/// `mzi_ml_trace` and `spike-divider-block::ml_trace` were already
/// hand-rolling.
#[derive(Clone, Copy, Debug)]
pub enum LogSchedule {
    /// Log every step. Use for short runs (<= a few hundred steps)
    /// where the trace table will be readable inline.
    Every,
    /// Log step 1, 2, 4, 8, …, plus the last step. Use for long
    /// runs where the late tail is uninteresting.
    Logarithmic,
    /// Log every Nth step plus the last step.
    Stride(u32),
}

impl Default for LogSchedule {
    fn default() -> Self { LogSchedule::Every }
}

impl LogSchedule {
    /// Returns `true` if this step should appear in the trace.
    pub fn should_log(&self, step: u32, total: u32) -> bool {
        if step == 0 || step == total { return true; }
        match self {
            LogSchedule::Every => true,
            LogSchedule::Logarithmic => step.is_power_of_two(),
            LogSchedule::Stride(n) => *n > 0 && step % *n == 0,
        }
    }
}

/// Run-level configuration.
#[derive(Clone, Debug)]
pub struct TraceCfg {
    /// Used as the asset-folder slug (`docs/assets/<name>/`) and the
    /// run identifier in the report.
    pub name: String,
    /// Total number of steps the harness will drive the optimizer
    /// for. The closure decides whether to early-stop by ignoring
    /// later step indices.
    pub steps: u32,
    /// Which steps end up in the trace table.
    pub log_at: LogSchedule,
}

impl TraceCfg {
    pub fn new(name: impl Into<String>, steps: u32) -> Self {
        Self { name: name.into(), steps, log_at: LogSchedule::default() }
    }

    pub fn with_log_schedule(mut self, sched: LogSchedule) -> Self {
        self.log_at = sched;
        self
    }
}

/// In-memory trace — the result of one [`Trace::run`] invocation.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub rows: Vec<TraceRow>,
    /// Order in which series first appeared. Used by the report's
    /// step-by-step table to keep columns deterministic across runs.
    pub series_order: Vec<String>,
}

impl Trace {
    /// Drive the step closure `cfg.steps` times and collect the rows
    /// the [`LogSchedule`] keeps. Each closure invocation gets the
    /// current step counter; the harness preserves whatever values
    /// the row carries.
    pub fn run<S: OptStep>(cfg: &TraceCfg, mut optimizer: S) -> Self {
        let mut rows = Vec::new();
        let mut series_order: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for step in 0..=cfg.steps {
            let row = optimizer.step(step);
            if cfg.log_at.should_log(step, cfg.steps) {
                for k in row.values.keys() {
                    if seen.insert(k.clone()) {
                        series_order.push(k.clone());
                    }
                }
                rows.push(row);
            }
        }
        Self { rows, series_order }
    }

    /// Pull a single series as `(step, value)` pairs. Missing keys
    /// turn into `0.0` per [`TraceRow::get`].
    pub fn series(&self, name: &str) -> Vec<(f64, f64)> {
        self.rows
            .iter()
            .map(|r| (r.step as f64, r.get(name)))
            .collect()
    }

    /// Pull multiple series sharing the same x axis (typically `step`).
    /// Returns `(xs, [(label, ys)])`. Missing values fill 0.0 to keep
    /// the lengths consistent for chart rendering.
    pub fn series_many(&self, names: &[&str]) -> (Vec<f64>, Vec<(String, Vec<f64>)>) {
        let xs: Vec<f64> = self.rows.iter().map(|r| r.step as f64).collect();
        let ys: Vec<(String, Vec<f64>)> = names
            .iter()
            .map(|n| (n.to_string(), self.rows.iter().map(|r| r.get(n)).collect()))
            .collect();
        (xs, ys)
    }

    /// Build a CSV string with `step` as the first column and every
    /// series in `series_order` as following columns. Used for the
    /// `*.csv` artifact every existing trace bin already emits.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("step");
        for s in &self.series_order {
            out.push(',');
            out.push_str(s);
        }
        out.push('\n');
        for r in &self.rows {
            out.push_str(&r.step.to_string());
            for s in &self.series_order {
                out.push(',');
                out.push_str(&format_csv_f64(r.get(s)));
            }
            out.push('\n');
        }
        out
    }
}

fn format_csv_f64(v: f64) -> String {
    // Match the existing trace bins' format: scientific for very
    // small / very large magnitudes, fixed otherwise.
    if v == 0.0 {
        "0".to_string()
    } else if v.abs() < 1e-3 || v.abs() >= 1e6 {
        format!("{v:.6e}")
    } else {
        format!("{v:.6}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closure_optstep_records_rows() {
        let cfg = TraceCfg::new("test", 5);
        let trace = Trace::run(&cfg, |step| {
            TraceRow::new(step)
                .with("loss", (step as f64 + 1.0).recip())
                .with("param", step as f64 * 0.5)
        });
        assert_eq!(trace.rows.len(), 6);
        assert!(trace.series_order.contains(&"loss".to_string()));
        assert!(trace.series_order.contains(&"param".to_string()));
        assert_eq!(trace.rows[0].step, 0);
        assert_eq!(trace.rows[5].step, 5);
    }

    #[test]
    fn log_schedule_keeps_first_last_powers_of_two() {
        let cfg = TraceCfg::new("log", 100).with_log_schedule(LogSchedule::Logarithmic);
        let trace = Trace::run(&cfg, |s| TraceRow::new(s).with("x", s as f64));
        let steps: Vec<u32> = trace.rows.iter().map(|r| r.step).collect();
        assert!(steps.contains(&0));
        assert!(steps.contains(&100));
        assert!(steps.contains(&1));
        assert!(steps.contains(&64));
        assert!(!steps.contains(&3)); // not a power of two
    }

    #[test]
    fn csv_round_trip_has_step_first_then_series_order() {
        let cfg = TraceCfg::new("csv", 1);
        let trace = Trace::run(&cfg, |s| {
            TraceRow::new(s).with("loss", 1.0).with("param", 2.0)
        });
        let csv = trace.to_csv();
        let header = csv.lines().next().unwrap();
        assert!(header.starts_with("step,"));
        // BTreeMap is sorted, so series_order comes alphabetic.
        assert_eq!(header, "step,loss,param");
    }
}
