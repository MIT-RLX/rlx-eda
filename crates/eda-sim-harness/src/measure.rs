//! Measurements: declare a `.meas` line, parse the result.
//!
//! ngspice prints meas results to stdout as `<name> = <value>` (sometimes
//! with units). We parse those into a [`MeasureLog`] map for downstream
//! spec checking.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One `.meas` directive and how to render it for ngspice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    /// Measure name; matches `Spec::name` and the stdout key.
    pub name: String,
    /// Body of the `.meas` line *after* `meas <analysis> <name>` —
    /// the harness prepends the analysis kind.
    ///
    /// Example: `"find v(out) at=10n"` becomes
    /// `meas tran <name> find v(out) at=10n`.
    pub body: String,
    /// Optional unit string for reporting (`"V"`, `"A"`, `"s"`, …).
    #[serde(default)]
    pub unit: Option<String>,
}

impl Measurement {
    pub fn tran(name: impl Into<String>, body: impl Into<String>, unit: Option<&str>) -> Self {
        Self {
            name: name.into(),
            body: body.into(),
            unit: unit.map(|s| s.to_string()),
        }
    }

    /// Average current drawn from a supply, integrated over the
    /// transient window. Emits `meas tran <name> avg i(v<vdd_source>)`.
    ///
    /// `vdd_source` is the SPICE name of the voltage source (e.g.
    /// `"vdd"` for `Vvdd vdd 0 1.8`). ngspice exposes the branch
    /// current as `i(vvdd)`.
    ///
    /// **Why current, not power:** ngspice's `.meas tran avg` only
    /// accepts a single vector name, not an arithmetic expression.
    /// `avg v(vdd)*i(vvdd)` parses but fails at evaluation time on
    /// every ngspice we've tested. Two productive paths to power
    /// from here:
    ///
    /// 1. **Constant supply (the common case).** Power = `Vsupply
    ///    × i_avg`. Compute in `Testbench::derive()` from this
    ///    measurement plus the known supply voltage. See
    ///    [`Self::power_from_const_supply`] for a one-liner that
    ///    builds the matching derive entry.
    /// 2. **Time-varying supply.** Wrap a `.control` block around
    ///    `let p_t = v(node)*i(vsrc); meas tran p avg p_t`. Outside
    ///    the scope of this helper today.
    pub fn supply_current(name: impl Into<String>, vdd_source: &str) -> Self {
        Self::tran(
            name,
            format!("avg i(v{vdd_source})"),
            Some("A"),
        )
    }

    /// Total charge drawn from a supply over the transient window
    /// (`integ i`). Coulombs. Multiply by a constant supply voltage
    /// to get energy.
    pub fn supply_charge(name: impl Into<String>, vdd_source: &str) -> Self {
        Self::tran(
            name,
            format!("integ i(v{vdd_source})"),
            Some("C"),
        )
    }

    /// Helper for `Testbench::derive()`: given a previously-measured
    /// average supply current and the known constant supply voltage,
    /// return the `(power_name, power_watts)` tuple to fold into the
    /// MeasureLog.
    ///
    /// ```ignore
    /// fn derive(&self, m: &MeasureLog) -> Vec<(String, f64)> {
    ///     Measurement::power_from_const_supply(m, "i_vbias", "p_vbias", VBIAS)
    ///         .into_iter().collect()
    /// }
    /// ```
    pub fn power_from_const_supply(
        log: &MeasureLog,
        current_name: &str,
        power_name: &str,
        vdd_v: f64,
    ) -> Option<(String, f64)> {
        let i = log.get(current_name)?.as_number()?;
        Some((power_name.to_string(), i.abs() * vdd_v))
    }

    /// Render as a `.meas` line for `analysis_kind` (`"tran"`, `"ac"`,
    /// `"dc"`).
    pub fn to_meas_line(&self, analysis_kind: &str) -> String {
        format!("meas {} {} {}", analysis_kind, self.name, self.body)
    }
}

/// Result of one parsed measure value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementValue {
    /// Successful numeric result.
    Number(f64),
    /// `.meas` printed `failed` or didn't appear in stdout.
    Failed,
}

impl MeasurementValue {
    pub fn as_number(self) -> Option<f64> {
        match self {
            MeasurementValue::Number(v) => Some(v),
            MeasurementValue::Failed => None,
        }
    }
}

/// Map of measure name → value. Iteration is ordered.
#[derive(Debug, Default, Clone)]
pub struct MeasureLog {
    pub values: BTreeMap<String, MeasurementValue>,
}

impl MeasureLog {
    /// Parse ngspice stdout for measure results. ngspice writes lines in
    /// a few shapes — pick whichever variant matches each requested name:
    ///
    /// ```text
    /// ibn                =  4.987654e-06
    /// vgs_m1             =  6.234500e-01
    /// failed_meas_name   = failed
    /// ```
    ///
    /// We anchor on the requested name as a word at line start; this
    /// avoids matching the body text of `.meas` echoes.
    pub fn parse(stdout: &str, requested: &[Measurement]) -> Self {
        let mut values = BTreeMap::new();
        for m in requested {
            let v = scan_one(stdout, &m.name);
            values.insert(m.name.clone(), v);
        }
        Self { values }
    }

    pub fn get(&self, name: &str) -> Option<MeasurementValue> {
        self.values.get(name).copied()
    }
}

fn scan_one(stdout: &str, name: &str) -> MeasurementValue {
    let lname = name.to_lowercase();
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        let lower = trimmed.to_lowercase();
        if !lower.starts_with(&lname) {
            continue;
        }
        let after = &trimmed[lname.len()..];
        let after_ok = after.is_empty()
            || after.starts_with(|c: char| c.is_whitespace() || c == '=');
        if !after_ok {
            continue;
        }
        let tail = after.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
        let tok = tail.split_whitespace().next().unwrap_or("");
        if tok.eq_ignore_ascii_case("failed") {
            return MeasurementValue::Failed;
        }
        if let Ok(v) = tok.parse::<f64>() {
            return MeasurementValue::Number(v);
        }
    }
    MeasurementValue::Failed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meas(name: &str) -> Measurement {
        Measurement::tran(name, "find v(x) at=1n", None)
    }

    #[test]
    fn parses_numeric_meas_line() {
        let s = "ibn                =  4.987654e-06\nbye\n";
        let log = MeasureLog::parse(s, &[meas("ibn")]);
        assert_eq!(log.get("ibn"), Some(MeasurementValue::Number(4.987654e-6)));
    }

    #[test]
    fn parses_multiple_lines() {
        let s = "ibn       = 5.0e-6\nvgs_m1   =  6.2e-1\n";
        let log = MeasureLog::parse(s, &[meas("ibn"), meas("vgs_m1")]);
        assert_eq!(log.get("ibn"), Some(MeasurementValue::Number(5e-6)));
        assert_eq!(log.get("vgs_m1"), Some(MeasurementValue::Number(0.62)));
    }

    #[test]
    fn marks_missing_as_failed() {
        let log = MeasureLog::parse("nothing here", &[meas("ibn")]);
        assert_eq!(log.get("ibn"), Some(MeasurementValue::Failed));
    }

    #[test]
    fn marks_failed_keyword_as_failed() {
        let log = MeasureLog::parse("ibn = failed\n", &[meas("ibn")]);
        assert_eq!(log.get("ibn"), Some(MeasurementValue::Failed));
    }

    #[test]
    fn supply_current_emits_avg_directive_with_a_unit() {
        let m = Measurement::supply_current("i_vdd", "vdd");
        assert_eq!(m.unit.as_deref(), Some("A"));
        assert_eq!(m.to_meas_line("tran"), "meas tran i_vdd avg i(vvdd)");
    }

    #[test]
    fn supply_charge_emits_integ_directive_with_c_unit() {
        let m = Measurement::supply_charge("q_vdd", "vdd");
        assert_eq!(m.unit.as_deref(), Some("C"));
        assert!(m.to_meas_line("tran").contains("integ i(vvdd)"));
    }

    #[test]
    fn power_from_const_supply_multiplies_avg_current_by_vdd() {
        let s = "i_vdd              = -1.000000e-03\n";
        let m = Measurement::supply_current("i_vdd", "vdd");
        let log = MeasureLog::parse(s, &[m]);
        let (name, watts) =
            Measurement::power_from_const_supply(&log, "i_vdd", "p_vdd", 1.8).unwrap();
        assert_eq!(name, "p_vdd");
        // |i| × V = 1 mA × 1.8 V = 1.8 mW. Sign of i is dropped — the
        // helper measures power dissipated, not delivered.
        assert!((watts - 1.8e-3).abs() < 1e-9, "got {watts}");
    }

    #[test]
    fn anchors_on_word_boundary() {
        // "vgs" should not match "vgs_m1".
        let s = "vgs_m1 = 0.62\nvgs    = 0.5\n";
        let log = MeasureLog::parse(s, &[meas("vgs")]);
        assert_eq!(log.get("vgs"), Some(MeasurementValue::Number(0.5)));
    }
}
