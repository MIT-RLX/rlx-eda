//! Pass/fail spec, `tran.yaml`-shaped.
//!
//! cicsim writes specs as YAML:
//!
//! ```yaml
//! - name: ibn
//!   min: 4.5e-6
//!   typ: 5.0e-6
//!   max: 5.5e-6
//!   unit: A
//! - name: vgs_m1
//!   min: 0.55
//!   max: 0.70
//!   unit: V
//! ```
//!
//! We mirror that 1:1 so cicsim users can hand us their existing
//! `tran.yaml` files. `min` / `max` / `typ` / `unit` are all optional;
//! omitting `min` and `max` makes the spec informational only.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spec {
    pub name: String,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub typ: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub unit: Option<String>,
}

/// How a Monte Carlo distribution is summarized in reports + spec checks.
///
/// Mirrors cicsim's `summary.yaml` `method:` field exactly so a cicsim
/// user can map our output to theirs without translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum McSummaryStyle {
    /// `(min, mean, max)` of observed draws. Statistically weak — the
    /// observed extremes depend heavily on n and don't generalize.
    /// Fine for sanity-checking; bad for yield gating.
    MinMax,
    /// `(µ - 3σ, µ, µ + 3σ)` — the ~99.7 % confidence corner. This is
    /// what production analog uses to gate spec pass/fail because it
    /// extrapolates beyond the observed sample. Default for MC corners.
    #[default]
    ThreeStd,
}

impl McSummaryStyle {
    pub fn as_str(self) -> &'static str {
        match self { McSummaryStyle::MinMax => "minmax", McSummaryStyle::ThreeStd => "3std" }
    }

    /// Compute `(low, typ, high)` from a slice of MC draws.
    /// Returns `None` if the slice is empty.
    pub fn summarize(self, values: &[f64]) -> Option<(f64, f64, f64)> {
        if values.is_empty() { return None; }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        match self {
            McSummaryStyle::MinMax => {
                let mn = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                Some((mn, mean, mx))
            }
            McSummaryStyle::ThreeStd => {
                // Sample variance, n-1 normalization. Falls back to 0
                // for n=1 so we don't NaN out.
                let var = if values.len() >= 2 {
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
                } else { 0.0 };
                let sigma = var.sqrt();
                Some((mean - 3.0 * sigma, mean, mean + 3.0 * sigma))
            }
        }
    }
}

impl Spec {
    /// Apply this spec to a measured value. Returns `Pass` when the value
    /// is within `[min, max]` (open bounds when either is `None`).
    /// Returns `Skipped` if the value is missing (failed measurement).
    pub fn check(&self, measured: Option<f64>) -> SpecCheck {
        let Some(v) = measured else { return SpecCheck::Skipped };
        if let Some(lo) = self.min {
            if v < lo { return SpecCheck::Fail { measured: v, reason: SpecFail::BelowMin(lo) }; }
        }
        if let Some(hi) = self.max {
            if v > hi { return SpecCheck::Fail { measured: v, reason: SpecFail::AboveMax(hi) }; }
        }
        SpecCheck::Pass { measured: v }
    }

    /// Spec check against a Monte Carlo distribution. Picks a worst-case
    /// representative (`low` for "above min" tests, `high` for "below
    /// max" tests) and runs the standard [`Self::check`] on each side.
    /// Returns the *worst* outcome (Fail wins over Pass).
    ///
    /// Use with [`McSummaryStyle::ThreeStd`] for production yield gating
    /// — checks that the 99.7%-confidence tail still meets spec.
    pub fn check_mc(&self, values: &[f64], style: McSummaryStyle) -> SpecCheck {
        let Some((low, _typ, high)) = style.summarize(values) else { return SpecCheck::Skipped };
        let lo_check = self.check(Some(low));
        let hi_check = self.check(Some(high));
        match (lo_check, hi_check) {
            (SpecCheck::Fail { .. }, _) => lo_check,
            (_, SpecCheck::Fail { .. }) => hi_check,
            (SpecCheck::Pass { .. }, _) => SpecCheck::Pass { measured: high },
            _ => SpecCheck::Skipped,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecFail {
    BelowMin(f64),
    AboveMax(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecCheck {
    Pass { measured: f64 },
    Fail { measured: f64, reason: SpecFail },
    /// Measurement was missing or `.meas` failed.
    Skipped,
}

impl SpecCheck {
    pub fn is_pass(self) -> bool { matches!(self, SpecCheck::Pass { .. }) }
    pub fn is_fail(self) -> bool { matches!(self, SpecCheck::Fail { .. }) }
}

/// Bundle = ordered list of specs, deserialized as the top-level YAML
/// sequence cicsim writes.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpecBundle {
    pub specs: Vec<Spec>,
}

impl SpecBundle {
    pub fn from_yaml_str(s: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(s)
    }

    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self, SpecLoadError> {
        let s = std::fs::read_to_string(path.as_ref())?;
        Ok(Self::from_yaml_str(&s)?)
    }

    pub fn find(&self, name: &str) -> Option<&Spec> {
        self.specs.iter().find(|s| s.name == name)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SpecLoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cicsim_shaped_yaml() {
        let s = r#"
- name: ibn
  min: 4.5e-6
  typ: 5.0e-6
  max: 5.5e-6
  unit: A
- name: vgs_m1
  min: 0.55
  max: 0.70
  unit: V
"#;
        let b = SpecBundle::from_yaml_str(s).unwrap();
        assert_eq!(b.specs.len(), 2);
        assert_eq!(b.specs[0].name, "ibn");
        assert_eq!(b.specs[0].typ, Some(5e-6));
        assert_eq!(b.specs[1].max, Some(0.70));
    }

    #[test]
    fn check_pass_within_bounds() {
        let s = Spec { name: "x".into(), min: Some(0.0), typ: None, max: Some(1.0), unit: None };
        assert!(s.check(Some(0.5)).is_pass());
    }

    #[test]
    fn check_fail_below_min() {
        let s = Spec { name: "x".into(), min: Some(0.0), typ: None, max: Some(1.0), unit: None };
        let r = s.check(Some(-0.1));
        assert!(matches!(r, SpecCheck::Fail { reason: SpecFail::BelowMin(_), .. }));
    }

    #[test]
    fn check_fail_above_max() {
        let s = Spec { name: "x".into(), min: Some(0.0), typ: None, max: Some(1.0), unit: None };
        let r = s.check(Some(2.0));
        assert!(matches!(r, SpecCheck::Fail { reason: SpecFail::AboveMax(_), .. }));
    }

    #[test]
    fn missing_measurement_is_skipped() {
        let s = Spec { name: "x".into(), min: Some(0.0), typ: None, max: Some(1.0), unit: None };
        assert!(matches!(s.check(None), SpecCheck::Skipped));
    }

    #[test]
    fn open_min_or_max_is_optional() {
        let s = Spec { name: "x".into(), min: None, typ: None, max: Some(1.0), unit: None };
        assert!(s.check(Some(-100.0)).is_pass()); // no min bound
        assert!(s.check(Some(2.0)).is_fail());
    }
}
