//! Verification-side report: DRC + LVS + EM violation counts, plus
//! a representative first-finding for each so the report can show
//! "what went wrong" at a glance without dragging in klayout types.
//!
//! ## Why counts, not the rich klayout types
//!
//! `eda-sim-harness` deliberately doesn't depend on klayout. DRC
//! violations live in `klayout-drc::Violation`, LVS mismatches in
//! `eda-extract::LvsMismatch`, EM violations in `eda-em::Violation`
//! — three different crates, all rooted at klayout. Pulling any of
//! them into the harness pulls the whole tree.
//!
//! Instead, callers (testbenches that *do* know about klayout) run
//! their verifier of choice and call `VerifyReport::set_drc(count,
//! first_message)` etc. The harness stores counts + previews and
//! renders them; the caller keeps the rich data alongside if they
//! want to drill in.
//!
//! ## Pass / fail
//!
//! `VerifyReport::is_clean()` is true iff every count is zero. The
//! `Reporter` flips a corner's pass/fail badge red when verification
//! is dirty, alongside (and after) the existing measurement-side
//! `Spec` checks.

use serde::{Deserialize, Serialize};

/// One verifier's contribution to the report. `count == 0` means
/// "ran clean"; any positive value flips the corner's verify-clean
/// gate red.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VerifierResult {
    pub count: usize,
    /// Short message shown on the report next to the count — the
    /// first / worst violation in human-readable form. Empty when
    /// `count == 0` or the caller didn't bother formatting one.
    pub first_message: String,
}

impl VerifierResult {
    pub fn clean() -> Self {
        Self { count: 0, first_message: String::new() }
    }

    pub fn from_count(count: usize, first_message: impl Into<String>) -> Self {
        Self { count, first_message: first_message.into() }
    }

    pub fn is_clean(&self) -> bool { self.count == 0 }
}

/// Per-corner verification snapshot. Each verifier is `None` when not
/// run for the corner — distinct from `Some(clean)`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VerifyReport {
    pub drc: Option<VerifierResult>,
    pub lvs: Option<VerifierResult>,
    pub em:  Option<VerifierResult>,
}

impl VerifyReport {
    pub fn empty() -> Self { Self::default() }

    pub fn set_drc(mut self, r: VerifierResult) -> Self { self.drc = Some(r); self }
    pub fn set_lvs(mut self, r: VerifierResult) -> Self { self.lvs = Some(r); self }
    pub fn set_em(mut self,  r: VerifierResult) -> Self { self.em  = Some(r); self }

    /// True iff every verifier that ran came back with zero
    /// violations. Verifiers that didn't run (`None`) don't count
    /// against cleanness — the caller chooses which checks apply to
    /// which corner. Schematic corners typically run LVS-equivalent
    /// only; layout-extracted corners run all three.
    pub fn is_clean(&self) -> bool {
        [self.drc.as_ref(), self.lvs.as_ref(), self.em.as_ref()]
            .iter()
            .filter_map(|x| *x)
            .all(|r| r.is_clean())
    }

    /// Total violations across every verifier that ran. Convenience
    /// for the CSV's `verify_count` column.
    pub fn total(&self) -> usize {
        self.drc.as_ref().map_or(0, |r| r.count)
            + self.lvs.as_ref().map_or(0, |r| r.count)
            + self.em.as_ref().map_or(0, |r| r.count)
    }

    /// Render as a compact "DRC=0 LVS=0 EM=2" string. `-` for
    /// verifiers that didn't run.
    pub fn summary_string(&self) -> String {
        fn one(label: &str, r: Option<&VerifierResult>) -> String {
            match r {
                None => format!("{label}=-"),
                Some(v) => format!("{label}={}", v.count),
            }
        }
        format!(
            "{} {} {}",
            one("DRC", self.drc.as_ref()),
            one("LVS", self.lvs.as_ref()),
            one("EM",  self.em.as_ref()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_is_clean() {
        // No verifiers ran — vacuously clean.
        assert!(VerifyReport::empty().is_clean());
    }

    #[test]
    fn all_zero_is_clean() {
        let r = VerifyReport::empty()
            .set_drc(VerifierResult::clean())
            .set_lvs(VerifierResult::clean())
            .set_em(VerifierResult::clean());
        assert!(r.is_clean());
        assert_eq!(r.total(), 0);
    }

    #[test]
    fn any_violation_flips_clean_to_false() {
        let r = VerifyReport::empty()
            .set_drc(VerifierResult::clean())
            .set_em(VerifierResult::from_count(2, "MET1.W: 50/0..50 too narrow"));
        assert!(!r.is_clean());
        assert_eq!(r.total(), 2);
    }

    #[test]
    fn summary_string_marks_unrun_verifiers() {
        let r = VerifyReport::empty()
            .set_drc(VerifierResult::clean())
            .set_em(VerifierResult::from_count(3, "—"));
        assert_eq!(r.summary_string(), "DRC=0 LVS=- EM=3");
    }

    #[test]
    fn round_trips_through_serde() {
        let r = VerifyReport::empty()
            .set_drc(VerifierResult::from_count(1, "MET1.S: 0,0..1k,1k"))
            .set_lvs(VerifierResult::clean());
        let s = serde_json::to_string(&r).unwrap();
        let r2: VerifyReport = serde_json::from_str(&s).unwrap();
        assert_eq!(r2.is_clean(), r.is_clean());
        assert_eq!(r2.total(), r.total());
    }
}
