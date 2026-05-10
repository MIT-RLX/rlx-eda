//! Golden-file harness for waveform regression tests.
//!
//! The pattern: a test runs the simulator, gets a `Waveform`, and
//! either records it as the blessed reference (first run, or when the
//! engineer accepts a deliberate change) or diffs it against the
//! reference (every other run). One call handles both:
//!
//! ```ignore
//! let candidate = run_simulation();
//! golden::assert_matches_golden(
//!     "tests/golden/divider.csv",
//!     &candidate,
//!     diff::Tol::new(1e-6, 1e-9),
//!     "divider transient",
//! );
//! ```
//!
//! ## Bless mode
//!
//! Set `RLX_BLESS=1` in the environment to overwrite the golden with
//! the current candidate. Use this when the change in the candidate is
//! intentional (e.g. you tightened a model and the trace shifted by
//! 0.5 % everywhere). Commit the regenerated golden file alongside
//! the model change.
//!
//! ## Storage
//!
//! CSV via [`crate::csv`]. Pandas-friendly, opens in cicwave, and the
//! diff stays human-readable in code review. Real-valued waveforms only
//! — AC golden traces aren't a flow we have today; add a complex
//! variant when one shows up.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::csv as wcsv;
use crate::diff::{self, DiffReport, Tol};
use crate::Waveform;

#[derive(Debug, Error)]
pub enum GoldenError {
    #[error("io error reading {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("csv error on {path:?}: {source}")]
    Csv {
        path: PathBuf,
        #[source]
        source: wcsv::CsvError,
    },
    #[error("diff error: {0}")]
    Diff(#[from] diff::DiffError),
    #[error("golden file {0:?} is missing and mode is Compare (no record fallback)")]
    MissingGolden(PathBuf),
    #[error("golden harness only supports Waveform::Real (got Complex)")]
    NotReal,
}

/// Bless or compare? The default flow honors `RLX_BLESS` env var; tests
/// can pin a specific mode to stay deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Bless if `RLX_BLESS` is set or the file doesn't exist; else compare.
    Auto,
    /// Always write the candidate to the path.
    Bless,
    /// Always read + diff. Errors if the file doesn't exist.
    Compare,
}

/// Outcome of a `record_or_compare` call.
#[derive(Debug)]
pub enum GoldenReport {
    /// Candidate was written to disk. No comparison performed.
    Recorded { path: PathBuf },
    /// File existed and was diffed against the candidate.
    Compared { path: PathBuf, diff: DiffReport },
}

impl GoldenReport {
    pub fn path(&self) -> &Path {
        match self {
            GoldenReport::Recorded { path } | GoldenReport::Compared { path, .. } => path,
        }
    }
}

/// Record (in bless mode) or compare (otherwise) a candidate waveform
/// against a CSV file at `path`.
pub fn record_or_compare(
    path: impl AsRef<Path>,
    candidate: &Waveform,
    tol: Tol,
    mode: Mode,
) -> Result<GoldenReport, GoldenError> {
    if !matches!(candidate, Waveform::Real { .. }) {
        return Err(GoldenError::NotReal);
    }
    let path = path.as_ref().to_path_buf();
    let exists = path.exists();
    let blessing = match mode {
        Mode::Bless => true,
        Mode::Compare => false,
        Mode::Auto => env_bless() || !exists,
    };

    if !blessing && !exists {
        return Err(GoldenError::MissingGolden(path));
    }

    if blessing {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| GoldenError::Io {
                    path: parent.to_path_buf(),
                    source: e,
                })?;
            }
        }
        let f = std::fs::File::create(&path).map_err(|e| GoldenError::Io {
            path: path.clone(),
            source: e,
        })?;
        wcsv::write(candidate, f).map_err(|e| GoldenError::Csv {
            path: path.clone(),
            source: e,
        })?;
        Ok(GoldenReport::Recorded { path })
    } else {
        let f = std::fs::File::open(&path).map_err(|e| GoldenError::Io {
            path: path.clone(),
            source: e,
        })?;
        let golden = wcsv::read_real(f).map_err(|e| GoldenError::Csv {
            path: path.clone(),
            source: e,
        })?;
        let report = diff::diff(&golden, candidate, tol, &Default::default())?;
        Ok(GoldenReport::Compared { path, diff: report })
    }
}

/// Wrapper that records on first run / bless mode, and panics with a
/// pinpoint diff message otherwise.
#[track_caller]
pub fn assert_matches_golden(
    path: impl AsRef<Path>,
    candidate: &Waveform,
    tol: Tol,
    label: &str,
) {
    match record_or_compare(path, candidate, tol, Mode::Auto) {
        Ok(GoldenReport::Recorded { path }) => {
            // First-time bless or RLX_BLESS=1 — silently accept and tell
            // the engineer the file was (re)written.
            eprintln!("[golden:{label}] recorded {}", path.display());
        }
        Ok(GoldenReport::Compared { diff, .. }) => diff.assert_ok(label),
        Err(e) => panic!("[golden:{label}] {e}"),
    }
}

fn env_bless() -> bool {
    matches!(
        std::env::var("RLX_BLESS").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn fixture(scale: f64) -> Waveform {
        let mut signals = BTreeMap::new();
        signals.insert(
            "v(out)".to_string(),
            vec![0.0, 0.5 * scale, 1.0 * scale, 0.5 * scale, 0.0],
        );
        Waveform::Real {
            axis_name: "time".into(),
            axis: vec![0.0, 1e-9, 2e-9, 3e-9, 4e-9],
            signals,
        }
    }

    #[test]
    fn first_run_records_then_subsequent_compares() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("golden.csv");

        // First call: file doesn't exist → Recorded.
        let r1 = record_or_compare(&path, &fixture(1.0), Tol::new(1e-9, 1e-12), Mode::Auto)
            .unwrap();
        assert!(matches!(r1, GoldenReport::Recorded { .. }));
        assert!(path.exists());

        // Second call (same data): Compared and clean.
        let r2 = record_or_compare(&path, &fixture(1.0), Tol::new(1e-9, 1e-12), Mode::Auto)
            .unwrap();
        match r2 {
            GoldenReport::Compared { diff, .. } => assert!(diff.is_ok()),
            _ => panic!("expected Compared on second run"),
        }
    }

    #[test]
    fn detects_drift_on_compare() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("golden.csv");
        // Bless once.
        record_or_compare(&path, &fixture(1.0), Tol::new(1e-9, 1e-12), Mode::Bless).unwrap();
        // Drift in candidate.
        let r = record_or_compare(&path, &fixture(1.5), Tol::new(1e-6, 1e-9), Mode::Compare)
            .unwrap();
        match r {
            GoldenReport::Compared { diff, .. } => {
                assert!(!diff.is_ok());
                assert_eq!(diff.divergent(), vec!["v(out)"]);
            }
            _ => panic!("expected Compared"),
        }
    }

    #[test]
    fn bless_mode_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("golden.csv");
        record_or_compare(&path, &fixture(1.0), Tol::new(1e-9, 1e-12), Mode::Bless).unwrap();
        let len1 = std::fs::metadata(&path).unwrap().len();

        // Re-bless with a longer signal — file content must change.
        let mut signals = BTreeMap::new();
        signals.insert("v(out)".into(), vec![0.0; 100]);
        let bigger = Waveform::Real {
            axis_name: "time".into(),
            axis: (0..100).map(|i| i as f64 * 1e-9).collect(),
            signals,
        };
        record_or_compare(&path, &bigger, Tol::new(1e-9, 1e-12), Mode::Bless).unwrap();
        let len2 = std::fs::metadata(&path).unwrap().len();
        assert!(len2 > len1, "expected re-blessed file to grow ({len1} → {len2})");
    }

    #[test]
    fn compare_mode_errors_if_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("never_recorded.csv");
        let r = record_or_compare(&path, &fixture(1.0), Tol::new(1e-9, 1e-12), Mode::Compare);
        assert!(matches!(r, Err(GoldenError::MissingGolden(_))));
    }

    #[test]
    fn rejects_complex_waveform() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("golden.csv");
        let w = Waveform::Complex {
            axis_name: "frequency".into(),
            axis: vec![1e3],
            signals: BTreeMap::new(),
        };
        let r = record_or_compare(&path, &w, Tol::new(1e-9, 1e-12), Mode::Auto);
        assert!(matches!(r, Err(GoldenError::NotReal)));
    }

    // Note: `assert_matches_golden` consults RLX_BLESS, which is racy
    // to test alongside other tests in the same process. The compare
    // path is exercised directly via `detects_drift_on_compare`; the
    // wrapper is a 4-line transform on top of it.
}
