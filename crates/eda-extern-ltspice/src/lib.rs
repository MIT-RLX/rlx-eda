//! LTspice external-validation driver.
//!
//! Mirror of [`eda_extern_ngspice`] for LTspice — same `Invoker` shape,
//! same `OutputRequest` / `DcResults` / `TransientTrace` / `AcTrace`
//! types, same shared waveform parsing via [`eda_waveform::nutmeg`].
//!
//! ## Why a separate crate
//!
//! ngspice and LTspice **mostly** speak the same SPICE deck — but
//! diverge at the analysis-control surface:
//!
//! - ngspice runs deck-level analysis directives via a `.control … .endc`
//!   block (`op`, `tran`, `print`). LTspice ignores `.control` and
//!   reads `.op` / `.tran` / `.ac` / `.save` as top-level deck
//!   directives.
//! - ngspice writes the raw output via an explicit `write <path>
//!   <signals>` command in the control block. LTspice always writes
//!   `<input>.raw` next to the input file when given `-b`.
//! - ngspice can read the deck from `/dev/stdin` (we exploit this for
//!   the temp-file-free path). LTspice requires a real input file.
//!
//! These differences are small but structural — a single Invoker would
//! end up with a `dialect: Dialect` enum fork at every method. Two
//! crates, both implementing a same-shaped `Invoker`, is cleaner.
//!
//! ## Pinning ASCII output
//!
//! We always pass `-ascii` to LTspice. Reasons:
//!   - LTspice's default binary RAW uses single-precision for dependent
//!     values (with double-precision for the time axis). The shared
//!     nutmeg parser assumes f64 for both. ASCII sidesteps this.
//!   - ASCII RAW is portable across LTspice versions; binary RAW has
//!     "fastaccess" variants that differ between LTspice IV and XVII.
//!   - File sizes for validation circuits are small; the parse cost
//!     is negligible.
//!
//! Full binary cross-sim support is a follow-on once we need million-
//! point AC sweeps.
//!
//! ## Finding the binary
//!
//! In order:
//!   1. `LTSPICE_BIN` env var (full path).
//!   2. `LTspice` on PATH (Linux package builds use this name).
//!   3. `ltspice` on PATH (some package managers lowercase).
//!   4. `/Applications/LTspice.app/Contents/MacOS/LTspice` (macOS app
//!      bundle — stable across LTspice versions).
//!
//! Tests use `from_env_optional()` so they soft-skip when LTspice is
//! not installed (CI matrix can run with / without).

use std::path::PathBuf;
use std::process::Command;
use thiserror::Error;

pub use eda_waveform::nutmeg;

/// What we ask LTspice to report at the end of an analysis. Same shape
/// as ngspice's `OutputRequest`.
#[derive(Debug, Clone)]
pub enum OutputRequest {
    NodeVoltage(String),
}

#[derive(Debug, Default, Clone)]
pub struct DcResults {
    pub node_voltages: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Default, Clone)]
pub struct TransientTrace {
    pub time: Vec<f64>,
    pub node_voltages: std::collections::HashMap<String, Vec<f64>>,
}

#[derive(Debug, Default, Clone)]
pub struct AcTrace {
    pub frequency: Vec<f64>,
    pub node_voltages: std::collections::HashMap<String, Vec<(f64, f64)>>,
}

#[derive(Debug, Clone, Copy)]
pub struct TransientAnalysis {
    pub t_step: f64,
    pub t_stop: f64,
    /// Maps to LTspice's `uic` keyword on `.tran`. See ngspice driver
    /// docs for why this defaults to `true` for rlx-eda comparisons.
    pub use_initial_conditions: bool,
}

impl TransientAnalysis {
    pub fn new(t_step: f64, t_stop: f64) -> Self {
        Self { t_step, t_stop, use_initial_conditions: true }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AcAnalysis {
    pub points_per_decade: usize,
    pub f_start: f64,
    pub f_stop: f64,
}

impl AcAnalysis {
    pub fn dec(points_per_decade: usize, f_start: f64, f_stop: f64) -> Self {
        Self { points_per_decade, f_start, f_stop }
    }
}

#[derive(Debug, Error)]
pub enum LtspiceError {
    #[error(
        "LTspice binary not found (set LTSPICE_BIN or install LTspice; tried \
         $LTSPICE_BIN, PATH:LTspice/ltspice, /Applications/LTspice.app/Contents/MacOS/LTspice)"
    )]
    BinaryNotFound,
    #[error("LTspice exited non-zero ({code:?}); stderr:\n{stderr}")]
    NonZero { code: Option<i32>, stderr: String },
    #[error("LTspice did not produce expected raw file at {0}")]
    RawMissing(PathBuf),
    #[error("requested signal '{0}' not found in LTspice raw output")]
    OutputMissing(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("nutmeg parse error: {0}")]
    Nutmeg(#[from] nutmeg::NutmegError),
    #[error("LTspice produced wrong plot type: expected {expected}, got {got:?}")]
    WrongPlotKind { expected: &'static str, got: nutmeg::NutmegFlavor },
}

pub type Result<T> = std::result::Result<T, LtspiceError>;

/// Same trait shape as `eda_extern_ngspice::Invoker`. We deliberately
/// don't share the type so the two drivers stay independently
/// versioned; the triangulation harness takes whichever it needs.
pub trait Invoker: Send + Sync {
    fn run_dc(&self, deck: &str, requests: &[OutputRequest]) -> Result<DcResults>;
    fn run_transient_trace(
        &self,
        deck: &str,
        analysis: &TransientAnalysis,
        requests: &[OutputRequest],
    ) -> Result<TransientTrace>;
    fn run_ac(
        &self,
        deck: &str,
        analysis: &AcAnalysis,
        requests: &[OutputRequest],
    ) -> Result<AcTrace>;
}

/// Shells out to a native LTspice binary.
pub struct LocalBinary {
    pub binary: PathBuf,
}

impl LocalBinary {
    /// Resolve LTspice via env var → PATH (`LTspice`, `ltspice`) →
    /// macOS app bundle. Errors if none exist; use [`LocalBinary::from_env_optional`]
    /// in tests if you want a soft skip instead.
    pub fn from_env() -> Result<Self> {
        Self::from_env_optional().ok_or(LtspiceError::BinaryNotFound)
    }

    /// `Some(_)` if any of the candidate paths resolve to an executable;
    /// `None` if LTspice is not installed. For tests that skip when
    /// LTspice is absent.
    pub fn from_env_optional() -> Option<Self> {
        if let Ok(p) = std::env::var("LTSPICE_BIN") {
            let pb = PathBuf::from(p);
            if pb.is_file() {
                return Some(Self { binary: pb });
            }
        }
        for name in ["LTspice", "ltspice"] {
            if let Some(p) = eda_container::which(name) {
                return Some(Self { binary: p });
            }
        }
        let mac = PathBuf::from("/Applications/LTspice.app/Contents/MacOS/LTspice");
        if mac.is_file() {
            return Some(Self { binary: mac });
        }
        None
    }

    /// Write `deck + extra_directives` to a temp `<base>.cir`, run
    /// LTspice in batch+ASCII mode, and return `(tmp_dir,
    /// path_to_raw)`. The tempdir handle keeps the directory alive for
    /// the caller — drop it to clean up.
    fn run_to_raw(
        &self,
        deck: &str,
        extra_directives: &str,
    ) -> Result<(tempfile::TempDir, PathBuf)> {
        let dir = tempfile::Builder::new().prefix("rlx-eda-lt-").tempdir()?;
        let base = dir.path().join("deck");
        let cir = base.with_extension("cir");
        let raw = base.with_extension("raw");

        let mut full = String::new();
        full.push_str(deck.trim_end());
        full.push('\n');
        full.push_str(extra_directives);
        // LTspice requires `.end` to terminate the deck; ngspice tolerates
        // either. Emit it here so SpiceEmit decks don't need to.
        full.push_str(".end\n");
        std::fs::write(&cir, &full)?;

        let out = Command::new(&self.binary)
            .arg("-b")
            .arg("-ascii")
            .arg(&cir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()?;

        if !out.status.success() {
            return Err(LtspiceError::NonZero {
                code: out.status.code(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        if !raw.is_file() {
            return Err(LtspiceError::RawMissing(raw));
        }
        Ok((dir, raw))
    }

    /// Build the `.save` lines for a request set so LTspice's RAW only
    /// contains what we asked for. Without this LTspice saves every
    /// node voltage and every device current — fine for tiny decks,
    /// wasteful at scale.
    fn save_directives(requests: &[OutputRequest]) -> String {
        let mut s = String::new();
        for req in requests {
            match req {
                OutputRequest::NodeVoltage(n) => {
                    s.push_str(&format!(".save V({n})\n"));
                }
            }
        }
        s
    }
}

impl Invoker for LocalBinary {
    fn run_dc(&self, deck: &str, requests: &[OutputRequest]) -> Result<DcResults> {
        let directives = format!(".op\n{}", Self::save_directives(requests));
        let (_dir, raw) = self.run_to_raw(deck, &directives)?;
        let bytes = std::fs::read(&raw)?;
        let plot = nutmeg::parse_bytes(&bytes)?;
        if plot.flavor != nutmeg::NutmegFlavor::Real {
            return Err(LtspiceError::WrongPlotKind {
                expected: "real (operating point)",
                got: plot.flavor,
            });
        }
        let mut results = DcResults::default();
        for req in requests {
            match req {
                OutputRequest::NodeVoltage(n) => {
                    let key = format!("V({n})");
                    let v = plot
                        .real_trace(&key)
                        .ok_or_else(|| LtspiceError::OutputMissing(n.clone()))?;
                    let scalar = *v.first().ok_or_else(|| LtspiceError::OutputMissing(n.clone()))?;
                    results.node_voltages.insert(n.clone(), scalar);
                }
            }
        }
        Ok(results)
    }

    fn run_transient_trace(
        &self,
        deck: &str,
        analysis: &TransientAnalysis,
        requests: &[OutputRequest],
    ) -> Result<TransientTrace> {
        let mut directives = format!(
            ".tran {} {}{}\n",
            analysis.t_step,
            analysis.t_stop,
            if analysis.use_initial_conditions { " uic" } else { "" },
        );
        directives.push_str(&Self::save_directives(requests));
        let (_dir, raw) = self.run_to_raw(deck, &directives)?;
        let bytes = std::fs::read(&raw)?;
        let plot = nutmeg::parse_bytes(&bytes)?;
        if plot.flavor != nutmeg::NutmegFlavor::Real {
            return Err(LtspiceError::WrongPlotKind {
                expected: "real (transient)",
                got: plot.flavor,
            });
        }
        let time = plot
            .real_trace("time")
            .ok_or_else(|| LtspiceError::OutputMissing("time".into()))?
            .to_vec();
        let mut node_voltages = std::collections::HashMap::new();
        for req in requests {
            match req {
                OutputRequest::NodeVoltage(n) => {
                    let key = format!("V({n})");
                    let v = plot
                        .real_trace(&key)
                        .ok_or_else(|| LtspiceError::OutputMissing(n.clone()))?
                        .to_vec();
                    node_voltages.insert(n.clone(), v);
                }
            }
        }
        Ok(TransientTrace { time, node_voltages })
    }

    fn run_ac(
        &self,
        deck: &str,
        analysis: &AcAnalysis,
        requests: &[OutputRequest],
    ) -> Result<AcTrace> {
        let mut directives = format!(
            ".ac dec {} {} {}\n",
            analysis.points_per_decade, analysis.f_start, analysis.f_stop,
        );
        directives.push_str(&Self::save_directives(requests));
        let (_dir, raw) = self.run_to_raw(deck, &directives)?;
        let bytes = std::fs::read(&raw)?;
        let plot = nutmeg::parse_bytes(&bytes)?;
        if plot.flavor != nutmeg::NutmegFlavor::Complex {
            return Err(LtspiceError::WrongPlotKind {
                expected: "complex (ac)",
                got: plot.flavor,
            });
        }
        let frequency = plot
            .complex_trace("frequency")
            .ok_or_else(|| LtspiceError::OutputMissing("frequency".into()))?
            .iter()
            .map(|(re, _)| *re)
            .collect();
        let mut node_voltages = std::collections::HashMap::new();
        for req in requests {
            match req {
                OutputRequest::NodeVoltage(n) => {
                    let key = format!("V({n})");
                    let v = plot
                        .complex_trace(&key)
                        .ok_or_else(|| LtspiceError::OutputMissing(n.clone()))?
                        .to_vec();
                    node_voltages.insert(n.clone(), v);
                }
            }
        }
        Ok(AcTrace { frequency, node_voltages })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_directives_emit_per_request() {
        let s = LocalBinary::save_directives(&[
            OutputRequest::NodeVoltage("vout".into()),
            OutputRequest::NodeVoltage("mid".into()),
        ]);
        assert!(s.contains(".save V(vout)"));
        assert!(s.contains(".save V(mid)"));
    }

    #[test]
    fn discovery_is_safe_to_call_when_absent() {
        // Best-effort: discovery either finds LTspice or returns None,
        // never panics.
        let _ = LocalBinary::from_env_optional();
    }
}
