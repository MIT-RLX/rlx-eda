//! Cicsim-shaped simulation harness.
//!
//! Wraps `eda-extern-ngspice` with the simulate → measure → aggregate →
//! report loop that production analog verification needs:
//!
//! - [`Testbench`] — emits a per-corner SPICE deck (via `eda-spice-emit`)
//!   and declares its `.meas` measurements.
//! - [`Corner`] / [`CornerSet`] — typical, extreme test condition (etc),
//!   Monte Carlo. Each corner injects a `.lib` section + `.options`.
//! - [`Spec`] — `tran.yaml`-shaped pass/fail spec (name/min/typ/max/unit).
//! - [`MeasureLog`] — parses ngspice's `.meas` output lines.
//! - [`Cache`] — SHA-of-deck skip-rerun, mirroring cicsim's `--no-sha` /
//!   `--no-run` flags.
//! - [`Reporter`] — HTML + Markdown + PNG, matching cicsim's
//!   `make summary` output.
//! - [`Harness`] — the builder that ties it all together.

pub mod cache;
pub mod corner;
pub mod harness;
pub mod measure;
pub mod report;
pub mod spec;
pub mod testbench;
pub mod verify;

pub use cache::{Cache, CacheMode};
pub use corner::{Corner, CornerKind, CornerSet, View};
pub use harness::{Harness, HarnessError, RunOutcome};
pub use measure::{MeasureLog, Measurement, MeasurementValue};
pub use report::{docs_dir_for_crate, Reporter, SummaryRow};
pub use spec::{McSummaryStyle, Spec, SpecBundle, SpecCheck};
pub use testbench::{Analysis, Testbench};
pub use verify::{VerifierResult, VerifyReport};
