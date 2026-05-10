//! Three backends, each with a distinct role:
//!
//! - [`inhouse`]  — design under test. Differentiable, fast iteration.
//! - [`orfs`]     — physical ground truth (Yosys/OpenROAD/OpenRCX/PDN
//!                   in docker). Industry-validated, slow (minutes).
//! - [`fpga`]     — functional ground truth at scale. Only backend
//!                   that can run the full 10k test set under any
//!                   candidate config in seconds.
//!
//! Designed so a future `SiliconBackend` (real chip in loop) can drop
//! in cleanly. PLAN.md "Deferred" calls this out explicitly.

pub mod fpga;
pub mod inhouse;
pub mod orfs;
pub mod rtl_sim;
pub mod yosys_sky130;

use crate::metrics::{Functional, Physical};

pub trait Backend {
    fn name(&self) -> &'static str;

    /// Synthesize/lower the design under this backend, returning
    /// physical metrics. Slow for ORFS (minutes); fast for in-house.
    fn measure_physical(&self) -> Result<Physical, BackendError>;

    /// Run inference at the requested validation level on the supplied
    /// image set. PLAN.md "Validation" defines which level each backend
    /// can serve.
    fn measure_functional(
        &self,
        level: crate::metrics::functional::Level,
        images: &[u32],
    ) -> Result<Functional, BackendError>;
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("backend {0} not enabled (build with --features {1})")]
    NotEnabled(&'static str, &'static str),
    #[error("validation level not supported by this backend")]
    LevelUnsupported,
    #[error("toolchain failure: {0}")]
    Toolchain(String),
}
