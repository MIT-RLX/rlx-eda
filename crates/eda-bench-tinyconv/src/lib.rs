//! `eda-bench-tinyconv` — TinyConv-MNIST silicon bench harness.
//!
//! Three backends (in-house code-defined sky130, Yosys/OpenROAD in
//! docker, FPGA via `rlx-fpga`) measured on two metric arms (physical:
//! area/power/freq/parasitics/thermal; functional: top-1 accuracy under
//! PVT × MC). Functional accuracy is the load-bearing metric; physical
//! metrics ride on top.
//!
//! Full plan + scope + build order in `PLAN.md` at the crate root.
//!
//! ## Status
//!
//! Scaffolding. Module surface mirrors PLAN.md "Bench harness layout"
//! so the call sites are visible from day one; bodies are
//! `unimplemented!()`.

pub mod backends;
pub mod bisect;
pub mod bundle;
pub mod config;
pub mod inference;
pub mod manifest;
pub mod metrics;
pub mod optimization;
pub mod pnr;
pub mod report;

pub use config::BenchConfig;

pub use manifest::{Manifest, ManifestInputs};
pub use metrics::{Functional, Physical};
pub use report::Report;
