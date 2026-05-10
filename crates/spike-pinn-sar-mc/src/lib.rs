//! spike-pinn-sar-mc — high-D SAR-with-mismatch PINN/surrogate
//! experiment. 10-D inputs (vin + 8 bit-weight mismatches +
//! comparator offset). Polynomial baselines only — lookup is
//! infeasible at this dimension. See `preregistration.md`.

pub mod baselines;
pub mod config;
pub mod graph;
pub mod inference;
pub mod metrics;
pub mod oracle;
pub mod runner;
pub mod sample;
pub mod sampling;
pub mod stats;
pub mod train;
