//! spike-pinn-sar — methodologically rigorous PINN/surrogate
//! experiment on the 8-bit SAR ADC.
//!
//! Mirrors `spike-pinn-diode`'s pre-registration → parity → ablation
//! → baselines → stats → results pipeline. SAR-specific:
//!
//! - 1-D input `vin/vref ∈ [0, 1]`, 1-D output `code/256 ∈ [0, 1]`.
//! - Discrete 256-step staircase output (not smooth in input).
//! - No physics term (SAR is a discrete iterative algorithm).
//! - Only Row B (pure surrogate); no Row A or H.
//! - Baselines: polynomial (deg 4/8/16) + lookup (16/64/256-node).
//!
//! See `preregistration.md` for the locked protocol; the in-code
//! constants are mirrored as `pub const` items in `config.rs` and
//! parity is enforced by `tests/pre_registration_check.rs`.

pub mod baselines;
pub mod config;
pub mod graph;
pub mod inference;
pub mod metrics;
pub mod oracle;
pub mod runner;
pub mod sampling;
pub mod stats;
pub mod train;
