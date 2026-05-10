//! spike-pinn-diode — methodologically rigorous PINN demo on the
//! nonlinear diode-RC transient.
//!
//! This crate exists because the `spike-pinn-rc` demo would not
//! survive a hostile review (linear problem with closed form,
//! hyperparameters fitted to the test, single seed, no ablation, no
//! OOD, no baselines beyond MNA). See `preregistration.md` for the
//! full methodological protocol — and §16 for the recorded
//! pre-training amendment that swapped the data oracle from ngspice
//! to `spike_diode::ref_transient`.
//!
//! Status: **pre-registration locked, smoke implementation landed,
//! full-protocol runner pending.** The `runner` module exposes a
//! smoke run (`hybrid` ablation, K=2 seeds, reduced `N_TRAIN` /
//! `N_STEPS`) that exercises the pipeline end-to-end on either CPU
//! or MLX. The full K=10 × 3-ablation × 5-baseline run lands in a
//! follow-on PR against the locked protocol.

pub mod baselines;
pub mod config;
pub mod encoding;
pub mod graph;
pub mod inference;
pub mod metrics;
pub mod oracle;
pub mod polynomial;
pub mod runner;
pub mod sampling;
pub mod stats;
pub mod train;
