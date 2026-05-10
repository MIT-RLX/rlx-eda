//! Tier 3 of the validation pyramid — ngspice tt + Monte Carlo
//! cross-validation of the analytic closed form.
//!
//! Uses the `mc_*_switch` override pattern (per the workspace memory
//! note `sky130_mc_composition.md`: chaining `.lib tt` + `.lib mc`
//! chokes on $-placeholder lines). Always writes the deck to a temp
//! file (per `ngspice_stdin_title.md`).
//!
//! Soft-skips when the sky130 library isn't checked out, mirroring
//! `eda-pdks::HAS_SKY130`.

#![cfg(feature = "ngspice")]

#[test]
#[ignore = "scaffold — bodies land in PLAN.md step 3"]
fn tt_corner_matches_analytic_within_1pct() {}

#[test]
#[ignore = "scaffold — bodies land in PLAN.md step 3"]
fn mc_sigma_matches_noise_model_within_2x() {}
