//! Tier 1 of the validation pyramid — analytic closed-form gain /
//! delay / static power as a function of `TileParams`.
//!
//! Runs in every CI cycle (no foundry library / ngspice required).
//! Closed-form values cross-check against the `DcBehavioral` residual
//! at a handful of representative `TileParams` points; any mismatch
//! means the residual or the closed form has drifted.

#[test]
#[ignore = "scaffold — bodies land in PLAN.md step 3"]
fn analytic_gain_matches_residual_at_nominal_params() {}

#[test]
#[ignore = "scaffold — bodies land in PLAN.md step 3"]
fn analytic_delay_monotone_in_w_l_n() {}

#[test]
#[ignore = "scaffold — bodies land in PLAN.md step 3"]
fn static_power_scales_with_vdd_squared() {}
