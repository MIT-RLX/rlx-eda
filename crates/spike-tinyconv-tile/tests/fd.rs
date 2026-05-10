//! Tier 2 of the validation pyramid — finite-difference sensitivities
//! cross-checked against autodiff through `DcBehavioral::add_to_dc`.
//!
//! Witness for "the gradient the Adam loop sees is the gradient that
//! actually exists." Same shape as `cpu_sqrt_grad.rs` over in `../rlx`
//! (per the user memory note that activation/Sqrt grads are
//! FD-validated).

#[test]
#[ignore = "scaffold — bodies land in PLAN.md step 3"]
fn fd_matches_ad_on_w_l_n() {}

#[test]
#[ignore = "scaffold — bodies land in PLAN.md step 3"]
fn fd_matches_ad_on_vdd() {}

#[test]
#[ignore = "scaffold — bodies land in PLAN.md step 3"]
fn fd_matches_ad_on_bias_v() {}
