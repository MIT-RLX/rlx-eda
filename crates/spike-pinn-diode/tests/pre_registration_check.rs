//! Pre-registration enforcement test.
//!
//! Reads `preregistration.md` and asserts each documented constant
//! matches the corresponding `pub const` in `config.rs`. This is the
//! crate's methodological backbone — without it, "pre-registered"
//! reduces to a polite docstring and silent drift defeats the whole
//! point.
//!
//! The test is deliberately strict: drift in either direction fails.
//! If the document is updated, the constant must be too. If a
//! constant is changed mid-experiment, the document forces an
//! explicit, dated edit. Either way, the change becomes visible in
//! version control instead of slipping in as a typo.
//!
//! Implementation note: the test parses the markdown loosely (regex-
//! free, line-by-line) rather than mounting a full Markdown parser.
//! The scope is small enough that brittle parsing is the right
//! tradeoff — anyone editing the protocol sees this test as the
//! canary.

use std::fs;
use std::path::PathBuf;

use spike_pinn_diode::config::*;

fn read_preregistration() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("preregistration.md");
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()))
}

/// Looks for a substring `needle` in the doc and panics with a
/// descriptive message if it is missing. Intent: assert the doc
/// literally contains the same numbers the code uses, so any drift
/// shows up.
fn require(doc: &str, needle: &str, label: &str) {
    assert!(
        doc.contains(needle),
        "preregistration.md missing `{needle}` ({label}); did you change the constant without updating the doc?"
    );
}

#[test]
fn parameter_ranges_match_doc() {
    let doc = read_preregistration();

    // §3 in-dist ranges. The doc renders as `[1e3, 1e5]` and similar.
    require(&doc, "`[1e3, 1e5]`", "R range");
    require(&doc, "`[1e-14, 1e-12]`", "Is range");
    require(&doc, "`[1e-10, 1e-8]`", "C range");
    require(&doc, "`[0.5, 1.5]`", "V_dc range");
    require(&doc, "`[0.01, 5.0]`", "t/τ range");

    // §3 OOD ranges.
    require(&doc, "`[1e2, 1e3]`", "R OOD");
    require(&doc, "`[1e-12, 1e-11]`", "Is OOD");
    require(&doc, "`[1e-8, 1e-7]`", "C OOD");
    require(&doc, "`[1.5, 2.0]`", "V_dc OOD");

    // Numeric parity.
    assert_eq!(R_LO, 1.0e3);   assert_eq!(R_HI, 1.0e5);
    assert_eq!(IS_LO, 1.0e-14); assert_eq!(IS_HI, 1.0e-12);
    assert_eq!(C_LO, 1.0e-10); assert_eq!(C_HI, 1.0e-8);
    assert_eq!(VDC_LO, 0.5);   assert_eq!(VDC_HI, 1.5);
    assert_eq!(T_OVER_TAU_LO, 0.01); assert_eq!(T_OVER_TAU_HI, 5.0);
}

#[test]
fn splits_match_doc() {
    let doc = read_preregistration();
    require(&doc, "12,000", "N_TRAIN");
    require(&doc, "4,000", "N_VAL/TEST/OOD");
    require(&doc, "0xD10DE_5EED", "LHS seed");
    require(&doc, "0xCAFE_BABE", "test seed");
    require(&doc, "0xDEAD_BEEF", "OOD seed");

    assert_eq!(N_TRAIN, 12_000);
    assert_eq!(N_VAL,   4_000);
    assert_eq!(N_TEST,  4_000);
    assert_eq!(N_OOD,   4_000);
    assert_eq!(SPLIT_SEED_LHS,  0xD10DE_5EED);
    assert_eq!(SPLIT_SEED_TEST, 0xCAFE_BABE);
    assert_eq!(SPLIT_SEED_OOD,  0xDEAD_BEEF);
}

#[test]
fn architecture_matches_doc() {
    let doc = read_preregistration();
    require(&doc, "(5 → 64 → 64 → 64 → 1)", "MLP shape");
    require(&doc, "**8,769**", "param count");

    assert_eq!(ARCH_DIMS, &[5, 64, 64, 64, 1]);
    assert_eq!(TOTAL_PARAMS, 8_769);

    let computed: usize = ARCH_DIMS
        .windows(2)
        .map(|ab| ab[0] * ab[1] + ab[1])
        .sum();
    assert_eq!(
        computed, TOTAL_PARAMS,
        "TOTAL_PARAMS const drifted from ARCH_DIMS-derived count"
    );
}

#[test]
fn hyperparameters_match_doc() {
    let doc = read_preregistration();
    require(&doc, "`3e-4`", "LR");
    require(&doc, "20,000", "N_STEPS");
    require(&doc, "FD ε (normalised) | `1e-3`", "FD eps");
    require(&doc, "λ_ic (all configs) | 10.0", "lambda_ic");
    require(&doc, "Seeds | `[1..=10]`", "seeds");

    assert_eq!(LR, 3.0e-4);
    assert_eq!(BATCH, 256);
    assert_eq!(N_STEPS, 20_000);
    assert_eq!(EPS_T_NORM, 1.0e-3);
    assert_eq!(LAMBDA_IC, 10.0);
    assert_eq!(N_SEEDS, 10);
}

#[test]
fn ablation_grid_matches_doc() {
    let doc = read_preregistration();
    // Grep for the table rows verbatim.
    require(&doc, "| A | 1.0 | 0.0 | 10.0", "ablation A");
    require(&doc, "| B | 0.0 | 1.0 | 0.0", "ablation B");
    require(&doc, "| H | 1.0 | 1.0 | 10.0", "ablation H");

    assert_eq!(ABLATIONS.len(), 3);
    assert_eq!(ABL_PURE_PINN.lambda_phys, 1.0);
    assert_eq!(ABL_PURE_PINN.lambda_data, 0.0);
    assert_eq!(ABL_PURE_PINN.lambda_ic,   LAMBDA_IC);
    assert_eq!(ABL_PURE_SURROGATE.lambda_phys, 0.0);
    assert_eq!(ABL_PURE_SURROGATE.lambda_data, 1.0);
    assert_eq!(ABL_PURE_SURROGATE.lambda_ic,   0.0);
    assert_eq!(ABL_HYBRID.lambda_phys, 1.0);
    assert_eq!(ABL_HYBRID.lambda_data, 1.0);
    assert_eq!(ABL_HYBRID.lambda_ic,   LAMBDA_IC);
}

#[test]
fn baseline_configs_match_doc() {
    let doc = read_preregistration();
    require(&doc, "M-coarse",  "MNA coarse");
    require(&doc, "M-default", "MNA default");
    require(&doc, "M-fine",    "MNA fine");
    require(&doc, "16⁵ grid",  "lookup grid");
    require(&doc, "degree 4",  "polynomial degree");

    assert_eq!(MNA_BASELINES.len(), 3);
    assert_eq!(LOOKUP_GRID_PER_AXIS, 16);
    assert_eq!(LOOKUP_TABLE_BYTES, 16usize.pow(5) * 4);
    assert_eq!(POLY_DEGREE, 4);
}

#[test]
fn acceptance_criteria_thresholds_match_doc() {
    let doc = read_preregistration();
    require(&doc, "ratio (#9) ≤ 2.0", "C2 threshold");
    require(&doc, "≥ 1σ on max-abs-err", "C3 threshold");
    require(&doc, "100× lower memory", "C4 threshold");
    require(&doc, "< 10% full-scale",   "C5 threshold");

    assert_eq!(C2_OOD_RATIO_MAX,       2.0);
    assert_eq!(C3_HYBRID_BEATS_DATA_BY, 1.0);
    assert_eq!(C4_LOOKUP_MEMORY_RATIO, 100.0);
    assert_eq!(C5_OOD_MAX_ABS_ERR_FS,  0.10);
}

#[test]
fn amendment_2026_05_10a_recorded() {
    let doc = read_preregistration();
    require(&doc, "Amendment 2026-05-10a", "oracle amendment");
    require(&doc, "spike_diode::ref_transient", "oracle source");
}

#[test]
fn amendment_2026_05_10b_recorded() {
    let doc = read_preregistration();
    require(&doc, "Amendment 2026-05-10b", "warmup amendment");
    require(&doc, "λ_phys(step) = 0", "warmup formula");
    require(&doc, "step < N_STEPS/2", "warmup midpoint");
    require(&doc, "Lookup baseline (§10 row L) deferred", "lookup deferral");
}


#[test]
fn statistics_match_doc() {
    let doc = read_preregistration();
    require(&doc, "α = 0.05", "alpha");
    require(&doc, "Holm-Bonferroni", "correction");
    assert_eq!(ALPHA, 0.05);
    assert_eq!(HOLM_FAMILY_SIZE, 5);
}
