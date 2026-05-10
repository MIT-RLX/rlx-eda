//! Parity test: every documented constant matches `config.rs`.

use std::fs;
use std::path::PathBuf;

use spike_pinn_sar::config::*;

fn read_doc() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("preregistration.md");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()))
}

fn require(doc: &str, needle: &str, label: &str) {
    assert!(
        doc.contains(needle),
        "preregistration.md missing `{needle}` ({label}); did you change the constant without updating the doc?"
    );
}

#[test]
fn problem_locked_to_8_bit_ideal_sar() {
    let doc = read_doc();
    require(&doc, "BehavioralSar", "oracle source");
    require(&doc, "8-bit", "8-bit lock");
    require(&doc, "Realization::ideal", "ideal realization");
    assert_eq!(N_BITS, 8);
    assert_eq!(VREF, 1.0);
    assert_eq!(LEVELS, 256);
}

#[test]
fn splits_match_doc() {
    let doc = read_doc();
    require(&doc, "12,000", "N_TRAIN");
    require(&doc, "4,000", "N_VAL/TEST");
    require(&doc, "0xCAB5_5AAD", "train seed");
    require(&doc, "0xC0DE_BABE", "test seed");
    assert_eq!(N_TRAIN, 12_000);
    assert_eq!(N_VAL,   4_000);
    assert_eq!(N_TEST,  4_000);
    assert_eq!(SPLIT_SEED_TRAIN, 0xCAB5_5AAD);
    assert_eq!(SPLIT_SEED_TEST,  0xC0DE_BABE);
}

#[test]
fn architecture_matches_doc() {
    let doc = read_doc();
    require(&doc, "32 → 32", "MLP hidden width");
    require(&doc, "1,153", "param count");
    assert_eq!(ARCH_DIMS, &[1, 32, 32, 1]);
    assert_eq!(TOTAL_PARAMS, 1_153);

    let computed: usize = ARCH_DIMS
        .windows(2)
        .map(|ab| ab[0] * ab[1] + ab[1])
        .sum();
    assert_eq!(computed, TOTAL_PARAMS);
}

#[test]
fn hyperparameters_match_doc() {
    let doc = read_doc();
    require(&doc, "`3e-4`", "LR");
    require(&doc, "20,000", "N_STEPS");
    require(&doc, "Batch size | 128", "batch");
    require(&doc, "`[1..=10]`", "seed range");
    assert_eq!(LR, 3.0e-4);
    assert_eq!(BATCH, 128);
    assert_eq!(N_STEPS, 20_000);
    assert_eq!(N_SEEDS, 10);
}

#[test]
fn baselines_match_doc() {
    let doc = read_doc();
    for d in [4, 8, 16].iter() {
        require(&doc, &format!("Poly-d{d}"), "polynomial baseline");
    }
    for n in [16, 64, 256].iter() {
        require(&doc, &format!("Lookup-{n}"), "lookup baseline");
    }
    assert_eq!(POLY_DEGREES, &[4, 8, 16]);
    assert_eq!(LOOKUP_SIZES, &[16, 64, 256]);
}

#[test]
fn statistics_match_doc() {
    let doc = read_doc();
    require(&doc, "α = 0.05", "alpha");
    require(&doc, "Holm-Bonferroni", "correction");
    require(&doc, "K = 10", "seed count");
    assert_eq!(ALPHA, 0.05);
    assert_eq!(HOLM_FAMILY_SIZE, 6);
}

#[test]
fn acceptance_criteria_thresholds_match_doc() {
    let doc = read_doc();
    require(&doc, "C1' (capacity)", "C1' label");
    require(&doc, "C2' (memory)",   "C2' label");
    require(&doc, "C5' (sub-LSB)",  "C5' label");
    require(&doc, "1/512", "half-LSB threshold");
    assert!((C5_HALF_LSB - 1.0 / 512.0).abs() < 1e-9);
}
