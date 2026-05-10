use std::fs;
use std::path::PathBuf;

use spike_pinn_sar_mc::config::*;

fn read_doc() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("preregistration.md");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()))
}

fn require(doc: &str, needle: &str, label: &str) {
    assert!(doc.contains(needle),
        "preregistration.md missing `{needle}` ({label}); did you change the constant without updating the doc?");
}

#[test]
fn problem_locks() {
    let doc = read_doc();
    require(&doc, "BehavioralSar",  "oracle source");
    require(&doc, "n_bits=8",       "8-bit lock");
    require(&doc, "σ_R = 5e-2",     "sigma R");
    require(&doc, "σ_offset = 5e-3","sigma offset");
    assert_eq!(N_BITS, 8);
    assert_eq!(VREF, 1.0);
    assert_eq!(LEVELS, 256);
    assert!((SIGMA_R - 5.0e-2).abs() < 1e-12);
    assert!((SIGMA_OFFSET - 5.0e-3).abs() < 1e-12);
}

#[test]
fn input_dim_is_10() {
    let doc = read_doc();
    require(&doc, "10-dimensional", "10-D claim");
    assert_eq!(INPUT_DIM, 10);
}

#[test]
fn splits_match_doc() {
    let doc = read_doc();
    require(&doc, "12,000", "N_TRAIN");
    require(&doc, "4,000",  "N_VAL/TEST");
    require(&doc, "0xCAB5_5AAD_5AAD_BEEF", "train seed");
    require(&doc, "0xC0DE_BABE_BABE_C0DE", "test seed");
    assert_eq!(N_TRAIN, 12_000);
    assert_eq!(N_VAL,   4_000);
    assert_eq!(N_TEST,  4_000);
    assert_eq!(SPLIT_SEED_TRAIN, 0xCAB5_5AAD_5AAD_BEEF);
    assert_eq!(SPLIT_SEED_TEST,  0xC0DE_BABE_BABE_C0DE);
}

#[test]
fn architecture_matches_doc() {
    let doc = read_doc();
    require(&doc, "10 → 64 → 64 → 1", "MLP shape");
    require(&doc, "4,929", "param count");
    assert_eq!(ARCH_DIMS, &[10, 64, 64, 1]);

    let computed: usize = ARCH_DIMS
        .windows(2)
        .map(|ab| ab[0] * ab[1] + ab[1])
        .sum();
    assert_eq!(computed, TOTAL_PARAMS);
    assert_eq!(TOTAL_PARAMS, 4_929);
}

#[test]
fn hyperparameters_match_doc() {
    let doc = read_doc();
    require(&doc, "`3e-4`", "LR");
    require(&doc, "20,000", "N_STEPS");
    require(&doc, "Batch | 256", "batch");
    require(&doc, "`[1..=10]`", "seed range");
    assert_eq!(LR, 3e-4);
    assert_eq!(BATCH, 256);
    assert_eq!(N_STEPS, 20_000);
    assert_eq!(N_SEEDS, 10);
}

#[test]
fn baselines_match_doc() {
    let doc = read_doc();
    for d in [1, 2, 4].iter() {
        require(&doc, &format!("Poly-d{d}"), "polynomial baseline");
    }
    assert_eq!(POLY_DEGREES, &[1, 2, 4]);
}

#[test]
fn acceptance_criteria() {
    let doc = read_doc();
    require(&doc, "C1'' (capacity)",  "C1''");
    require(&doc, "C2'' (capacity-progression)", "C2''");
    require(&doc, "C5'' (functional)", "C5''");
    require(&doc, "1 LSB",        "C5'' threshold");
    assert!((C5_ONE_LSB - 1.0 / 256.0).abs() < 1e-9);
    assert_eq!(HOLM_FAMILY_SIZE, 3);
    assert_eq!(ALPHA, 0.05);
}
