//! Pure-function tests for the ORFS metrics-JSON parser.
//!
//! Does NOT require docker, ORFS, or sky130 to run. Exercises the
//! contract that `run_orfs.sh` and `Physical::from_orfs_json` agree
//! on. If `run_orfs.sh` changes its emit shape, these tests fail.

use eda_bench_tinyconv::Physical;

const ORFS_FULL_OUTPUT: &str = r#"{
    "area_um2": 4321.5,
    "max_freq_mhz": 250.0,
    "wns_ns": 0.12,
    "dynamic_power_mw": 1.7,
    "leakage_power_mw": null,
    "parasitic_cap_ff": 880.0,
    "peak_temp_c": null,
    "energy_pj_per_inference": null
}"#;

#[test]
fn parses_full_orfs_metrics_json() {
    let p = Physical::from_orfs_json(ORFS_FULL_OUTPUT).expect("parse");
    assert_eq!(p.area_um2, Some(4321.5));
    assert_eq!(p.max_freq_mhz, Some(250.0));
    assert_eq!(p.wns_ns, Some(0.12));
    assert_eq!(p.dynamic_power_mw, Some(1.7));
    assert_eq!(p.leakage_power_mw, None);
    assert_eq!(p.parasitic_cap_ff, Some(880.0));
    assert_eq!(p.peak_temp_c, None);
    assert_eq!(p.energy_pj_per_inference, None);
}

#[test]
fn parses_partial_flow_with_all_nulls() {
    // What `run_orfs.sh` emits when ORFS errored before producing
    // most reports — every field becomes `null`. The bench harness
    // can detect "partial flow" by counting `Some(_)` fields.
    let all_null = r#"{
        "area_um2": null,
        "max_freq_mhz": null,
        "wns_ns": null,
        "dynamic_power_mw": null,
        "leakage_power_mw": null,
        "parasitic_cap_ff": null,
        "peak_temp_c": null,
        "energy_pj_per_inference": null
    }"#;
    let p = Physical::from_orfs_json(all_null).expect("parse");
    assert!(p.area_um2.is_none());
    assert!(p.max_freq_mhz.is_none());
}

#[test]
fn rejects_malformed_json() {
    let bad = r#"{ "area_um2": "not a number" }"#;
    assert!(Physical::from_orfs_json(bad).is_err());
}
