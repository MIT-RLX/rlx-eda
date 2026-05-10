//! `ModelCard` tests — round-trip + drift detection.

use spike_tinyconv_tile::ModelCard;

#[test]
fn default_card_is_marked_placeholder() {
    assert!(ModelCard::default().is_placeholder());
}

#[test]
fn modified_card_is_not_placeholder() {
    let mut c = ModelCard::default();
    c.k_leak = 0.001;
    assert!(!c.is_placeholder());
}

#[test]
fn toml_round_trip_preserves_all_fields() {
    let original = ModelCard::default();
    let s = original.to_toml();
    let restored = ModelCard::from_toml(&s).expect("toml parses");
    assert_eq!(original, restored);
}

#[test]
fn toml_round_trip_preserves_user_calibration() {
    // Simulate what a calibration run produces: non-default values
    // for every constant.
    let calibrated = ModelCard {
        n_cells_digital: 202.0, // unchanged — comes from floorplan, not calibration
        c_per_cell: 7.5,
        activity_factor: 0.18,
        f_clk_scale: 1.2,
        k_leak: 0.07,
        n_critical_stages: 40.0,
        k_delay: 0.55,
        a0_per_cell: 0.045,
        a_per_wl: 0.018,
    };
    let restored = ModelCard::from_toml(&calibrated.to_toml()).unwrap();
    assert_eq!(calibrated, restored);
    assert!(!restored.is_placeholder(), "calibrated card must not match default");
}

#[test]
fn toml_format_is_readable() {
    // Quick sanity: the emitted TOML names every field as the
    // contributor would expect (no opaque indices).
    let s = ModelCard::default().to_toml();
    for field in [
        "n_cells_digital",
        "c_per_cell",
        "activity_factor",
        "f_clk_scale",
        "k_leak",
        "n_critical_stages",
        "k_delay",
        "a0_per_cell",
        "a_per_wl",
    ] {
        assert!(s.contains(field), "field {field} missing in TOML output:\n{s}");
    }
}
