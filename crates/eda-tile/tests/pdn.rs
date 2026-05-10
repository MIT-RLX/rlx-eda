//! Tests for `current_density_check`. Pure function — no toolchain
//! required.

use eda_tile::{current_density_check, RailSpec};
use klayout_core::{LayerIndex, LayerInfo, Library};

/// Build a rail with given width (in DBU) on a fresh test library.
fn rail(width_dbu: i64) -> RailSpec {
    let lib = Library::new("pdn-test", 1000);
    let m1: LayerIndex = lib.layer(LayerInfo::gds(10, 0));
    let m2: LayerIndex = lib.layer(LayerInfo::gds(11, 0));
    RailSpec {
        vdd_layer: m1,
        gnd_layer: m2,
        width_dbu,
        dbu_per_um: 1000, // sky130 convention: 1 nm/DBU → 1000 DBU/µm
        vdd_tracks: vec![0],
        gnd_tracks: vec![1_000],
    }
}

/// sky130 metal1 Jmax is roughly ~1 mA/µm for DC current. Use a
/// round 1.0 here; the PDK constant lives at the caller (typically
/// alongside the foundry library bindings).
const JMAX_MA_PER_UM: f64 = 1.0;

#[test]
fn passes_when_actual_within_budget() {
    // Strap 2 µm wide → budget 2 mA. 16 tiles × 0.1 mA = 1.6 mA → pass.
    let r = rail(2_000);
    assert!(current_density_check(&r, 0.1, 16, JMAX_MA_PER_UM).is_ok());
}

#[test]
fn fails_when_actual_exceeds_budget() {
    // Strap 1 µm wide → budget 1 mA. 256 tiles × 0.1 mA = 25.6 mA → fail.
    let r = rail(1_000);
    let err = current_density_check(&r, 0.1, 256, JMAX_MA_PER_UM).unwrap_err();
    let s = err.to_string();
    assert!(s.contains("exceeds budget"), "got: {s}");
    assert!(s.contains("vdd"), "should localize to a rail: {s}");
}

#[test]
fn passes_at_zero_tiles() {
    // Edge case: composing an empty grid is trivially within budget.
    let r = rail(1_000);
    assert!(current_density_check(&r, 1.0, 0, JMAX_MA_PER_UM).is_ok());
}

#[test]
fn rejects_non_positive_width() {
    let r = rail(0);
    assert!(current_density_check(&r, 0.1, 1, JMAX_MA_PER_UM).is_err());
}

#[test]
fn boundary_at_exact_budget_passes() {
    // 1 µm strap, 1 mA budget. 10 tiles × 0.1 mA = exactly 1.0 mA →
    // pass (≤, not <).
    let r = rail(1_000);
    assert!(current_density_check(&r, 0.1, 10, JMAX_MA_PER_UM).is_ok());
}
