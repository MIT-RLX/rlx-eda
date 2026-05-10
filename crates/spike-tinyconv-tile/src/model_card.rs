//! `ModelCard` — calibration constants for the closed-form Digital
//! MAC residuals (energy, delay, area). v1 ships placeholder values
//! that match the `behavioral.rs` `const`s exactly; production
//! workflows write their own card from ngspice-characterization
//! data and pass it to `Mac8x8Tile::add_loss_to_dc_with_card`.
//!
//! ## Why a struct, not just `const`s
//!
//! 1. **Per-PDK calibration**: sky130 and gf180mcu have different
//!    process numbers; one card per PDK, swapped at construction.
//! 2. **Round-trip via serde**: card files (TOML / JSON) become the
//!    canonical artifact a calibration run produces, version-controlled
//!    alongside test data.
//! 3. **Inspection**: tests can compare default against an
//!    arbitrary card to detect drift (drift-detection mirrors the
//!    `eda-pdks` lyp conformance test).
//!
//! ## What this is NOT
//!
//! - Not the full PDK card — that lives upstream in `eda-pdks`.
//! - Not wired into `behavioral.rs` yet (would churn loss
//!   magnitudes across existing tests). The shape is here so a
//!   later opt-in commit reads from it; default behavior unchanged.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelCard {
    // ── Cell + activity constants ────────────────────────────────
    /// Total cell count per tile (PLAN.md "Internal blocks" table).
    pub n_cells_digital: f32,
    /// Per-cell switched capacitance scale (normalized).
    pub c_per_cell: f32,
    /// Activity factor α — fraction of cells switching per clock.
    pub activity_factor: f32,
    /// Clock-frequency scale (normalized — not literal Hz).
    pub f_clk_scale: f32,
    /// Per-cell leakage scale (normalized).
    pub k_leak: f32,

    // ── Delay model: τ ∝ N_stages · C_load / ((W/L) · V_dd) ─────
    /// Number of gate stages on the critical path of an 8×8 MAC.
    pub n_critical_stages: f32,
    /// Delay scaling (normalized).
    pub k_delay: f32,

    // ── Area model: linear in average sizing ────────────────────
    /// Per-cell baseline area (normalized).
    pub a0_per_cell: f32,
    /// Per-cell area sensitivity to W/L.
    pub a_per_wl: f32,
}

impl Default for ModelCard {
    /// **v1 placeholder card.** Bit-exact match to the `const`s in
    /// `spike_tinyconv_tile::behavioral` so that callers swapping
    /// `add_loss_to_dc(...)` for `add_loss_to_dc_with_card(...,
    /// &ModelCard::default())` produce identical loss values.
    /// Calibration runs replace these with foundry-characterized
    /// numbers later.
    fn default() -> Self {
        Self {
            n_cells_digital: 202.0,
            c_per_cell: 3.0,
            activity_factor: 0.15,
            f_clk_scale: 1.0,
            k_leak: 0.05,
            n_critical_stages: 40.0,
            k_delay: 0.5,
            a0_per_cell: 0.05,
            a_per_wl: 0.015,
        }
    }
}

impl ModelCard {
    /// Round-trip via TOML — round-trips by construction since every
    /// field is f32, but exists so calibration pipelines can write
    /// `cards/sky130_tt_25c.toml` and a contributor can inspect
    /// the values without grepping `const`s.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("ModelCard serializes")
    }

    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Quick predicate: is this card the v1 placeholder? Useful in
    /// the bench reporter to flag "uncalibrated" runs.
    pub fn is_placeholder(&self) -> bool {
        self == &Self::default()
    }
}
