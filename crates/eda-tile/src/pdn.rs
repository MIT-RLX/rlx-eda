//! Power-distribution-network helpers — rail spec + coarse current-
//! density sanity check.
//!
//! Not a substitute for ORFS PDN analysis (which is the production
//! ground truth, PLAN.md cross-cutting #4). This is the in-house
//! "fail loudly at array compose time" check so single-tile bring-up
//! doesn't quietly produce a layout that browns out at 256-tile scale.

use klayout_core::LayerIndex;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RailSpec {
    pub vdd_layer: LayerIndex,
    pub gnd_layer: LayerIndex,
    /// Rail width in DBU. Used by `current_density_check` together
    /// with the per-tile peak current estimate to bound array size.
    pub width_dbu: i64,
    /// Database units per micron. Sky130 uses 1000 (1 nm/DBU).
    /// Carried on `RailSpec` so the density check is a pure function
    /// — caller passed it in when they built the rail and the tile's
    /// home library knows it.
    pub dbu_per_um: i64,
    /// Track positions (Y for horizontal rails) in DBU, relative to
    /// tile origin. Two tiles abut cleanly only if their `vdd_tracks`
    /// and `gnd_tracks` agree on the abutting side.
    pub vdd_tracks: Vec<i64>,
    pub gnd_tracks: Vec<i64>,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum PdnError {
    #[error("strap current {actual_ma:.3} mA exceeds budget {budget_ma:.3} mA on rail {rail}")]
    OverBudget {
        rail: &'static str,
        actual_ma: f64,
        budget_ma: f64,
    },
    #[error("RailSpec.width_dbu must be > 0 (was {0})")]
    NonPositiveWidth(i64),
    #[error("RailSpec.dbu_per_um must be > 0 (was {0})")]
    NonPositiveDbu(i64),
}

/// Coarse check: per-tile peak current × tiles-per-strap ≤ rail
/// current budget (derived from rail width and per-PDK Jmax).
///
/// Run for both VDD and GND straps — return-current magnitude on
/// GND equals supply current on VDD for a single-supply digital MAC,
/// and Jmax on the GND strap is identical to VDD's. (Mixed-signal
/// arrays with separate analog supplies need a richer check.)
///
/// Pure function: no I/O, no library lookups. Catches "1 tile fine,
/// 256 tiles browns out" at compose time. Real PDN analysis still
/// requires ORFS (PLAN.md cross-cutting #4).
pub fn current_density_check(
    rails: &RailSpec,
    per_tile_peak_ma: f64,
    tiles_per_strap: usize,
    jmax_ma_per_um: f64,
) -> Result<(), PdnError> {
    if rails.width_dbu <= 0 {
        return Err(PdnError::NonPositiveWidth(rails.width_dbu));
    }
    if rails.dbu_per_um <= 0 {
        return Err(PdnError::NonPositiveDbu(rails.dbu_per_um));
    }

    let width_um = rails.width_dbu as f64 / rails.dbu_per_um as f64;
    let budget_ma = jmax_ma_per_um * width_um;
    let actual_ma = per_tile_peak_ma * tiles_per_strap as f64;

    // Same actual / budget on both rails today; check both anyway so
    // any future per-rail divergence (e.g. wider VDD strap) is one
    // line of code, not a refactor.
    for rail in ["vdd", "gnd"] {
        if actual_ma > budget_ma {
            return Err(PdnError::OverBudget {
                rail,
                actual_ma,
                budget_ma,
            });
        }
    }
    Ok(())
}
