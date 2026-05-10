//! Physical metric arm — area, power, frequency, parasitics, thermal,
//! energy-per-inference. Cross-backend definitions live here so the
//! three backends project onto the same units (cross-cutting #6).

use serde::{Deserialize, Serialize};

/// Per-run physical metrics. Fields are `Option<f64>` when the
/// emitting backend may not have a value (e.g. in-house has no
/// thermal model in v1; FPGA backend has no parasitic-cap number).
/// `None` and `0.0` are not interchangeable — `None` means "not
/// measured by this backend," `0.0` means "measured and equal to
/// zero." The bench reporter renders them differently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Physical {
    pub area_um2: Option<f64>,
    pub max_freq_mhz: Option<f64>,
    /// Worst negative slack at `max_freq_mhz`; ≥ 0 means timing-clean.
    pub wns_ns: Option<f64>,
    pub dynamic_power_mw: Option<f64>,
    pub leakage_power_mw: Option<f64>,
    /// Total parasitic capacitance from extraction. Backend must use
    /// the shared OpenRCX path (cross-cutting #2) for this to be
    /// comparable across backends.
    pub parasitic_cap_ff: Option<f64>,
    /// Peak die temperature under nominal activity. `None` for
    /// in-house backend in v1 (HotSpot/FEM not wired).
    pub peak_temp_c: Option<f64>,
    /// Switching + leakage over one-image latency window, integrated.
    /// One operational definition across all backends.
    pub energy_pj_per_inference: Option<f64>,
}

impl Physical {
    /// Parse the JSON shape that `eda-bench-tinyconv/docker/run_orfs.sh`
    /// emits to `/work/metrics.json`. Pure function — no docker, no
    /// I/O. Used internally by `backends::orfs::measure_physical`
    /// after the docker invocation reads the file back.
    pub fn from_orfs_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// All-fields-`None` constructor for backends that only fill in
    /// a subset of metrics. Lets backends use struct-update syntax:
    /// `Physical { area_um2: Some(x), ..Physical::empty() }`.
    pub fn empty() -> Self {
        Self {
            area_um2: None,
            max_freq_mhz: None,
            wns_ns: None,
            dynamic_power_mw: None,
            leakage_power_mw: None,
            parasitic_cap_ff: None,
            peak_temp_c: None,
            energy_pj_per_inference: None,
        }
    }
}

/// Release-gate parity bands relative to ORFS on the same lowered IR.
pub struct ParityBands {
    pub area_pct: f64,
    pub freq_pct: f64,
    pub power_pct: f64,
}

impl ParityBands {
    /// PLAN.md "Bench harness layout — Parity bands".
    pub const RELEASE: Self = Self {
        area_pct: 15.0,
        freq_pct: 20.0,
        power_pct: 25.0,
    };
}
