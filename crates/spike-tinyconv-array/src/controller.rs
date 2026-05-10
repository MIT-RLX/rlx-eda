//! `ControllerFsm` — TinyConv's control plane. Address generation,
//! ping-pong scratch banking, layer sequencing, MAC-array dispatch.
//!
//! Built from `eda-stdcells` instances. Tiny in v1 (~thousands of
//! gates, single clock domain). PLAN.md "Failure modes the plan is
//! designed to prevent" — the controller is the one piece that isn't
//! a regular tile, so it's the most likely place for hand-author
//! divergence vs ORFS synthesis. Keep it tiny, validate aggressively.

use serde::{Deserialize, Serialize};

use crate::array::ArrayConfig;

/// Controller FSM description — derived from `ArrayConfig` so it
/// stays in sync with the array geometry.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControllerFsm {
    /// Number of bits in the layer-counter — enough to address every
    /// layer in the lowered `Model`.
    pub layer_counter_bits: u8,
    /// Width of the tile-dispatch one-hot bus.
    pub dispatch_width: usize,
    /// Pipeline depth — must match `ArrayConfig::pipeline_depth`.
    pub pipeline_depth: usize,
}

impl ControllerFsm {
    pub fn for_config(cfg: &ArrayConfig) -> Self {
        let (nx, ny) = cfg.grid;
        Self {
            // Placeholder — real layer count comes from the lowered Model.
            layer_counter_bits: 4,
            dispatch_width: nx * ny,
            pipeline_depth: cfg.pipeline_depth,
        }
    }
}
