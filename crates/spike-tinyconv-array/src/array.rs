//! `ArrayBlock` — the top-level silicon block. Composes a tile grid
//! plus a controller FSM into one `Block + Layout<P>`.
//!
//! `ArrayConfig` carries the architectural knobs the outer DADO loop
//! optimizes over — these are exactly the same knobs `rlx_fpga::tune`
//! sweeps, so an `ArrayConfig` value can be derived from any
//! `tune::Variant` without information loss.

use eda_hir::{Block, Layout};
use eda_tile::tile_grid;
use klayout_core::{CellId, Library};
use serde::{Deserialize, Serialize};
use spike_divider_block::MosfetPdk;
use spike_tinyconv_tile::{Mac8x8Tile, MacTopology, TileParams};
use std::collections::HashMap;

/// How weights are baked into the silicon. Picked by the outer
/// DADO loop alongside `grid` and `pipeline_depth`.
///
/// - **`Bram`** (default, FPGA-style): weights live in `block_ram`,
///   addressed by the controller each cycle. Sequential MAC.
///   Cheapest area, highest cycle count. What `rlx_fpga::codegen`
///   emits.
/// - **`MaskRom`**: weights live in mask-programmable ROM, still
///   addressed each cycle but no R/W BRAM port. Saves area;
///   cycle count similar to `Bram`. Stub today — emit pending.
/// - **`BakedConstants`**: weights become `localparam` constants
///   wired into dedicated multipliers, all running in parallel.
///   No memory at all — silicon is the network. Largest area,
///   smallest cycle count (≈ pipeline depth, not work-count).
///   The ASIC-target form.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightStrategy {
    Bram,
    MaskRom,
    BakedConstants,
}

impl Default for WeightStrategy {
    fn default() -> Self {
        Self::Bram
    }
}

impl WeightStrategy {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Bram => "bram",
            Self::MaskRom => "mask_rom",
            Self::BakedConstants => "baked",
        }
    }
}

use crate::controller::ControllerFsm;

/// Architectural knobs — the outer DADO loop's discrete search space.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ArrayConfig {
    /// Tile grid extent (nx, ny). Total tiles = nx · ny.
    pub grid: (usize, usize),
    /// Outer pipeline depth — how many layers in flight.
    pub pipeline_depth: usize,
    /// MAC topology to use across the array. All tiles share one
    /// topology in v1; mixed-topology arrays are out of scope.
    pub topology: MacTopology,
    /// Tile parameters at construction. Inner Adam loop refines these
    /// for a fixed `ArrayConfig`.
    pub tile_params: TileParams,
    /// How weights are baked into silicon. Selects the codegen
    /// path: `Bram` → rlx-fpga style sequential controller;
    /// `BakedConstants` → unrolled parallel ASIC.
    pub weight_strategy: WeightStrategy,
}

impl Default for ArrayConfig {
    /// Conservative v1 default — small grid, default tile, shallow
    /// pipeline. Enough to validate end-to-end lowering before any
    /// DADO sweep happens.
    fn default() -> Self {
        Self {
            grid: (4, 4),
            pipeline_depth: 1,
            topology: MacTopology::Digital,
            tile_params: TileParams::default(),
            weight_strategy: WeightStrategy::default(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ArrayBlock {
    pub instance_id: String,
    pub config: ArrayConfig,
    pub controller: ControllerFsm,
}

impl ArrayBlock {
    pub fn new(instance_id: impl Into<String>, config: ArrayConfig) -> Self {
        let controller = ControllerFsm::for_config(&config);
        Self {
            instance_id: instance_id.into(),
            config,
            controller,
        }
    }

    /// Per-cell-type inventory aggregated across the entire array:
    /// tile inventory × (nx · ny). Controller FSM cells get added
    /// once the FSM is actually built out (currently a stub —
    /// `ControllerFsm` carries counts but no cell instances).
    ///
    /// Same shape as `Mac8x8Tile::cell_inventory()` — caller can
    /// pass the result straight to `ScHdLibrary::sum_area_um2_x1000`.
    pub fn cell_inventory(&self) -> Vec<(&'static str, usize)> {
        let (nx, ny) = self.config.grid;
        let scale = nx * ny;
        let tile = Mac8x8Tile::with_topology(
            format!("{}_tile_inv", self.instance_id),
            self.config.tile_params,
            self.config.topology,
        );
        // Sum tile inventory × scale; coalesce across cell names so
        // controller cells (when added) don't produce duplicates.
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for (name, count) in tile.cell_inventory() {
            *counts.entry(name).or_default() += count * scale;
        }
        // Sort by name for deterministic ordering — same shape as
        // `Mac8x8Tile::cell_inventory` produces.
        let mut out: Vec<(&'static str, usize)> = counts.into_iter().collect();
        out.sort_by_key(|(name, _)| *name);
        out
    }
}

impl Block for ArrayBlock {
    fn name(&self) -> String {
        let (nx, ny) = self.config.grid;
        format!(
            "TinyConvArray_{}_{}_{}x{}_d{}",
            self.instance_id,
            self.config.topology.tag(),
            nx,
            ny,
            self.config.pipeline_depth,
        )
    }
}

impl<P> Layout<P> for ArrayBlock
where
    P: MosfetPdk,
{
    /// **v1 array layout** — tile_grid composition only. Controller
    /// FSM placement and inter-block routing land in the next pass
    /// (PLAN.md step 6+); for now the grid cell IS the array cell.
    ///
    /// **Caller responsibility**: populate `lib` with the foundry
    /// (or mock) sc_hd cell library before calling — `tile_grid`
    /// recurses into `Mac8x8Tile::layout` which expects the cells
    /// present.
    fn layout(&self, lib: &Library, pdk: &P) -> CellId {
        let (nx, ny) = self.config.grid;
        let tile = Mac8x8Tile::with_topology(
            format!("{}_tile", self.instance_id),
            self.config.tile_params,
            self.config.topology,
        );
        // PDN check skipped at array compose time — rolled up to
        // the bench harness, which knows the per-PDK Jmax. Here we
        // just need the geometry.
        tile_grid(&tile, nx, ny, lib, pdk, None)
            .expect("array layout: tile_grid abutment / rail check failed")
    }
}
