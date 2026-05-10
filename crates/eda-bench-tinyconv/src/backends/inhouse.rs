//! In-house backend: rlx-eda `Block` composition → klayout GDS.
//!
//! Uses `spike-tinyconv-tile` (custom analog MAC) tiled via `eda-tile`,
//! with `eda-stdcells` providing the digital glue from foundry
//! `sky130_fd_sc_hd`. Parasitics delegated to OpenRCX (cross-cutting
//! #2) — do not pretend `klayout_geom::Region` is an independent
//! extractor.
//!
//! Serves L1, L2, and L4 functional levels. Cannot serve L3 (no SDF
//! without synthesis) or L5 at scale (post-layout sim too slow).
//!
//! ## v1 status
//!
//! - **`measure_physical`**: returns area only, computed from the
//!   tile's cell inventory × sc_hd Liberty areas. Other fields stay
//!   `None` until OpenRCX (parasitics) and ngspice (power/timing)
//!   wire through.
//! - **`measure_functional`**: still stubbed — the gate-level / SDF
//!   sim path is the next domino.

use eda_stdcells::ScHdLibrary;
use spike_tinyconv_array::array::ArrayBlock;
use spike_tinyconv_tile::Mac8x8Tile;

use super::{Backend, BackendError};
use crate::metrics::{functional::Level, Functional, Physical};

pub struct InhouseBackend {
    /// Cell inventory at whatever scope the caller wired in (tile
    /// or array). The backend doesn't care which — it just sums
    /// areas through `ScHdLibrary::sum_area_um2_x1000`.
    pub inventory: Vec<(&'static str, usize)>,
    pub library: ScHdLibrary,
    /// Friendly label used for reporting / debugging — typically
    /// `"tile"` or `"array"`. Doesn't affect correctness.
    pub scope: &'static str,
}

impl InhouseBackend {
    /// Tile-scope: report area for one Mac8x8Tile.
    pub fn from_tile(tile: &Mac8x8Tile, library: ScHdLibrary) -> Self {
        Self {
            inventory: tile.cell_inventory(),
            library,
            scope: "tile",
        }
    }

    /// Array-scope: report area for an `nx × ny` ArrayBlock — sums
    /// the tile inventory × grid count + (eventually) the controller
    /// FSM cells.
    pub fn from_array(array: &ArrayBlock, library: ScHdLibrary) -> Self {
        Self {
            inventory: array.cell_inventory(),
            library,
            scope: "array",
        }
    }

    /// Backwards-compat alias for callers that want the tile path.
    /// Deprecated in favour of the explicit `from_tile`.
    pub fn new(tile: Mac8x8Tile, library: ScHdLibrary) -> Self {
        Self::from_tile(&tile, library)
    }
}

impl Backend for InhouseBackend {
    fn name(&self) -> &'static str {
        "inhouse"
    }

    fn measure_physical(&self) -> Result<Physical, BackendError> {
        let area_x1000 = self
            .library
            .sum_area_um2_x1000(&self.inventory)
            .ok_or_else(|| {
                BackendError::Toolchain(format!(
                    "inhouse {}: inventory references cell or area metadata \
                     missing from ScHdLibrary",
                    self.scope
                ))
            })?;
        Ok(Physical {
            area_um2: Some(area_x1000 as f64 / 1000.0),
            ..Physical::empty()
        })
    }

    fn measure_functional(
        &self,
        _level: Level,
        _images: &[u32],
    ) -> Result<Functional, BackendError> {
        unimplemented!("in-house functional sim (L1/L2/L4)")
    }
}
