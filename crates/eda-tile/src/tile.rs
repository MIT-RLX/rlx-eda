//! `Tile` trait — a `Block + Layout<P>` that abuts neighbours by
//! contract: fixed pitch, declared power-rail track positions, and
//! declared edge-port positions per side.

use eda_hir::{Block, Layout};
use klayout_core::Vec2;

use crate::pdn::RailSpec;

/// One of the four pitch-matched sides of a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    North,
    South,
    East,
    West,
}

/// A port that must align with the matching `EdgePort` on the abutting
/// neighbour. Position is along the side's axis, in DBU, measured from
/// the tile origin.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EdgePort {
    pub name: String,
    pub side: Side,
    /// Offset along the side, in DBU.
    pub offset_dbu: i64,
    /// Layer this port reaches the edge on. The abutting neighbour
    /// must expose the same name on the same layer at the matching
    /// offset, or `tile_grid` rejects the composition.
    pub layer: klayout_core::LayerIndex,
}

/// A `Block + Layout<P>` whose layout abuts cleanly with copies of
/// itself (or other tiles with matching pitch + rails + edge ports).
pub trait Tile<P>: Block + Layout<P> {
    /// Tile bounding box in DBU. `tile_grid` instantiates copies on
    /// integer multiples of `(pitch.x, pitch.y)`.
    fn pitch(&self) -> Vec2;

    /// Power-rail tracks running across the tile. The grid composer
    /// concatenates rails across abutted tiles and runs a current-
    /// density check on the resulting strap.
    ///
    /// Takes `&P` because `RailSpec` carries a per-PDK `LayerIndex`
    /// — the layer doesn't exist before a library / PDK is in scope.
    fn rails(&self, pdk: &P) -> RailSpec;

    /// Edge ports on the requested side, in offset order. Takes
    /// `&P` for the same reason as `rails` — each `EdgePort` carries
    /// a `LayerIndex` that has to come from somewhere.
    fn edge_ports(&self, side: Side, pdk: &P) -> Vec<EdgePort>;
}
