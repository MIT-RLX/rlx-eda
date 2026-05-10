//! `eda-tile` — pitch-matched abutment + power-rail helpers.
//!
//! See `eda-bench-tinyconv/PLAN.md` build-order step 2.

pub mod grid;
pub mod pdn;
pub mod tile;

pub use grid::{tile_grid, GridError, PdnCheck};
pub use pdn::{current_density_check, RailSpec};
pub use tile::{EdgePort, Side, Tile};
