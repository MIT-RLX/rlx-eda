//! `Schematic<Sky130>` for `Mac8x8Tile` — symbolic schematic for the
//! `eda-viz` renderer. Drifts away from layout reality unless this
//! impl exists, hence the explicit trait obligation.

use crate::tile::Mac8x8Tile;

// `eda_hir::Schematic` is the trait — fill this in once the
// schematic-side IR shape is finalized for analog mixed-signal blocks.
// For now, schematic generation is `unimplemented!()` and the bench
// harness routes around this at the array level.

impl Mac8x8Tile {
    pub fn emit_schematic(&self) {
        unimplemented!("MAC tile schematic — eda-viz binding")
    }
}
