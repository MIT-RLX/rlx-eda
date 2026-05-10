//! `StdCell` — wrapper that lifts an imported foundry cell into the
//! `eda-hir` trait family.
//!
//! Each `StdCell` carries:
//!   - the foundry cell name (`"sky130_fd_sc_hd__nand2_1"`),
//!   - a per-instance designator so two placements of the same
//!     foundry cell are distinct as `Block` values.
//!
//! `Layout::layout` is a pure lookup: the foundry cell *is* the
//! cell. The caller must have loaded the foundry library into the
//! target `Library` (via `ScHdLibrary::load(...)`) before calling
//! — the imported `ScHdLibrary.library` is the natural target to
//! pass.

use eda_hir::{Block, Layout};
use eda_pdks::Sky130;
use klayout_core::{CellBuilder, CellId, Library, Point, Trans, Vec2};

use crate::liberty::LibertyMetadata;

/// A reference to a foundry cell that has been imported once into a
/// shared `klayout_core::Library`. Cheap to clone.
#[derive(Debug, Clone)]
pub struct StdCellRef {
    /// Foundry cell name, e.g. `"sky130_fd_sc_hd__nand2_1"`.
    pub name: String,
    /// `CellId` inside the imported library.
    pub cell_id: CellId,
    /// Liberty metadata. `None` if the cell was loaded from GDS only.
    pub metadata: Option<LibertyMetadata>,
}

/// `StdCell` — an instance-site of a foundry cell. Equal under `Eq`
/// when the foundry cell + instance designator match.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StdCell {
    pub cell_name: String,
    pub instance_id: String,
}

impl StdCell {
    pub fn new(cell_name: impl Into<String>, instance_id: impl Into<String>) -> Self {
        Self {
            cell_name: cell_name.into(),
            instance_id: instance_id.into(),
        }
    }
}

impl Block for StdCell {
    fn name(&self) -> String {
        format!("{}__{}", self.cell_name, self.instance_id)
    }
}

impl Layout<Sky130> for StdCell {
    /// Resolve the foundry cell by name. The caller must have loaded
    /// the foundry library into `lib` (e.g. by passing
    /// `ScHdLibrary.library` here, or by importing into a shared
    /// library that the array uses).
    ///
    /// Returns the foundry cell's `CellId` directly — there is no
    /// wrapper cell. Cells are content-addressable in klayout-core,
    /// so multiple `StdCell` instance-sites of the same foundry
    /// cell all resolve to the same `CellId` and are placed via
    /// per-instance `Trans` at the parent.
    fn layout(&self, lib: &Library, _pdk: &Sky130) -> CellId {
        lib.by_name(&self.cell_name).unwrap_or_else(|| {
            panic!(
                "StdCell {:?}: foundry cell {:?} not found in library — \
                 load it via `ScHdLibrary::load(...)` and pass that \
                 library here",
                self.instance_id, self.cell_name,
            )
        })
    }
}

/// Compose multiple `StdCell` instances into a parent cell-builder
/// at given placement points. Caller is responsible for creating
/// the `CellBuilder`, loading the foundry library into `lib`, and
/// inserting the result via `lib.insert(builder)`.
///
/// Used by the controller-FSM lowering in `spike-tinyconv-array`.
pub fn build_composite(
    b: &mut CellBuilder,
    lib: &Library,
    pdk: &Sky130,
    children: &[(StdCell, Point)],
) {
    for (child, origin) in children {
        let cell_id = child.layout(lib, pdk);
        b.instantiate(cell_id, Trans::translate(Vec2::new(origin.x, origin.y)));
    }
}
