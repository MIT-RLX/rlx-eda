//! `eda-stdcells` — foundry standard-cell ingest.
//!
//! Reads a foundry library (sky130_fd_sc_hd in v1) and exposes each
//! cell as a `StdCell` that implements the `eda-hir` traits. Used by
//! the digital glue (controller FSM, address decoders, ping-pong
//! control) in `spike-tinyconv-array`.
//!
//! See `eda-bench-tinyconv/PLAN.md` build-order step 1.

pub mod cell;
pub mod library;
pub mod liberty;

#[cfg(feature = "mock-stdcells")]
pub mod mock;

pub use cell::{StdCell, StdCellRef};
pub use library::{ScHdLibrary, default_sc_hd_path};
pub use liberty::{LibertyMetadata, PinDirection};

#[cfg(feature = "mock-stdcells")]
pub use mock::{mock_cell_names, populate_mock_sc_hd};
