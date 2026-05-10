//! Foundry-library loader.
//!
//! Reads `sky130_fd_sc_hd.gds` once via `klayout_io::read_gds_path`
//! and indexes every cell by its foundry name. Subsequent
//! `StdCell::layout()` calls just instantiate the imported `CellId`
//! rather than re-importing.
//!
//! Soft-skip pattern: if the foundry library isn't checked out, the
//! loader returns `LibraryError::NotFound` and downstream tests
//! gracefully `eprintln + return`, mirroring the
//! `eda_pdks::HAS_SKY130` build-time pattern.

use klayout_core::Library;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::cell::StdCellRef;
use crate::liberty::{parse_lib, LibertyMetadata};

#[derive(Debug, thiserror::Error)]
pub enum LibraryError {
    #[error("foundry library not found at {0}")]
    NotFound(PathBuf),
    #[error("GDS read failure: {0}")]
    Gds(String),
    #[error("Liberty read failure: {0}")]
    LibertyIo(#[from] std::io::Error),
    #[error("Liberty parse failure: {0}")]
    Liberty(String),
    #[error("cell {0} not found in imported library")]
    UnknownCell(String),
}

/// In-memory index of an imported foundry library.
pub struct ScHdLibrary {
    pub library: Library,
    pub cells: HashMap<String, StdCellRef>,
}

impl ScHdLibrary {
    /// Import the foundry GDS at `gds_path` and (optionally) Liberty
    /// metadata at `lib_path` into a fresh `Library`. The returned
    /// index maps foundry cell name → `StdCellRef`.
    ///
    /// Caller decides which file to load — typically
    /// `default_sc_hd_path()` for v1 sky130 work, or a project-local
    /// override for downstream forks.
    pub fn load(
        gds_path: &Path,
        lib_path: Option<&Path>,
    ) -> Result<Self, LibraryError> {
        if !gds_path.exists() {
            return Err(LibraryError::NotFound(gds_path.to_path_buf()));
        }

        let library = klayout_io::gds::read_gds_path(gds_path)
            .map_err(|e| LibraryError::Gds(e.to_string()))?;

        // Optional Liberty metadata. Errors here are soft — a GDS-only
        // load is still useful (you just lose pin/area info).
        let metadata_by_cell: HashMap<String, LibertyMetadata> = match lib_path {
            Some(p) if p.exists() => {
                let text = std::fs::read_to_string(p)?;
                parse_lib(&text)
                    .map_err(|e| LibraryError::Liberty(e.to_string()))?
                    .into_iter()
                    .map(|m| (m.cell_name.clone(), m))
                    .collect()
            }
            _ => HashMap::new(),
        };

        let cells = library
            .all_cells()
            .into_iter()
            .map(|(cell_id, cell)| {
                let name = cell.name().as_str().to_string();
                let metadata = metadata_by_cell.get(&name).cloned();
                (
                    name.clone(),
                    StdCellRef {
                        name,
                        cell_id,
                        metadata,
                    },
                )
            })
            .collect();

        Ok(Self { library, cells })
    }

    pub fn get(&self, cell_name: &str) -> Result<&StdCellRef, LibraryError> {
        self.cells
            .get(cell_name)
            .ok_or_else(|| LibraryError::UnknownCell(cell_name.to_string()))
    }

    /// Iterate cell names — useful for printing what's available when
    /// a build references an unknown cell.
    pub fn cell_names(&self) -> impl Iterator<Item = &str> {
        self.cells.keys().map(String::as_str)
    }

    /// Sum the per-cell areas of the supplied inventory. Returns
    /// `None` if any cell name is missing from the library, or if
    /// any cell lacks Liberty metadata (caller decides whether to
    /// treat that as fatal or fall back to a placeholder).
    ///
    /// Result is in **µm² × 1000** to match `LibertyMetadata`'s
    /// integer-millimicron units (avoids `f64` in equality keys).
    pub fn sum_area_um2_x1000(
        &self,
        inventory: &[(&str, usize)],
    ) -> Option<u64> {
        let mut total = 0u64;
        for (cell_name, count) in inventory {
            let cell_ref = self.cells.get(*cell_name)?;
            let meta = cell_ref.metadata.as_ref()?;
            total = total.checked_add(meta.area_um2_x1000.checked_mul(*count as u64)?)?;
        }
        Some(total)
    }
}

/// Resolve a path to the `sky130_fd_sc_hd` standard-cell GDS. Search
/// order mirrors `eda-pdks::build.rs::resolve_lyp`:
///
///   1. `RLX_EDA_SKY130_FD_SC_HD_GDS` env var.
///   2. `$PDK_ROOT` / `~/.ciel` / `~/.volare` install trees, walking
///      `<root>/<bin>/sky130/versions/*/sky130A/libs.ref/sky130_fd_sc_hd/gds/sky130_fd_sc_hd.gds`
///      (the open_pdks-built layout `rlx-eda-cli pdk install sky130A`
///      produces).
///   3. Legacy hardcoded dev paths (transitional).
///
/// Returns the first path that exists; falls back to the first
/// candidate (which then fails NotFound at load time, triggering
/// soft-skip in callers).
pub fn default_sc_hd_path() -> PathBuf {
    if let Ok(p) = std::env::var("RLX_EDA_SKY130_FD_SC_HD_GDS") {
        let path = PathBuf::from(p);
        if path.is_file() { return path; }
    }

    // Install-tree probe.
    let install_relpaths = [
        "sky130A/libs.ref/sky130_fd_sc_hd/gds/sky130_fd_sc_hd.gds",
        "sky130B/libs.ref/sky130_fd_sc_hd/gds/sky130_fd_sc_hd.gds",
    ];
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("PDK_ROOT") {
        roots.push(PathBuf::from(p));
    }
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(&home).join(".ciel"));
        roots.push(PathBuf::from(&home).join(".volare"));
    }
    for root in &roots {
        for bin in &["ciel", "volare"] {
            let versions = root.join(bin).join("sky130").join("versions");
            let Ok(iter) = std::fs::read_dir(&versions) else { continue };
            for ver in iter.flatten() {
                for rel in &install_relpaths {
                    let p = ver.path().join(rel);
                    if p.is_file() { return p; }
                }
            }
        }
    }

    // Legacy dev paths — last resort.
    let legacy = [
        "/Users/Shared/mtl/skywater130/sky130/src/sky130_fd_sc_hd/sky130_fd_sc_hd.gds",
        "/Users/Shared/mtl/skywater130/libraries/sky130_fd_sc_hd/latest/cells/sky130_fd_sc_hd.gds",
    ];
    for c in &legacy {
        let p = PathBuf::from(c);
        if p.exists() { return p; }
    }
    PathBuf::from(legacy[0])
}
