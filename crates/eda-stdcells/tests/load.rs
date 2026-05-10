//! Soft-skip GDS-load smoke test.
//!
//! Mirrors `eda-pdks/tests/drc_smoke.rs`: skip cleanly when the
//! foundry library isn't checked out so contributors without it
//! don't see false test failures.

use eda_stdcells::library::{default_sc_hd_path, LibraryError, ScHdLibrary};

#[test]
fn loads_sky130_fd_sc_hd_when_checked_out() {
    let gds = default_sc_hd_path();
    match ScHdLibrary::load(&gds, None) {
        Ok(lib) => {
            assert!(
                lib.cells.len() > 50,
                "expected hundreds of standard cells, got {}",
                lib.cells.len(),
            );
            // sky130_fd_sc_hd ships with named cells like
            // `sky130_fd_sc_hd__nand2_1` and `sky130_fd_sc_hd__inv_1`.
            // We check at least one canonical name is present.
            let any_canonical = lib
                .cell_names()
                .any(|n| n.starts_with("sky130_fd_sc_hd__"));
            assert!(
                any_canonical,
                "no `sky130_fd_sc_hd__*` cells in imported library"
            );
        }
        Err(LibraryError::NotFound(p)) => {
            eprintln!("skipping: sky130_fd_sc_hd GDS not present at {p:?}");
        }
        Err(e) => panic!("unexpected load failure: {e}"),
    }
}
