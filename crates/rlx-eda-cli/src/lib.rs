//! `rlx-eda-cli` library half — PDK manager surface that other crates
//! (spike testbenches, harness clients) can call without going through
//! the binary.

pub mod dashboard;
pub mod doctor;
pub mod pdk;

/// Convenience: look up a PDK by name in the registry / ciel scan.
/// Equivalent to running `rlx-eda pdk show <name>` and reading the
/// resolved `lib_path` + `sections` programmatically.
pub fn resolve_pdk(name: &str) -> Result<pdk::PdkEntry, pdk::Error> {
    pdk::resolve(name)
}
