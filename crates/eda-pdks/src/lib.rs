//! `eda-pdks` — shared foundry PDK definitions, generated at build time
//! from each foundry's KLayout `.lyp`.
//!
//! Each PDK is gated behind a Cargo feature so downstream crates only
//! pull in the foundries they need. The struct name and field names
//! follow a uniform convention so consumers can write traits that
//! abstract over PDKs (e.g. an `RcLikePdk` trait that requires `RES`,
//! `METAL1`, `VIA1` — see `spike-divider-block`).
//!
//! ## Foundries
//!
//! CMOS:
//! - `Sky130` (feature `sky130`)
//! - `Gf180mcu` (feature `gf180mcu`)
//!
//! Photonic:
//! - `GdsfactoryGeneric` (feature `gdsfactory-generic`)
//! - `CornerstoneSi220` (feature `cornerstone-si220`)
//! - `SiepicEbeam` (feature `siepic-ebeam`)
//!
//! For each enabled foundry, the build also emits a
//! `pub const HAS_<FOUNDRY>: bool` indicating whether the foundry's
//! `.lyp` was present at build time. Tests typically gate on this so
//! contributors without the foundry file checked out get clean skips
//! instead of panics.
//!
//! ## Conformance tests
//!
//! `build.rs` additionally emits one generated `#[test]` per enabled
//! foundry into `OUT_DIR/pdks_conformance.rs`, each calling into the
//! [`__conformance`] helpers below. They check that registered layers
//! map back to the expected GDS pair, no two logical fields collide on
//! one pair, port-kind ids are distinct + non-zero, and that the
//! generated struct still agrees with the foundry's current `.lyp`
//! (drift detection).

include!(concat!(env!("OUT_DIR"), "/pdks_generated.rs"));

#[doc(hidden)]
pub mod __conformance {
    //! Helpers for the build-script-generated conformance tests. Public
    //! so the generated test code (which lives in `OUT_DIR`, outside
    //! this crate's source tree) can call them, but `#[doc(hidden)]`
    //! because they aren't part of the user-facing API.

    use eda_pdk_ingest::{LayerProps, parse_lyp};
    use klayout_core::{LayerIndex, Library, PortKindId};
    use std::collections::HashMap;

    /// A single (logical-name, expected-(L,D), registered-LayerIndex)
    /// row. Generated test code constructs an array of these from each
    /// PDK's known layers.
    pub struct LayerRow {
        pub name: &'static str,
        pub layer: u16,
        pub datatype: u16,
        pub idx: LayerIndex,
    }

    /// Each registered `LayerIndex` should resolve back to a `LayerInfo`
    /// with the expected GDS pair. Catches build-time corruption and
    /// "I edited the macro by hand" mistakes.
    pub fn check_layers_match_expected(lib: &Library, rows: &[LayerRow]) {
        for r in rows {
            let info = lib.layer_info(r.idx);
            assert_eq!(
                info.layer, r.layer,
                "{}: registered LayerIndex resolved to layer {} but expected {}",
                r.name, info.layer, r.layer,
            );
            assert_eq!(
                info.datatype, r.datatype,
                "{}: registered LayerIndex resolved to datatype {} but expected {}",
                r.name, info.datatype, r.datatype,
            );
        }
    }

    /// No two logical fields should share a `(layer, datatype)` pair —
    /// otherwise downstream code that iterates by logical name produces
    /// duplicate shapes when it reads back by GDS pair.
    pub fn check_pairwise_distinct_gds_pairs(rows: &[LayerRow]) {
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                assert!(
                    rows[i].layer != rows[j].layer || rows[i].datatype != rows[j].datatype,
                    "{} and {} share GDS pair ({}, {})",
                    rows[i].name, rows[j].name, rows[i].layer, rows[i].datatype,
                );
            }
        }
    }

    /// Anti-drift: re-parse the foundry's current `.lyp` and verify the
    /// short-name → (L, D) mapping it gives still matches the build-time
    /// generated values. If the foundry updates its file but the build
    /// cache is stale, this is the test that catches it.
    ///
    /// `mapping_with_expected` rows are `(logical, short_in_lyp, expected_L, expected_D)`.
    pub fn check_lyp_drift(
        lyp_path: &str,
        mapping_with_expected: &[(&str, &str, u16, u16)],
    ) {
        let xml = match std::fs::read_to_string(lyp_path) {
            Ok(x) => x,
            // No file at test time → not a drift, just an absent foundry
            // distribution. The build already noted this via HAS_*; tests
            // gate on that. Reaching here means the test ran anyway, so
            // surface the missing-file as an explicit fail.
            Err(e) => panic!(".lyp expected at {} but couldn't be read: {}", lyp_path, e),
        };
        let layers = parse_lyp(&xml).expect(".lyp re-parse failed");
        let by_short: HashMap<&str, &LayerProps> =
            layers.iter().map(|p| (p.short_name(), p)).collect();
        for (logical, short, expected_l, expected_d) in mapping_with_expected {
            let p = by_short.get(short).unwrap_or_else(|| {
                panic!(
                    "{}: short name {:?} no longer present in {} — foundry may have renamed it",
                    logical, short, lyp_path,
                )
            });
            assert_eq!(
                p.layer, *expected_l,
                "{}: lyp drift — {:?} now at layer {} (was {})",
                logical, short, p.layer, expected_l,
            );
            assert_eq!(
                p.datatype, *expected_d,
                "{}: lyp drift — {:?} now at datatype {} (was {})",
                logical, short, p.datatype, expected_d,
            );
        }
    }

    /// Port-kind ids must be non-zero (zero is `PortKindId::ANY`, the
    /// wildcard) and pairwise distinct (FNV collisions on identifier
    /// names — vanishingly unlikely but worth pinning down).
    pub fn check_port_kinds_distinct(kinds: &[(&'static str, PortKindId)]) {
        for (name, k) in kinds {
            assert_ne!(
                k.0, 0,
                "port kind {:?} hashed to 0 (PortKindId::ANY); rename it",
                name,
            );
        }
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                assert_ne!(
                    kinds[i].1, kinds[j].1,
                    "port kinds {:?} and {:?} hash to the same id ({})",
                    kinds[i].0, kinds[j].0, kinds[i].1.0,
                );
            }
        }
    }
}

#[cfg(test)]
mod conformance {
    //! Auto-generated per-foundry conformance tests. The actual test
    //! functions are emitted by `build.rs` based on the FOUNDRIES table
    //! and the `HAS_<FOUNDRY>` build-time presence flags.
    include!(concat!(env!("OUT_DIR"), "/pdks_conformance.rs"));
}
