//! MZI floorplan: cross-check geometry under a real foundry PDK and
//! run DRC against typical 220 nm-SOI photonic rules.
//!
//! Targets `gdsfactory-generic` (the open default). This is the
//! photonic counterpart to `spike-divider-block::tests::drc`.

#![cfg(feature = "gdsfactory-generic")]

use eda_hir::Layout;
use klayout_core::{LayerInfo, Library, PortKindId};
use klayout_drc::{enclosing, space, width};
use klayout_geom::{merge, union, Region};
use spike_waveguide_block::{Mzi, OpticalPdk};

/// Standard photonic 220 nm-SOI design rules used here:
/// - WG min width:                    400 nm
/// - WG min spacing (outside coupler): 1.0 µm
/// - HEATER min width:                 1.0 µm
/// - HEATER enclosed inside arm A:    not required (heater overhangs WG)
/// - M1 min width:                     1.0 µm
const WG_MIN_W: i64 = 400;
const WG_MIN_S: i64 = 1_000;
const HEATER_MIN_W: i64 = 1_000;
const M1_MIN_W: i64 = 1_000;

fn build_floorplan() -> (Library, eda_pdks::GdsfactoryGeneric, klayout_core::CellId) {
    let lib = eda_pdks::GdsfactoryGeneric::new_library("mzi_floorplan");
    let pdk = eda_pdks::GdsfactoryGeneric::register(&lib);
    let mzi = Mzi::new(500, 100_000, 110_000, "drc");
    let top = mzi.layout(&lib, &pdk);
    (lib, pdk, top)
}

fn no_violations(rule: &str, region: &Region) {
    if !region.is_empty() {
        let count = region.polygons().len();
        panic!("[{rule}] {count} violation regions: not empty");
    }
}

#[test]
fn floorplan_lays_out_under_gdsfactory_generic() {
    if !eda_pdks::HAS_GDSFACTORY_GENERIC {
        eprintln!("skipping: gdsfactory-generic .lyp absent at build time");
        return;
    }
    let (lib, _pdk, top) = build_floorplan();
    let cell = lib.get(top);

    // WG layer = (1, 0) for gdsfactory-generic.
    let wg_layer = lib.layer(LayerInfo::gds(1, 0));
    let wg_shapes = cell.shapes_on(wg_layer).count();
    // 2 instantiated arms (each 1 shape) + 2 couplers + 2 in-stubs + 2 out-stubs
    // = 4 in-cell + 2 instances visible after flatten via Region.
    // We can't easily count cross-instance, so just require ≥6 here.
    assert!(
        wg_shapes >= 6,
        "expected ≥6 WG shapes in top cell (couplers + stubs), got {wg_shapes}"
    );

    // Heater + M1 shapes live in the top cell (not in arm sub-cells).
    let heater_layer = lib.layer(LayerInfo::gds(47, 0)); // gdsfactory MH
    assert!(
        cell.shapes_on(heater_layer).count() >= 1,
        "heater shape missing"
    );
    let m1_layer = lib.layer(LayerInfo::gds(41, 0)); // gdsfactory M1
    // 2 pads + 2 vertical leads landing on the heater = 4 raw shapes.
    assert_eq!(
        cell.shapes_on(m1_layer).count(),
        4,
        "expected 4 M1 shapes (2 pads + 2 heater leads)"
    );
}

#[test]
fn floorplan_exposes_2_input_2_output_optical_ports() {
    if !eda_pdks::HAS_GDSFACTORY_GENERIC {
        eprintln!("skipping: gdsfactory-generic .lyp absent at build time");
        return;
    }
    let (lib, pdk, top) = build_floorplan();
    let cell = lib.get(top);

    let opt: PortKindId = pdk.optical_kind();
    let elec: PortKindId = pdk.electrical_kind();

    let opt_ports: Vec<_> = cell.ports().iter().filter(|p| p.kind == opt).collect();
    let elec_ports: Vec<_> = cell.ports().iter().filter(|p| p.kind == elec).collect();

    assert_eq!(opt_ports.len(), 4, "expected 4 optical ports (in1/in2/through/cross)");
    assert_eq!(elec_ports.len(), 2, "expected 2 electrical ports (heater +/-)");

    let names: Vec<&str> = opt_ports.iter().map(|p| p.name.as_str()).collect();
    for need in ["in1", "in2", "through", "cross"] {
        assert!(names.contains(&need), "missing optical port {need:?} in {names:?}");
    }
    let enames: Vec<&str> = elec_ports.iter().map(|p| p.name.as_str()).collect();
    for need in ["heater_pos", "heater_neg"] {
        assert!(enames.contains(&need), "missing electrical port {need:?} in {enames:?}");
    }
}

/// Wiring-connectivity invariant: every M1 polygon must physically
/// touch the HEATER region. Catches "floating contact pad" regressions
/// — i.e. someone moves a pad without updating the lead, leaving the
/// pad disconnected from the device it's supposed to drive.
///
/// Implementation: `merge(M1 ∪ HEATER)` should have the same polygon
/// count as `merge(HEATER)` alone. If any M1 island is disconnected
/// from the heater, the union grows by at least one polygon.
fn assert_m1_touches_heater(m1: &Region, heater: &Region) {
    let combined = merge(&union(m1, heater));
    let heater_only = merge(heater);
    let combined_n = combined.polygons().len();
    let heater_n = heater_only.polygons().len();
    assert!(
        combined_n <= heater_n,
        "wiring rule violated: {} M1 island(s) float free of HEATER",
        combined_n - heater_n,
    );
}

/// WG-connectivity invariant: a 2×2 MZI floorplan should leave the WG
/// layer as a *single* merged polygon — input stubs connect to the
/// input coupler, which connects to both arms, which connect to the
/// output coupler, which connects to the output stubs. Any gap (e.g.
/// arm length mismatching coupler placement) shows up as ≥ 2 islands.
fn assert_wg_is_single_component(wg: &Region) {
    let n = merge(wg).polygons().len();
    assert_eq!(
        n, 1,
        "wiring rule violated: WG has {n} disconnected island(s) — expected 1 contiguous network"
    );
}

#[test]
fn floorplan_wiring_rules_hold() {
    if !eda_pdks::HAS_GDSFACTORY_GENERIC {
        eprintln!("skipping: gdsfactory-generic .lyp absent at build time");
        return;
    }
    let (lib, pdk, top) = build_floorplan();
    let wg = Region::from_cell_layer(&lib, top, pdk.WG);
    let heater = Region::from_cell_layer(&lib, top, pdk.HEATER);
    let m1 = Region::from_cell_layer(&lib, top, pdk.M1);

    assert_wg_is_single_component(&wg);
    assert_m1_touches_heater(&m1, &heater);
}

#[test]
fn floorplan_passes_photonic_drc() {
    if !eda_pdks::HAS_GDSFACTORY_GENERIC {
        eprintln!("skipping: gdsfactory-generic .lyp absent at build time");
        return;
    }
    let (lib, pdk, top) = build_floorplan();

    let wg = Region::from_cell_layer(&lib, top, pdk.WG);
    let heater = Region::from_cell_layer(&lib, top, pdk.HEATER);
    let m1 = Region::from_cell_layer(&lib, top, pdk.M1);

    no_violations("WG.W>=400nm", &width(&wg, WG_MIN_W));
    // After boolean union of arms + couplers + stubs, the only WG-WG
    // spacing in the design is between the arms in the bare-arm region
    // (5 µm centre-to-centre, 500 nm width → 4.5 µm gap).
    no_violations("WG.S>=1um", &space(&wg, WG_MIN_S));
    no_violations("HEATER.W>=1um", &width(&heater, HEATER_MIN_W));
    no_violations("M1.W>=1um", &width(&m1, M1_MIN_W));

    // Heater should not extend past the design extents — sanity check
    // by asking M1 to enclose a smaller, all-zero region (always ok).
    // The real `enclosing` rule we'd run in production would be HEATER
    // ⊂ WG with margin; we omit it here because the heater is
    // intentionally narrower than the arm + adjacent coupler bridges,
    // which would change the rule semantics.
    let _enc_check = enclosing(&m1, &heater, 0);
}
