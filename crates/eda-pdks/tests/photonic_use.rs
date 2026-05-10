//! Layer-2 conformance for the photonic PDKs: register the PDK, place a
//! waveguide rectangle on `pdk.WG`, freeze the cell, and verify the
//! shape lands on the GDS pair the foundry's `.lyp` declared. Each test
//! is feature-gated and presence-gated (soft-skip when the foundry's
//! `.lyp` wasn't on disk at build time).
//!
//! These pin "the PDK is wired correctly enough that I can lay out a
//! waveguide on it" — the smallest end-to-end check that exercises
//! `register` + `add_shape` + GDS-pair readback. Trait-driven photonic
//! abstractions (`OpticalPdk` analog of the CMOS `RcLikePdk`) are a
//! follow-up; until that lands, this is the substitute.

#![allow(unused_imports)]

use klayout_core::{Bbox, CellBuilder, LayerInfo, Point, Rect, Shape};

/// Place a 1µm × 10µm rectangle on `wg_idx` inside a fresh cell and
/// return the cell id. Library DBU is 1000 (1 µm = 1000 DBU).
fn place_wg_rect(lib: &klayout_core::Library, wg_idx: klayout_core::LayerIndex, name: &str) -> klayout_core::CellId {
    let mut cb = CellBuilder::new(name);
    let rect = Rect::new(Bbox::new(Point::new(0, 0), Point::new(10_000, 1_000)));
    cb.add_shape(wg_idx, Shape::Box(rect));
    lib.insert(cb)
}

fn assert_wg_shape_on_gds_pair(
    lib: &klayout_core::Library,
    cell_id: klayout_core::CellId,
    expected_l: u16,
    expected_d: u16,
    pdk_label: &str,
) {
    let layer = lib.layer(LayerInfo::gds(expected_l, expected_d));
    let cell = lib.get(cell_id);
    let count = cell.shapes_on(layer).count();
    assert!(
        count > 0,
        "{}: expected ≥1 shape on WG GDS pair ({}, {}), found {}",
        pdk_label, expected_l, expected_d, count,
    );
}

#[cfg(feature = "gdsfactory-generic")]
#[test]
fn gdsfactory_generic_lays_out_a_waveguide() {
    if !eda_pdks::HAS_GDSFACTORY_GENERIC {
        eprintln!("skipping: gdsfactory generic .lyp absent at build time");
        return;
    }
    let lib = eda_pdks::GdsfactoryGeneric::new_library("photonic_use_gf");
    let pdk = eda_pdks::GdsfactoryGeneric::register(&lib);
    let cell = place_wg_rect(&lib, pdk.WG, "wg_test");
    // gdsfactory generic_tech: Waveguide = (1, 0).
    assert_wg_shape_on_gds_pair(&lib, cell, 1, 0, "GdsfactoryGeneric");
}

#[cfg(feature = "cornerstone-si220")]
#[test]
fn cornerstone_si220_lays_out_a_waveguide() {
    if !eda_pdks::HAS_CORNERSTONE_SI220 {
        eprintln!("skipping: cornerstone-si220 .lyp absent at build time");
        return;
    }
    let lib = eda_pdks::CornerstoneSi220::new_library("photonic_use_cs");
    let pdk = eda_pdks::CornerstoneSi220::register(&lib);
    let cell = place_wg_rect(&lib, pdk.WG, "wg_test");
    // Cornerstone si220: WG = (3, 0).
    assert_wg_shape_on_gds_pair(&lib, cell, 3, 0, "CornerstoneSi220");
}

#[cfg(feature = "siepic-ebeam")]
#[test]
fn siepic_ebeam_lays_out_a_waveguide() {
    if !eda_pdks::HAS_SIEPIC_EBEAM {
        eprintln!("skipping: siepic-ebeam .lyp absent at build time");
        return;
    }
    let lib = eda_pdks::SiepicEbeam::new_library("photonic_use_eb");
    let pdk = eda_pdks::SiepicEbeam::register(&lib);
    let cell = place_wg_rect(&lib, pdk.WG, "wg_test");
    // SiEPIC EBeam: Si = (1, 0).
    assert_wg_shape_on_gds_pair(&lib, cell, 1, 0, "SiepicEbeam");
}
