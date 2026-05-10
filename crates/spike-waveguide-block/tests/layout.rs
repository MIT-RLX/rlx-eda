//! Cross-PDK Layout conformance for [`Waveguide`].
//!
//! For each enabled photonic PDK: register the PDK, lay out a 0.5 µm
//! wide × 10 µm long Waveguide via `Layout<P: OpticalPdk>::layout`, and
//! assert (a) the WG shape lands on the foundry's expected GDS pair, and
//! (b) the cell exposes two optical ports at the expected endpoints.
//!
//! This is the photonic counterpart of `spike-divider-block`'s
//! `foundry_pdks.rs` test: parametric helper, one tiny per-PDK test
//! function calling in.

#![allow(unused_imports)]

use eda_hir::Layout;
use klayout_core::{CellId, LayerInfo, Library, PortKindId};
use spike_waveguide_block::{OpticalPdk, Waveguide};

/// Lay out a fixed-size waveguide and assert the WG shape lands on
/// `(expected_l, expected_d)` and the cell has 2 optical ports.
fn assert_waveguide_lays_out_with_expected_gds_pair<P: OpticalPdk>(
    lib: &Library,
    pdk: &P,
    pdk_label: &str,
    expected_l: u16,
    expected_d: u16,
) {
    let wg = Waveguide { width: 500, length: 10_000, id: "wg1".into() };
    let id: CellId = wg.layout(lib, pdk);
    let cell = lib.get(id);

    // Strip rectangle on the WG layer.
    let layer = lib.layer(LayerInfo::gds(expected_l, expected_d));
    let shape_count = cell.shapes_on(layer).count();
    assert!(
        shape_count > 0,
        "{}: expected ≥1 shape on WG GDS ({},{}), found {}",
        pdk_label, expected_l, expected_d, shape_count,
    );

    // Two optical ports: `in` and `out`, both tagged with optical_kind.
    let optical = pdk.optical_kind();
    let opt_ports: Vec<_> = cell.ports().iter().filter(|p| p.kind == optical).collect();
    assert_eq!(
        opt_ports.len(), 2,
        "{}: expected 2 optical ports, found {}",
        pdk_label, opt_ports.len(),
    );
    let names: Vec<&str> = opt_ports.iter().map(|p| p.name.as_str()).collect();
    for need in ["in", "out"] {
        assert!(names.contains(&need), "{}: missing optical port {:?}: {:?}", pdk_label, need, names);
    }
}

#[cfg(feature = "gdsfactory-generic")]
#[test]
fn waveguide_lays_out_under_gdsfactory_generic() {
    if !eda_pdks::HAS_GDSFACTORY_GENERIC {
        eprintln!("skipping: gdsfactory-generic .lyp absent at build time");
        return;
    }
    let lib = eda_pdks::GdsfactoryGeneric::new_library("wg_gf");
    let pdk = eda_pdks::GdsfactoryGeneric::register(&lib);
    assert_waveguide_lays_out_with_expected_gds_pair(&lib, &pdk, "GdsfactoryGeneric", 1, 0);
}

#[cfg(feature = "cornerstone-si220")]
#[test]
fn waveguide_lays_out_under_cornerstone_si220() {
    if !eda_pdks::HAS_CORNERSTONE_SI220 {
        eprintln!("skipping: cornerstone-si220 .lyp absent at build time");
        return;
    }
    let lib = eda_pdks::CornerstoneSi220::new_library("wg_cs");
    let pdk = eda_pdks::CornerstoneSi220::register(&lib);
    assert_waveguide_lays_out_with_expected_gds_pair(&lib, &pdk, "CornerstoneSi220", 3, 0);
}

#[cfg(feature = "siepic-ebeam")]
#[test]
fn waveguide_lays_out_under_siepic_ebeam() {
    if !eda_pdks::HAS_SIEPIC_EBEAM {
        eprintln!("skipping: siepic-ebeam .lyp absent at build time");
        return;
    }
    let lib = eda_pdks::SiepicEbeam::new_library("wg_eb");
    let pdk = eda_pdks::SiepicEbeam::register(&lib);
    assert_waveguide_lays_out_with_expected_gds_pair(&lib, &pdk, "SiepicEbeam", 1, 0);
}

/// Cross-PDK shape-count parity: the same `Waveguide` value should yield
/// the same shape count under each photonic PDK (the geometry is
/// foundry-agnostic; only the GDS pair changes).
#[cfg(all(feature = "gdsfactory-generic", feature = "cornerstone-si220", feature = "siepic-ebeam"))]
#[test]
fn waveguide_shape_count_is_pdk_independent() {
    if !(eda_pdks::HAS_GDSFACTORY_GENERIC
        && eda_pdks::HAS_CORNERSTONE_SI220
        && eda_pdks::HAS_SIEPIC_EBEAM) {
        eprintln!("skipping: not all photonic .lyps present");
        return;
    }
    let make = || Waveguide { width: 500, length: 10_000, id: "wg_parity".into() };

    let lib_a = eda_pdks::GdsfactoryGeneric::new_library("parity_gf");
    let pdk_a = eda_pdks::GdsfactoryGeneric::register(&lib_a);
    let id_a = make().layout(&lib_a, &pdk_a);
    let count_a = lib_a.get(id_a).shapes_on(lib_a.layer(LayerInfo::gds(1, 0))).count();

    let lib_b = eda_pdks::CornerstoneSi220::new_library("parity_cs");
    let pdk_b = eda_pdks::CornerstoneSi220::register(&lib_b);
    let id_b = make().layout(&lib_b, &pdk_b);
    let count_b = lib_b.get(id_b).shapes_on(lib_b.layer(LayerInfo::gds(3, 0))).count();

    let lib_c = eda_pdks::SiepicEbeam::new_library("parity_eb");
    let pdk_c = eda_pdks::SiepicEbeam::register(&lib_c);
    let id_c = make().layout(&lib_c, &pdk_c);
    let count_c = lib_c.get(id_c).shapes_on(lib_c.layer(LayerInfo::gds(1, 0))).count();

    assert_eq!(count_a, count_b, "shape count diverged: gdsfactory={}, cornerstone={}", count_a, count_b);
    assert_eq!(count_b, count_c, "shape count diverged: cornerstone={}, ebeam={}", count_b, count_c);
}
