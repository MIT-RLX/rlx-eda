//! 8-bit DAC validation: enumerate codes through ngspice (and LTspice
//! when present), assert each Vout matches the analytic formula, and
//! render the full 0..255 staircase as a PNG.
//!
//! The staircase plot is the textbook DAC visualization: code on the x
//! axis, Vout on the y axis. A perfectly linear ladder produces 256
//! evenly-spaced steps; INL would show as deviation from the diagonal.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};
use eda_spice_emit::{Netlist, SpiceEmit};
use spike_dac_r2r::{ideal_vout, R2RDac};

const VREF: f64 = 1.0;
const VLOW: f64 = 0.0;
const N_BITS: u32 = 8;

/// Build a deck with the DAC plus DC sources for each input bit.
fn deck_for_code(code: u32) -> String {
    let dac: R2RDac<8> = R2RDac::default();
    let mut net = Netlist::new("8-bit R-2R DAC code probe");
    net.add_dc_source("ref", "vref", "0", VREF);
    net.add_dc_source("low", "vlow", "0", VLOW);
    // Drive each input from a controlled DC source. Bit i = (code >> i) & 1.
    for i in 0..N_BITS as usize {
        let bit = (code >> i) & 1;
        let v = if bit == 1 { VREF } else { VLOW };
        net.add_dc_source(&format!("b{i}"), &format!("in{i}"), "0", v);
    }
    let nets: Vec<String> = (0..N_BITS as usize).map(|i| format!("in{i}")).collect();
    let mut nets_refs: Vec<&str> = nets.iter().map(String::as_str).collect();
    nets_refs.push("vlow");
    nets_refs.push("vout");
    dac.emit_spice(&mut net, &nets_refs, "u1").unwrap();
    net.deck()
}

fn ngspice_vout(ng: &LocalBinary, deck: &str) -> f64 {
    let res = ng
        .run_dc(deck, &[OutputRequest::NodeVoltage("vout".into())])
        .expect("ngspice .op");
    res.node_voltages["vout"]
}

#[test]
fn paper_fig_8_1_example_code_164_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let v = ngspice_vout(&ng, &deck_for_code(164));
    let expected = ideal_vout(164, N_BITS, VREF, VLOW);
    assert!(
        (v - expected).abs() < 1e-3,
        "ngspice vout for code=164 = {v:.6}, expected {expected:.6} (the paper says 0.6406)",
    );
}

#[test]
fn boundary_codes_zero_half_full_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    for code in [0u32, 128, 255] {
        let v = ngspice_vout(&ng, &deck_for_code(code));
        let expected = ideal_vout(code, N_BITS, VREF, VLOW);
        assert!(
            (v - expected).abs() < 1e-3,
            "ngspice vout for code={code} = {v:.6}, expected {expected:.6}",
        );
    }
}

#[test]
fn checkerboard_codes_aa_55_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    for code in [0xAAu32, 0x55] {
        let v = ngspice_vout(&ng, &deck_for_code(code));
        let expected = ideal_vout(code, N_BITS, VREF, VLOW);
        assert!(
            (v - expected).abs() < 1e-3,
            "ngspice vout for code=0x{code:X} = {v:.6}, expected {expected:.6}",
        );
    }
}

/// Full 0..255 sweep, render the staircase PNG, and assert max INL <
/// 0.5 LSB. The R-2R ladder has zero theoretical INL (it's a linear
/// network of fixed resistors); any deviation is purely SPICE
/// floating-point + DC-solver tolerance, so 0.5 LSB ≈ 2 mV is wildly
/// generous.
#[test]
fn full_staircase_renders_and_is_linear_ngspice() {
    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let mut codes = Vec::with_capacity(256);
    let mut vouts = Vec::with_capacity(256);
    let mut ideals = Vec::with_capacity(256);
    let mut max_inl_lsb = 0.0_f64;
    let lsb = (VREF - VLOW) / (1u64 << N_BITS) as f64;

    for code in 0u32..256 {
        let v = ngspice_vout(&ng, &deck_for_code(code));
        let ideal = ideal_vout(code, N_BITS, VREF, VLOW);
        let inl_lsb = (v - ideal) / lsb;
        if inl_lsb.abs() > max_inl_lsb { max_inl_lsb = inl_lsb.abs(); }
        codes.push(code as f64);
        vouts.push(v);
        ideals.push(ideal);
    }
    eprintln!("max INL across 256 codes = {max_inl_lsb:.4} LSB");
    assert!(
        max_inl_lsb < 0.5,
        "DAC INL exceeded 0.5 LSB: {max_inl_lsb:.4} LSB",
    );

    render_staircase(&codes, &vouts, &ideals);
}

/// Cross-sim triangulation on a handful of codes — same deck through
/// ngspice and LTspice, assert agreement.
#[cfg(feature = "ltspice")]
#[test]
fn ngspice_and_ltspice_agree_on_dac_ngspice() {
    use eda_extern_ltspice::{Invoker as LtInvoker, LocalBinary as LtLocal, OutputRequest as LtReq};

    let Ok(ng) = LocalBinary::from_env() else { eprintln!("ngspice missing"); return; };
    let Some(lt) = LtLocal::from_env_optional() else {
        eprintln!("LTspice missing; skipping triangulation");
        return;
    };

    for code in [0u32, 1, 64, 127, 128, 164, 200, 255] {
        let deck = deck_for_code(code);
        let ng_v = ngspice_vout(&ng, &deck);
        let lt_v = lt
            .run_dc(&deck, &[LtReq::NodeVoltage("vout".into())])
            .expect("LTspice .op")
            .node_voltages["vout"];
        assert!(
            (ng_v - lt_v).abs() < 1e-3,
            "code={code}: ngspice={ng_v:.6} vs LTspice={lt_v:.6}",
        );
    }
}

fn render_staircase(codes: &[f64], measured: &[f64], ideal: &[f64]) {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use eda_waveform::{plot, Waveform};

    let mut signals = BTreeMap::new();
    signals.insert("vout (ngspice)".into(), measured.to_vec());
    signals.insert("ideal".into(), ideal.to_vec());
    let wave = Waveform::Real {
        axis_name: "code (0..255)".into(),
        axis: codes.to_vec(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title("8-bit R-2R DAC: Vout vs code (vref = 1V)")
        .with_size(900, 600)
        .add_marker(plot::Marker::Vertical { x: 164.0, label: Some("paper Fig 8.1 (code 164)".into()) })
        .add_marker(plot::Marker::Horizontal { y: 0.640625, label: Some("0.6406 V".into()) });
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("dac_staircase.png");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, out_dir.join("dac_staircase.svg"), &cfg).expect("svg");
    eprintln!("DAC staircase at {}", png.display());
}
