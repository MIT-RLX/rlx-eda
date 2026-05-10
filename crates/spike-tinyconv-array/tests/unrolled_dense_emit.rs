//! Smoke test for the unrolled Dense SV emitter — verifies the
//! produced top.sv contains the expected `localparam` weights and
//! parsable structure. Doesn't run Verilator (that's in
//! `eda-bench-tinyconv`'s `unrolled_rtl_sim` test, gated by
//! `bench-rtl-sim`).

use rlx_fpga::model::{tinyconv_mnist_from_cortexm, Layer};
use spike_tinyconv_array::codegen::{emit_unrolled_dense_tb, emit_unrolled_dense_top};

fn dense_layer() -> Layer {
    let model = tinyconv_mnist_from_cortexm();
    model
        .layers
        .iter()
        .find(|l| matches!(l, Layer::Dense { .. }))
        .expect("TinyConv has a Dense layer")
        .clone()
}

#[test]
fn emits_top_with_baked_weights() {
    let layer = dense_layer();
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("unrolled-dense-{}.sv", std::process::id()));
    let n = emit_unrolled_dense_top(&layer, &tmp).expect("emit");
    // 400 in × 10 out = 4000 weights baked in.
    assert_eq!(n, 4000);

    let s = std::fs::read_to_string(&tmp).unwrap();
    assert!(s.contains("module top"));
    // Weights/bias/requant emit as packed bit-vectors (Yosys-friendly)
    // — sliced via `+:` at use-time inside the always_comb.
    assert!(s.contains("W_FLAT"), "weight flat-vector localparam absent");
    assert!(s.contains("B_FLAT"), "bias flat-vector localparam absent");
    assert!(s.contains("M0_FLAT"), "M0 flat-vector localparam absent");
    assert!(s.contains("BakedConstants"), "strategy marker absent in emitted SV");
    // Sanity: file holds 4000 weight constants packed as 8'hXX hex
    // bytes plus bias / requant tables → ~30 KB at minimum.
    assert!(
        s.len() > 25_000,
        "emitted SV unexpectedly small ({} bytes) — weights may not have been inlined",
        s.len()
    );
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn emits_tb_with_correct_cycle_counter() {
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("unrolled-tb-{}.sv", std::process::id()));
    emit_unrolled_dense_tb(400, &tmp).expect("emit tb");
    let s = std::fs::read_to_string(&tmp).unwrap();
    assert!(s.contains("RESULT pred=%0d cycles=%0d"));
    assert!(s.contains("cycles_counter"));
    assert!(s.contains("$readmemh(\"tb_image.mem\""));
    let _ = std::fs::remove_file(&tmp);
}
