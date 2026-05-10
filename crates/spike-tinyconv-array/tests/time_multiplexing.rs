//! Time-multiplexing helpers — `cycles_per_layer`, `total_cycles`,
//! `lower_time_multiplexed`. Where strict `lower` rejects the real
//! TinyConv-MNIST model on small grids, the time-multiplexed
//! variant keeps the grid small and reports how many cycles the
//! workload takes — DADO's natural input for "throughput vs area"
//! Pareto exploration.

use eda_hir::Block;
use rlx_fpga::model::{tinyconv_mnist_from_cortexm, Layer, Model};
use spike_tinyconv_array::{
    array::ArrayConfig,
    lower::{cycles_per_layer, lower_time_multiplexed, total_cycles, LowerError},
};

fn synth_dense(name: &'static str, ifeats: usize, ofeats: usize) -> Layer {
    Layer::Dense {
        name,
        in_features: ifeats,
        out_features: ofeats,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: 8,
        requant: vec![(0, 0); ofeats],
        weights: vec![0; ifeats * ofeats],
        bias: None,
    }
}

fn model(name: &str, layers: Vec<Layer>) -> Model {
    Model {
        name: name.to_string(),
        input_len: 16,
        layers,
    }
}

#[test]
fn cycles_per_layer_returns_one_when_layer_fits() {
    // 50 weights ≤ 100 budget → single pass.
    assert_eq!(cycles_per_layer(50, 100), 1);
    // Equal-fit case: 100 weights / 100 budget → still 1.
    assert_eq!(cycles_per_layer(100, 100), 1);
}

#[test]
fn cycles_per_layer_ceil_divides_when_layer_exceeds_budget() {
    // 4000 weights / 16 tile budget → 250 cycles.
    assert_eq!(cycles_per_layer(4000, 16), 250);
    // 101 weights / 100 budget → 2 cycles (ceil).
    assert_eq!(cycles_per_layer(101, 100), 2);
}

#[test]
fn cycles_per_layer_handles_zero_budget_safely() {
    // Degenerate case — usize::MAX signals "infinite cycles" so a
    // caller chaining `total_cycles` doesn't panic.
    assert_eq!(cycles_per_layer(1, 0), usize::MAX);
}

#[test]
fn total_cycles_sums_across_compute_layers_only() {
    // Conv(needs 100) + Relu(0) + Dense(needs 200) on 50-tile
    // budget → 2 + 0 + 4 = 6 cycles.
    let m = model(
        "tm-test",
        vec![
            synth_dense("a", 10, 10), // 100 weights
            Layer::Relu {
                name: "r",
                len: 8,
                zero_point: 0,
            },
            synth_dense("b", 20, 10), // 200 weights
        ],
    );
    assert_eq!(total_cycles(&m, 50), 2 + 0 + 4);
}

#[test]
fn lower_time_multiplexed_does_not_error_on_overbudget_layer() {
    // Where strict `lower` would error, time-multiplexed accepts
    // and reports cycles instead.
    let m = model("big", vec![synth_dense("d", 100, 100)]); // 10_000 weights
    let cfg = ArrayConfig::default(); // 4×4 = 16 tiles
    let (array, cycles) = lower_time_multiplexed(&m, cfg).expect("no error");
    assert!(array.name().contains("lowered_tm_big"));
    assert_eq!(cycles, 10_000_usize.div_ceil(16));
}

#[test]
fn lower_time_multiplexed_still_errors_on_empty_model() {
    let m = model("empty", vec![]);
    match lower_time_multiplexed(&m, ArrayConfig::default()) {
        Err(LowerError::EmptyModel) => {}
        other => panic!("expected EmptyModel, got {other:?}"),
    }
}

#[test]
fn real_tinyconv_mnist_total_cycles_on_default_grid() {
    // Default 4×4 = 16 tile budget. TinyConv layer cycles:
    //   conv1   72 weights → ceil(72 / 16)   = 5
    //   relu1                                 = 0
    //   pool1                                 = 0
    //   conv2 1152 weights → ceil(1152 / 16) = 72
    //   relu2                                 = 0
    //   pool2                                 = 0
    //   fc    4000 weights → ceil(4000 / 16) = 250
    //   argmax                                = 0
    // total = 5 + 72 + 250 = 327 cycles
    let model = tinyconv_mnist_from_cortexm();
    assert_eq!(total_cycles(&model, 16), 327);
}

#[test]
fn real_tinyconv_mnist_lowers_under_time_multiplexing_on_default_grid() {
    // Strict `lower` rejects the same model on the same grid
    // (covered by `lower_real_mnist::default_4x4_grid_rejects_real_model_with_overbudget`).
    let model = tinyconv_mnist_from_cortexm();
    let (_array, cycles) =
        lower_time_multiplexed(&model, ArrayConfig::default()).expect("tm accepts");
    assert!(
        cycles > 100,
        "real MNIST on a 16-tile array should take > 100 cycles; got {cycles}"
    );
}
