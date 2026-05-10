//! Lowers the real TinyConv-MNIST model from `rlx_fpga` → `ArrayBlock`,
//! confirming that the v1 lowering catches the realistic
//! "default 4×4 grid is way too small" sizing tension.
//!
//! This is the first test that connects the silicon flow to the
//! exact `Model` value the FPGA backend already validates against.
//! When DADO outer search lands, this test grows into a sweep that
//! finds the smallest grid satisfying every layer.

use eda_hir::Block;
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use spike_tinyconv_array::{
    array::ArrayConfig,
    lower::{lower, min_required_tiles, LowerError},
};
use spike_tinyconv_tile::TileParams;

#[test]
fn real_tinyconv_mnist_largest_layer_is_dense_4000() {
    // TinyConv-MNIST: conv1(1×8×3×3=72) + relu + pool + conv2(8×16×3×3=1152)
    //               + relu + pool + dense(5·5·16 × 10 = 4000) + argmax.
    let model = tinyconv_mnist_from_cortexm();
    let needed = min_required_tiles(&model);
    assert_eq!(
        needed, 4000,
        "Dense(in_features=400, out_features=10) should dominate"
    );
}

#[test]
fn default_4x4_grid_rejects_real_model_with_overbudget() {
    // Default 4×4 = 16 tile budget. First Conv2d layer alone (72)
    // already overflows → OverBudget on layer 0.
    let model = tinyconv_mnist_from_cortexm();
    match lower(&model, ArrayConfig::default()) {
        Err(LowerError::OverBudget {
            layer,
            needed,
            budget,
        }) => {
            assert_eq!(layer, 0, "conv1 is layer 0");
            assert_eq!(needed, 72);
            assert_eq!(budget, 16);
        }
        other => panic!("expected OverBudget on layer 0, got {other:?}"),
    }
}

#[test]
fn grid_sized_to_min_required_lowers_successfully() {
    // Pick a grid that exactly accommodates the largest layer.
    // 64×64 = 4096 ≥ 4000.
    let model = tinyconv_mnist_from_cortexm();
    let cfg = ArrayConfig {
        grid: (64, 64),
        pipeline_depth: 1,
        topology: spike_tinyconv_tile::MacTopology::Digital,
        tile_params: TileParams::default(),
    };
    let array = lower(&model, cfg).expect("64×64 grid fits");
    assert!(array.name().contains("lowered_tinyconv_mnist"));
}
