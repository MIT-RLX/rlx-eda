//! `lower(model, config)` tests — Model → ArrayBlock conversion.

use eda_hir::Block;
use rlx_fpga::model::{Layer, Model};
use spike_tinyconv_array::{
    array::ArrayConfig,
    lower::{lower, min_required_tiles, weight_count, LowerError},
};

fn synthetic_conv(name: &'static str, c_in: usize, c_out: usize, kh: usize, kw: usize) -> Layer {
    Layer::Conv2d {
        name,
        h_in: 4,
        w_in: 4,
        c_in,
        c_out,
        kh,
        kw,
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: 8,
        requant: vec![(0, 0); c_out],
        weights: vec![0; c_in * c_out * kh * kw],
        bias: None,
    }
}

fn synthetic_dense(name: &'static str, ifeats: usize, ofeats: usize) -> Layer {
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
fn empty_model_returns_empty_model_error() {
    match lower(&model("nope", vec![]), ArrayConfig::default()) {
        Err(LowerError::EmptyModel) => {}
        other => panic!("expected EmptyModel, got {other:?}"),
    }
}

#[test]
fn weight_count_matches_per_layer_formula() {
    // Conv2d(c_in=2, c_out=4, kh=3, kw=3) = 2 * 4 * 3 * 3 = 72.
    assert_eq!(weight_count(&synthetic_conv("c", 2, 4, 3, 3)), 72);
    // Dense(in=10, out=5) = 50.
    assert_eq!(weight_count(&synthetic_dense("d", 10, 5)), 50);
    // Relu / MaxPool = 0.
    assert_eq!(
        weight_count(&Layer::Relu {
            name: "r",
            len: 8,
            zero_point: 0
        }),
        0
    );
}

#[test]
fn lower_succeeds_when_every_layer_fits() {
    // Default 4×4 → budget 16. Synthetic conv with 1·2·2·2 = 8 fits.
    let m = model("fits", vec![synthetic_conv("c", 1, 2, 2, 2)]);
    let array = lower(&m, ArrayConfig::default()).expect("fits");
    assert!(array.name().contains("lowered_fits"));
}

#[test]
fn lower_returns_overbudget_with_layer_index() {
    // Default 4×4 = 16 tile budget. Dense(10, 10) needs 100.
    let m = model(
        "too-big",
        vec![
            synthetic_conv("c1", 1, 2, 2, 2), // 8, fits
            synthetic_dense("d1", 10, 10),    // 100, doesn't fit
        ],
    );
    match lower(&m, ArrayConfig::default()) {
        Err(LowerError::OverBudget {
            layer,
            needed,
            budget,
        }) => {
            assert_eq!(layer, 1);
            assert_eq!(needed, 100);
            assert_eq!(budget, 16);
        }
        other => panic!("expected OverBudget, got {other:?}"),
    }
}

#[test]
fn min_required_tiles_returns_largest_layer_count() {
    let m = model(
        "mixed",
        vec![
            synthetic_conv("c1", 1, 8, 3, 3), // 72
            synthetic_conv("c2", 8, 16, 3, 3), // 1152 ← largest
            synthetic_dense("d1", 400, 10),    // 4000 ← actually largest
        ],
    );
    assert_eq!(min_required_tiles(&m), 4000);
}

#[test]
fn relu_and_maxpool_layers_consume_zero_tiles() {
    let m = model(
        "with-glue",
        vec![
            synthetic_conv("c", 1, 2, 2, 2), // 8
            Layer::Relu {
                name: "r",
                len: 32,
                zero_point: 0,
            },
            Layer::MaxPool2d {
                name: "p",
                h_in: 4,
                w_in: 4,
                c: 2,
                kh: 2,
                kw: 2,
                stride_h: 2,
                stride_w: 2,
            },
        ],
    );
    // Even on a 1-tile budget, the Relu + MaxPool don't push over.
    let cfg_3_tiles = ArrayConfig {
        grid: (1, 3), // budget 3
        ..ArrayConfig::default()
    };
    assert!(lower(&m, cfg_3_tiles).is_err()); // Conv2d needs 8
    let cfg_8_tiles = ArrayConfig {
        grid: (2, 4), // budget 8
        ..ArrayConfig::default()
    };
    assert!(lower(&m, cfg_8_tiles).is_ok());
}
