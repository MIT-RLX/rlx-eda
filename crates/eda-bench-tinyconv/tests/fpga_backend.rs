//! `FpgaBackend::measure_functional(L1Reference, ...)` — runs the
//! bit-exact pure-Rust reference inference and reports top-1
//! accuracy. No toolchain, no docker, no foundry GDS required.

use eda_bench_tinyconv::{
    backends::{fpga::FpgaBackend, Backend, BackendError},
    metrics::functional::Level,
};
use rlx_fpga::model::tinyconv_mnist_from_cortexm;

/// Pull the cortexm-shipped single test image + label.
fn one_test_pair() -> (Vec<i8>, u8) {
    use rlx_cortexm::model_weights::{TEST_IMAGE, TEST_LABEL};
    (TEST_IMAGE.to_vec(), TEST_LABEL)
}

#[test]
fn measure_l1_classifies_canonical_test_image_correctly() {
    let model = tinyconv_mnist_from_cortexm();
    let backend = FpgaBackend::new("/tmp/fpga-test".into(), model)
        .with_test_set(vec![one_test_pair()]);
    let f = backend
        .measure_functional(Level::L1Reference, &[])
        .expect("L1 runs");
    assert_eq!(f.level, Level::L1Reference);
    assert_eq!(f.n_images, 1);
    assert_eq!(
        f.top1_acc, 1.0,
        "TinyConv-MNIST should classify TEST_IMAGE (label 7) correctly"
    );
    // Per-class accuracy: only class 7 had a sample (1.0); rest = 0.
    assert_eq!(f.per_class_acc[7], 1.0);
}

#[test]
fn measure_l1_with_empty_test_set_returns_toolchain_error() {
    let model = tinyconv_mnist_from_cortexm();
    let backend = FpgaBackend::new("/tmp/fpga-test".into(), model);
    match backend.measure_functional(Level::L1Reference, &[]) {
        Err(BackendError::Toolchain(msg)) => {
            assert!(msg.contains("empty test set"), "msg: {msg}");
        }
        other => panic!("expected Toolchain error, got {other:?}"),
    }
}

#[test]
fn measure_l1_index_out_of_bounds_surfaces_toolchain_error() {
    let model = tinyconv_mnist_from_cortexm();
    let backend = FpgaBackend::new("/tmp/fpga-test".into(), model)
        .with_test_set(vec![one_test_pair()]);
    let err = backend
        .measure_functional(Level::L1Reference, &[7])
        .unwrap_err();
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn measure_l1_aggregates_correctly_across_repeated_image() {
    // Run the same image 5 times → 5/5 correct → top-1 = 1.0,
    // n_images = 5.
    let model = tinyconv_mnist_from_cortexm();
    let backend = FpgaBackend::new("/tmp/fpga-test".into(), model)
        .with_test_set(vec![one_test_pair(); 5]);
    let f = backend
        .measure_functional(Level::L1Reference, &[])
        .unwrap();
    assert_eq!(f.n_images, 5);
    assert_eq!(f.top1_acc, 1.0);
}

#[test]
fn measure_l1_subsetting_via_indices_works() {
    let model = tinyconv_mnist_from_cortexm();
    // Two-image set: same TEST_IMAGE twice.
    let backend = FpgaBackend::new("/tmp/fpga-test".into(), model)
        .with_test_set(vec![one_test_pair(), one_test_pair()]);
    // Eval just the second.
    let f = backend
        .measure_functional(Level::L1Reference, &[1])
        .unwrap();
    assert_eq!(f.n_images, 1);
    assert_eq!(f.top1_acc, 1.0);
}

#[test]
fn measure_physical_returns_not_enabled_without_bench_fpga_feature() {
    let model = tinyconv_mnist_from_cortexm();
    let backend = FpgaBackend::new("/tmp/fpga-test".into(), model);
    match backend.measure_physical() {
        Err(BackendError::NotEnabled("fpga", "bench-fpga")) => {}
        other => panic!("expected NotEnabled, got {other:?}"),
    }
}
