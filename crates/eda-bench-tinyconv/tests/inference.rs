//! End-to-end inference perf tests. Times the bit-exact pure-Rust
//! reference path; same wrapper plugs into FPGA / silicon backends
//! once they run real images.

use eda_bench_tinyconv::inference::{
    run_inference_bench, InferenceConfig, InferenceError, InferenceMetrics, SimulatedLatency,
};
use rlx_fpga::model::tinyconv_mnist_from_cortexm;

fn one_test_pair() -> (Vec<i8>, u8) {
    use rlx_cortexm::model_weights::{TEST_IMAGE, TEST_LABEL};
    (TEST_IMAGE.to_vec(), TEST_LABEL)
}

#[test]
fn empty_test_set_returns_error() {
    let model = tinyconv_mnist_from_cortexm();
    match run_inference_bench(&model, &[], &InferenceConfig::default()) {
        Err(InferenceError::EmptyTestSet) => {}
        other => panic!("expected EmptyTestSet, got {other:?}"),
    }
}

#[test]
fn zero_n_images_returns_empty_metrics() {
    let model = tinyconv_mnist_from_cortexm();
    let cfg = InferenceConfig {
        n_images: 0,
        ..InferenceConfig::default()
    };
    let m = run_inference_bench(&model, &[one_test_pair()], &cfg).unwrap();
    assert_eq!(m.n_images, 0);
    assert_eq!(m.throughput_per_sec, 0.0);
}

#[test]
fn small_bench_produces_finite_positive_stats() {
    // Keep n_images low so the test runs in <100ms.
    let model = tinyconv_mnist_from_cortexm();
    let cfg = InferenceConfig {
        n_images: 5,
        warmup: 2,
        repetitions: 2,
    };
    let m: InferenceMetrics = run_inference_bench(&model, &[one_test_pair()], &cfg).unwrap();

    assert_eq!(m.n_images, 5);
    assert_eq!(m.repetitions, 2);
    assert!(m.mean_us > 0.0, "mean_us should be > 0; got {}", m.mean_us);
    assert!(m.p50_us > 0.0);
    assert!(m.p99_us >= m.p50_us, "p99 should be ≥ p50");
    assert!(m.min_us <= m.mean_us);
    assert!(m.max_us >= m.mean_us);
    assert!(m.throughput_per_sec > 0.0);
    // Sanity: throughput ≈ 1e6 / mean_us
    let derived = 1_000_000.0 / m.mean_us;
    assert!(
        (m.throughput_per_sec - derived).abs() < 1e-3,
        "throughput {} should match 1e6/mean_us {}",
        m.throughput_per_sec,
        derived
    );
}

#[test]
fn l1_reference_has_no_simulated_latency() {
    // L1 is pure Rust — no clock, no silicon time. The
    // `simulated` field MUST be None so consumers don't
    // confuse host wall-clock for silicon performance.
    let model = tinyconv_mnist_from_cortexm();
    let cfg = InferenceConfig {
        n_images: 2,
        warmup: 0,
        repetitions: 1,
    };
    let m = run_inference_bench(&model, &[one_test_pair()], &cfg).unwrap();
    assert!(m.simulated.is_none(), "L1 reference must not populate simulated latency");
    assert!(m.mean_us > 0.0, "but wall-clock should still be populated");
}

#[test]
fn simulated_latency_round_trips_through_with_simulated() {
    // L2+ backend pattern: build wall-clock metrics, attach
    // simulated cycles, verify both fields land.
    let model = tinyconv_mnist_from_cortexm();
    let cfg = InferenceConfig {
        n_images: 2,
        warmup: 0,
        repetitions: 1,
    };
    let m = run_inference_bench(&model, &[one_test_pair()], &cfg)
        .unwrap()
        .with_simulated(SimulatedLatency::from_cycles(327, 10.0));
    let sim = m.simulated.expect("simulated populated");
    assert_eq!(sim.cycles_per_inference, 327);
    assert_eq!(sim.period_ns, 10.0);
    assert_eq!(sim.total_ns, 3270.0);
    // 1e9 / 3270 = 305_810 inferences/sec at 100 MHz with 327 cycles.
    assert!((sim.silicon_throughput_per_sec - 305_810.397).abs() < 1.0);
}

#[test]
fn simulated_latency_zero_cycles_returns_zero_throughput() {
    // Edge case: zero-cycle "inference" (degenerate, but
    // shouldn't div-by-zero).
    let sim = SimulatedLatency::from_cycles(0, 10.0);
    assert_eq!(sim.total_ns, 0.0);
    assert_eq!(sim.silicon_throughput_per_sec, 0.0);
}

#[test]
fn percentiles_are_ordered_min_p50_p99_max() {
    let model = tinyconv_mnist_from_cortexm();
    let cfg = InferenceConfig {
        n_images: 20,
        warmup: 5,
        repetitions: 2,
    };
    let m = run_inference_bench(&model, &[one_test_pair()], &cfg).unwrap();
    assert!(m.min_us <= m.p50_us);
    assert!(m.p50_us <= m.p99_us);
    assert!(m.p99_us <= m.max_us);
}
