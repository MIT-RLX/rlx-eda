//! Real RTL inference via Verilator inside docker. Drives one
//! MNIST image through the rlx-fpga-emitted SystemVerilog, parses
//! the simulator's stdout for prediction + cycle count, asserts
//! both match the bit-exact reference.
//!
//! `#[ignore]` by default — Verilator compile + sim takes ~30-60 s
//! the first time and needs docker on PATH. Run explicitly with:
//!
//! ```sh
//! cargo test -p eda-bench-tinyconv --features bench-rtl-sim \
//!   --test rtl_sim_real -- --ignored --nocapture
//! ```

#![cfg(feature = "bench-rtl-sim")]

use eda_bench_tinyconv::{
    backends::rtl_sim::RtlSimBackend,
    inference::SimulatedLatency,
};
use std::path::PathBuf;

fn rlx_fpga_hw_dir() -> PathBuf {
    // Walk from this crate's manifest up to the rlx workspace.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("rlx")
        .join("rlx-fpga")
        .join("hw")
        .join("tinyconv_mnist")
}

fn one_test_pair() -> (Vec<i8>, u8) {
    use rlx_cortexm::model_weights::{TEST_IMAGE, TEST_LABEL};
    (TEST_IMAGE.to_vec(), TEST_LABEL)
}

#[test]
#[ignore = "requires docker + verilator pull + ~30-60s wall-clock; opt-in via --ignored"]
fn rtl_sim_inference_classifies_test_image_correctly() {
    let hw_dir = rlx_fpga_hw_dir();
    if !hw_dir.join("top.sv").exists() {
        eprintln!(
            "skipping: rlx-fpga emit output not present at {hw_dir:?}; \
             run `cargo run -p rlx-fpga --bin rlx-fpga-emit` first"
        );
        return;
    }

    let backend = RtlSimBackend::new(hw_dir);
    let (image, label) = one_test_pair();

    eprintln!("running RTL sim through Verilator (this takes ~30-60s)…");
    let result = backend
        .measure_inference_one(&image)
        .expect("RTL sim succeeds");

    eprintln!(
        "RTL sim result: prediction={}, cycles={}",
        result.prediction, result.cycles
    );

    assert_eq!(
        result.prediction as u8, label,
        "RTL sim should predict the same class as the bit-exact reference"
    );
    assert!(
        result.cycles > 0,
        "cycle count should be positive (got {})",
        result.cycles
    );

    // Project to silicon time at 100 MHz target.
    let sim: SimulatedLatency = result.to_simulated(10.0);
    eprintln!(
        "simulated silicon latency: {} cycles × {:.1} ns = {:.2} µs / inference, \
         throughput {:.0} inferences/s",
        sim.cycles_per_inference,
        sim.period_ns,
        sim.total_ns / 1000.0,
        sim.silicon_throughput_per_sec
    );

    // Sanity bounds: TinyConv on a sequential controller should
    // complete in 100k–10M cycles (rough order of magnitude for the
    // ~10k MAC operations in the model).
    assert!(
        result.cycles > 1_000 && result.cycles < 100_000_000,
        "cycle count should be in a sane range: {}",
        result.cycles
    );
}
