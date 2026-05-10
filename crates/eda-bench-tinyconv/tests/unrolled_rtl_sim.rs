//! End-to-end demo of the **2×2 architecture × target matrix**:
//!
//! ```
//!                FPGA target            ASIC target
//!              ┌──────────────────────┬──────────────────────┐
//! BRAM         │ rlx-fpga sequential  │ same SV → sky130 GDS │
//! (loaded)     │ 672 533 cycles       │ 672 533 cycles       │
//!              │ (Verilator-measured) │ (architecture-only)  │
//!              ├──────────────────────┼──────────────────────┤
//! BakedConst   │ unrolled SV via      │ same SV → sky130 GDS │
//! (burned)     │ rlx-eda::codegen     │ N cycles (one cycle  │
//!              │ N cycles measured    │ per pipeline stage)  │
//!              │ (this test)          │                      │
//!              └──────────────────────┴──────────────────────┘
//! ```
//!
//! Cycles depend on the SV (architecture); area / period / energy
//! depend on the target. So the cycle column collapses to a 1×2:
//! BRAM full-network = 672 533 cycles; BakedConst Dense-only = ?.
//! This test measures the `?` and asserts it's far smaller than
//! the BRAM Dense slice.
//!
//! `#[ignore]` by default — needs Verilator + ~30 s wall time.
//! Run with:
//!
//! ```sh
//! cargo test -p eda-bench-tinyconv --features bench-rtl-sim \
//!   --test unrolled_rtl_sim -- --ignored --nocapture
//! ```

#![cfg(feature = "bench-rtl-sim")]

use eda_bench_tinyconv::backends::rtl_sim::RtlSimBackend;
use eda_bench_tinyconv::backends::yosys_sky130::{synth_sky130, SynthMetrics};
use rlx_fpga::codegen::emit_model;
use rlx_fpga::model::{tinyconv_mnist_from_cortexm, Layer, Model};
use spike_tinyconv_array::codegen::{emit_unrolled_dense_tb, emit_unrolled_dense_top};
use std::path::Path;

/// Use a small representative Dense (32 in × 4 out = 128 weights)
/// instead of the full TinyConv-MNIST FC (400 × 10 = 4 000). The
/// 4 000-multiplier combinational netlist is tractable for Verilator
/// (seconds) but pathological for Yosys 0.64 + ABC under x86 docker
/// emulation (process exhausts the 8 GB container limit during
/// flatten/abc). 128 weights synthesizes in seconds; the
/// architectural conclusion (BRAM cycles ≫ BakedConst cycles, with
/// proportional area trade) holds at any scale.
const TEST_IN_FEATURES: usize = 32;
const TEST_OUT_FEATURES: usize = 4;

fn dense_layer() -> Layer {
    // Borrow a real Dense to keep the requantization scheme honest;
    // truncate weights / requant table to the small dimensions.
    let real = tinyconv_mnist_from_cortexm()
        .layers
        .iter()
        .find(|l| matches!(l, Layer::Dense { .. }))
        .expect("Dense layer present")
        .clone();
    match real {
        Layer::Dense {
            name,
            x_zp,
            w_zp,
            out_zp,
            weight_bits,
            mut requant,
            mut weights,
            mut bias,
            in_features,
            out_features,
            ..
        } => {
            // Keep first TEST_OUT_FEATURES rows, first TEST_IN_FEATURES cols.
            let mut new_w = Vec::with_capacity(TEST_OUT_FEATURES * TEST_IN_FEATURES);
            for oc in 0..TEST_OUT_FEATURES {
                for ic in 0..TEST_IN_FEATURES {
                    new_w.push(weights[oc * in_features + ic]);
                }
            }
            weights = new_w;
            requant.truncate(TEST_OUT_FEATURES);
            if let Some(b) = &mut bias {
                b.truncate(TEST_OUT_FEATURES);
            }
            let _ = out_features;
            Layer::Dense {
                name,
                in_features: TEST_IN_FEATURES,
                out_features: TEST_OUT_FEATURES,
                x_zp,
                w_zp,
                out_zp,
                weight_bits,
                requant,
                weights,
                bias,
            }
        }
        _ => unreachable!(),
    }
}

/// Build a single-layer Model containing only the Dense layer, so
/// `rlx_fpga::codegen::emit_model` produces a Dense-only BRAM-style
/// emit we can run through the same Verilator harness for a true
/// apples-to-apples cycle comparison.
fn dense_only_model() -> Model {
    let dense = dense_layer();
    Model {
        name: "tinyconv_dense_only".to_string(),
        input_len: TEST_IN_FEATURES,
        layers: vec![dense],
    }
}

#[test]
#[ignore = "requires docker + verilator + ~60s wall-clock; opt-in via --ignored"]
fn unrolled_dense_far_fewer_cycles_than_bram_baseline() {
    let dense = dense_layer();
    let in_features = match &dense {
        Layer::Dense { in_features, .. } => *in_features,
        _ => unreachable!(),
    };

    // Apples-to-apples input: same i8 vector drives both architectures.
    // Deterministic but non-trivial pattern (a tiny ramp + one hot)
    // exercises every weight column at least once.
    let synthetic_input: Vec<i8> =
        (0..in_features).map(|i| if i == 0 { 1 } else { (i % 7) as i8 - 3 }).collect();

    // ── Emit + simulate both architectures, keeping their dirs alive
    // so the same SV trees can be pushed through Yosys → sky130
    // afterwards (silicon area + ABC delay estimate).
    let baked = build_baked_dense(&dense, in_features, &synthetic_input);
    let bram  = build_bram_dense_only(&synthetic_input);

    // Reference: previously-measured BRAM cycles on the full 8-layer
    // model — keeps the original baseline visible for context.
    let bram_full_model_cycles: u64 = 672_533;

    // ── (3) Sky130 silicon: synthesize each SV against
    //        sky130_fd_sc_hd (tt_025C_1v80) via Yosys + ABC. ABC
    //        itself is single-threaded, so we spawn the two synths
    //        in parallel — independent designs, independent docker
    //        containers — to roughly halve wall time on multi-core
    //        hosts.
    eprintln!("[sky130] synthesizing both designs in parallel against sky130_fd_sc_hd…");
    let baked_dir = baked.dir.clone();
    let bram_dir  = bram.dir.clone();
    let h_baked = std::thread::spawn(move || synth_or_skip(&baked_dir, "top", "[baked-sky130]"));
    let h_bram  = std::thread::spawn(move || synth_or_skip(&bram_dir,  "top", "[bram-sky130] "));
    let baked_silicon = h_baked.join().expect("baked synth thread panicked");
    let bram_silicon  = h_bram.join().expect("bram synth thread panicked");

    print_matrix(
        bram.cycles,
        baked.cycles,
        bram_full_model_cycles,
        bram_silicon.as_ref(),
        baked_silicon.as_ref(),
    );

    // Apples-to-apples cycle assertion (kept from prior step).
    assert!(
        baked.cycles < bram.cycles / 100,
        "BakedConst should be ≥100× faster than BRAM for the same Dense layer; \
         got {} vs {}",
        baked.cycles,
        bram.cycles
    );

    // ── Cleanup. Leave dirs around if either silicon synth failed
    // so the user can inspect what Yosys saw.
    if baked_silicon.is_some() && bram_silicon.is_some() {
        let _ = std::fs::remove_dir_all(&baked.dir);
        let _ = std::fs::remove_dir_all(&bram.dir);
    } else {
        eprintln!(
            "(synth skipped or failed — keeping {} and {} for inspection)",
            baked.dir.display(),
            bram.dir.display()
        );
    }
}

struct BuiltDesign {
    dir: std::path::PathBuf,
    cycles: u64,
}

fn build_baked_dense(dense: &Layer, in_features: usize, input: &[i8]) -> BuiltDesign {
    let mut hw_dir = std::env::temp_dir();
    hw_dir.push(format!("rlx-eda-baked-dense-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&hw_dir);
    std::fs::create_dir_all(&hw_dir).unwrap();

    let n = emit_unrolled_dense_top(dense, &hw_dir.join("top.sv"))
        .expect("emit unrolled top");
    eprintln!(
        "[baked] emitted top.sv with {n} weight constants ({:.1} KB)",
        std::fs::metadata(hw_dir.join("top.sv")).unwrap().len() as f64 / 1024.0
    );
    emit_unrolled_dense_tb(in_features, &hw_dir.join("tb_bench.sv")).expect("emit tb");

    let mut backend = RtlSimBackend::new(hw_dir.clone());
    backend.input_len = in_features;
    eprintln!("[baked] running BakedConst Dense through Verilator…");
    let r = backend.measure_inference_one(input).expect("baked RTL sim");
    BuiltDesign { dir: hw_dir, cycles: r.cycles }
}

fn build_bram_dense_only(input: &[i8]) -> BuiltDesign {
    let mut hw_dir = std::env::temp_dir();
    hw_dir.push(format!("rlx-eda-bram-dense-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&hw_dir);
    std::fs::create_dir_all(&hw_dir).unwrap();

    let model = dense_only_model();
    emit_model(&model, &hw_dir).expect("rlx-fpga emit Dense-only model");
    eprintln!(
        "[bram]  emitted rlx-fpga Dense-only tree under {}",
        hw_dir.display()
    );

    let mut backend = RtlSimBackend::new(hw_dir.clone());
    backend.input_len = model.input_len;
    eprintln!("[bram]  running BRAM-loaded Dense through Verilator…");
    let r = backend.measure_inference_one(input).expect("bram RTL sim");
    BuiltDesign { dir: hw_dir, cycles: r.cycles }
}

fn synth_or_skip(dir: &Path, top: &str, label: &str) -> Option<SynthMetrics> {
    match synth_sky130(dir, top) {
        Ok(m) => {
            let dly = m
                .abc_delay_ps
                .map(|p| format!("{:.0} ps", p))
                .unwrap_or_else(|| "n/a".into());
            eprintln!(
                "{label} cells={} area={:.1} µm²  ABC-delay={dly}",
                m.cells, m.area_um2
            );
            Some(m)
        }
        Err(e) => {
            eprintln!("{label} synth FAILED — {e}");
            None
        }
    }
}

fn print_matrix(
    bram_cycles: u64,
    baked_cycles: u64,
    bram_full_model_cycles: u64,
    bram_silicon: Option<&SynthMetrics>,
    baked_silicon: Option<&SynthMetrics>,
) {
    let fmt_area = |m: Option<&SynthMetrics>| match m {
        Some(s) => format!("{:>10.0} µm²", s.area_um2),
        None => "       n/a    ".into(),
    };
    let fmt_period = |m: Option<&SynthMetrics>| match m.and_then(|s| s.abc_delay_ps) {
        Some(ps) => format!("{:>7.2} ns", ps / 1000.0),
        None => "    n/a   ".into(),
    };
    let fmt_silicon_time = |cycles: u64, m: Option<&SynthMetrics>| {
        match m.and_then(|s| s.abc_delay_ps) {
            Some(ps) => {
                let total_ns = cycles as f64 * ps / 1000.0;
                if total_ns >= 1_000.0 {
                    format!("{:>7.1} µs", total_ns / 1000.0)
                } else if total_ns > 0.0 {
                    format!("{:>7.1} ns", total_ns)
                } else {
                    "  ~0 (comb)".into()
                }
            }
            None => "    n/a   ".into(),
        }
    };

    eprintln!();
    eprintln!("══════════════════════ TinyConv Dense layer · sky130_fd_sc_hd (tt 25°C 1.8V) ══════════════════════");
    eprintln!("┌──────────────────────┬────────────┬────────────────┬───────────────┬──────────────────┐");
    eprintln!("│  Strategy            │   cycles   │   sky130 area  │  ABC period   │  silicon time    │");
    eprintln!("├──────────────────────┼────────────┼────────────────┼───────────────┼──────────────────┤");
    eprintln!(
        "│  BRAM   (loaded)     │ {:>10} │ {} │ {} │ {} │",
        bram_cycles,
        fmt_area(bram_silicon),
        fmt_period(bram_silicon),
        fmt_silicon_time(bram_cycles, bram_silicon),
    );
    eprintln!(
        "│  BakedConst (burned) │ {:>10} │ {} │ {} │ {} │",
        baked_cycles,
        fmt_area(baked_silicon),
        fmt_period(baked_silicon),
        fmt_silicon_time(baked_cycles, baked_silicon),
    );
    eprintln!("└──────────────────────┴────────────┴────────────────┴───────────────┴──────────────────┘");
    eprintln!(
        "(For reference: BRAM full 8-layer model = {bram_full_model_cycles} cycles.)"
    );
    eprintln!("Cycles are architecture-only (same on FPGA & ASIC).");
    eprintln!("Area + period are sky130 silicon — Yosys+ABC mapping; STA closure via ORFS for sign-off.");
    eprintln!();

    if let (Some(b), Some(k)) = (bram_silicon, baked_silicon) {
        let area_ratio = k.area_um2 / b.area_um2;
        eprintln!(
            "⇒ BakedConst trades {:.1}× more silicon area for ~{}× fewer compute cycles —",
            area_ratio,
            if baked_cycles == 0 { "∞".to_string() } else { format!("{}", bram_cycles / baked_cycles.max(1)) }
        );
        eprintln!("  the canonical 'burn weights into the network' area↔latency knob.");
    }
}
