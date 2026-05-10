//! Per-channel parallel synth of the BakedConst Dense layer.
//!
//! Path (c) of the throughput-optimization plan: the monolithic
//! `top.sv` with all OUT × IN multipliers chokes single-threaded
//! ABC. Splitting the design into one self-contained module per
//! output channel lets us run OUT independent ABC processes in
//! parallel — true OUT× wall-time speedup on a multi-core host.
//!
//! ```sh
//! cargo test -p eda-bench-tinyconv --features bench-rtl-sim \
//!   --test baked_per_channel_synth -- --ignored --nocapture
//! ```

#![cfg(feature = "bench-rtl-sim")]

use eda_bench_tinyconv::backends::yosys_sky130::{synth_sky130, SynthMetrics};
use rlx_fpga::model::{tinyconv_mnist_from_cortexm, Layer};
use spike_tinyconv_array::codegen::emit_unrolled_dense_per_channel;
use std::path::PathBuf;
use std::time::Instant;

const TEST_IN_FEATURES: usize = 32;
const TEST_OUT_FEATURES: usize = 4;

fn small_dense() -> Layer {
    let real = tinyconv_mnist_from_cortexm()
        .layers
        .iter()
        .find(|l| matches!(l, Layer::Dense { .. }))
        .expect("Dense layer present")
        .clone();
    match real {
        Layer::Dense {
            name, x_zp, w_zp, out_zp, weight_bits,
            mut requant, mut weights, mut bias,
            in_features, out_features, ..
        } => {
            let mut new_w = Vec::with_capacity(TEST_OUT_FEATURES * TEST_IN_FEATURES);
            for oc in 0..TEST_OUT_FEATURES {
                for ic in 0..TEST_IN_FEATURES {
                    new_w.push(weights[oc * in_features + ic]);
                }
            }
            weights = new_w;
            requant.truncate(TEST_OUT_FEATURES);
            if let Some(b) = &mut bias { b.truncate(TEST_OUT_FEATURES); }
            let _ = out_features;
            Layer::Dense {
                name,
                in_features: TEST_IN_FEATURES,
                out_features: TEST_OUT_FEATURES,
                x_zp, w_zp, out_zp, weight_bits,
                requant, weights, bias,
            }
        }
        _ => unreachable!(),
    }
}

#[test]
#[ignore = "needs docker + openroad/orfs:latest; ~3-10 min wall (4 parallel ABCs)"]
fn baked_per_channel_parallel_synth() {
    let dense = small_dense();

    // Each channel gets its own dir so it's a self-contained synth target.
    let base = std::env::temp_dir().join(format!("rlx-eda-baked-pc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let paths = emit_unrolled_dense_per_channel(&dense, &base).expect("per-channel emit");
    eprintln!(
        "[per-channel] emitted {} channel modules under {}",
        paths.len(),
        base.display()
    );

    // Stage each channel's SV in its own subdir so synth_sky130's
    // glob (`/design/*.sv`) picks up exactly one file.
    let mut channel_dirs: Vec<PathBuf> = Vec::new();
    for (oc, p) in paths.iter().enumerate() {
        let dir = base.join(format!("ch{oc}"));
        std::fs::create_dir_all(&dir).unwrap();
        let dst = dir.join(p.file_name().unwrap());
        std::fs::copy(p, &dst).unwrap();
        channel_dirs.push(dir);
    }

    // ── 4-way parallel synth.
    let t0 = Instant::now();
    let handles: Vec<_> = channel_dirs
        .iter()
        .enumerate()
        .map(|(oc, d)| {
            let d = d.clone();
            std::thread::spawn(move || (oc, synth_sky130(&d, &format!("dense_oc_{oc}"))))
        })
        .collect();

    let mut results: Vec<(usize, Result<SynthMetrics, _>)> =
        handles.into_iter().map(|h| h.join().expect("thread panic")).collect();
    results.sort_by_key(|(oc, _)| *oc);

    let elapsed = t0.elapsed();

    // ── Sum + report.
    let mut total_cells = 0u64;
    let mut total_area = 0.0_f64;
    let mut max_delay = 0.0_f64;
    let mut all_ok = true;
    eprintln!();
    eprintln!("┌──────────────┬──────────┬────────────────┬──────────────┐");
    eprintln!("│  channel     │  cells   │   area (µm²)   │ ABC delay ps │");
    eprintln!("├──────────────┼──────────┼────────────────┼──────────────┤");
    for (oc, r) in &results {
        match r {
            Ok(m) => {
                total_cells += m.cells;
                total_area += m.area_um2;
                if let Some(d) = m.abc_delay_ps {
                    if d > max_delay { max_delay = d; }
                }
                eprintln!(
                    "│  dense_oc_{oc}  │  {:>6}  │   {:>10.1}   │   {:>8}   │",
                    m.cells,
                    m.area_um2,
                    m.abc_delay_ps.map(|d| format!("{:.0}", d)).unwrap_or_else(|| "n/a".into()),
                );
            }
            Err(e) => {
                all_ok = false;
                eprintln!("│  dense_oc_{oc}  │   FAIL   │      n/a       │     n/a      │  ({e})");
            }
        }
    }
    eprintln!("├──────────────┼──────────┼────────────────┼──────────────┤");
    eprintln!(
        "│  Σ (BakedConst total)    │  {:>10}  │   {:>8} (max critical) │",
        total_cells, format!("{:.0}", max_delay)
    );
    eprintln!("│  area sum    │  {:>6}  │   {:>10.1}   │              │", total_cells, total_area);
    eprintln!("└──────────────┴──────────┴────────────────┴──────────────┘");
    eprintln!(
        "Wall: {:.1}s  ({}-way parallel ABC, single-threaded inside each)",
        elapsed.as_secs_f64(),
        results.len()
    );

    if all_ok {
        let _ = std::fs::remove_dir_all(&base);
    } else {
        eprintln!("(keeping {} for inspection)", base.display());
    }
    assert!(all_ok, "at least one channel synth failed; see log");
    assert!(total_area > 0.0);
}
