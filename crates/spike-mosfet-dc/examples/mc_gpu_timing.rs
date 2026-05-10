//! Timing comparison: scalar f64 MC vs auto (GPU+CPU concurrent) MC.
//!
//! Run release-mode for representative numbers:
//!   cargo run --example mc_gpu_timing -p spike-mosfet-dc --release
//!
//! Knobs:
//!   RLX_EDA_MC_GPU_SPLIT=0.0   — force CPU rayon path (no GPU)
//!   RLX_EDA_MC_GPU_SPLIT=1.0   — force GPU-only path
//!   RLX_EDA_MC_DISABLE_GPU=1   — same as split=0.0, but exercises the
//!                                "MLX unavailable" fallback branch
//!
//! Reports wall-clock per N for each path and the speedup. We discard
//! the first run at each N to amortize the MLX compile cost.

use spike_mosfet_dc::mc::run_mc_sweep;
use spike_mosfet_dc::mc_gpu::run_mc_sweep_auto;
use std::time::Instant;

const IREF: f64    = 5e-6;
const VBIAS: f64   = 0.9;
const VTH: f64     = 0.5;
const KP_UNIT: f64 = 100e-6 * 5.0;
const LAM: f64     = 0.02;
const W_UM: f64    = 2.0;
const L_UM: f64    = 2.0;
const AVT: f64     = 5e-3;

fn time_ms<F: FnMut() -> R, R>(mut f: F) -> (f64, R) {
    let t0 = Instant::now();
    let r = f();
    (t0.elapsed().as_secs_f64() * 1000.0, r)
}

fn run_at_n(n: usize) {
    // Warmup: pays compile + kernel-launch initialization cost so the
    // measured runs are steady-state. Discarded.
    let _ = run_mc_sweep_auto(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, n.min(256), 0);
    let _ = run_mc_sweep(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, n.min(256), 0);

    // Scalar: original per-draw rlx-CPU graph eval (the canonical baseline).
    let (t_scalar, r_scalar) = time_ms(|| {
        run_mc_sweep(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, n, 1)
    });

    // Auto: macOS GPU + CPU rayon split (or pure rayon CPU on fallback).
    let (t_auto, r_auto) = time_ms(|| {
        run_mc_sweep_auto(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, n, 1)
    });

    let speedup = t_scalar / t_auto.max(1e-9);
    let mean_drift = ((r_auto.mean - r_scalar.mean) / r_scalar.mean).abs();
    println!(
        "N={n:>7}  scalar={t_scalar:>9.2} ms   auto={t_auto:>9.2} ms   speedup={speedup:>6.2}×   mean drift={mean_drift:.2e}",
    );
}

fn main() {
    println!("Monte Carlo timing — 1:4 NMOS mirror, Pelgrom Vth mismatch");
    println!("Operating point: iref={IREF} A, vbias={VBIAS} V, W={W_UM} µm, L={L_UM} µm");
    println!(
        "GPU split: {} (set RLX_EDA_MC_GPU_SPLIT to override; RLX_EDA_MC_DISABLE_GPU=1 to force CPU)",
        std::env::var("RLX_EDA_MC_GPU_SPLIT").unwrap_or_else(|_| "1.00 (default)".into()),
    );
    println!();

    for &n in &[100usize, 1_000, 10_000, 100_000] {
        run_at_n(n);
    }
}
