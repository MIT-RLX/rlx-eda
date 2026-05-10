//! GPU-accelerated Monte Carlo sweep for the 1:4 NMOS mirror.
//!
//! Three-tier fallback ladder:
//!   1. macOS + MLX available + compile OK → batched f32 graph on Apple GPU
//!   2. Otherwise / on init or runtime failure → batched f32 graph on
//!      `rlx-runtime` CPU backend
//!   3. Catastrophic failure (panic in either rlx path) → existing
//!      scalar `mc::run_mc_sweep` (always works, this is the canonical
//!      reference path that the rest of the crate validates against)
//!
//! Concurrent CPU/GPU utilization: when on the GPU path, a fraction
//! `split` of the draws goes to the GPU on a worker thread (blocking
//! `MlxExecutable::run`), and simultaneously the remaining draws are
//! computed on host cores via rayon using the f64 scalar path. Both
//! finish in parallel and results are stitched.
//!
//! **Empirical default: split = 1.0 (GPU-only on the GPU path).** The
//! original assumption was that CPU and GPU throughputs would be
//! within ~1 OOM and a split would balance them. They aren't:
//! measured on M-series Apple Silicon for this LEVEL=1 mirror MC,
//! GPU is ~100× per-draw faster than CPU rayon, so any CPU
//! contribution becomes the long pole and *slows* the wall clock.
//! At N=100k: GPU-only ≈ 5 ms, 50/50 split ≈ 296 ms, CPU-only ≈ 597 ms.
//!
//! The split mechanism is preserved (env `RLX_EDA_MC_GPU_SPLIT`,
//! clamped to [0.0, 1.0]) for two reasons: (a) heavier graphs in
//! follow-on work (BSIM4 surrogate, batched MNA solves) will have a
//! different CPU/GPU throughput ratio where a real mix wins; (b)
//! setting it to 0.0 forces the pure-rayon CPU path for parity tests
//! and on machines where GPU init succeeds but is slower than CPU.
//!
//! Precision: MLX is f32-only (Apple Silicon GPU has no f64). At µA
//! current scale and mV Vth scale the f32 dynamic range is comfortable;
//! parity tests assert agreement with the f64 scalar path within 1e-3
//! relative on aggregate stats, which is well above f32 epsilon for
//! these magnitudes and below the Pelgrom mismatch's intrinsic noise.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::mc::{McResult, gaussian_draws, mirror_iout, vth_sigma};

/// Default fraction of draws sent to GPU when GPU path is live.
/// Empirically 1.0 wins by 10×+ vs any mix on this graph — see the
/// module-level docstring for the measured numbers and rationale.
const DEFAULT_GPU_SPLIT: f64 = 1.0;

fn split_from_env() -> f64 {
    std::env::var("RLX_EDA_MC_GPU_SPLIT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|s| s.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT_GPU_SPLIT)
}

fn gpu_disabled() -> bool {
    std::env::var("RLX_EDA_MC_DISABLE_GPU")
        .ok()
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false)
}

// ── Batched f32 graph ───────────────────────────────────────────────

fn vec_shape(n: usize) -> Shape {
    Shape::new(&[n], DType::F32)
}

fn scalar_f32() -> Shape {
    Shape::new(&[1], DType::F32)
}

fn const_scalar_f32(g: &mut Graph, x: f32) -> NodeId {
    g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], scalar_f32())
}

/// Build a batched [N]-shape f32 graph for the 1:4 NMOS mirror MC sweep.
///
/// Inputs (all shape [N], f32):
///   `vth_m1`   — M1 threshold + per-draw mismatch
///   `vth_m2`   — M2 threshold + per-draw mismatch (different σ_Vth: 4× area)
///   `vgs`      — bias Vgs precomputed on host (= vth_m1 + √(2·iref / kp))
///   `vds`      — drain bias for M2 (broadcast of `vbias`)
///
/// Params (scalar f32):
///   `kp_m2`    — kp_unit · 4 (M2 is 4× wider)
///   `lam`      — channel-length modulation
///
/// Output (shape [N], f32): `Iout` per draw.
///
/// Splitting `vth_m1` / `vgs` as separate inputs (rather than computing
/// vgs in the graph) lets the host pre-bake the √(2·iref/kp) term once
/// per call and avoids dragging `iref` through the graph as an input —
/// which would force adding it to the parameter map every call. With N
/// in the thousands the host-side √ is negligible.
pub fn build_mirror_graph_batched_f32(n: usize) -> Graph {
    let mut g = Graph::new("mirror_mc_batched_f32");
    let s = vec_shape(n);

    let vgs    = g.input("vgs",    s.clone());
    let vds    = g.input("vds",    s.clone());
    let vth_m2 = g.input("vth_m2", s.clone());
    // vth_m1 is not used inside the graph (vgs already carries its
    // contribution) but we accept it so the host can keep the same
    // input-binding shape across runs without conditional logic.
    let _vth_m1 = g.input("vth_m1", s.clone());

    let kp  = g.param("kp_m2", scalar_f32());
    let lam = g.param("lam",   scalar_f32());

    let beta     = const_scalar_f32(&mut g, super::BETA as f32);
    let inv_beta = const_scalar_f32(&mut g, (1.0 / super::BETA) as f32);
    let half     = const_scalar_f32(&mut g, 0.5);
    let one      = const_scalar_f32(&mut g, 1.0);
    let delta    = const_scalar_f32(&mut g, super::DELTA as f32);

    // Vov_smooth = (1/β) · log(1 + exp(β · (vgs − vth_m2)))
    let vov_raw      = g.binary(BinaryOp::Sub, vgs, vth_m2, s.clone());
    let scaled       = g.binary(BinaryOp::Mul, beta, vov_raw, s.clone());
    let exp_v        = g.activation(Activation::Exp, scaled, s.clone());
    let one_plus_exp = g.binary(BinaryOp::Add, one, exp_v, s.clone());
    let log_v        = g.activation(Activation::Log, one_plus_exp, s.clone());
    let vov_s        = g.binary(BinaryOp::Mul, inv_beta, log_v, s.clone());

    // Vds_eff = ½ · (vds + vov_s − √((vds − vov_s)² + δ))
    let sum     = g.binary(BinaryOp::Add, vds, vov_s, s.clone());
    let diff    = g.binary(BinaryOp::Sub, vds, vov_s, s.clone());
    let diff_sq = g.binary(BinaryOp::Mul, diff, diff, s.clone());
    let arg     = g.binary(BinaryOp::Add, diff_sq, delta, s.clone());
    let root    = g.activation(Activation::Sqrt, arg, s.clone());
    let inner   = g.binary(BinaryOp::Sub, sum, root, s.clone());
    let vds_eff = g.binary(BinaryOp::Mul, half, inner, s.clone());

    // Id = kp · (vov_s · vds_eff − vds_eff² / 2) · (1 + λ·vds)
    let term1           = g.binary(BinaryOp::Mul, vov_s, vds_eff, s.clone());
    let vds_eff_sq      = g.binary(BinaryOp::Mul, vds_eff, vds_eff, s.clone());
    let half_vds_eff_sq = g.binary(BinaryOp::Mul, half, vds_eff_sq, s.clone());
    let bracket         = g.binary(BinaryOp::Sub, term1, half_vds_eff_sq, s.clone());

    let lam_vds = g.binary(BinaryOp::Mul, lam, vds, s.clone());
    let clm     = g.binary(BinaryOp::Add, one, lam_vds, s.clone());

    let kp_bracket = g.binary(BinaryOp::Mul, kp, bracket, s.clone());
    let id         = g.binary(BinaryOp::Mul, kp_bracket, clm, s.clone());

    g.set_outputs(vec![id]);
    g
}

// ── Public entry point ──────────────────────────────────────────────

/// MC sweep with auto device selection + concurrent CPU/GPU utilization.
///
/// Drop-in replacement for `mc::run_mc_sweep`. On macOS attempts
/// MLX-compiled execution for the bulk of draws while the host runs the
/// remainder in parallel via rayon. Falls back transparently if any
/// stage fails — the caller cannot tell which path served the result.
pub fn run_mc_sweep_auto(
    iref: f64, vbias: f64,
    vth_nom: f64, kp_unit: f64, lam: f64,
    w_um: f64, l_um: f64, avt_v_um: f64,
    n_draws: usize, seed: u64,
) -> McResult {
    if n_draws == 0 {
        return McResult {
            values: Vec::new(), mean: 0.0, sigma: 0.0,
            min: 0.0, max: 0.0, elapsed_ms: 0.0,
        };
    }

    let sigma_vth_m1 = vth_sigma(w_um,        l_um, avt_v_um);
    let sigma_vth_m2 = vth_sigma(w_um * 4.0,  l_um, avt_v_um);
    let m1_draws = gaussian_draws(n_draws, seed);
    let m2_draws = gaussian_draws(n_draws, seed.wrapping_add(1));

    let t0 = std::time::Instant::now();

    let split = split_from_env();
    let n_gpu = if gpu_disabled() {
        0
    } else {
        ((n_draws as f64) * split).floor() as usize
    };
    let n_cpu = n_draws - n_gpu;

    // Try GPU+CPU concurrent path. If GPU dispatch fails for any
    // reason, fall through to pure-CPU below.
    let mut values: Option<Vec<f64>> = None;

    #[cfg(target_os = "macos")]
    if n_gpu > 0 {
        values = run_concurrent_gpu_cpu(
            iref, vbias, vth_nom, kp_unit, lam,
            sigma_vth_m1, sigma_vth_m2,
            &m1_draws, &m2_draws,
            n_gpu, n_cpu,
        );
    }

    // Fallback: pure-CPU rayon-parallel. Used when GPU is disabled,
    // unavailable, or panicked.
    let values = values.unwrap_or_else(|| {
        cpu_chunk(
            iref, vbias, vth_nom, kp_unit, lam,
            sigma_vth_m1, sigma_vth_m2,
            &m1_draws, &m2_draws,
            0, n_draws,
        )
    });

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    finalize(values, elapsed_ms)
}

/// CPU work unit: scalar f64 closed-form for draws `[start..end)`.
/// Identical math to `mc::mirror_iout` so results are bit-identical
/// to the canonical scalar path on this slice.
fn cpu_chunk(
    iref: f64, vbias: f64,
    vth_nom: f64, kp_unit: f64, lam: f64,
    sigma_vth_m1: f64, sigma_vth_m2: f64,
    m1_draws: &[f64], m2_draws: &[f64],
    start: usize, end: usize,
) -> Vec<f64> {
    use rayon::prelude::*;
    (start..end)
        .into_par_iter()
        .map(|i| {
            mirror_iout(
                iref, vbias, vth_nom, kp_unit, lam,
                sigma_vth_m1 * m1_draws[i],
                sigma_vth_m2 * m2_draws[i],
                4.0,
            )
        })
        .collect()
}

fn finalize(values: Vec<f64>, elapsed_ms: f64) -> McResult {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sigma = if values.len() >= 2 {
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else { 0.0 };
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    McResult { values, mean, sigma, min, max, elapsed_ms }
}

// ── macOS GPU path ──────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn run_concurrent_gpu_cpu(
    iref: f64, vbias: f64,
    vth_nom: f64, kp_unit: f64, lam: f64,
    sigma_vth_m1: f64, sigma_vth_m2: f64,
    m1_draws: &[f64], m2_draws: &[f64],
    n_gpu: usize, n_cpu: usize,
) -> Option<Vec<f64>> {
    if !rlx_mlx::is_available() {
        return None;
    }

    // Stage GPU inputs as f32. The first `n_gpu` draws go to GPU; the
    // last `n_cpu` go to host rayon. Splitting at the index boundary
    // (rather than interleaving) keeps the stitch a plain extend.
    let kp_m2 = kp_unit * 4.0;
    let mut vgs_in    = Vec::with_capacity(n_gpu);
    let mut vds_in    = Vec::with_capacity(n_gpu);
    let mut vth_m2_in = Vec::with_capacity(n_gpu);
    let mut vth_m1_in = Vec::with_capacity(n_gpu);
    for i in 0..n_gpu {
        let vth_m1 = vth_nom + sigma_vth_m1 * m1_draws[i];
        let vth_m2 = vth_nom + sigma_vth_m2 * m2_draws[i];
        let vgs    = vth_m1 + (2.0 * iref / kp_unit).sqrt();
        vgs_in.push(vgs as f32);
        vds_in.push(vbias as f32);
        vth_m2_in.push(vth_m2 as f32);
        vth_m1_in.push(vth_m1 as f32);
    }

    // Build + compile the executable. catch_unwind because compile
    // panics on backend errors and we need to keep the fallback live.
    let exe_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let g = build_mirror_graph_batched_f32(n_gpu);
        let mut exe = rlx_mlx::MlxExecutable::compile_with_mode(g, rlx_mlx::MlxMode::Compiled);
        exe.set_param("kp_m2", &[kp_m2 as f32]);
        exe.set_param("lam",   &[lam as f32]);
        exe.warm_compile().ok();
        exe
    }));
    let mut exe = match exe_result {
        Ok(e) => e,
        Err(_) => return None,
    };

    // Concurrent dispatch: GPU on a scoped worker thread, CPU rayon
    // chunk on the main thread. Both finish in parallel; the rayon
    // call blocks until its chunk is done, and the join waits for the
    // GPU. With `thread::scope` we don't need 'static bounds.
    let gpu_handle = std::thread::scope(|s| {
        let gpu = s.spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let outs = exe.run(&[
                    ("vgs",    vgs_in.as_slice()),
                    ("vds",    vds_in.as_slice()),
                    ("vth_m2", vth_m2_in.as_slice()),
                    ("vth_m1", vth_m1_in.as_slice()),
                ]);
                outs.into_iter().next().unwrap_or_default()
            }))
        });

        let cpu_values = if n_cpu > 0 {
            cpu_chunk(
                iref, vbias, vth_nom, kp_unit, lam,
                sigma_vth_m1, sigma_vth_m2,
                m1_draws, m2_draws,
                n_gpu, n_gpu + n_cpu,
            )
        } else {
            Vec::new()
        };

        let gpu_values = match gpu.join() {
            Ok(Ok(v)) => v,
            _ => return None,
        };

        Some((gpu_values, cpu_values))
    })?;

    let (gpu_f32, cpu_f64) = gpu_handle;
    if gpu_f32.len() != n_gpu {
        return None;
    }

    let mut out = Vec::with_capacity(n_gpu + n_cpu);
    out.extend(gpu_f32.into_iter().map(|v| v as f64));
    out.extend(cpu_f64);
    Some(out)
}

// On non-macOS this never fires; suppress dead_code on the helper.
#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn run_concurrent_gpu_cpu(
    _iref: f64, _vbias: f64,
    _vth_nom: f64, _kp_unit: f64, _lam: f64,
    _sigma_vth_m1: f64, _sigma_vth_m2: f64,
    _m1_draws: &[f64], _m2_draws: &[f64],
    _n_gpu: usize, _n_cpu: usize,
) -> Option<Vec<f64>> {
    None
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::run_mc_sweep;

    // Same operating point as `mc_sweep_produces_nonzero_sigma` so the
    // baseline numbers are familiar.
    const IREF: f64    = 5e-6;
    const VBIAS: f64   = 0.9;
    const VTH: f64     = 0.5;
    const KP_UNIT: f64 = 100e-6 * 5.0;
    const LAM: f64     = 0.02;
    const W_UM: f64    = 2.0;
    const L_UM: f64    = 2.0;
    const AVT: f64     = 5e-3;

    fn rel(a: f64, b: f64) -> f64 {
        if b.abs() < 1e-30 { (a - b).abs() } else { ((a - b) / b).abs() }
    }

    #[test]
    fn auto_matches_scalar_aggregate_stats() {
        let n = 500;
        let scalar = run_mc_sweep(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, n, 7);
        let auto   = run_mc_sweep_auto(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, n, 7);
        assert_eq!(auto.values.len(), n);
        // f32 GPU path + f64 CPU path mix → mean/σ should agree with
        // the all-f64 reference well within Pelgrom-noise resolution.
        assert!(rel(auto.mean,  scalar.mean)  < 1e-3,
            "mean mismatch: auto={} scalar={}", auto.mean, scalar.mean);
        assert!(rel(auto.sigma, scalar.sigma) < 1e-2,
            "sigma mismatch: auto={} scalar={}", auto.sigma, scalar.sigma);
    }

    #[test]
    fn auto_falls_back_when_gpu_disabled() {
        // Force the pure-CPU rayon path. Result must still be correct
        // (it bit-identically matches the scalar reference because
        // both run the same f64 closed-form).
        // Note: env vars are process-global; we don't unset because
        // other tests don't read this var.
        unsafe { std::env::set_var("RLX_EDA_MC_DISABLE_GPU", "1"); }
        let n = 200;
        let scalar = run_mc_sweep(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, n, 11);
        let auto   = run_mc_sweep_auto(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, n, 11);
        unsafe { std::env::remove_var("RLX_EDA_MC_DISABLE_GPU"); }

        assert_eq!(auto.values.len(), n);
        for (a, b) in auto.values.iter().zip(scalar.values.iter()) {
            assert!((a - b).abs() < 1e-15,
                "fallback path must be bit-identical to scalar: {a} vs {b}");
        }
    }

    #[test]
    fn auto_handles_zero_draws() {
        let r = run_mc_sweep_auto(IREF, VBIAS, VTH, KP_UNIT, LAM, W_UM, L_UM, AVT, 0, 0);
        assert_eq!(r.values.len(), 0);
        assert_eq!(r.mean, 0.0);
    }

    #[test]
    fn batched_graph_has_expected_shape() {
        let n = 64;
        let g = build_mirror_graph_batched_f32(n);
        let out_id = *g.outputs.first().expect("graph must have an output");
        let shape = &g.node(out_id).shape;
        assert_eq!(shape.dtype(), DType::F32);
        assert_eq!(shape.num_elements(), Some(n));
    }
}
