//! End-to-end inference performance benchmark.
//!
//! Times `rlx_fpga::reference::run` (bit-exact pure Rust) over a
//! configurable batch of test images and produces latency + throughput
//! statistics. Currently calls the reference path; the same wrapper
//! plugs into `FpgaBackend` (board-in-loop) and `InhouseBackend`
//! (post-layout sim) once those run real images, by swapping the
//! `infer_one` callback.
//!
//! Why measure inference here: the bench's existing `Functional`
//! metric only reports top-1 accuracy. Top-1 is necessary but not
//! sufficient — a chip that classifies correctly at 1 inference/sec
//! is useless for streaming workloads. Adam co-design needs a
//! latency number alongside the energy number to balance the
//! `α · E + β · delay` terms.

use std::time::{Duration, Instant};

use rlx_fpga::model::Model;
use rlx_fpga::reference;
use serde::{Deserialize, Serialize};

/// Configuration for the inference benchmark. Lives in
/// `BenchConfig::inference`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// How many images to time end-to-end per repetition.
    pub n_images: usize,
    /// Warmup inferences before timing starts (excluded from stats).
    /// Smooths out cold-cache jitter on the first runs.
    pub warmup: usize,
    /// Number of full passes to average over. >1 catches GC / OS
    /// scheduling jitter.
    pub repetitions: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_images: 100,
            warmup: 10,
            repetitions: 3,
        }
    }
}

/// Latency + throughput summary.
///
/// Two distinct kinds of time live here:
///
/// - **Wall-clock**: `mean_us` etc. — how long the host (Rust /
///   Verilator / ngspice) took to *produce* the answer. Useful for
///   bench-runner ergonomics; **not** a silicon performance
///   number. L1 reference (pure Rust) populates these; L2 / L3 /
///   L4 simulators populate them too as a diagnostic.
///
/// - **Simulated silicon**: `simulated` — the latency the actual
///   silicon will exhibit when fabricated and clocked. **This is
///   the real performance number.** L1 reference can't produce
///   one (no clock); L2 reports cycles, L3 reports cycles ×
///   back-annotated period, L4 reports the ngspice transient
///   end-time. Headline metric for the `β·delay` term in the
///   co-design loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub n_images: usize,
    pub repetitions: usize,
    /// Mean wall-clock per-image latency across all repetitions.
    pub mean_us: f64,
    /// Median (p50) wall-clock per-image latency.
    pub p50_us: f64,
    /// 99th-percentile wall-clock per-image latency. Tail-latency
    /// sentinel for "did anything stutter on the host."
    pub p99_us: f64,
    /// Min observed wall-clock per-image latency.
    pub min_us: f64,
    /// Max observed wall-clock per-image latency.
    pub max_us: f64,
    /// Wall-clock throughput in inferences per second, derived from
    /// `mean_us`. Tells you how fast the bench can iterate; says
    /// nothing about silicon throughput.
    pub throughput_per_sec: f64,
    /// Silicon-side simulated latency. `None` for L1 (pure Rust
    /// reference has no clock). L2+ backends populate this — this
    /// is the *real* answer to "how fast does inference run on
    /// silicon."
    pub simulated: Option<SimulatedLatency>,
}

/// Silicon-time latency from a sim backend. Decoupled from
/// wall-clock so the bench reporter never confuses the two.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimulatedLatency {
    /// Number of clock cycles per inference (integer, RTL/L2+).
    pub cycles_per_inference: u64,
    /// Clock period in nanoseconds. L2 typically uses a target
    /// (e.g. 10 ns @ 100 MHz); L3 uses the back-annotated value
    /// from OpenSTA after timing closure; L4 derives from the
    /// transient sim's clock-edge spacing.
    pub period_ns: f64,
    /// Total simulated silicon time per inference =
    /// `cycles_per_inference · period_ns`. Reported as the
    /// headline simulated-latency number.
    pub total_ns: f64,
    /// Silicon-side throughput at this clock period
    /// (inferences/second). Decoupled from `throughput_per_sec`
    /// (which is host-side wall-clock).
    pub silicon_throughput_per_sec: f64,
}

impl SimulatedLatency {
    /// Construct from cycle count + period. Computes total + throughput.
    pub fn from_cycles(cycles: u64, period_ns: f64) -> Self {
        let total_ns = cycles as f64 * period_ns;
        let throughput = if total_ns > 0.0 {
            1.0e9 / total_ns
        } else {
            0.0
        };
        Self {
            cycles_per_inference: cycles,
            period_ns,
            total_ns,
            silicon_throughput_per_sec: throughput,
        }
    }
}

impl InferenceMetrics {
    /// Empty-input sentinel (zero throughput, zero stats). Used when
    /// the bench is configured to skip inference benchmarking.
    pub fn empty() -> Self {
        Self {
            n_images: 0,
            repetitions: 0,
            mean_us: 0.0,
            p50_us: 0.0,
            p99_us: 0.0,
            min_us: 0.0,
            max_us: 0.0,
            throughput_per_sec: 0.0,
            simulated: None,
        }
    }

    /// Builder helper: attach a simulated-silicon latency. Used by
    /// L2/L3/L4 backends after the Rust-side timing wrapper has
    /// finished collecting wall-clock samples.
    pub fn with_simulated(mut self, sim: SimulatedLatency) -> Self {
        self.simulated = Some(sim);
        self
    }
}

/// Run the inference benchmark. `test_set` supplies `(image, label)`
/// pairs; the bench cycles through them as needed if it has fewer
/// than `cfg.n_images`.
///
/// Returns metrics across `cfg.repetitions` × `cfg.n_images` total
/// inferences (after `cfg.warmup` warmup runs).
pub fn run_inference_bench(
    model: &Model,
    test_set: &[(Vec<i8>, u8)],
    cfg: &InferenceConfig,
) -> Result<InferenceMetrics, InferenceError> {
    if test_set.is_empty() {
        return Err(InferenceError::EmptyTestSet);
    }
    if cfg.n_images == 0 || cfg.repetitions == 0 {
        return Ok(InferenceMetrics::empty());
    }

    // Warmup — discard timings, just heat up caches / branch
    // predictors / any lazy-init in the reference path.
    for i in 0..cfg.warmup {
        let (img, _label) = &test_set[i % test_set.len()];
        let _ = reference::run(model, img);
    }

    // Time each inference individually so we can compute
    // percentiles across the full sample (rather than mean across
    // repetitions only).
    let total_samples = cfg.n_images * cfg.repetitions;
    let mut samples: Vec<f64> = Vec::with_capacity(total_samples);
    for _rep in 0..cfg.repetitions {
        for i in 0..cfg.n_images {
            let (img, _label) = &test_set[i % test_set.len()];
            let t0 = Instant::now();
            let _ = reference::run(model, img);
            let dur: Duration = t0.elapsed();
            samples.push(dur.as_micros() as f64);
        }
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let p50 = samples[n / 2];
    let p99_idx = ((n as f64) * 0.99).floor() as usize;
    let p99 = samples[p99_idx.min(n - 1)];
    let min = samples[0];
    let max = samples[n - 1];
    let throughput = if mean > 0.0 { 1_000_000.0 / mean } else { 0.0 };

    Ok(InferenceMetrics {
        n_images: cfg.n_images,
        repetitions: cfg.repetitions,
        mean_us: mean,
        p50_us: p50,
        p99_us: p99,
        min_us: min,
        max_us: max,
        throughput_per_sec: throughput,
        // L1 reference is pure Rust — no clock, no simulated
        // silicon time. L2+ backends use `with_simulated` to
        // attach their cycle count.
        simulated: None,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("inference: test set is empty")]
    EmptyTestSet,
}
