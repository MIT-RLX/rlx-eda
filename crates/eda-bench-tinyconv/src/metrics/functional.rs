//! Functional metric arm — does the chip classify MNIST correctly?
//!
//! The load-bearing metric. Five validation levels (L1–L5) defined in
//! PLAN.md "Validation". Yield gate (cross-cutting #3) lives here.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Level {
    /// Q31 reference (Rust). Defines "correct".
    L1Reference,
    /// RTL functional sim against golden subset.
    L2Rtl,
    /// Gate-level sim with SDF back-annotation.
    L3GateSdf,
    /// Mixed-signal post-layout sim with parasitics.
    L4PostLayout,
    /// PVT × MC sweep on full 10k test set.
    L5PvtMc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Functional {
    pub level: Level,
    pub top1_acc: f64,
    pub per_class_acc: [f64; 10],
    /// First layer index where this backend's per-image activation
    /// diverged from the reference. `None` if no divergence on the
    /// evaluated subset.
    pub divergence_first_layer: Option<usize>,
    pub n_images: usize,
}

/// Release condition: `P(top-1 ≥ 97% | PVT × MC) ≥ 99%`.
/// Mean is informational; this yield number is the gate.
pub struct YieldGate {
    pub min_top1: f64,
    pub min_pvt_mc_pass_rate: f64,
}

impl YieldGate {
    pub const RELEASE: Self = Self {
        min_top1: 0.97,
        min_pvt_mc_pass_rate: 0.99,
    };

    /// Yield gate evaluation. `true` iff at least
    /// `min_pvt_mc_pass_rate` of the runs hit `top1_acc ≥ min_top1`.
    /// Empty input is `false` — no claim possible from zero runs.
    pub fn evaluate(&self, l5_runs: &[Functional]) -> bool {
        if l5_runs.is_empty() {
            return false;
        }
        let passed = l5_runs.iter().filter(|r| r.top1_acc >= self.min_top1).count();
        let rate = passed as f64 / l5_runs.len() as f64;
        rate >= self.min_pvt_mc_pass_rate
    }

    /// Pass rate as a fraction in `[0.0, 1.0]`. Reported alongside
    /// the boolean gate so the report can show how close a failing
    /// run got.
    pub fn pass_rate(&self, l5_runs: &[Functional]) -> f64 {
        if l5_runs.is_empty() {
            return 0.0;
        }
        l5_runs.iter().filter(|r| r.top1_acc >= self.min_top1).count() as f64
            / l5_runs.len() as f64
    }
}
