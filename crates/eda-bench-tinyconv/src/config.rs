//! `BenchConfig` — single TOML file describing one bench run.
//!
//! Aggregates every configurable knob in the bench (loss weights,
//! inner Adam, outer DADO, PnR, noise model, array geometry,
//! manifest sources). Defaults match the in-code `Default` impls
//! exactly, so an empty `bench.toml` reproduces the demo binary's
//! current behavior.
//!
//! ## Why TOML
//!
//! Cargo + workspace-wide convention (`Cargo.toml`, `clippy.toml`,
//! `rustfmt.toml`); `serde_toml` already in workspace deps for the
//! `ModelCard` round-trip. YAML is a sibling option; pick this file
//! to switch.
//!
//! ## Adding a new section
//!
//! 1. Add a `pub struct XConfig { ... }` with `serde(default)` and
//!    a `Default` impl.
//! 2. Add `#[serde(default)] pub x: XConfig` to `BenchConfig`.
//! 3. Document the field in the example file at
//!    `crates/eda-bench-tinyconv/bench.example.toml`.
//!
//! That's it — round-trip + per-section override comes for free
//! from serde.

use eda_config::Configurable;
use serde::{Deserialize, Serialize};

use crate::bundle::BundleConfig;
use crate::inference::InferenceConfig;
use crate::optimization::{
    inner::InnerConfig,
    outer::OuterConfig,
    LossWeights,
};
use crate::pnr::{PnrAdamConfig, PnrMode};
use spike_tinyconv_tile::NoiseModel;

/// One bench run's worth of knobs. Loading, writing, env-var
/// override, and soft-fallback come for free via [`Configurable`]
/// — see `eda-config`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchConfig {
    pub run: RunConfig,
    pub loss_weights: LossWeights,
    pub inner: InnerConfig,
    pub outer: OuterConfig,
    pub pnr: PnrConfigToml,
    pub noise: NoiseModel,
    pub inference: InferenceConfig,
    pub bundle: BundleConfig,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            run: RunConfig::default(),
            loss_weights: LossWeights::default(),
            inner: InnerConfig::default(),
            outer: OuterConfig::default(),
            pnr: PnrConfigToml::default(),
            noise: NoiseModel::default(),
            inference: InferenceConfig::default(),
            bundle: BundleConfig::default(),
        }
    }
}

impl Configurable for BenchConfig {
    const FILENAME: &'static str = "bench.toml";
}

impl BenchConfig {
    /// Project to the runtime `PnrMode` consumed by `crate::pnr::run`.
    pub fn pnr_mode(&self) -> PnrMode {
        if self.pnr.enabled {
            PnrMode::AdamHpwl(self.pnr.adam)
        } else {
            PnrMode::Disabled
        }
    }
}

// ── Per-section structs ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RunConfig {
    /// Optimizer seed — recorded in the manifest, threaded into
    /// every Adam loop.
    pub seed: u64,
    /// Where the bench writes its markdown report. Demo binary
    /// resolves this relative to the workspace root.
    pub output_path: String,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            output_path: "target/bench/demo/report.md".into(),
        }
    }
}

/// Adjacent-tagged TOML shape for `PnrMode`. Cleaner than serde
/// `enum`-tagging because the `[pnr.adam]` sub-table works whether
/// PnR is enabled or not — TOML doesn't support conditional
/// sections elegantly.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct PnrConfigToml {
    /// Toggle: `true` runs `PnrMode::AdamHpwl(self.adam)`,
    /// `false` is `PnrMode::Disabled` (no-op).
    pub enabled: bool,
    /// Schedule used when `enabled = true`. Always present so
    /// callers can flip `enabled` without re-supplying every
    /// hyperparameter.
    pub adam: PnrAdamConfig,
}

impl Default for PnrConfigToml {
    fn default() -> Self {
        Self {
            enabled: false,
            adam: PnrAdamConfig::default(),
        }
    }
}
