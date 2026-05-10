//! `eda-config` — shared TOML config loading machinery with
//! template inheritance.
//!
//! Each domain (bench, calibration, training, deployment, …) defines
//! its own config struct and impls [`Configurable`] by declaring a
//! conventional filename. Loading, writing, soft-fallback, env-var
//! override, and **template inheritance** come for free.
//!
//! ## Convention
//!
//! Configs live at `<workspace>/configs/<filename>` by default.
//! Override per-call via [`load_strict`] / [`load_or_default`] or at
//! runtime via the `EDA_CONFIG_<NAME>` env var
//! (e.g. `EDA_CONFIG_BENCH=bench-tt.toml`).
//!
//! ## Inheritance
//!
//! A TOML file may declare `extends = "<path>"` at the top level.
//! The base file is loaded first, then the child file's keys are
//! deep-merged on top — child keys override parent keys at every
//! nesting level. Multi-level chains and same-directory siblings
//! both work; cycles are detected and surface as
//! [`ConfigError::CycleDetected`].
//!
//! ```toml
//! # configs/bench.toml
//! extends = "templates/base.toml"
//!
//! [run]
//! seed = 99      # overrides templates/base.toml::run.seed
//!
//! [pnr]
//! enabled = true # overrides templates/base.toml::pnr.enabled;
//!                # other [pnr.*] keys inherited from base
//! ```
//!
//! Use cases: shared "tt corner" / "ss corner" templates, per-PDK
//! defaults, calibration-result overlays, CI/dev variants.
//!
//! ## Defining a new config
//!
//! ```ignore
//! use eda_config::Configurable;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Debug, Clone, Default, Serialize, Deserialize)]
//! #[serde(default)]
//! pub struct CalibrationConfig {
//!     pub corner: String,
//!     pub mc_runs: usize,
//! }
//!
//! impl Configurable for CalibrationConfig {
//!     const FILENAME: &'static str = "calibration.toml";
//! }
//!
//! let cfg = CalibrationConfig::load_or_default();
//! cfg.write_default_to_disk()?;
//! ```

use serde::{de::DeserializeOwned, Serialize};
use std::path::{Path, PathBuf};

/// Conventional filename for the config. The blanket impls under
/// this trait derive load/write paths from it.
pub trait Configurable: Default + Serialize + DeserializeOwned {
    /// Filename relative to `<workspace>/configs/`. Conventionally
    /// snake-case, `.toml`-suffixed.
    const FILENAME: &'static str;

    /// SCREAMING_SNAKE form of `FILENAME` minus the extension.
    /// Used to derive the env var that overrides the config's
    /// path per-run (`EDA_CONFIG_<NAME>`).
    fn env_var_name() -> String {
        let stem = Self::FILENAME.split('.').next().unwrap_or("");
        format!("EDA_CONFIG_{}", stem.to_uppercase())
    }

    /// Resolve the config's path: env var override → conventional
    /// `<workspace>/configs/<filename>` → none.
    fn resolve_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var(Self::env_var_name()) {
            return Some(PathBuf::from(p));
        }
        let root = std::env::var("CARGO_WORKSPACE_DIR")
            .or_else(|_| std::env::var("CARGO_MANIFEST_DIR"))
            .map(PathBuf::from)
            .ok()?;
        Some(root.join("configs").join(Self::FILENAME))
    }

    /// Load from the resolved path with template inheritance
    /// resolved. Falls back to `Default` (with stderr warning)
    /// when the path is missing or unparseable — non-fatal so
    /// demo / CI runs always succeed.
    fn load_or_default() -> Self {
        match Self::resolve_path() {
            Some(p) => load_or_default(&p),
            None => Self::default(),
        }
    }

    /// Strict load — propagates errors including inheritance
    /// resolution failures. Useful for CI gates.
    fn load_strict() -> Result<Self, ConfigError> {
        let p = Self::resolve_path().ok_or(ConfigError::NoPathResolvable)?;
        load_strict(&p)
    }

    /// Write the config to its conventional path, creating parent
    /// directories if missing.
    fn write_default_to_disk(&self) -> Result<(), ConfigError> {
        let p = Self::resolve_path().ok_or(ConfigError::NoPathResolvable)?;
        write_to(self, &p)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml parse: {0}")]
    TomlDe(#[from] toml::de::Error),
    #[error("toml serialize: {0}")]
    TomlSer(#[from] toml::ser::Error),
    #[error("could not resolve config path (no env var, no workspace root)")]
    NoPathResolvable,
    #[error("inheritance cycle detected: {0}")]
    CycleDetected(String),
    #[error("inheritance depth exceeded {0} levels (loop suspected)")]
    DepthExceeded(usize),
}

// ── Bare-function API for callers who don't want to impl Configurable ──

/// Parse a TOML config from text. Pure — no I/O, no inheritance
/// (the `extends` key is ignored at this level).
pub fn from_toml<T: DeserializeOwned>(text: &str) -> Result<T, ConfigError> {
    Ok(toml::from_str(text)?)
}

/// Render a config to TOML.
pub fn to_toml<T: Serialize>(cfg: &T) -> Result<String, ConfigError> {
    Ok(toml::to_string_pretty(cfg)?)
}

/// Load from a path with `extends = "..."` inheritance resolved.
/// Strict — propagates parse / I/O / cycle errors.
pub fn load_strict<T: DeserializeOwned>(path: &Path) -> Result<T, ConfigError> {
    let merged = load_merged_value(path)?;
    Ok(merged.try_into()?)
}

/// Load from a path with `extends = "..."` inheritance resolved.
/// Soft — falls back to `Default` (with stderr warning on parse
/// error) when missing or unparseable.
pub fn load_or_default<T: DeserializeOwned + Default>(path: &Path) -> T {
    match load_strict::<T>(path) {
        Ok(cfg) => cfg,
        Err(ConfigError::Io(_)) => T::default(),
        Err(e) => {
            eprintln!(
                "warning: failed to load config at {}: {e}; using defaults",
                path.display()
            );
            T::default()
        }
    }
}

/// Write a config to `path`, creating parent directories if missing.
pub fn write_to<T: Serialize>(cfg: &T, path: &Path) -> Result<(), ConfigError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, to_toml(cfg)?)?;
    Ok(())
}

// ── Template inheritance ────────────────────────────────────────────

/// Maximum chain depth. Anything deeper is treated as a misconfigured
/// loop even when paths don't repeat exactly (e.g. symlinks or
/// canonicalization edge cases).
const MAX_INHERIT_DEPTH: usize = 16;

/// Resolve `extends = "..."` chains and deep-merge into a single
/// `toml::Value` ready for `try_into::<T>()`. Pure-ish (only
/// reads files); idempotent given the same on-disk content.
fn load_merged_value(path: &Path) -> Result<toml::Value, ConfigError> {
    let canonical = canonical_or_path(path);
    let mut visited: Vec<PathBuf> = Vec::new();
    load_merged_value_inner(&canonical, &mut visited, 0)
}

fn load_merged_value_inner(
    path: &Path,
    visited: &mut Vec<PathBuf>,
    depth: usize,
) -> Result<toml::Value, ConfigError> {
    if depth > MAX_INHERIT_DEPTH {
        return Err(ConfigError::DepthExceeded(MAX_INHERIT_DEPTH));
    }
    if visited.iter().any(|p| p == path) {
        let chain: Vec<String> = visited
            .iter()
            .chain(std::iter::once(&path.to_path_buf()))
            .map(|p| p.display().to_string())
            .collect();
        return Err(ConfigError::CycleDetected(chain.join(" → ")));
    }
    visited.push(path.to_path_buf());

    let text = std::fs::read_to_string(path)?;
    let mut value: toml::Value = toml::from_str(&text)?;

    // Look for `extends = "..."` at the top level. If present,
    // load the parent first and overlay this file on top.
    let extends = value
        .as_table_mut()
        .and_then(|t| t.remove("extends"))
        .and_then(|v| v.as_str().map(str::to_string));

    if let Some(rel) = extends {
        let parent_path = path
            .parent()
            .map(|p| p.join(&rel))
            .unwrap_or_else(|| PathBuf::from(&rel));
        let parent_canonical = canonical_or_path(&parent_path);
        let parent_value = load_merged_value_inner(&parent_canonical, visited, depth + 1)?;
        Ok(deep_merge(parent_value, value))
    } else {
        Ok(value)
    }
}

/// Best-effort canonicalization. Falls back to the input path when
/// canonicalize fails (file doesn't exist yet — `load_strict` will
/// surface the I/O error from `read_to_string`).
fn canonical_or_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Deep-merge two `toml::Value`s. Child keys override parent keys
/// at every nesting level. Tables merge recursively; arrays /
/// primitives in the child REPLACE the parent's value (no
/// concatenation — opt in via a future `merge_strategy` config if
/// needed).
fn deep_merge(parent: toml::Value, child: toml::Value) -> toml::Value {
    match (parent, child) {
        (toml::Value::Table(mut p), toml::Value::Table(c)) => {
            for (k, child_v) in c {
                let merged = match p.remove(&k) {
                    Some(parent_v) => deep_merge(parent_v, child_v),
                    None => child_v,
                };
                p.insert(k, merged);
            }
            toml::Value::Table(p)
        }
        // Child wins for any non-table / type-mismatch case.
        (_, child) => child,
    }
}
