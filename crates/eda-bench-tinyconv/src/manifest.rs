//! Per-run reproducibility manifest.
//!
//! Every bench run records the full toolchain + input fingerprint so
//! "1.2 mm² Tuesday → 1.4 mm² Friday" is debuggable. Without this the
//! parity bands in PLAN.md are meaningless.
//!
//! Cross-cutting requirement #1 in PLAN.md.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

/// Inputs to `Manifest::capture`. Optional sources resolve to
/// `"(unavailable)"` on the manifest if missing — the manifest still
/// builds so a bench run is never blocked by a missing tool, but the
/// gap is visible in every report.
pub struct ManifestInputs<'a> {
    /// Path to the sky130 PDK checkout (typically the same path
    /// `eda-pdks::build.rs` reads `.lyp` from). `git rev-parse HEAD`
    /// resolves the commit.
    pub sky130_repo: Option<&'a Path>,
    /// ORFS docker image, e.g. `openroad/orfs@sha256:<digest>` or a
    /// tag. `docker inspect <image> --format '{{.Id}}'` resolves the
    /// digest.
    pub orfs_image: Option<&'a str>,
    /// Weight blob path — `sha256` over file bytes.
    pub weights: Option<&'a Path>,
    /// Workspace `Cargo.lock`. `sha256` over file bytes.
    pub cargo_lock: &'a Path,
    /// Seed passed to the optimizer for this run.
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// sky130 PDK git commit hash, or `"(unavailable)"`.
    pub sky130_commit: String,
    /// ORFS docker image digest (`sha256:...`), or `"(unavailable)"`.
    pub orfs_image: String,
    /// `ngspice --version` first-line output, or `"(unavailable)"`.
    pub ngspice_version: String,
    /// SHA-256 hex of the weight blob, or `"(unavailable)"`.
    pub weights_sha256: String,
    /// Seed passed to the optimizer for this run.
    pub optimizer_seed: u64,
    /// SHA-256 hex of the workspace `Cargo.lock`. Required.
    pub cargo_lock_sha256: String,
}

impl Manifest {
    /// Capture toolchain + input state. Optional sources soft-fail to
    /// `"(unavailable)"`; only `cargo_lock` is required.
    pub fn capture(inputs: ManifestInputs) -> Result<Self, ManifestError> {
        Ok(Self {
            sky130_commit: inputs
                .sky130_repo
                .map(git_head_or_unavailable)
                .unwrap_or_else(unavailable),
            orfs_image: inputs
                .orfs_image
                .map(docker_digest_or_unavailable)
                .unwrap_or_else(unavailable),
            ngspice_version: ngspice_version_or_unavailable(),
            weights_sha256: inputs
                .weights
                .map(sha256_file_or_unavailable)
                .unwrap_or_else(unavailable),
            optimizer_seed: inputs.seed,
            cargo_lock_sha256: sha256_file(inputs.cargo_lock)
                .map_err(|_| ManifestError::NoCargoLock)?,
        })
    }
}

fn unavailable() -> String {
    "(unavailable)".to_string()
}

fn git_head_or_unavailable(repo: &Path) -> String {
    Command::new("git")
        .args(["-C"])
        .arg(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(unavailable)
}

fn docker_digest_or_unavailable(image: &str) -> String {
    eda_container::inspect_digest(image).unwrap_or_else(unavailable)
}

fn ngspice_version_or_unavailable() -> String {
    Command::new("ngspice")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(str::to_string))
        .unwrap_or_else(unavailable)
}

fn sha256_file_or_unavailable(p: &Path) -> String {
    sha256_file(p).unwrap_or_else(|_| unavailable())
}

/// Pure: SHA-256 hex of file bytes. `pub` so tests + downstream
/// crates can reuse it for blob fingerprinting.
pub fn sha256_file(p: &Path) -> Result<String, std::io::Error> {
    let bytes = std::fs::read(p)?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("Cargo.lock not readable (manifest cannot be captured without it)")]
    NoCargoLock,
}
