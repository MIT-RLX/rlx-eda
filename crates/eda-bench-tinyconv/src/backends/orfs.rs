//! ORFS backend: SystemVerilog (from `rlx-fpga` emit) → OpenROAD Flow
//! Scripts in docker → parsed `metrics.json`.
//!
//! Physical ground truth. Yosys synthesis + OpenROAD PnR + OpenSTA
//! timing + OpenRCX parasitics + OpenROAD PDN analysis. Magic + netgen
//! for DRC/LVS. Slow (minutes per run) so it gates only the outer DADO
//! loop and final tile validation, not the inner Adam loop.
//!
//! Serves all five functional levels but only L3 and L5 are
//! cost-effective; L1 belongs to the Rust reference and L2 to RTL sim
//! outside docker.
//!
//! Gated by `bench-orfs` feature for the docker invocation; the JSON
//! parser is feature-free so it can be unit-tested without docker.

use super::{Backend, BackendError};
use crate::metrics::{functional::Level, Functional, Physical};
use std::path::PathBuf;

pub struct OrfsBackend {
    /// Pinned ORFS image, e.g. `openroad/orfs@sha256:<digest>`.
    /// Recorded in the bench manifest (cross-cutting #1).
    pub image_digest: String,
    /// `config.mk` for the design under test, mounted into `/work`.
    pub config_mk: PathBuf,
    /// Verilog sources directory, mounted into `/work`.
    pub verilog_dir: PathBuf,
    /// Host path that gets mounted at `/work`. The container writes
    /// `metrics.json` here; we read it back on success.
    pub work_dir: PathBuf,
}

impl Backend for OrfsBackend {
    fn name(&self) -> &'static str {
        "orfs"
    }

    fn measure_physical(&self) -> Result<Physical, BackendError> {
        #[cfg(not(feature = "bench-orfs"))]
        {
            return Err(BackendError::NotEnabled("orfs", "bench-orfs"));
        }
        #[cfg(feature = "bench-orfs")]
        {
            run_orfs_and_parse(self)
        }
    }

    fn measure_functional(
        &self,
        _level: Level,
        _images: &[u32],
    ) -> Result<Functional, BackendError> {
        #[cfg(not(feature = "bench-orfs"))]
        {
            return Err(BackendError::NotEnabled("orfs", "bench-orfs"));
        }
        #[cfg(feature = "bench-orfs")]
        {
            unimplemented!("gate-level / SDF sim in docker — PLAN.md L3/L5")
        }
    }
}

#[cfg(feature = "bench-orfs")]
fn run_orfs_and_parse(b: &OrfsBackend) -> Result<Physical, BackendError> {
    // Mount the work dir, pass config_mk + verilog_dir as the two
    // positional args run_orfs.sh expects. Container writes
    // /work/metrics.json which lives at b.work_dir/metrics.json on
    // the host. Docker plumbing — image build, mount syntax, error
    // surface — lives in `eda-container`.
    let config_in_container = format!("/work/{}", file_name(&b.config_mk)?);
    let verilog_in_container = format!("/work/{}", file_name(&b.verilog_dir)?);

    let status = eda_container::DockerRun::new(&b.image_digest)
        .mount(b.work_dir.clone(), std::path::PathBuf::from("/work"))
        .arg(config_in_container)
        .arg(verilog_in_container)
        .status()
        .map_err(|e| BackendError::Toolchain(format!("docker run: {e}")))?;

    if !status.success() {
        return Err(BackendError::Toolchain(format!(
            "docker exited with {status}"
        )));
    }

    let metrics_path = b.work_dir.join("metrics.json");
    let text = std::fs::read_to_string(&metrics_path).map_err(|e| {
        BackendError::Toolchain(format!(
            "read {}: {e}",
            metrics_path.display()
        ))
    })?;
    Physical::from_orfs_json(&text)
        .map_err(|e| BackendError::Toolchain(format!("parse metrics.json: {e}")))
}

#[cfg(feature = "bench-orfs")]
fn file_name(p: &PathBuf) -> Result<String, BackendError> {
    p.file_name()
        .and_then(|s| s.to_str())
        .map(String::from)
        .ok_or_else(|| {
            BackendError::Toolchain(format!("non-UTF8 path: {}", p.display()))
        })
}
