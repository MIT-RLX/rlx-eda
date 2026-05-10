//! Shared container plumbing for rlx-eda.
//!
//! Two pieces:
//!
//! - The [`images`] registry: every Dockerfile we ship lives under the
//!   workspace's top-level `docker/` directory, and its tag + build
//!   context path is named here so callers don't hard-code paths.
//! - [`DockerRun`]: a thin builder around `docker run …` that handles
//!   bind mounts, env vars, stdin piping, and stdout capture. Both the
//!   ngspice driver and the ORFS bench backend route through this so
//!   they share docker-availability checks, error reporting, and image
//!   auto-build.
//!
//! Workspace layout assumed:
//!
//! ```text
//! <workspace>/
//!   crates/eda-container/      ← this crate
//!   docker/
//!     ngspice/Dockerfile
//!     orfs/Dockerfile
//!     orfs/run_orfs.sh
//! ```
//!
//! The workspace root is anchored at compile time via
//! `env!("CARGO_MANIFEST_DIR")` of *this* crate, so consumers don't
//! need their own way to find `docker/`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContainerError {
    #[error("docker not found on PATH (install Docker, or set the relevant *_IMAGE env var to a pre-built tag)")]
    DockerNotFound,
    #[error("docker build failed for image '{image}'\n{stderr}")]
    BuildFailed { image: String, stderr: String },
    #[error("docker run for '{image}' exited non-zero ({code:?})\nstderr:\n{stderr}")]
    RunFailed {
        image: String,
        code: Option<i32>,
        stderr: String,
    },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ContainerError>;

/// Workspace root, resolved from this crate's `CARGO_MANIFEST_DIR`.
/// `crates/eda-container` → `..`.
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Path to the centralized `docker/` directory at the workspace root.
pub fn docker_root() -> PathBuf {
    workspace_root().join("docker")
}

/// Central registry of the docker images rlx-eda ships. Every entry
/// names a tag + a Dockerfile context dir under `docker/`. Add new
/// images here, not in consumer crates.
pub mod images {
    use super::*;

    /// Logical handle for one image: default tag, env-var override
    /// name, and Dockerfile context dir.
    pub struct ImageSpec {
        pub default_tag: &'static str,
        pub env_var: &'static str,
        /// Path component under `docker/` (e.g. `"ngspice"`).
        pub context_subdir: &'static str,
    }

    impl ImageSpec {
        /// Resolve the image tag, honoring `self.env_var` if set.
        pub fn tag(&self) -> String {
            std::env::var(self.env_var).unwrap_or_else(|_| self.default_tag.to_string())
        }

        /// Absolute path to the Dockerfile context (passed to
        /// `docker build`).
        pub fn context_dir(&self) -> PathBuf {
            docker_root().join(self.context_subdir)
        }
    }

    pub const NGSPICE: ImageSpec = ImageSpec {
        default_tag: "rlx-ngspice:local",
        env_var: "RLX_NGSPICE_IMAGE",
        context_subdir: "ngspice",
    };

    pub const ORFS: ImageSpec = ImageSpec {
        default_tag: "rlx-eda-orfs:local",
        env_var: "RLX_ORFS_IMAGE",
        context_subdir: "orfs",
    };

    /// Standalone Yosys synthesis (no PnR / STA / DRC). Lighter than
    /// ORFS for digital flows that only need RTL→netlist.
    pub const YOSYS: ImageSpec = ImageSpec {
        default_tag: "rlx-yosys:local",
        env_var: "RLX_YOSYS_IMAGE",
        context_subdir: "yosys",
    };

    /// Standalone Magic — DRC, parasitic extraction, LVS via netgen.
    /// For analog / full-custom flows where ORFS isn't a fit.
    pub const MAGIC: ImageSpec = ImageSpec {
        default_tag: "rlx-magic:local",
        env_var: "RLX_MAGIC_IMAGE",
        context_subdir: "magic",
    };

    /// KLayout — headless GDS/OASIS I/O, scripted DRC, PNG renders.
    /// Pairs with the `klayout-rs` Rust integration.
    pub const KLAYOUT: ImageSpec = ImageSpec {
        default_tag: "rlx-klayout:local",
        env_var: "RLX_KLAYOUT_IMAGE",
        context_subdir: "klayout",
    };

    /// All known images, in the order `just deps-docker` builds them
    /// (small first, ORFS last because it's heaviest).
    pub const ALL: &[&ImageSpec] = &[&NGSPICE, &YOSYS, &MAGIC, &KLAYOUT, &ORFS];
}

/// Tiny dependency-free `which`.
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// `docker` is on PATH.
pub fn docker_available() -> bool {
    which("docker").is_some()
}

/// True if a local image with `tag` is present (`docker image inspect`).
pub fn image_exists(tag: &str) -> bool {
    Command::new("docker")
        .args(["image", "inspect", tag])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// `docker inspect <image> --format {{.Id}}`. Returns `None` on any
/// failure; recorded by the bench manifest for reproducibility.
pub fn inspect_digest(image: &str) -> Option<String> {
    let out = Command::new("docker")
        .args(["inspect", image, "--format", "{{.Id}}"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Make sure `tag` exists locally; build from `context_dir` if not.
/// Idempotent — no-op when the image is already present.
pub fn ensure_image(tag: &str, context_dir: &Path) -> Result<()> {
    if !docker_available() {
        return Err(ContainerError::DockerNotFound);
    }
    if image_exists(tag) {
        return Ok(());
    }
    let out = Command::new("docker")
        .args(["build", "-t", tag])
        .arg(context_dir)
        .output()?;
    if !out.status.success() {
        return Err(ContainerError::BuildFailed {
            image: tag.to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    Ok(())
}

/// Convenience: ensure a registered [`images::ImageSpec`] is built.
pub fn ensure_image_spec(spec: &images::ImageSpec) -> Result<()> {
    ensure_image(&spec.tag(), &spec.context_dir())
}

/// Builder for `docker run …` invocations. Captures the bits both
/// drivers care about: image, mounts, env, command args, optional
/// `-i` for stdin piping. Two terminal methods:
///
/// - [`DockerRun::run_with_stdin`] feeds bytes to the container's
///   stdin and captures stdout (used by the ngspice deck pipe).
/// - [`DockerRun::status`] runs with inherited stdio and returns the
///   exit status (used by the ORFS bench, which writes
///   `/work/metrics.json` rather than streaming results).
pub struct DockerRun {
    image: String,
    mounts: Vec<(PathBuf, PathBuf)>,
    env: Vec<(String, String)>,
    args: Vec<String>,
    interactive: bool,
    entrypoint: Option<String>,
    workdir: Option<PathBuf>,
    platform: Option<String>,
}

impl DockerRun {
    /// Start a builder for `image` with sensible defaults: `--rm`,
    /// no mounts, no env overrides, non-interactive.
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            mounts: Vec::new(),
            env: Vec::new(),
            args: Vec::new(),
            interactive: false,
            entrypoint: None,
            workdir: None,
            platform: None,
        }
    }

    /// Force a specific Docker platform (e.g. `"linux/amd64"`). Needed
    /// when running an x86 image on an arm64 host through Rosetta —
    /// without `--platform`, Docker prints a warning and may still run
    /// but emulation behavior is fragile.
    pub fn platform(mut self, p: impl Into<String>) -> Self {
        self.platform = Some(p.into());
        self
    }

    /// Override the container's `ENTRYPOINT` (`docker run
    /// --entrypoint <bin>`). Useful when the image ships with a
    /// strict entrypoint (e.g. `verilator/verilator` defaults to
    /// `verilator`) and you need to run a wrapper instead.
    pub fn entrypoint(mut self, ep: impl Into<String>) -> Self {
        self.entrypoint = Some(ep.into());
        self
    }

    /// Set the container's working directory (`docker run -w <dir>`).
    /// Resolved inside the container, not on the host — combine
    /// with `mount(host, container)` and pass `container` here.
    pub fn workdir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.workdir = Some(dir.into());
        self
    }

    /// Add a `-v host:container` bind mount.
    pub fn mount(mut self, host: impl Into<PathBuf>, container: impl Into<PathBuf>) -> Self {
        self.mounts.push((host.into(), container.into()));
        self
    }

    /// Add an `-e KEY=VALUE` environment override.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Add `-i` so the container's stdin is open for piping.
    pub fn interactive(mut self, yes: bool) -> Self {
        self.interactive = yes;
        self
    }

    /// Append one positional arg (passed to the container's
    /// entrypoint / cmd).
    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.args.push(a.into());
        self
    }

    /// Append several positional args.
    pub fn args<I, S>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for a in items {
            self.args.push(a.into());
        }
        self
    }

    fn build_command(&self) -> Command {
        let mut cmd = Command::new("docker");
        cmd.args(["run", "--rm"]);
        if let Some(p) = &self.platform {
            cmd.arg("--platform").arg(p);
        }
        if self.interactive {
            cmd.arg("-i");
        }
        if let Some(ep) = &self.entrypoint {
            cmd.arg("--entrypoint").arg(ep);
        }
        if let Some(wd) = &self.workdir {
            cmd.arg("-w").arg(wd);
        }
        for (host, container) in &self.mounts {
            cmd.arg("-v")
                .arg(format!("{}:{}", host.display(), container.display()));
        }
        for (k, v) in &self.env {
            cmd.arg("-e").arg(format!("{k}={v}"));
        }
        cmd.arg(&self.image);
        for a in &self.args {
            cmd.arg(a);
        }
        cmd
    }

    /// Pipe `stdin_bytes` into the container, capture stdout. Errors
    /// on non-zero exit, surfacing stderr in the message.
    pub fn run_with_stdin(self, stdin_bytes: &[u8]) -> Result<String> {
        if !docker_available() {
            return Err(ContainerError::DockerNotFound);
        }
        let mut cmd = self.build_command();
        let out = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child.stdin.as_mut().unwrap().write_all(stdin_bytes)?;
                child.wait_with_output()
            })?;
        if !out.status.success() {
            return Err(ContainerError::RunFailed {
                image: self.image,
                code: out.status.code(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// Run with inherited stdio (the container's output streams to
    /// the parent process), return the raw exit status. Use when the
    /// container deposits results on a mounted volume rather than on
    /// stdout.
    pub fn status(self) -> Result<ExitStatus> {
        if !docker_available() {
            return Err(ContainerError::DockerNotFound);
        }
        let mut cmd = self.build_command();
        Ok(cmd.status()?)
    }

    /// Spawn the container with `--name` set to a deterministic
    /// label, returning a [`RunningContainer`] handle that exposes
    /// `logs()`, `top()`, `stats()`, and `wait()`. This lets long-
    /// running synth/simulation jobs be observed live (matches how
    /// the `synth_sky130` ABC pass needed peeking into during the
    /// 3×3-matrix runs).
    ///
    /// `name` must be unique across the docker daemon; conventional
    /// choice is `format!("rlx-{tool}-{pid}")` so concurrent benches
    /// don't clash. The handle's `wait()` collects stdout + exit
    /// status the same way `run_with_stdin` does.
    pub fn spawn_named(mut self, name: impl Into<String>) -> Result<RunningContainer> {
        if !docker_available() {
            return Err(ContainerError::DockerNotFound);
        }
        let name = name.into();
        // Inject `--name <name>` ahead of any positional args.
        // build_command emits `docker run --rm [...]<image>[args]`;
        // we want `--name` in the flags section, so rebuild.
        let mut cmd = Command::new("docker");
        cmd.args(["run", "--rm", "--name", &name]);
        if let Some(p) = &self.platform {
            cmd.arg("--platform").arg(p);
        }
        if self.interactive {
            cmd.arg("-i");
        }
        if let Some(ep) = self.entrypoint.take() {
            cmd.arg("--entrypoint").arg(ep);
        }
        if let Some(wd) = self.workdir.take() {
            cmd.arg("-w").arg(wd);
        }
        for (host, container) in self.mounts.drain(..) {
            cmd.arg("-v")
                .arg(format!("{}:{}", host.display(), container.display()));
        }
        for (k, v) in self.env.drain(..) {
            cmd.arg("-e").arg(format!("{k}={v}"));
        }
        cmd.arg(&self.image);
        for a in self.args.drain(..) {
            cmd.arg(a);
        }
        let child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        Ok(RunningContainer { name, child })
    }
}

/// Handle to a container started via [`DockerRun::spawn_named`].
///
/// While alive, supports out-of-band introspection — `logs()`
/// shells `docker logs --tail N <name>`; `top()` shells `docker top
/// <name>` for in-container PID + CPU info; `stats()` snapshots
/// memory + CPU%. None of these block on the container itself.
///
/// `wait()` consumes the handle and returns `(stdout, exit_status)`
/// — analogous to `run_with_stdin` but for the spawn-then-monitor
/// pattern.
pub struct RunningContainer {
    pub name: String,
    child: std::process::Child,
}

impl RunningContainer {
    /// Tail the container's combined stdout+stderr. Returns up to
    /// `lines` most recent lines (whatever `docker logs --tail N`
    /// prints — works whether the container is running or not).
    pub fn logs(&self, lines: usize) -> Result<String> {
        let out = Command::new("docker")
            .args(["logs", "--tail", &lines.to_string(), &self.name])
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .output()?;
        // docker logs writes both streams to its own stdout/stderr;
        // merge for the caller's convenience.
        let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
        s.push_str(&String::from_utf8_lossy(&out.stderr));
        Ok(s)
    }

    /// Snapshot of in-container processes (PID / CPU / CMD).
    /// Surfaces what's actually consuming time — invaluable when a
    /// silent ABC mapping pass is the bottleneck.
    pub fn top(&self) -> Result<String> {
        let out = Command::new("docker")
            .args(["top", &self.name])
            .output()?;
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// One-shot resource snapshot: CPU%, memory, I/O. Wraps
    /// `docker stats --no-stream <name>`.
    pub fn stats(&self) -> Result<String> {
        let out = Command::new("docker")
            .args(["stats", "--no-stream", &self.name])
            .output()?;
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// Block until the container exits. Returns captured stdout +
    /// the exit status (mirrors `run_with_stdin` for spawn-mode).
    pub fn wait(self) -> Result<(String, ExitStatus)> {
        let out = self.child.wait_with_output()?;
        Ok((
            String::from_utf8_lossy(&out.stdout).into_owned(),
            out.status,
        ))
    }

    /// Send `docker kill` to the container; useful for timeouts.
    pub fn kill(&self) -> Result<()> {
        Command::new("docker").args(["kill", &self.name]).status()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_root_contains_docker_dir() {
        let root = workspace_root();
        assert!(root.join("Cargo.toml").exists(), "{}", root.display());
        assert!(root.join("docker").is_dir(), "{}", root.join("docker").display());
    }

    #[test]
    fn registered_images_have_context_dirs() {
        for spec in images::ALL {
            assert!(
                spec.context_dir().join("Dockerfile").exists(),
                "missing Dockerfile for {}: {}",
                spec.default_tag,
                spec.context_dir().display()
            );
        }
    }

    #[test]
    fn image_spec_env_var_override() {
        let key = "RLX_TEST_IMAGE_OVERRIDE_5fbc7";
        let spec = images::ImageSpec {
            default_tag: "default:tag",
            env_var: key,
            context_subdir: "ngspice",
        };
        std::env::remove_var(key);
        assert_eq!(spec.tag(), "default:tag");
        std::env::set_var(key, "override:tag");
        assert_eq!(spec.tag(), "override:tag");
        std::env::remove_var(key);
    }
}
