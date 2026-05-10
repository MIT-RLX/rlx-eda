//! ngspice external-validation driver.
//!
//! Two pieces:
//! - [`Invoker`] trait: backend-agnostic "run an ngspice deck, get results".
//!   `LocalBinary` shells out to a `ngspice` on PATH; a future `DockerImage`
//!   impl will pin via image digest. Tests pick at runtime via env var so the
//!   same test code can run native or containerized.
//! - A minimal stdout parser sufficient for `.op` operating-point voltages.
//!   Just enough to extract `v(node) = X` from a `print` line emitted by a
//!   `.control ... .endc` block.
//! - Raw waveform parsing comes from `eda-waveform::nutmeg` (shared with
//!   the LTspice driver); `run_transient_trace` and `run_ac` use it to
//!   return full-trace data instead of single-time scalars.
//!
//! Every later "extern" crate (klayout, magic, sax, …) implements the same
//! `Invoker` shape.

/// Re-export of the shared Nutmeg parser for back-compat. New code should
/// `use eda_waveform::nutmeg` directly.
pub use eda_waveform::nutmeg;

use eda_container::{self as container, DockerRun};
use std::path::PathBuf;
use std::process::Command;
use thiserror::Error;

/// What we ask ngspice to report at the end of an analysis.
#[derive(Debug, Clone)]
pub enum OutputRequest {
    /// Operating-point voltage at a named node.
    NodeVoltage(String),
}

#[derive(Debug, Default, Clone)]
pub struct DcResults {
    /// `node_name → volts` for each requested NodeVoltage.
    pub node_voltages: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Error)]
pub enum NgspiceError {
    #[error("ngspice binary not found on PATH (set NGSPICE_BIN to override)")]
    BinaryNotFound,
    #[error("ngspice exited non-zero ({code:?}); stderr:\n{stderr}")]
    NonZero { code: Option<i32>, stderr: String },
    #[error("requested output '{0}' not found in ngspice stdout")]
    OutputMissing(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("nutmeg parse error: {0}")]
    Nutmeg(#[from] nutmeg::NutmegError),
    #[error("ngspice produced wrong plot type: expected {expected}, got {got:?}")]
    WrongPlotKind { expected: &'static str, got: nutmeg::NutmegFlavor },
    #[error(transparent)]
    Container(#[from] container::ContainerError),
}

pub type Result<T> = std::result::Result<T, NgspiceError>;

/// Transient analysis spec: uniform timestep, run to `t_stop`.
///
/// `use_initial_conditions` adds the `uic` flag to the `.tran` line, so
/// ngspice starts from explicit `IC=` / `.ic` values rather than the DC
/// operating point. Default true — matches what circuit-as-residual-graph
/// transient comparisons want (start from discharged caps / zero state),
/// otherwise ngspice's pre-DC charges everything to steady state.
///
/// `t_max` (when `Some`) caps ngspice's adaptive substepping. The default
/// LTE-controlled stepper picks `h_substep` per the local solution rate;
/// in transient comparisons against rlx's uniform-h outer loop, that
/// dispersion shows up as O(h) drift around step edges. Setting
/// `t_max = Some(t_step)` forces ngspice into a similar uniform-h regime.
#[derive(Debug, Clone, Copy)]
pub struct TransientAnalysis {
    pub t_step: f64,
    pub t_stop: f64,
    pub use_initial_conditions: bool,
    pub t_max: Option<f64>,
}

impl TransientAnalysis {
    pub fn new(t_step: f64, t_stop: f64) -> Self {
        Self { t_step, t_stop, use_initial_conditions: true, t_max: None }
    }

    /// Cap ngspice's adaptive substep size at `tmax`. Pass `t_step` to
    /// pin ngspice to the same grid resolution rlx uses.
    pub fn with_t_max(mut self, tmax: f64) -> Self {
        self.t_max = Some(tmax);
        self
    }
}

/// AC analysis spec: log-spaced frequency sweep `[fstart, fstop]`.
///
/// `dec` semantics matches ngspice: `points_per_decade` ticks per decade of
/// frequency. Octave / linear sweeps fit the same struct shape behind a
/// `Mode` enum if/when we need them; for now `dec` covers every Bode plot
/// we'd want to validate.
#[derive(Debug, Clone, Copy)]
pub struct AcAnalysis {
    pub points_per_decade: usize,
    pub f_start: f64,
    pub f_stop: f64,
}

impl AcAnalysis {
    pub fn dec(points_per_decade: usize, f_start: f64, f_stop: f64) -> Self {
        Self { points_per_decade, f_start, f_stop }
    }
}

/// Full-trace transient result: one shared time axis + per-node voltage
/// vectors aligned to it.
#[derive(Debug, Default, Clone)]
pub struct TransientTrace {
    pub time: Vec<f64>,
    pub node_voltages: std::collections::HashMap<String, Vec<f64>>,
}

/// Full-trace AC result: shared frequency axis + per-node complex
/// `(real, imag)` voltage vectors aligned to it.
#[derive(Debug, Default, Clone)]
pub struct AcTrace {
    pub frequency: Vec<f64>,
    pub node_voltages: std::collections::HashMap<String, Vec<(f64, f64)>>,
}

/// Pluggable backend: `LocalBinary` today, `DockerImage` later. Tests should
/// take a `&dyn Invoker` so they're agnostic.
pub trait Invoker: Send + Sync {
    fn run_dc(&self, deck: &str, requests: &[OutputRequest]) -> Result<DcResults>;

    /// Run a transient analysis and report node voltages at `t_stop` via
    /// `.meas tran ... find v(node) at=<t_stop>`. Same `OutputRequest` shape
    /// as `run_dc`; the variant carries the node name and the analysis spec
    /// supplies the time grid.
    fn run_transient_final(
        &self,
        deck: &str,
        analysis: &TransientAnalysis,
        requests: &[OutputRequest],
    ) -> Result<DcResults>;

    /// Run a transient and return the **full waveform** for each requested
    /// node, sampled on ngspice's chosen time grid (which may not match the
    /// caller's `t_step` exactly — ngspice uses LTE-controlled adaptive
    /// stepping). For comparison against rlx output, interpolate via
    /// `eda-validate::assert_traces_close`.
    fn run_transient_trace(
        &self,
        deck: &str,
        analysis: &TransientAnalysis,
        requests: &[OutputRequest],
    ) -> Result<TransientTrace>;

    /// Run an AC sweep and return the full complex `(re, im)` response per
    /// requested node, on ngspice's frequency grid.
    fn run_ac(
        &self,
        deck: &str,
        analysis: &AcAnalysis,
        requests: &[OutputRequest],
    ) -> Result<AcTrace>;
}

/// Shells out to a native `ngspice` binary on PATH.
pub struct LocalBinary {
    pub binary: PathBuf,
}

impl LocalBinary {
    /// Resolve `ngspice` on PATH (or honor the `NGSPICE_BIN` env var).
    pub fn from_env() -> Result<Self> {
        if let Ok(p) = std::env::var("NGSPICE_BIN") {
            return Ok(Self { binary: PathBuf::from(p) });
        }
        container::which("ngspice")
            .map(|binary| Self { binary })
            .ok_or(NgspiceError::BinaryNotFound)
    }
}

/// Internal extension point shared by `LocalBinary` and `DockerInvoker`.
/// The four `Invoker` methods above all assemble a deck-with-control,
/// pipe it to ngspice somewhere, then parse stdout or a `.raw` file —
/// only the *somewhere* differs between native and containerized runs.
trait NgspiceRunner: Send + Sync {
    /// Build a deck wrapped in `<header>\n<deck>\n.control\n<body>\n
    /// .endc\n.end`, pipe it to ngspice's stdin, return stdout.
    fn run_with_control(&self, deck: &str, header: &str, body: &str) -> Result<String>;
}

// ---- Free helpers shared by both backends -------------------------------

fn invoker_run_dc<R: NgspiceRunner + ?Sized>(
    r: &R, deck: &str, requests: &[OutputRequest],
) -> Result<DcResults> {
    let mut body = String::from("op\n");
    for req in requests {
        match req {
            OutputRequest::NodeVoltage(n) => body.push_str(&format!("print v({n})\n")),
        }
    }
    let stdout = r.run_with_control(deck, "* rlx-eda ngspice driver (dc)\n", &body)?;
    let mut results = DcResults::default();
    for req in requests {
        match req {
            OutputRequest::NodeVoltage(n) => {
                let v = parse_node_voltage(&stdout, n)
                    .ok_or_else(|| NgspiceError::OutputMissing(n.clone()))?;
                results.node_voltages.insert(n.clone(), v);
            }
        }
    }
    Ok(results)
}

fn invoker_run_transient_trace<R: NgspiceRunner + ?Sized>(
    r: &R, deck: &str, analysis: &TransientAnalysis, requests: &[OutputRequest],
) -> Result<TransientTrace> {
    let raw = tempfile::Builder::new()
        .prefix("rlx-eda-tran-")
        .suffix(".raw")
        .tempfile()?;
    let raw_path = raw.path().to_path_buf();

    let signals = signals_for_request(requests);
    let body = format!(
        "{}\nwrite {} {}\n",
        tran_line(analysis), raw_path.display(), signals,
    );
    let _ = r.run_with_control(deck, "* rlx-eda ngspice driver (tran trace)\n", &body)?;

    let bytes = std::fs::read(&raw_path)?;
    drop(raw);
    let plot = nutmeg::parse_bytes(&bytes)?;
    if plot.flavor != nutmeg::NutmegFlavor::Real {
        return Err(NgspiceError::WrongPlotKind { expected: "real (transient)", got: plot.flavor });
    }

    let time = plot
        .real_trace("time")
        .ok_or_else(|| NgspiceError::OutputMissing("time".into()))?
        .to_vec();
    let mut node_voltages = std::collections::HashMap::new();
    for req in requests {
        match req {
            OutputRequest::NodeVoltage(n) => {
                let key = format!("v({n})");
                let v = plot
                    .real_trace(&key)
                    .ok_or_else(|| NgspiceError::OutputMissing(n.clone()))?
                    .to_vec();
                node_voltages.insert(n.clone(), v);
            }
        }
    }
    Ok(TransientTrace { time, node_voltages })
}

fn invoker_run_ac<R: NgspiceRunner + ?Sized>(
    r: &R, deck: &str, analysis: &AcAnalysis, requests: &[OutputRequest],
) -> Result<AcTrace> {
    let raw = tempfile::Builder::new()
        .prefix("rlx-eda-ac-")
        .suffix(".raw")
        .tempfile()?;
    let raw_path = raw.path().to_path_buf();
    let signals = signals_for_request(requests);
    let body = format!(
        "ac dec {} {} {}\nwrite {} {}\n",
        analysis.points_per_decade, analysis.f_start, analysis.f_stop,
        raw_path.display(), signals,
    );
    let _ = r.run_with_control(deck, "* rlx-eda ngspice driver (ac)\n", &body)?;

    let bytes = std::fs::read(&raw_path)?;
    drop(raw);
    let plot = nutmeg::parse_bytes(&bytes)?;
    if plot.flavor != nutmeg::NutmegFlavor::Complex {
        return Err(NgspiceError::WrongPlotKind { expected: "complex (ac)", got: plot.flavor });
    }
    let frequency = plot
        .complex_trace("frequency")
        .ok_or_else(|| NgspiceError::OutputMissing("frequency".into()))?
        .iter().map(|(re, _)| *re).collect();
    let mut node_voltages = std::collections::HashMap::new();
    for req in requests {
        match req {
            OutputRequest::NodeVoltage(n) => {
                let key = format!("v({n})");
                let v = plot
                    .complex_trace(&key)
                    .ok_or_else(|| NgspiceError::OutputMissing(n.clone()))?
                    .to_vec();
                node_voltages.insert(n.clone(), v);
            }
        }
    }
    Ok(AcTrace { frequency, node_voltages })
}

fn invoker_run_transient_final<R: NgspiceRunner + ?Sized>(
    r: &R, deck: &str, analysis: &TransientAnalysis, requests: &[OutputRequest],
) -> Result<DcResults> {
    let mut body = format!("{}\n", tran_line(analysis));
    let mut measure_names: Vec<(String, String)> = Vec::new();
    for req in requests {
        match req {
            OutputRequest::NodeVoltage(n) => {
                let mname = format!("m_{}", n);
                body.push_str(&format!("meas tran {mname} find v({n}) at={}\n", analysis.t_stop));
                body.push_str(&format!("print {mname}\n"));
                measure_names.push((n.clone(), mname));
            }
        }
    }
    let stdout = r.run_with_control(deck, "* rlx-eda ngspice driver (tran)\n", &body)?;
    let mut results = DcResults::default();
    for (node, mname) in &measure_names {
        let v = parse_named_value(&stdout, mname)
            .ok_or_else(|| NgspiceError::OutputMissing(node.clone()))?;
        results.node_voltages.insert(node.clone(), v);
    }
    Ok(results)
}

// ---- LocalBinary --------------------------------------------------------

impl NgspiceRunner for LocalBinary {
    fn run_with_control(&self, deck: &str, header: &str, body: &str) -> Result<String> {
        let full = assemble_deck(deck, header, body);
        let out = Command::new(&self.binary)
            .args(["-b", "-n"])
            .arg("/dev/stdin")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child.stdin.as_mut().unwrap().write_all(full.as_bytes())?;
                child.wait_with_output()
            })?;
        if !out.status.success() {
            return Err(NgspiceError::NonZero {
                code: out.status.code(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

impl Invoker for LocalBinary {
    fn run_dc(&self, deck: &str, reqs: &[OutputRequest]) -> Result<DcResults> {
        invoker_run_dc(self, deck, reqs)
    }
    fn run_transient_trace(&self, deck: &str, a: &TransientAnalysis, reqs: &[OutputRequest])
        -> Result<TransientTrace> { invoker_run_transient_trace(self, deck, a, reqs) }
    fn run_ac(&self, deck: &str, a: &AcAnalysis, reqs: &[OutputRequest]) -> Result<AcTrace> {
        invoker_run_ac(self, deck, a, reqs)
    }
    fn run_transient_final(&self, deck: &str, a: &TransientAnalysis, reqs: &[OutputRequest])
        -> Result<DcResults> { invoker_run_transient_final(self, deck, a, reqs) }
}

// ---- DockerInvoker ------------------------------------------------------

/// Runs ngspice inside a Docker container. Useful for reproducibility
/// (image pinning) and for environments without a host ngspice. /tmp is
/// bind-mounted by default so deck-side `write <tempfile>.raw` lands on
/// a host path the parser reads.
///
/// Image registry, build, and `docker run` plumbing live in
/// `eda-container`. This struct just holds the per-invocation knobs
/// (image tag + mounts) and forwards.
pub struct DockerInvoker {
    /// Image tag (e.g. `"rlx-ngspice:local"`).
    pub image: String,
    /// Bind mounts as `(host_path, container_path)`. Defaults to one
    /// entry, `(/tmp, /tmp)` — required for the `write <raw>` path to
    /// round-trip between deck and host.
    pub mounts: Vec<(PathBuf, PathBuf)>,
}

impl DockerInvoker {
    /// Construct an invoker for `image` with the default `/tmp:/tmp` mount.
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            mounts: vec![(PathBuf::from("/tmp"), PathBuf::from("/tmp"))],
        }
    }

    /// Resolve the image from the central registry (honoring
    /// `RLX_NGSPICE_IMAGE`), check docker is installed, return a
    /// ready-to-use invoker. Does **not** auto-build — call
    /// `ensure_image` if you want that.
    pub fn from_env() -> Result<Self> {
        if !container::docker_available() {
            return Err(container::ContainerError::DockerNotFound.into());
        }
        Ok(Self::new(container::images::NGSPICE.tag()))
    }

    /// Make sure `self.image` exists locally; if not, build it from
    /// the centralized Dockerfile at `<workspace>/docker/ngspice/`.
    /// Idempotent.
    pub fn ensure_image(&self) -> Result<()> {
        container::ensure_image(&self.image, &container::images::NGSPICE.context_dir())?;
        Ok(())
    }
}

impl NgspiceRunner for DockerInvoker {
    fn run_with_control(&self, deck: &str, header: &str, body: &str) -> Result<String> {
        let full = assemble_deck(deck, header, body);
        let mut run = DockerRun::new(&self.image).interactive(true);
        for (host, cont) in &self.mounts {
            run = run.mount(host.clone(), cont.clone());
        }
        let stdout = run
            .args(["ngspice", "-b", "-n", "/dev/stdin"])
            .run_with_stdin(full.as_bytes())?;
        Ok(stdout)
    }
}

impl Invoker for DockerInvoker {
    fn run_dc(&self, deck: &str, reqs: &[OutputRequest]) -> Result<DcResults> {
        invoker_run_dc(self, deck, reqs)
    }
    fn run_transient_trace(&self, deck: &str, a: &TransientAnalysis, reqs: &[OutputRequest])
        -> Result<TransientTrace> { invoker_run_transient_trace(self, deck, a, reqs) }
    fn run_ac(&self, deck: &str, a: &AcAnalysis, reqs: &[OutputRequest]) -> Result<AcTrace> {
        invoker_run_ac(self, deck, a, reqs)
    }
    fn run_transient_final(&self, deck: &str, a: &TransientAnalysis, reqs: &[OutputRequest])
        -> Result<DcResults> { invoker_run_transient_final(self, deck, a, reqs) }
}

fn assemble_deck(deck: &str, header: &str, body: &str) -> String {
    let mut full = String::new();
    full.push_str(header);
    full.push_str(deck.trim_end());
    full.push_str("\n.control\n");
    full.push_str(body);
    full.push_str(".endc\n.end\n");
    full
}

/// Build the ngspice `.tran` directive from a [`TransientAnalysis`].
///
/// ngspice grammar: `tran tstep tstop [tstart [tmax]] [uic]`. When
/// `t_max` is set we always emit `tstart=0` so the parser sees a 4-arg
/// form; otherwise we use the short 2-arg form to avoid changing default
/// behavior.
fn tran_line(analysis: &TransientAnalysis) -> String {
    let uic_suffix = if analysis.use_initial_conditions { " uic" } else { "" };
    match analysis.t_max {
        Some(tmax) => format!(
            "tran {} {} {} {}{uic_suffix}",
            analysis.t_step, analysis.t_stop, 0.0, tmax,
        ),
        None => format!("tran {} {}{uic_suffix}", analysis.t_step, analysis.t_stop),
    }
}

/// Format the `requests` slice as a space-separated list of ngspice signal
/// names suitable for a `write` command (`v(node1) v(node2) ...`).
fn signals_for_request(requests: &[OutputRequest]) -> String {
    let mut s = String::new();
    for req in requests {
        match req {
            OutputRequest::NodeVoltage(n) => {
                if !s.is_empty() { s.push(' '); }
                s.push_str(&format!("v({n})"));
            }
        }
    }
    s
}

/// Extract the numeric value from an ngspice line of form
/// `v(node) = <number>` or `v(node)\t<number>` (some builds use whitespace).
/// Case-insensitive, lenient about whitespace.
fn parse_node_voltage(stdout: &str, node: &str) -> Option<f64> {
    let needle = format!("v({})", node.to_lowercase());
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        if let Some(idx) = lower.find(&needle) {
            let tail = &line[idx + needle.len()..];
            let tail = tail.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
            let tok = tail.split_whitespace().next()?;
            if let Ok(v) = tok.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

/// Extract a scalar value emitted by `print <name>` after a `.meas` (or by
/// any `let name = ...; print name` form). Looks for `<name> = <number>`,
/// case-insensitive, anchored on word boundary so a measure named `m_vout`
/// doesn't match `v(m_vout)`.
fn parse_named_value(stdout: &str, name: &str) -> Option<f64> {
    let needle = name.to_lowercase();
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        let mut from = 0usize;
        while let Some(rel) = lower[from..].find(&needle) {
            let idx = from + rel;
            let before_ok = idx == 0
                || lower.as_bytes()[idx - 1] == b' '
                || lower.as_bytes()[idx - 1] == b'\t';
            let end = idx + needle.len();
            let after_ok = end == lower.len()
                || lower.as_bytes()[end] == b'='
                || lower.as_bytes()[end] == b' '
                || lower.as_bytes()[end] == b'\t';
            if before_ok && after_ok {
                let tail = &line[end..];
                let tail = tail.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
                if let Some(tok) = tail.split_whitespace().next() {
                    if let Ok(v) = tok.parse::<f64>() {
                        return Some(v);
                    }
                }
            }
            from = idx + needle.len();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_voltage_eq_form() {
        let s = "v(vout) = 5.000000e-01\nbye\n";
        assert_eq!(parse_node_voltage(s, "vout"), Some(0.5));
    }

    #[test]
    fn parse_voltage_tab_form() {
        let s = "v(vout)\t5.000000e-01\n";
        assert_eq!(parse_node_voltage(s, "vout"), Some(0.5));
    }

    #[test]
    fn parse_voltage_missing() {
        assert_eq!(parse_node_voltage("nothing here", "vout"), None);
    }

    #[test]
    fn parse_voltage_case_insensitive() {
        let s = "V(VOUT) = 1.234e+00";
        assert_eq!(parse_node_voltage(s, "vout"), Some(1.234));
    }

    #[test]
    fn parse_named_meas_value() {
        let s = "vout_at_tend           =  9.500000e-01\n";
        assert_eq!(parse_named_value(s, "vout_at_tend"), Some(0.95));
    }

    #[test]
    fn parse_named_value_word_boundary() {
        // Must not match a substring that's part of a longer name like
        // `vsource_at_tend` when asked for `vsource`.
        let s = "vsource_at_tend = 1.0\nvsource = 0.5\n";
        assert_eq!(parse_named_value(s, "vsource"), Some(0.5));
    }
}
