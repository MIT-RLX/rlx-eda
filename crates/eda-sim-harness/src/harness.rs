//! The harness: drive a [`Testbench`] through every corner in a
//! [`CornerSet`], pulling results from cache or ngspice.
//!
//! Output is one [`RunOutcome`] per corner: the [`MeasureLog`] plus an
//! optional waveform (when transient/AC) and the rendered
//! [`SpecCheck`]es. Aggregation across corners and report generation
//! lives in [`crate::report`].

use std::path::PathBuf;
use std::process::Command;

extern crate tempfile;

use eda_spice_emit::Netlist;
use eda_waveform::nutmeg;
use eda_waveform::Waveform;

use crate::cache::{Cache, CacheError, CacheMode};
use crate::corner::Corner;
use crate::measure::{MeasureLog, Measurement, MeasurementValue};
use crate::spec::{SpecBundle, SpecCheck};
use crate::testbench::{Analysis, Testbench};
use crate::verify::VerifyReport;

/// One corner's results.
#[derive(Debug)]
pub struct RunOutcome {
    pub corner: Corner,
    pub deck: String,
    pub sha: String,
    pub from_cache: bool,
    pub stdout: String,
    pub measures: MeasureLog,
    pub waveform: Option<Waveform>,
    /// Per-spec result, ordered to match the [`SpecBundle`].
    pub spec_checks: Vec<(String, SpecCheck)>,
    /// DRC / LVS / EM verification results for this corner.
    /// `VerifyReport::empty()` when no verifier ran (the default).
    pub verify: VerifyReport,
    /// Wallclock time at which the simulation finished (or was loaded
    /// from cache). Used in the cicsim-compatible CSV's `time` column.
    pub ran_at: std::time::SystemTime,
}

impl RunOutcome {
    /// True iff every spec passed (or was skipped — i.e. never failed)
    /// AND every verifier that ran was clean. Mirrors cicsim's `OK`
    /// column in the per-corner CSV, extended for the verification
    /// pass: a measurement-clean corner with DRC violations is `false`.
    pub fn ok(&self) -> bool {
        self.spec_checks.iter().all(|(_, c)| !c.is_fail())
            && self.verify.is_clean()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    #[error("cache: {0}")]
    Cache(#[from] CacheError),
    /// `first_error` is the first interesting line scraped from ngspice
    /// stderr (e.g. `Error in netlist line N:`, `Syntax error: …`,
    /// `could not find a valid modelname`). When `None`, ngspice
    /// returned non-zero without a recognizable diagnostic — full
    /// stderr is in `stderr`.
    #[error("ngspice exited {code:?}{}\n  full stderr ({stderr_lines} lines suppressed; set RLX_EDA_VERBOSE=1)",
        first_error.as_deref().map(|s| format!("\n  → {s}")).unwrap_or_default(),
    )]
    NgspiceFailed {
        code: Option<i32>,
        stderr: String,
        stderr_lines: usize,
        first_error: Option<String>,
    },
    #[error("ngspice binary not found on PATH (set NGSPICE_BIN to override)")]
    BinaryNotFound,
    #[error("nutmeg: {0}")]
    Nutmeg(#[from] nutmeg::NutmegError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Pull the first diagnostic line out of an ngspice stderr blob.
/// ngspice emits a wall of text on a deck error; the first interesting
/// line tells you what's wrong. Looks for these patterns, in order:
///
/// - `Error in netlist line N: …`
/// - `Syntax error: …`
/// - `could not find a valid modelname`
/// - `incomplete or empty netlist`
/// - lines containing `Error:` or starting with `Error`
pub fn extract_first_ngspice_error(stderr: &str) -> Option<String> {
    let patterns = [
        "Error in netlist line",
        "Syntax error",
        "could not find a valid modelname",
        "incomplete or empty netlist",
        "fatal error",
    ];
    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        for p in &patterns {
            if trimmed.contains(p) {
                return Some(trimmed.to_string());
            }
        }
        if trimmed.starts_with("Error") || trimmed.starts_with("ERROR") {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Builder/runner.
pub struct Harness<'a, T: Testbench> {
    pub tb: &'a T,
    pub corners: Vec<Corner>,
    pub specs: SpecBundle,
    pub cache: Cache,
    pub output_dir: PathBuf,
    pub ngspice_bin: Option<PathBuf>,
}

impl<'a, T: Testbench> Harness<'a, T> {
    pub fn new(tb: &'a T) -> Self {
        Self {
            tb,
            corners: Vec::new(),
            specs: SpecBundle::default(),
            cache: Cache::off(),
            output_dir: PathBuf::from("results"),
            ngspice_bin: None,
        }
    }

    pub fn corners(mut self, set: crate::corner::CornerSet) -> Self {
        self.corners = set.corners;
        self
    }

    pub fn specs(mut self, b: SpecBundle) -> Self {
        self.specs = b;
        self
    }

    pub fn cache(mut self, c: Cache) -> Self {
        self.cache = c;
        self
    }

    pub fn output_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.output_dir = p.into();
        self
    }

    pub fn ngspice_bin(mut self, p: PathBuf) -> Self {
        self.ngspice_bin = Some(p);
        self
    }

    /// Resolve the ngspice binary; honor `$NGSPICE_BIN`, else PATH.
    fn resolve_bin(&self) -> Result<PathBuf, HarnessError> {
        if let Some(p) = &self.ngspice_bin { return Ok(p.clone()); }
        if let Ok(p) = std::env::var("NGSPICE_BIN") { return Ok(PathBuf::from(p)); }
        which("ngspice").ok_or(HarnessError::BinaryNotFound)
    }

    /// Run every corner. Returns one [`RunOutcome`] per corner, in input
    /// order. Errors out only on infrastructure failures; per-spec
    /// pass/fail is reported via [`RunOutcome::spec_checks`].
    pub fn run(&self) -> Result<Vec<RunOutcome>, HarnessError> {
        std::fs::create_dir_all(&self.output_dir)?;
        let bin = if self.cache.mode == CacheMode::ReuseOnly {
            None
        } else {
            Some(self.resolve_bin()?)
        };

        let mut out = Vec::with_capacity(self.corners.len());
        for corner in &self.corners {
            out.push(self.run_one(corner, bin.as_deref())?);
        }
        Ok(out)
    }

    fn run_one(&self, corner: &Corner, bin: Option<&std::path::Path>) -> Result<RunOutcome, HarnessError> {
        let measurements = self.tb.measurements();
        let analysis = self.tb.analysis();
        let nl = self.tb.build_netlist(corner);
        let deck = build_deck(&nl, corner, &analysis, &measurements);

        // Hash the deck *plus* corner-specific lib paths the testbench
        // declared via Netlist preamble lines starting with `.lib `.
        let lib_paths = collect_lib_paths(&nl);
        let lib_refs: Vec<&std::path::Path> = lib_paths.iter().map(|p| p.as_path()).collect();
        let sha = Cache::compute_sha(&deck, &lib_refs);
        let entry = self.cache.entry(&sha);

        let (stdout, raw_bytes, from_cache) = if self.cache.can_reuse(&entry) {
            let s = self.cache.load_stdout(&entry)?;
            let r = self.cache.load_raw(&entry);
            (s, r, true)
        } else if self.cache.mode == CacheMode::ReuseOnly {
            return Err(HarnessError::Cache(CacheError::NoEntry(sha.clone())));
        } else {
            let bin = bin.expect("bin must be Some when not ReuseOnly");
            let raw_path = self.output_dir.join(format!("{}_{}.raw", self.tb.name(), corner.label));
            let stdout = invoke_ngspice(bin, &deck, &analysis, &raw_path)?;
            let raw_bytes = if matches!(analysis, Analysis::Tran { .. } | Analysis::AcDec { .. }) {
                std::fs::read(&raw_path).ok()
            } else { None };
            let meta = serde_json::to_string(&serde_json::json!({
                "corner": corner.label,
                "kind": format!("{:?}", corner.kind),
                "vdd": corner.vdd,
                "temp_c": corner.temp_c,
                "lib_section": corner.lib_section,
            })).unwrap_or_else(|_| "{}".into());
            self.cache.store(&entry, &deck, &stdout, raw_bytes.as_deref(), &meta)?;
            (stdout, raw_bytes, false)
        };

        let mut measures = MeasureLog::parse(&stdout, &measurements);
        // Fold testbench-derived measurements (cicsim tran.py analogue)
        // into the log so they participate in spec checks and end up in
        // reports / CSVs alongside the raw .meas outputs.
        for (name, value) in self.tb.derive(&measures) {
            measures.values.insert(name, MeasurementValue::Number(value));
        }
        let extra_keep = self.tb.plot_signals();
        let waveform = raw_bytes.as_deref().and_then(|b| nutmeg_to_waveform(b, &extra_keep));

        let mut spec_checks = Vec::with_capacity(self.specs.specs.len());
        for spec in &self.specs.specs {
            let m = measures.get(&spec.name).and_then(|v| v.as_number());
            spec_checks.push((spec.name.clone(), spec.check(m)));
        }

        // Run any verifiers the testbench provides for this corner
        // (DRC / LVS / EM). Default is "no verifiers" — corners flow
        // through the harness exactly as before. The verifier sees
        // the parsed measurement log so EM checks can use real
        // simulated peak currents.
        let verify = self.tb.verify(corner, &measures);

        Ok(RunOutcome {
            corner: corner.clone(),
            deck,
            sha,
            from_cache,
            stdout,
            measures,
            waveform,
            spec_checks,
            verify,
            ran_at: std::time::SystemTime::now(),
        })
    }
}

/// Translate a nutmeg-binary blob into our `Waveform` IR. Returns `None`
/// when parsing fails or the plot kind isn't real/complex.
///
/// `extra_keep` is an explicit allow-list of signal names that the
/// testbench wants plotted regardless of the heuristic filter (e.g.
/// internal currents like `i(vmeas)` whose `(` makes them non-trivial
/// to spell as a top-level identifier). When empty, the heuristic
/// filter is the only gate.
fn nutmeg_to_waveform(bytes: &[u8], extra_keep: &[String]) -> Option<Waveform> {
    let plot = nutmeg::parse_bytes(bytes).ok()?;
    let mut signals_real = std::collections::BTreeMap::new();
    let mut signals_complex = std::collections::BTreeMap::new();
    let axis_name = plot.var_names.first().cloned().unwrap_or_else(|| "axis".into());
    let keep = |name: &str| -> bool {
        if extra_keep.iter().any(|k| k == name) { return true; }
        is_user_signal(name)
    };
    match plot.flavor {
        nutmeg::NutmegFlavor::Real => {
            let axis = plot.real_trace(&axis_name)?.to_vec();
            let axis_len = axis.len();
            for name in plot.var_names.iter().skip(1) {
                if !keep(name) { continue; }
                if let Some(samples) = plot.real_trace(name) {
                    // Drop length-1 scalars (`.meas` outputs land in
                    // the raw with dims=1) — they'd plot as a single
                    // point against the time axis and add noise.
                    if samples.len() != axis_len { continue; }
                    signals_real.insert(name.clone(), samples.to_vec());
                }
            }
            Some(Waveform::Real { axis_name, axis, signals: signals_real })
        }
        nutmeg::NutmegFlavor::Complex => {
            let axis_complex = plot.complex_trace(&axis_name)?;
            let axis: Vec<f64> = axis_complex.iter().map(|(re, _)| *re).collect();
            let axis_len = axis.len();
            for name in plot.var_names.iter().skip(1) {
                if !keep(name) { continue; }
                if let Some(samples) = plot.complex_trace(name) {
                    if samples.len() != axis_len { continue; }
                    signals_complex.insert(name.clone(), samples.to_vec());
                }
            }
            Some(Waveform::Complex { axis_name, axis, signals: signals_complex })
        }
    }
}

/// Heuristic: a "user-meaningful" signal name is a top-level circuit
/// quantity ngspice wraps as `v(node)` / `i(source)` (or a source's
/// internal `vname#branch` tag), with no hierarchical `.` or
/// BSIM-internal `#` inside the node identifier. Bare identifiers
/// (e.g. `.meas` outputs like `ibn`) are intentionally dropped — they
/// land in the raw as broadcast scalars and would clutter plots.
fn is_user_signal(name: &str) -> bool {
    if name == "time" || name == "frequency" { return true; }
    if let Some(branch) = name.strip_suffix("#branch") {
        return !branch.contains('.') && !branch.contains('#');
    }
    let lower = name.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("v(").or_else(|| lower.strip_prefix("i(")) {
        if let Some(inner) = rest.strip_suffix(')') {
            return !inner.contains('.') && !inner.contains('#');
        }
    }
    false
}

/// Pull `.lib <path> <section>` paths out of the netlist preamble.
fn collect_lib_paths(nl: &Netlist) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for line in &nl.preamble {
        let t = line.trim_start();
        for prefix in [".lib", ".LIB", ".include", ".INCLUDE"] {
            if let Some(rest) = t.strip_prefix(prefix) {
                let mut it = rest.split_whitespace();
                if let Some(p) = it.next() {
                    let p = p.trim_matches(|c: char| c == '"' || c == '\'');
                    out.push(PathBuf::from(p));
                }
            }
        }
    }
    out
}

/// Build the full deck text — preamble + body + `.control` block with
/// the analysis directive, `.meas` lines, and a `write` for the raw
/// file when it's tran/ac.
fn build_deck(
    nl: &Netlist,
    corner: &Corner,
    analysis: &Analysis,
    measurements: &[Measurement],
) -> String {
    use std::fmt::Write as _;
    let mut s = nl.deck();
    let _ = writeln!(s, ".options temp={}", corner.temp_c);
    let _ = writeln!(s, ".control");
    // HSPICE-A compatibility — sky130's montecarlo.spice and several
    // gf180mcu corner files contain HSPICE-flavored constructs (`$`
    // line-end comments, `$letter` escapes inside string params).
    // ngspice 46 only accepts them when this flag is set.
    let _ = writeln!(s, "set ngbehavior=hsa");
    // Reproducible Monte Carlo draws: ngspice's `set rndseed=<n>`
    // anchors the per-run RNG. Without this, two runs of the same
    // mc corner yield different parameters and the SHA cache becomes
    // semantically wrong (same hash, different output).
    if let Some(seed) = corner.seed {
        let _ = writeln!(s, "set rndseed={seed}");
    }

    match analysis {
        Analysis::Op => {
            let _ = writeln!(s, "op");
            for m in measurements {
                // Operating-point "measures" use `let <name> = <expr>; print`.
                // Caller can encode that in body; otherwise default to a
                // node voltage `print`.
                let _ = writeln!(s, "let {} = {}", m.name, m.body);
                let _ = writeln!(s, "print {}", m.name);
            }
        }
        Analysis::Tran { t_step, t_stop, uic } => {
            let uic_s = if *uic { " uic" } else { "" };
            let _ = writeln!(s, "tran {:.10e} {:.10e}{uic_s}", t_step, t_stop);
            for m in measurements {
                let _ = writeln!(s, "{}", m.to_meas_line("tran"));
                let _ = writeln!(s, "print {}", m.name);
            }
        }
        Analysis::AcDec { points_per_decade, f_start, f_stop } => {
            let _ = writeln!(s, "ac dec {} {:.10e} {:.10e}", points_per_decade, f_start, f_stop);
            for m in measurements {
                let _ = writeln!(s, "{}", m.to_meas_line("ac"));
                let _ = writeln!(s, "print {}", m.name);
            }
        }
        Analysis::DcSweep { source, start, stop, step } => {
            let _ = writeln!(s, "dc {} {:.10e} {:.10e} {:.10e}", source, start, stop, step);
            for m in measurements {
                let _ = writeln!(s, "{}", m.to_meas_line("dc"));
                let _ = writeln!(s, "print {}", m.name);
            }
        }
    }

    let _ = writeln!(s, ".endc");
    let _ = writeln!(s, ".end");
    s
}

fn invoke_ngspice(
    bin: &std::path::Path,
    deck: &str,
    analysis: &Analysis,
    raw_path: &std::path::Path,
) -> Result<String, HarnessError> {
    // Inject a `write` line so the raw waveform survives. We do this by
    // replacing `.endc` with `write <path> all\n.endc` only for tran/ac.
    let injected = match analysis {
        Analysis::Tran { .. } | Analysis::AcDec { .. } => deck.replace(
            ".endc",
            &format!("write {} all\n.endc", raw_path.display()),
        ),
        _ => deck.to_string(),
    };

    // Write deck *and* a `.spiceinit` to a temp directory, then run
    // ngspice from there without `-n`. The .spiceinit sets HSPICE-A
    // compatibility BEFORE the deck parse begins — necessary so
    // sky130's montecarlo.spice (and a handful of gf180mcu corners)
    // parse without hitting "Syntax error: letter [$]" on their
    // HSPICE-style `$` escapes.
    let workdir = tempfile::Builder::new().prefix("rlx-eda-run-").tempdir()?;
    let deck_path = workdir.path().join("deck.spice");
    std::fs::write(&deck_path, &injected)?;
    std::fs::write(
        workdir.path().join(".spiceinit"),
        "set ngbehavior=hsa\nset noaskquit\nset nomoremode\n",
    )?;

    let out = Command::new(bin)
        .arg("-b")
        .arg(&deck_path)
        .current_dir(workdir.path())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        let stderr_lines = stderr.lines().count();
        let first_error = extract_first_ngspice_error(&stderr);
        // Surface verbose stderr only when explicitly opted-in;
        // otherwise the Display impl shows just the first error line.
        if std::env::var("RLX_EDA_VERBOSE").is_ok() {
            eprintln!("--- full ngspice stderr ---\n{stderr}\n--- end ---");
        }
        return Err(HarnessError::NgspiceFailed {
            code: out.status.code(),
            stderr,
            stderr_lines,
            first_error,
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
