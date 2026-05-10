//! `run-all-traces` — workspace-wide orchestrator that discovers
//! every spike crate's report-emitting bin and runs each in turn,
//! letting the bin write its own artifacts to its own
//! `crates/<spike>/docs/` (the convention every existing trace bin
//! already follows).
//!
//! Discovery is filesystem-based: walks `crates/*/Cargo.toml`, reads
//! `[package].name`, lists `[[bin]]` entries plus anything in
//! `src/bin/*.rs`. By default the runner keeps only **report-emitting**
//! bins — names matching `*trace*`, `*characterization*`, `*_opt*`,
//! `*_match*`, `*_demo*`, `*_ml*`, `*sizing*`. Pass `--all` to
//! disable the auto-filter, or `--filter <substr>` to override it
//! explicitly.
//!
//! Each surviving bin is launched as
//! `cargo run --quiet -p <pkg> --bin <bin>` with the workspace root
//! as cwd. After every real run the orchestrator writes a markdown
//! summary to `crates/eda-trace/docs/run-summary.md` with per-bin
//! status, duration, and link to the artifact each bin emitted.
//!
//! Run:
//!
//! ```text
//!   cargo run -p eda-trace --bin run-all-traces -- --list
//!   cargo run -p eda-trace --bin run-all-traces -- --dry-run
//!   cargo run -p eda-trace --bin run-all-traces -- --filter trace --exclude mzi_ml
//!   cargo run -p eda-trace --bin run-all-traces -- --all --release
//! ```

use std::env;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

// ── Auto-filter heuristics ────────────────────────────────────────
//
// Bins matching any of these substrings are treated as
// "report-emitting" by default. Comprehensive enough to catch every
// existing spike's trace / opt / characterization / demo bin
// without dragging in the catch-all package binaries that just
// print debug noise.
const REPORT_PATTERNS: &[&str] = &[
    "trace", "characterization", "_opt", "_match", "_demo", "_ml", "sizing",
];

#[derive(Debug)]
struct Args {
    /// Explicit filter — overrides the report-pattern auto-filter.
    /// `Some("")` means "no substring filter", i.e. keep everything.
    filter: Option<String>,
    excludes: Vec<String>,
    release: bool,
    dry_run: bool,
    list_only: bool,
    all: bool,
    workspace_root: Option<PathBuf>,
}

impl Args {
    fn parse() -> Self {
        let mut filter: Option<String> = None;
        let mut excludes = Vec::new();
        let mut release = false;
        let mut dry_run = false;
        let mut list_only = false;
        let mut all = false;
        let mut workspace_root = None;

        let mut it = env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--filter" => filter = Some(it.next().unwrap_or_default()),
                "--exclude" => {
                    if let Some(v) = it.next() { excludes.push(v); }
                }
                "--release" => release = true,
                "--dry-run" => dry_run = true,
                "--list" => list_only = true,
                "--all" => all = true,
                "--workspace-root" => workspace_root = it.next().map(PathBuf::from),
                "--help" | "-h" => { print_help(); std::process::exit(0); }
                other => {
                    eprintln!("unknown argument: {other}");
                    print_help();
                    std::process::exit(2);
                }
            }
        }
        Self { filter, excludes, release, dry_run, list_only, all, workspace_root }
    }
}

fn print_help() {
    let p = Painter::detect();
    let bold = |s: &str| p.bold(s).to_string();
    eprintln!(
"{title} — discover and run every spike crate's report bin

{usage}
  cargo run -p eda-trace --bin run-all-traces -- [flags]

{flags}
  --list                  list discovered bins grouped by crate, then exit
  --dry-run               group + label what would run, then exit
  --filter <substring>    keep bins whose name contains this (overrides auto-filter)
  --exclude <substring>   skip bins matching this substring (repeatable)
  --all                   disable the report-pattern auto-filter
  --release               cargo --release (faster bins, slower compile)
  --workspace-root <dir>  override autodetected root
  -h, --help              this help

{examples}
  # default — auto-detect report bins, run them all
  cargo run -p eda-trace --bin run-all-traces

  # list everything that would run, grouped by crate, without running
  cargo run -p eda-trace --bin run-all-traces -- --dry-run

  # only the eda-trace-driven *_match_trace bins (lna + mzi)
  cargo run -p eda-trace --bin run-all-traces -- --filter _match_trace

  # everything except the slow MZI literature-validation run
  cargo run -p eda-trace --bin run-all-traces -- --exclude mzi_ml

  # exhaustive — every discoverable bin in the workspace
  cargo run -p eda-trace --bin run-all-traces -- --all

After every real run the orchestrator writes
crates/eda-trace/docs/run-summary.md with per-bin status, duration,
and links to each emitted artifact.",
        title = bold("run-all-traces"),
        usage = bold("Usage:"),
        flags = bold("Flags:"),
        examples = bold("Examples:"),
    );
}

#[derive(Debug, Clone)]
struct BinTarget {
    package: String,
    bin_name: String,
}

impl BinTarget {
    fn matches_any(&self, patterns: &[&str]) -> bool {
        patterns.iter().any(|p| self.bin_name.contains(p))
    }
}

fn main() {
    let args = Args::parse();
    let p = Painter::detect();

    let root = match args.workspace_root.clone() {
        Some(p) => p,
        None => find_workspace_root().unwrap_or_else(|| {
            eprintln!("could not locate workspace root from cwd");
            std::process::exit(2);
        }),
    };
    println!("{} {}", p.dim("workspace:"), root.display());

    let bins = discover_bins(&root);

    // ── Apply filter ──────────────────────────────────────────────
    // Precedence: --all > --filter (explicit) > REPORT_PATTERNS.
    // Self-exclusion (`run-all-traces`) is always applied so we
    // never recurse.
    let kept: Vec<&BinTarget> = bins
        .iter()
        .filter(|b| !(b.package == "eda-trace" && b.bin_name == "run-all-traces"))
        .filter(|b| {
            if args.all {
                true
            } else if let Some(f) = &args.filter {
                f.is_empty() || b.bin_name.contains(f)
            } else {
                b.matches_any(REPORT_PATTERNS)
            }
        })
        .filter(|b| !args.excludes.iter().any(|e| b.bin_name.contains(e)))
        .collect();

    let mode_desc = if args.all {
        "all".to_string()
    } else if let Some(f) = &args.filter {
        if f.is_empty() { "no filter".to_string() } else { format!("filter {f:?}") }
    } else {
        "auto (report patterns)".to_string()
    };

    if args.list_only {
        print_grouped_listing(&bins, &p, "all discovered bins", true);
        return;
    }

    // ── Dry-run ───────────────────────────────────────────────────
    if args.dry_run {
        println!(
            "\n{} discovered {} bins; {} pass {} (excludes {:?})\n",
            p.bold("→"), bins.len(), kept.len(), mode_desc, args.excludes,
        );
        print_grouped_listing(&kept_owned(&kept), &p, "would run", false);
        return;
    }

    if kept.is_empty() {
        eprintln!(
            "{} no bins matched. try `--list` to see all discovered, or `--all` to run them.",
            p.warn("warning:"),
        );
        std::process::exit(0);
    }

    // ── Pre-run banner ────────────────────────────────────────────
    let crates: std::collections::BTreeSet<&str> =
        kept.iter().map(|b| b.package.as_str()).collect();
    println!(
        "\n{} running {} bins from {} crates  ({})",
        p.bold("▶"),
        kept.len(),
        crates.len(),
        mode_desc,
    );
    if args.release {
        println!("  {}", p.dim("(--release: longer compile, faster bins)"));
    }
    println!();

    // ── Run each bin ──────────────────────────────────────────────
    let total_start = Instant::now();
    let mut results: Vec<RunResult> = Vec::new();
    let label_width = kept
        .iter()
        .map(|b| format!("{}::{}", b.package, b.bin_name).len())
        .max()
        .unwrap_or(0);

    for (i, b) in kept.iter().enumerate() {
        let label = format!(
            "[{:>2}/{}] {:<width$}",
            i + 1,
            kept.len(),
            format!("{}::{}", b.package, b.bin_name),
            width = label_width,
        );
        print!("  {label} ");
        use std::io::Write;
        let _ = std::io::stdout().flush();

        let start = Instant::now();
        let mut cmd = Command::new("cargo");
        cmd.current_dir(&root)
            .arg("run")
            .arg("--quiet")
            .arg("-p").arg(&b.package)
            .arg("--bin").arg(&b.bin_name);
        if args.release { cmd.arg("--release"); }
        let output = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).output();
        let elapsed = start.elapsed();

        match output {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let wrote = stdout
                    .lines()
                    .find(|l| l.contains("wrote:"))
                    .map(|l| l.trim().to_string());
                let artifact = wrote.as_deref().and_then(parse_artifact);
                println!(
                    "{} {}",
                    p.ok("ok"),
                    p.dim(&format!("({:.1}s)", elapsed.as_secs_f64())),
                );
                if let Some(a) = &artifact {
                    println!("           {} {}", p.dim("→"), p.cyan(a));
                }
                results.push(RunResult {
                    bin: (*b).clone(),
                    status: RunStatus::Ok,
                    elapsed,
                    artifact,
                    stderr_tail: Vec::new(),
                });
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let tail = stderr
                    .lines()
                    .rev()
                    .take(20)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                println!(
                    "{} {}",
                    p.fail("FAIL"),
                    p.dim(&format!("({:.1}s)", elapsed.as_secs_f64())),
                );
                for line in tail.iter().rev().take(3).rev() {
                    println!("           {} {}", p.dim("│"), line);
                }
                results.push(RunResult {
                    bin: (*b).clone(),
                    status: RunStatus::Fail,
                    elapsed,
                    artifact: None,
                    stderr_tail: tail,
                });
            }
            Err(e) => {
                println!("{} ({:.1}s): {e}", p.fail("ERROR"), elapsed.as_secs_f64());
                results.push(RunResult {
                    bin: (*b).clone(),
                    status: RunStatus::Fail,
                    elapsed,
                    artifact: None,
                    stderr_tail: vec![e.to_string()],
                });
            }
        }
    }

    let total_elapsed = total_start.elapsed();

    // ── Summary ───────────────────────────────────────────────────
    let ok = results.iter().filter(|r| r.status == RunStatus::Ok).count();
    let fail = results.len() - ok;
    println!(
        "\n{} {} ok, {} failed across {} bins  {}",
        p.bold("∎"),
        p.ok(&ok.to_string()),
        if fail > 0 { p.fail(&fail.to_string()) } else { p.dim(&fail.to_string()) },
        results.len(),
        p.dim(&format!("(total {:.1}s)", total_elapsed.as_secs_f64())),
    );

    // Always write a markdown summary so the run is itself a
    // browsable artifact alongside each spike's own report.
    if let Err(e) = write_summary(&root, &results, total_elapsed, &mode_desc) {
        eprintln!("{} failed to write summary: {e}", p.warn("warning:"));
    } else {
        println!(
            "  {} {}",
            p.dim("→ summary:"),
            p.cyan("crates/eda-trace/docs/run-summary.md"),
        );
    }

    if fail > 0 { std::process::exit(1); }
}

fn kept_owned(kept: &[&BinTarget]) -> Vec<BinTarget> {
    kept.iter().map(|b| (*b).clone()).collect()
}

/// Print bins grouped by crate. Adds a `[report]` tag on bins that
/// match a [`REPORT_PATTERNS`] entry — useful in `--list` to see at
/// a glance which bins the auto-filter would keep.
fn print_grouped_listing(bins: &[BinTarget], p: &Painter, header: &str, tag_reports: bool) {
    use std::collections::BTreeMap;
    let mut by_crate: BTreeMap<String, Vec<&BinTarget>> = BTreeMap::new();
    for b in bins {
        by_crate.entry(b.package.clone()).or_default().push(b);
    }
    println!("\n{} ({} bins, {} crates)\n", p.bold(header), bins.len(), by_crate.len());
    for (pkg, items) in &by_crate {
        println!("  {} {}", p.bold(pkg), p.dim(&format!("({} bins)", items.len())));
        for b in items {
            let tag = if tag_reports && b.matches_any(REPORT_PATTERNS) {
                p.cyan(" [report]").to_string()
            } else {
                String::new()
            };
            println!("    • {}{}", b.bin_name, tag);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum RunStatus { Ok, Fail }

#[derive(Debug, Clone)]
struct RunResult {
    bin: BinTarget,
    status: RunStatus,
    elapsed: std::time::Duration,
    /// Path the bin reported writing on its `wrote: ...` line.
    artifact: Option<String>,
    stderr_tail: Vec<String>,
}

/// Pulls the path out of a `wrote: <path> ...` line. Bins typically
/// emit `wrote: /abs/path/to/foo.md (+ csv + assets/...)` — we keep
/// just the first whitespace-delimited token after `wrote:`.
fn parse_artifact(line: &str) -> Option<String> {
    let after = line.split("wrote:").nth(1)?.trim();
    let path = after.split_whitespace().next()?;
    if path.is_empty() { None } else { Some(path.to_string()) }
}

// ── Markdown summary ──────────────────────────────────────────────────

fn write_summary(
    root: &Path,
    results: &[RunResult],
    total: std::time::Duration,
    mode_desc: &str,
) -> std::io::Result<()> {
    let docs = root.join("crates").join("eda-trace").join("docs");
    std::fs::create_dir_all(&docs)?;
    let path = docs.join("run-summary.md");

    let ok = results.iter().filter(|r| r.status == RunStatus::Ok).count();
    let fail = results.len() - ok;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut md = String::new();
    md.push_str("# `run-all-traces` — last run summary\n\n");
    md.push_str(&format!(
        "Mode: `{mode_desc}`  ·  Total: **{:.1}s**  ·  \
         Outcome: **{ok}** ok, **{fail}** failed  ·  \
         Unix time: `{now}`\n\n\
         _Regenerated every time `cargo run -p eda-trace --bin run-all-traces` is invoked._\n\n",
        total.as_secs_f64(),
    ));

    md.push_str("| Status | Crate | Bin | Duration | Artifact |\n");
    md.push_str("| :---: | --- | --- | ---: | --- |\n");
    for r in results {
        let s = match r.status {
            RunStatus::Ok => "✓",
            RunStatus::Fail => "✗",
        };
        let art = match &r.artifact {
            Some(a) => {
                // Try to render a relative-path link for the markdown.
                let rel = relative_to(root, Path::new(a)).unwrap_or_else(|| a.clone());
                format!("[`{rel}`](../../../{rel})")
            }
            None => "—".to_string(),
        };
        md.push_str(&format!(
            "| {} | `{}` | `{}` | {:.1}s | {} |\n",
            s, r.bin.package, r.bin.bin_name, r.elapsed.as_secs_f64(), art,
        ));
    }

    let failed: Vec<_> = results.iter().filter(|r| r.status == RunStatus::Fail).collect();
    if !failed.is_empty() {
        md.push_str("\n## Failures\n\n");
        for r in failed {
            md.push_str(&format!("### `{}::{}`\n\n", r.bin.package, r.bin.bin_name));
            md.push_str("```\n");
            for line in &r.stderr_tail {
                md.push_str(line);
                md.push('\n');
            }
            md.push_str("```\n\n");
        }
    }

    std::fs::write(&path, md)?;
    Ok(())
}

fn relative_to(root: &Path, abs: &Path) -> Option<String> {
    abs.strip_prefix(root).ok().map(|p| p.to_string_lossy().to_string())
}

// ── ANSI painter (TTY-aware) ──────────────────────────────────────────

struct Painter { color: bool }

impl Painter {
    fn detect() -> Self {
        // Honor NO_COLOR and TERM=dumb for accessibility, otherwise
        // colorize when stdout is a terminal.
        let no_color = env::var_os("NO_COLOR").is_some()
            || env::var("TERM").map(|t| t == "dumb").unwrap_or(false);
        Self { color: !no_color && std::io::stdout().is_terminal() }
    }
    fn paint(&self, code: &str, s: &str) -> String {
        if self.color { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
    }
    fn ok(&self, s: &str)   -> String { self.paint("32", s) }      // green
    fn fail(&self, s: &str) -> String { self.paint("31;1", s) }    // bold red
    fn warn(&self, s: &str) -> String { self.paint("33", s) }      // yellow
    fn dim(&self, s: &str)  -> String { self.paint("2", s) }       // dim
    fn bold(&self, s: &str) -> String { self.paint("1", s) }       // bold
    fn cyan(&self, s: &str) -> String { self.paint("36", s) }      // cyan
}

// ── Workspace + bin discovery ─────────────────────────────────────────

fn find_workspace_root() -> Option<PathBuf> {
    let mut cur = env::current_dir().ok()?;
    loop {
        let cargo = cur.join("Cargo.toml");
        if cargo.exists() {
            if let Ok(s) = std::fs::read_to_string(&cargo) {
                if s.contains("[workspace]") { return Some(cur); }
            }
        }
        if !cur.pop() { return None; }
    }
}

fn discover_bins(root: &Path) -> Vec<BinTarget> {
    let mut out = Vec::new();
    let crates_dir = root.join("crates");
    let entries = match std::fs::read_dir(&crates_dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    let mut crate_dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("Cargo.toml").exists())
        .collect();
    crate_dirs.sort();

    for cdir in crate_dirs {
        let manifest = match std::fs::read_to_string(cdir.join("Cargo.toml")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let parsed: toml::Value = match manifest.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let pkg_name = parsed
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        if pkg_name.is_empty() { continue; }

        let mut explicit = Vec::new();
        if let Some(bins) = parsed.get("bin").and_then(|b| b.as_array()) {
            for entry in bins {
                if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                    explicit.push(name.to_string());
                    out.push(BinTarget {
                        package: pkg_name.clone(),
                        bin_name: name.to_string(),
                    });
                }
            }
        }

        if let Ok(rd) = std::fs::read_dir(cdir.join("src").join("bin")) {
            for entry in rd.filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) != Some("rs") { continue; }
                let name = match p.file_stem().and_then(|s| s.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                if explicit.contains(&name) { continue; }
                out.push(BinTarget { package: pkg_name.clone(), bin_name: name });
            }
        }

        if cdir.join("src").join("main.rs").exists() && !explicit.contains(&pkg_name) {
            out.push(BinTarget { package: pkg_name.clone(), bin_name: pkg_name.clone() });
        }
    }
    out
}
