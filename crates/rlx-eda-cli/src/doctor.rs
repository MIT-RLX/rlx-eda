//! `rlx-eda doctor` — diagnose the local environment.
//!
//! Runs cheap precheck probes and prints a status table. Exits 0 when
//! everything's green; exits 1 if any *required* probe fails. Soft
//! probes (ciel installed but no PDKs registered, etc.) print warnings
//! but don't fail.
//!
//! Probes:
//!   - ngspice on PATH (required for any external simulation).
//!   - ciel / volare on PATH (recommended; needed for `pdk install`).
//!   - registry config readable (always reports the path).
//!   - each registered PDK's lib_path exists + has section headers.
//!   - ciel/volare auto-discovered installs reachable.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::pdk;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("doctor reported {0} required failure(s); fix them and re-run")]
    HasFailures(usize),
    #[error("pdk: {0}")]
    Pdk(#[from] pdk::Error),
}

pub fn run() -> Result<(), Error> {
    let mut report = Report::default();

    probe_binary(&mut report, "ngspice", true);
    probe_binary(&mut report, "ciel", false);
    probe_binary(&mut report, "volare", false);

    let cfg = pdk::Registry::config_path().map_err(Error::Pdk)?;
    report.note("registry", true, format!("config path: {}", cfg.display()));
    if cfg.exists() {
        report.note("registry", true, format!("file present ({} bytes)",
            std::fs::metadata(&cfg).map(|m| m.len()).unwrap_or(0)));
    } else {
        report.note("registry", true, "not yet created (run `rlx-eda pdk install …`)".into());
    }

    let registered = match pdk::Registry::load_or_default() {
        Ok(r) => r.entries,
        Err(e) => {
            report.fail("registry", format!("could not parse: {e}"));
            Vec::new()
        }
    };

    let ciel_scan = match pdk::install::scan_ciel_root() {
        Ok(v) => v,
        Err(e) => {
            report.warn("ciel-scan", format!("scan failed: {e}"));
            Vec::new()
        }
    };

    // Stitch registry + scan: registry is authoritative; ciel scan
    // surfaces installs not yet registered.
    let all: Vec<&pdk::PdkEntry> = registered.iter().collect();
    let already: std::collections::HashSet<&str> = registered.iter().map(|e| e.name.as_str()).collect();
    let mut extra = Vec::new();
    for e in &ciel_scan {
        if !already.contains(e.name.as_str()) { extra.push(e); }
    }

    if all.is_empty() && extra.is_empty() {
        report.warn("pdks", "none registered or discovered. Try `rlx-eda pdk install sky130A`.".into());
    } else {
        for e in &all {
            check_pdk_entry(&mut report, e);
        }
        for e in &extra {
            report.warn(
                &format!("pdk:{}", e.name),
                format!(
                    "ciel-discovered but not registered. Run `rlx-eda pdk install {}` to add it to the registry.",
                    e.name,
                ),
            );
        }
    }

    println!("{}", report.render());
    if report.failures > 0 {
        return Err(Error::HasFailures(report.failures));
    }
    Ok(())
}

fn probe_binary(report: &mut Report, name: &str, required: bool) {
    match which(name) {
        Some(p) => {
            // Try to grab a version string for diagnostic value.
            let ver = Command::new(&p).arg("--version").output().ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.lines().next().unwrap_or("").trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "(no --version)".into());
            report.ok(name, format!("{} ({})", p.display(), ver));
        }
        None => {
            if required {
                report.fail(name, "not found on PATH".into());
            } else {
                report.warn(name, "not on PATH (optional — only needed for `pdk install`)".into());
            }
        }
    }
}

fn check_pdk_entry(report: &mut Report, e: &pdk::PdkEntry) {
    let probe = format!("pdk:{}", e.name);
    if !e.lib_path.exists() {
        report.fail(&probe, format!("lib_path does not exist: {}", e.lib_path.display()));
        return;
    }
    let bytes = match std::fs::metadata(&e.lib_path) {
        Ok(m) => m.len(),
        Err(err) => { report.fail(&probe, format!("stat error: {err}")); return; }
    };
    if e.sections.is_empty() {
        report.warn(&probe, format!(
            "{} sections registered (lib OK, {} bytes — register --sections to add corner labels)",
            e.sections.len(), bytes,
        ));
    } else {
        report.ok(&probe, format!(
            "{} sections, {} kB at {}",
            e.sections.len(), bytes / 1024, e.lib_path.display(),
        ));
    }
}

fn which(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|p| {
        std::env::split_paths(&p).find_map(|d| {
            let c = d.join(name);
            if c.is_file() { Some(c) } else { None }
        })
    })
}

#[derive(Default)]
struct Report {
    rows: Vec<(Status, String, String)>,
    failures: usize,
}

#[derive(Clone, Copy)]
enum Status { Ok, Warn, Fail }

impl Report {
    fn ok(&mut self, probe: impl Into<String>, msg: String) {
        self.rows.push((Status::Ok, probe.into(), msg));
    }
    fn warn(&mut self, probe: impl Into<String>, msg: String) {
        self.rows.push((Status::Warn, probe.into(), msg));
    }
    fn fail(&mut self, probe: impl Into<String>, msg: String) {
        self.rows.push((Status::Fail, probe.into(), msg));
        self.failures += 1;
    }
    fn note(&mut self, probe: impl Into<String>, ok: bool, msg: String) {
        if ok { self.ok(probe, msg) } else { self.warn(probe, msg) }
    }

    fn render(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "rlx-eda doctor");
        let _ = writeln!(s, "──────────────");
        for (st, probe, msg) in &self.rows {
            let mark = match st {
                Status::Ok => "✓",
                Status::Warn => "!",
                Status::Fail => "✗",
            };
            let _ = writeln!(s, " {mark} {probe:<24}  {msg}");
        }
        let _ = writeln!(s);
        let total = self.rows.len();
        let oks = self.rows.iter().filter(|(s, _, _)| matches!(s, Status::Ok)).count();
        let warns = self.rows.iter().filter(|(s, _, _)| matches!(s, Status::Warn)).count();
        let _ = writeln!(s, " summary: {oks}/{total} ok, {warns} warning(s), {} failure(s)", self.failures);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_render_groups_status() {
        let mut r = Report::default();
        r.ok("ngspice", "/usr/bin/ngspice (ngspice-46)".into());
        r.warn("xschem", "not on PATH".into());
        r.fail("pdk:bad", "lib_path missing".into());
        let s = r.render();
        assert!(s.contains("✓ ngspice"));
        assert!(s.contains("! xschem"));
        assert!(s.contains("✗ pdk:bad"));
        assert!(s.contains("3/3 ok") == false); // 1/3 ok
        assert!(s.contains("1/3 ok"));
    }
}
