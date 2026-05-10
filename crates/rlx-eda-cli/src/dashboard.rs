//! `rlx-eda dashboard` — cross-test regression page.
//!
//! Walks `<root>/crates/*/docs/*/` for testbench-summary outputs, reads
//! the cicsim-shaped CSVs we emit, and writes a top-level
//! `<root>/docs/index.html` with a card per (crate, run) showing
//! pass/fail counts, last-run timestamp, and a link to the per-test
//! HTML summary.
//!
//! This is the regression-monitoring view: one URL to bookmark and
//! glance at after every commit.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("no testbench summaries found under {0}")]
    Empty(PathBuf),
}

#[derive(Debug, Clone)]
struct Run {
    crate_name: String,
    run_name: String,
    summary_path: PathBuf,    // absolute
    summary_rel: String,      // relative to dashboard's index.html
    csvs: Vec<CsvBucket>,
    latest_run: Option<SystemTime>,
}

#[derive(Debug, Clone)]
struct CsvBucket {
    /// e.g. "Sch_typical", "Lay_etc", "Sch_mc"
    bucket: String,
    n_total: usize,
    n_pass: usize,
    n_fail: usize,
}

pub fn run(root: Option<PathBuf>) -> Result<(), Error> {
    let root = root.unwrap_or_else(|| std::env::current_dir().unwrap());
    let crates_dir = root.join("crates");
    let dashboard_dir = root.join("docs");
    let index_path = dashboard_dir.join("index.html");

    let runs = discover_runs(&crates_dir, &dashboard_dir)?;
    if runs.is_empty() {
        return Err(Error::Empty(crates_dir));
    }

    std::fs::create_dir_all(&dashboard_dir)?;
    let html = render_dashboard(&runs);
    std::fs::write(&index_path, html)?;

    println!(
        "rlx-eda dashboard: {} testbench run(s) at {}",
        runs.len(), index_path.display(),
    );
    Ok(())
}

fn discover_runs(crates_dir: &Path, dashboard_dir: &Path) -> Result<Vec<Run>, Error> {
    let mut out = Vec::new();
    let crates = match std::fs::read_dir(crates_dir) {
        Ok(d) => d,
        Err(_) => return Ok(out),
    };
    for crate_entry in crates.flatten() {
        let crate_path = crate_entry.path();
        if !crate_path.is_dir() { continue; }
        let crate_name = crate_path.file_name().unwrap().to_string_lossy().into_owned();
        let docs_dir = crate_path.join("docs");
        let runs_iter = match std::fs::read_dir(&docs_dir) {
            Ok(d) => d,
            Err(_) => continue,
        };
        for run_entry in runs_iter.flatten() {
            let run_path = run_entry.path();
            if !run_path.is_dir() { continue; }
            // Look for any `*_summary.html` in the run dir.
            let Some(summary) = find_summary(&run_path) else { continue };
            let csvs = scan_csvs(&run_path);
            let latest = newest_mtime(&run_path);
            let rel = pathdiff_relative(&summary, dashboard_dir)
                .unwrap_or_else(|| summary.display().to_string());
            out.push(Run {
                crate_name: crate_name.clone(),
                run_name: run_path.file_name().unwrap().to_string_lossy().into_owned(),
                summary_path: summary,
                summary_rel: rel,
                csvs,
                latest_run: latest,
            });
        }
    }
    out.sort_by(|a, b| a.crate_name.cmp(&b.crate_name).then(a.run_name.cmp(&b.run_name)));
    Ok(out)
}

fn find_summary(run_path: &Path) -> Option<PathBuf> {
    for entry in std::fs::read_dir(run_path).ok()?.flatten() {
        let p = entry.path();
        if let Some(n) = p.file_name().and_then(|s| s.to_str()) {
            if n.ends_with("_summary.html") { return Some(p); }
        }
    }
    None
}

fn scan_csvs(run_path: &Path) -> Vec<CsvBucket> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(run_path) else { return out };
    for entry in entries.flatten() {
        let p = entry.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else { continue };
        if !name.ends_with(".csv") { continue; }
        // Filename pattern: `<tb>_<View>_<kind>.csv`. Bucket = view_kind.
        let stem = name.trim_end_matches(".csv");
        let parts: Vec<&str> = stem.rsplitn(3, '_').collect();
        if parts.len() < 3 { continue; }
        let bucket = format!("{}_{}", parts[1], parts[0]);
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        let Some(stats) = parse_ok_column(&text) else { continue };
        out.push(CsvBucket {
            bucket, n_total: stats.0, n_pass: stats.1, n_fail: stats.2,
        });
    }
    out.sort_by(|a, b| a.bucket.cmp(&b.bucket));
    out
}

/// Returns `(total, pass, fail)` from the CSV's `OK` column. Header
/// gives column order; we look for `OK` (cicsim) or `OK\n` end-of-line.
fn parse_ok_column(csv: &str) -> Option<(usize, usize, usize)> {
    let mut lines = csv.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split(',').collect();
    let ok_idx = cols.iter().position(|&c| c == "OK")?;
    let mut total = 0usize;
    let mut pass = 0usize;
    let mut fail = 0usize;
    for line in lines {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() <= ok_idx { continue; }
        total += 1;
        match cells[ok_idx].trim() {
            "True" => pass += 1,
            "False" => fail += 1,
            _ => {}
        }
    }
    Some((total, pass, fail))
}

fn newest_mtime(dir: &Path) -> Option<SystemTime> {
    let mut newest = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(t) = meta.modified() else { continue };
        if newest.map_or(true, |n: SystemTime| t > n) {
            newest = Some(t);
        }
    }
    newest
}

/// Compute a path that walks from `target_dir` to `from_path`. Both
/// must be absolute or share a common prefix; we don't canonicalize
/// (target_dir may not exist yet when the dashboard is the first
/// thing we write).
fn pathdiff_relative(from_path: &Path, target_dir: &Path) -> Option<String> {
    let from_components: Vec<_> = from_path.components().collect();
    let target_components: Vec<_> = target_dir.components().collect();
    let common = from_components.iter().zip(target_components.iter())
        .take_while(|(a, b)| a == b).count();
    let ups = target_components.len().saturating_sub(common);
    let mut s = String::new();
    for _ in 0..ups { s.push_str("../"); }
    for c in &from_components[common..] {
        s.push_str(&c.as_os_str().to_string_lossy());
        s.push('/');
    }
    s.pop();
    Some(s)
}

fn fmt_age(t: SystemTime) -> String {
    match t.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => {
            let secs_since_epoch = d.as_secs();
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs()).unwrap_or(secs_since_epoch);
            let age = now.saturating_sub(secs_since_epoch);
            if age < 60 { format!("{age}s ago") }
            else if age < 3600 { format!("{}m ago", age / 60) }
            else if age < 86400 { format!("{}h ago", age / 3600) }
            else { format!("{}d ago", age / 86400) }
        }
        Err(_) => "?".into(),
    }
}

fn render_dashboard(runs: &[Run]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    s.push_str("<!doctype html><meta charset=utf-8><title>rlx-eda dashboard</title>");
    s.push_str(STYLE);
    s.push_str("<h1>rlx-eda — testbench dashboard</h1>");

    // Top-line: total testbenches, total runs, total pass/fail.
    let n_runs = runs.len();
    let total_total: usize = runs.iter().flat_map(|r| r.csvs.iter()).map(|b| b.n_total).sum();
    let total_pass: usize = runs.iter().flat_map(|r| r.csvs.iter()).map(|b| b.n_pass).sum();
    let total_fail: usize = runs.iter().flat_map(|r| r.csvs.iter()).map(|b| b.n_fail).sum();
    let banner_cls = if total_fail == 0 { "ok" } else { "bad" };
    let banner_text = if total_fail == 0 {
        format!("ALL {total_total} CORNERS PASS")
    } else {
        format!("{total_fail} OF {total_total} CORNERS FAILING")
    };
    let _ = writeln!(s, "<div class=\"banner banner-{banner_cls}\">{banner_text}</div>");
    let _ = writeln!(s, "<p class=meta>{n_runs} testbench run(s) · {total_pass}/{total_total} passing</p>");

    // Group by crate.
    let mut current_crate: Option<&str> = None;
    s.push_str("<div class=grid>");
    for r in runs {
        if Some(r.crate_name.as_str()) != current_crate {
            if current_crate.is_some() { s.push_str("</section>"); }
            let _ = writeln!(s, "</div><h2>{}</h2><div class=grid>", html_escape(&r.crate_name));
            current_crate = Some(r.crate_name.as_str());
        }
        let card_cls = if r.csvs.iter().any(|b| b.n_fail > 0) { "fail" } else { "pass" };
        let _ = writeln!(s, "<a class=\"card card-{card_cls}\" href=\"{href}\">",
            href = html_escape(&r.summary_rel));
        let _ = writeln!(s, "<div class=card-head><strong>{}</strong></div>",
            html_escape(&r.run_name));
        if !r.csvs.is_empty() {
            s.push_str("<div class=buckets>");
            for b in &r.csvs {
                let bcls = if b.n_fail > 0 { "fail" } else { "pass" };
                let _ = writeln!(s, "<span class=\"bucket bucket-{bcls}\">{} {}/{}</span>",
                    html_escape(&b.bucket), b.n_pass, b.n_total);
            }
            s.push_str("</div>");
        } else {
            s.push_str("<div class=meta-mini>(no CSVs found)</div>");
        }
        if let Some(t) = r.latest_run {
            let _ = writeln!(s, "<div class=meta-mini>updated {}</div>", fmt_age(t));
        }
        s.push_str("</a>");
    }
    s.push_str("</div>");
    s
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

const STYLE: &str = "<style>
:root{--ok:#22863a;--bad:#b31d28;--mute:#586069}
body{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1.5em;color:#24292e}
h1{font-size:1.7em;margin:.2em 0}
h2{margin:1.6em 0 .5em;font-size:1.05em;color:var(--mute);font-weight:500;letter-spacing:.04em;text-transform:uppercase;border-bottom:1px solid #eee;padding-bottom:.3em}
.meta{color:var(--mute);font-size:90%}
.meta-mini{color:var(--mute);font-size:78%;margin-top:.3em}
.banner{display:inline-block;padding:.4em 1em;border-radius:.4em;font-weight:700;letter-spacing:.05em;font-size:.95em;margin:.5em 0}
.banner-ok{background:#dcffe4;color:var(--ok)}
.banner-bad{background:#ffd6d8;color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.7em;margin-bottom:1.5em}
.card{display:block;border:1px solid #e1e4e8;border-radius:.4em;padding:.7em .9em;background:#fff;text-decoration:none;color:inherit;transition:box-shadow .1s}
.card:hover{box-shadow:0 1px 4px rgba(0,0,0,.1)}
.card-pass{border-left:4px solid var(--ok)}
.card-fail{border-left:4px solid var(--bad);background:#fff7f8}
.card-head{margin-bottom:.5em}
.buckets{display:flex;flex-wrap:wrap;gap:.3em}
.bucket{font-size:79%;padding:.15em .55em;border-radius:1em;font-family:ui-monospace,Menlo,Consolas,monospace}
.bucket-pass{background:#e6ffed;color:var(--ok)}
.bucket-fail{background:#ffeef0;color:var(--bad);font-weight:600}
</style>";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ok_column_counts_pass_fail() {
        let csv = ",ibn,vgs_m1,name,type,time,OK\n0,1.2,0.6,typ,Sch,42.0,True\n1,2.4,0.7,ff,Sch,43.0,False\n2,1.8,0.65,ss,Sch,44.0,True\n";
        let (total, pass, fail) = parse_ok_column(csv).unwrap();
        assert_eq!(total, 3);
        assert_eq!(pass, 2);
        assert_eq!(fail, 1);
    }

    #[test]
    fn parses_ok_column_returns_none_when_missing() {
        let csv = ",x,y,name,type\n0,1,2,a,Sch\n";
        assert!(parse_ok_column(csv).is_none());
    }

    #[test]
    fn fmt_age_buckets_correctly() {
        let now = SystemTime::now();
        assert!(fmt_age(now).ends_with("s ago"));
        let yesterday = SystemTime::now() - std::time::Duration::from_secs(86400 * 2);
        assert!(fmt_age(yesterday).ends_with("d ago"));
    }

    #[test]
    fn dashboard_renders_when_no_runs_panic_safely() {
        let html = render_dashboard(&[]);
        assert!(html.contains("rlx-eda — testbench dashboard"));
        // Banner should still render even with 0 runs (everything passes vacuously).
        assert!(html.contains("ALL 0 CORNERS PASS"));
    }

    #[test]
    fn dashboard_groups_runs_by_crate() {
        let runs = vec![
            Run {
                crate_name: "spike-a".into(),
                run_name: "tt_ff_ss".into(),
                summary_path: PathBuf::from("/x/spike-a/docs/tt_ff_ss/summary.html"),
                summary_rel: "../crates/spike-a/docs/tt_ff_ss/summary.html".into(),
                csvs: vec![CsvBucket { bucket: "Sch_typical".into(), n_total: 1, n_pass: 1, n_fail: 0 }],
                latest_run: Some(SystemTime::now()),
            },
            Run {
                crate_name: "spike-b".into(),
                run_name: "mc".into(),
                summary_path: PathBuf::from("/x/spike-b/docs/mc/summary.html"),
                summary_rel: "../crates/spike-b/docs/mc/summary.html".into(),
                csvs: vec![CsvBucket { bucket: "Sch_mc".into(), n_total: 8, n_pass: 7, n_fail: 1 }],
                latest_run: Some(SystemTime::now()),
            },
        ];
        let html = render_dashboard(&runs);
        assert!(html.contains("spike-a"));
        assert!(html.contains("spike-b"));
        assert!(html.contains("Sch_typical 1/1"));
        assert!(html.contains("Sch_mc 7/8"));
        // Failures get the bad banner.
        assert!(html.contains("1 OF 9 CORNERS FAILING"));
    }
}
