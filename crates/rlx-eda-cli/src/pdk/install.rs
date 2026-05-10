//! `pdk install` — wrap ciel for the download, then auto-register.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::discover;
use super::known;
use super::registry::{PdkEntry, Registry, Source};
use super::{Error, InstallArgs};

pub fn run(args: InstallArgs) -> Result<(), Error> {
    let kp = known::lookup(&args.name).ok_or_else(|| Error::UnknownVariant(args.name.clone()))?;

    let bin = resolve_ciel_bin(args.ciel_bin.as_deref())?;
    let version = match args.version.as_deref() {
        Some(v) => v.to_string(),
        None => latest_remote_version(&bin, kp.ciel_family)?,
    };

    println!(
        "rlx-eda: installing {} via ciel (family={}, version={}…)",
        kp.variant, kp.ciel_family, &version[..version.len().min(12)],
    );
    enable(&bin, kp.ciel_family, &version)?;

    let lib_path = locate_variant_lib(&bin, kp.ciel_family, kp.variant, kp.lib_subpath)
        .ok_or_else(|| Error::Discover(
            PathBuf::from(format!("<pdk_root>/{}/.../{}", kp.variant, kp.lib_subpath)),
            std::io::Error::new(std::io::ErrorKind::NotFound, "could not locate lib in any ciel version dir"),
        ))?;
    let sections = discover::sections_from_lib(&lib_path)
        .map_err(|e| Error::Discover(lib_path.clone(), e))?;

    let entry = PdkEntry {
        name: kp.variant.to_string(),
        lib_path: lib_path.clone(),
        sections: sections.clone(),
        vdd_nom: kp.vdd_nom,
        source: Source::Ciel,
    };
    let mut reg = Registry::load_or_default()?;
    reg.upsert(entry);
    reg.save()?;

    let sections_summary = if sections.is_empty() {
        "(none auto-detected — register --sections to add corner labels)".to_string()
    } else {
        let head = sections.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
        let tail = if sections.len() > 8 { ", …" } else { "" };
        format!("{head}{tail} ({} corners)", sections.len())
    };
    println!(
        "rlx-eda: registered {}\n  lib:      {}\n  sections: {}\n  vdd_nom:  {} V",
        kp.variant,
        lib_path.display(),
        sections_summary,
        kp.vdd_nom,
    );
    Ok(())
}

/// Walk every known PDK root (`~/.ciel`, `~/.volare`, `$PDK_ROOT`) and
/// return one [`PdkEntry`] per variant whose `lib_subpath` exists.
/// Used by `pdk list` so installs done directly via `ciel enable` /
/// `volare enable` surface without re-running `rlx-eda pdk install`.
pub fn scan_ciel_root() -> Result<Vec<PdkEntry>, Error> {
    let mut roots: Vec<(PathBuf, &'static str)> = Vec::new();
    if let Ok(env_root) = std::env::var("PDK_ROOT") {
        roots.push((PathBuf::from(env_root), "ciel"));
    }
    if let Ok(home) = std::env::var("HOME") {
        roots.push((PathBuf::from(&home).join(".ciel"), "ciel"));
        roots.push((PathBuf::from(&home).join(".volare"), "volare"));
    }

    let mut out = Vec::new();
    for kp in known::ALL {
        for (root, basename) in &roots {
            let Some(lib) = locate_variant_lib_in_root(root, basename, kp.ciel_family, kp.variant, kp.lib_subpath) else { continue };
            let sections = discover::sections_from_lib(&lib).unwrap_or_default();
            out.push(PdkEntry {
                name: kp.variant.to_string(),
                lib_path: lib,
                sections,
                vdd_nom: kp.vdd_nom,
                source: Source::Ciel,
            });
            break;
        }
    }
    Ok(out)
}

fn resolve_ciel_bin(explicit: Option<&Path>) -> Result<PathBuf, Error> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    if let Ok(p) = std::env::var("CIEL_BIN") {
        return Ok(PathBuf::from(p));
    }
    if let Some(p) = which("ciel") { return Ok(p); }
    // Fall back to volare if present (legacy sky130-only installs).
    if let Some(p) = which("volare") { return Ok(p); }
    Err(Error::CielNotFound)
}

/// Resolve the PDK_ROOT directory ciel/volare writes into. Both honor
/// `$PDK_ROOT` and otherwise default to `~/.ciel` (ciel) or
/// `~/.volare` (volare). We don't shell out to `ciel path` because
/// recent ciel versions require `--pdk-family <…> <version>` for that.
fn pdk_root_for(bin: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("PDK_ROOT") { return PathBuf::from(p); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    match bin.file_name().and_then(|s| s.to_str()) {
        Some("volare") => PathBuf::from(home).join(".volare"),
        _ => PathBuf::from(home).join(".ciel"),
    }
}

fn bin_basename(bin: &Path) -> &'static str {
    match bin.file_name().and_then(|s| s.to_str()) {
        Some("volare") => "volare",
        _ => "ciel",
    }
}

/// Look up `<pdk_root>/<binname>/<family>/versions/*/<variant>/<lib_subpath>`
/// across every version dir. Picks the most-recently-modified version so
/// the "currently enabled" one wins on ties. Returns `None` if no
/// version contains the file.
fn locate_variant_lib(bin: &Path, family: &str, variant: &str, lib_subpath: &str) -> Option<PathBuf> {
    let root = pdk_root_for(bin);
    let bn = bin_basename(bin);
    locate_variant_lib_in_root(&root, bn, family, variant, lib_subpath)
}

fn locate_variant_lib_in_root(
    root: &Path,
    bin_basename: &str,
    family: &str,
    variant: &str,
    lib_subpath: &str,
) -> Option<PathBuf> {
    let versions_dir = root.join(bin_basename).join(family).join("versions");
    let entries = std::fs::read_dir(&versions_dir).ok()?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in entries.flatten() {
        let candidate = e.path().join(variant).join(lib_subpath);
        if !candidate.is_file() { continue; }
        let mtime = e.metadata().and_then(|m| m.modified()).ok().unwrap_or(std::time::UNIX_EPOCH);
        if best.as_ref().map_or(true, |(prev, _)| mtime > *prev) {
            best = Some((mtime, candidate));
        }
    }
    best.map(|(_, p)| p)
}

fn latest_remote_version(bin: &Path, family: &str) -> Result<String, Error> {
    let out = Command::new(bin)
        .args(["ls-remote", "--pdk", family])
        .output()?;
    if !out.status.success() {
        return Err(Error::CielFailed {
            code: out.status.code(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().to_string())
        .ok_or_else(|| Error::CielFailed {
            code: out.status.code(),
            stderr: "ls-remote returned no versions".into(),
        })
}

fn enable(bin: &Path, family: &str, version: &str) -> Result<(), Error> {
    let out = Command::new(bin)
        .args(["enable", "--pdk", family, version])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()?;
    if !out.success() {
        return Err(Error::CielFailed {
            code: out.code(),
            stderr: format!("ciel enable {family} {version} failed"),
        });
    }
    Ok(())
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
