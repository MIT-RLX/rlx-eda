//! `rlx-eda pdk …` — install / register / list PDKs.

use std::path::PathBuf;

use clap::{Args, Subcommand};

mod discover;
pub mod install;
mod known;
mod registry;

pub use registry::{PdkEntry, Registry};

#[derive(Subcommand, Debug)]
pub enum PdkCmd {
    /// Download and register a known PDK (sky130A/B, gf180mcuA-D, ihp-sg13g2).
    /// Delegates to `ciel` for the download; auto-discovers the SPICE lib
    /// path and corner sections after install.
    Install(InstallArgs),

    /// Manually register a PDK that lives outside ciel (custom open_pdks
    /// install, vendor PDK, etc.).
    Register(RegisterArgs),

    /// List installed/registered PDKs (registry + ciel index).
    List,

    /// Show one PDK's resolved details: lib path, valid corner sections,
    /// nominal supply, source (ciel-managed vs. user-registered).
    Show(ShowArgs),

    /// Print the registry config path.
    Path,

    /// Remove a registered PDK from the registry. Does NOT uninstall
    /// from ciel — use `ciel rm` for that.
    Forget(ForgetArgs),
}

#[derive(Args, Debug)]
pub struct InstallArgs {
    /// Variant: `sky130A`, `sky130B`, `gf180mcuA`..`gf180mcuD`, `ihp-sg13g2`.
    pub name: String,
    /// Pin a specific ciel version SHA. Default = latest remote.
    #[arg(long)]
    pub version: Option<String>,
    /// Override ciel binary lookup.
    #[arg(long)]
    pub ciel_bin: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct RegisterArgs {
    /// Friendly name (e.g. `vendor_x_180`).
    pub name: String,
    /// Path to the top-level `.lib` / `.spice` file (the one with
    /// `.lib <section>` headers).
    #[arg(long)]
    pub lib: PathBuf,
    /// Comma-separated section names. If omitted, auto-detected from the
    /// `.lib` file's `.lib <name>` headers.
    #[arg(long)]
    pub sections: Option<String>,
    /// Nominal supply voltage in volts. Used as the default vdd for
    /// `CornerSet::typical_etc`.
    #[arg(long, default_value_t = 1.8)]
    pub vdd: f64,
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    pub name: String,
}

#[derive(Args, Debug)]
pub struct ForgetArgs {
    pub name: String,
}

pub fn run(cmd: PdkCmd) -> Result<(), Error> {
    match cmd {
        PdkCmd::Install(a) => install::run(a),
        PdkCmd::Register(a) => register(a),
        PdkCmd::List => list(),
        PdkCmd::Show(a) => show(a),
        PdkCmd::Path => print_path(),
        PdkCmd::Forget(a) => forget(a),
    }
}

fn register(args: RegisterArgs) -> Result<(), Error> {
    let sections = match args.sections {
        Some(s) => s.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        None => discover::sections_from_lib(&args.lib)
            .map_err(|e| Error::Discover(args.lib.clone(), e))?,
    };
    if sections.is_empty() {
        return Err(Error::NoSections(args.lib));
    }
    let entry = PdkEntry {
        name: args.name.clone(),
        lib_path: args.lib.clone(),
        sections,
        vdd_nom: args.vdd,
        source: registry::Source::User,
    };
    let mut reg = Registry::load_or_default()?;
    reg.upsert(entry.clone());
    reg.save()?;
    println!(
        "registered {} → {}\n  sections: {}\n  vdd_nom:  {} V",
        entry.name,
        entry.lib_path.display(),
        entry.sections.join(", "),
        entry.vdd_nom,
    );
    Ok(())
}

fn list() -> Result<(), Error> {
    let reg = Registry::load_or_default()?;
    let ciel_entries = install::scan_ciel_root().unwrap_or_default();
    let mut all = reg.entries.clone();
    for e in ciel_entries {
        if !all.iter().any(|x| x.name == e.name) {
            all.push(e);
        }
    }
    if all.is_empty() {
        println!("no PDKs installed or registered");
        println!("  → try `rlx-eda pdk install sky130A` or `rlx-eda pdk register …`");
        return Ok(());
    }
    println!("{:<14} {:<8} {:<6} {}", "name", "source", "vdd", "lib");
    for e in &all {
        println!(
            "{:<14} {:<8} {:<6.2} {}",
            e.name,
            match e.source { registry::Source::Ciel => "ciel", registry::Source::User => "user" },
            e.vdd_nom,
            e.lib_path.display(),
        );
    }
    Ok(())
}

fn show(args: ShowArgs) -> Result<(), Error> {
    let entry = resolve(&args.name)?;
    println!("name:      {}", entry.name);
    println!("source:    {}", match entry.source {
        registry::Source::Ciel => "ciel-managed",
        registry::Source::User => "user-registered",
    });
    println!("lib_path:  {}", entry.lib_path.display());
    println!("vdd_nom:   {} V", entry.vdd_nom);
    println!("sections:  ({} total)", entry.sections.len());
    for s in &entry.sections {
        println!("  - {s}");
    }
    Ok(())
}

fn forget(args: ForgetArgs) -> Result<(), Error> {
    let mut reg = Registry::load_or_default()?;
    if !reg.remove(&args.name) {
        return Err(Error::NotRegistered(args.name));
    }
    reg.save()?;
    println!("forgot {}", args.name);
    Ok(())
}

fn print_path() -> Result<(), Error> {
    println!("{}", Registry::config_path()?.display());
    Ok(())
}

/// Look up a PDK across the registry first, then ciel's auto-discovered
/// installs. Returns the merged entry.
pub fn resolve(name: &str) -> Result<PdkEntry, Error> {
    let reg = Registry::load_or_default()?;
    if let Some(e) = reg.find(name) {
        return Ok(e.clone());
    }
    let ciel = install::scan_ciel_root().unwrap_or_default();
    if let Some(e) = ciel.into_iter().find(|e| e.name == name) {
        return Ok(e);
    }
    Err(Error::Unknown(name.into()))
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml: {0}")]
    TomlSer(#[from] toml::ser::Error),
    #[error("toml: {0}")]
    TomlDe(#[from] toml::de::Error),
    #[error("PDK '{0}' not registered")]
    NotRegistered(String),
    #[error("PDK '{0}' not found in registry or ciel root")]
    Unknown(String),
    #[error("could not discover sections in {0}: {1}")]
    Discover(PathBuf, std::io::Error),
    #[error("no .lib sections found in {0} — pass --sections explicitly")]
    NoSections(PathBuf),
    #[error("ciel binary not found on PATH (set with --ciel-bin or install ciel via pipx)")]
    CielNotFound,
    #[error("ciel exited non-zero ({code:?}); stderr:\n{stderr}")]
    CielFailed { code: Option<i32>, stderr: String },
    #[error("config home not resolvable (set HOME or XDG_CONFIG_HOME)")]
    NoConfigHome,
    #[error("unknown PDK family for `pdk install`: {0}. Known: {}", known::all_names().join(", "))]
    UnknownVariant(String),
}
