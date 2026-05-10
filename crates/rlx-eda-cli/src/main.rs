//! `rlx-eda` — top-level CLI for the rlx-eda harness + PDK manager.
//!
//! Mirror of cicsim's command surface, plus a PDK manager that wraps
//! `ciel` (and falls back to `volare` for legacy sky130 deployments) so
//! installs of sky130A / gf180mcu* / ihp-sg13g2 land registered and
//! ready to drive a [`Testbench`](eda_sim_harness::Testbench).

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use std::path::PathBuf;

use rlx_eda_cli::{dashboard, doctor, pdk};

#[derive(Parser, Debug)]
#[command(
    name = "rlx-eda",
    version,
    about = "rlx-eda — differentiable + SPICE simulation harness with a PDK manager",
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Manage process design kits.
    Pdk {
        #[command(subcommand)]
        sub: pdk::PdkCmd,
    },
    /// Diagnose the local install: ngspice, ciel/volare, registered PDKs.
    /// Run this first when something's broken.
    Doctor,
    /// Cross-test dashboard: roll every crate's docs/ into one
    /// top-level HTML. Run after the test suite to refresh
    /// `<root>/docs/index.html`.
    Dashboard {
        /// Workspace root (defaults to CWD).
        #[arg(long)]
        root: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Pdk { sub } => pdk::run(sub).map_err(|e| e.to_string()),
        Cmd::Doctor => doctor::run().map_err(|e| e.to_string()),
        Cmd::Dashboard { root } => dashboard::run(root).map_err(|e| e.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rlx-eda: {e}");
            ExitCode::from(1)
        }
    }
}
