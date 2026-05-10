//! Probe the host's PDK install state — sky130 GDS, sky130 PDK
//! repo, ngspice, docker, ORFS image. Prints a human-readable
//! report and exits non-zero if anything required for `cargo run
//! -p eda-bench-tinyconv --bin demo_report --features demo-bin`
//! is missing.
//!
//! ```sh
//! cargo run -p eda-bench-tinyconv --bin probe_pdk
//! ```
//!
//! No bench features required — the probe runs unconditionally so
//! contributors can diagnose "why doesn't the demo work" without
//! enabling docker / FPGA toolchains.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("eda-bench-tinyconv PDK / toolchain probe");
    println!("=========================================\n");

    let mut all_ok = true;

    // ── sky130_fd_sc_hd GDS ─────────────────────────────────────
    println!("sky130_fd_sc_hd GDS");
    let path = eda_stdcells::default_sc_hd_path();
    if path.is_file() {
        let bytes = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or(0);
        println!("  ✓ found: {} ({} bytes)", path.display(), bytes);
    } else {
        println!("  ✗ NOT FOUND. Probed paths:");
        println!("    - $RLX_EDA_SKY130_FD_SC_HD_GDS env var");
        println!("    - $PDK_ROOT / ~/.ciel / ~/.volare install trees");
        println!("    - legacy /Users/Shared/mtl/skywater130/...");
        println!("  Fix: `volare install sky130 --pdk sky130A`");
        println!("  (configures ~/.volare with sky130_fd_sc_hd cells)");
        all_ok = false;
    }

    // ── sky130 PDK repo (for `git rev-parse HEAD` in manifest) ─
    println!("\nsky130 PDK repo");
    let repo = PathBuf::from("/Users/Shared/mtl/skywater130");
    if repo.join(".git").exists() {
        let head = Command::new("git")
            .args(["-C"])
            .arg(&repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "(git failed)".into());
        println!("  ✓ found at {} @ {}", repo.display(), head);
    } else {
        println!("  ! optional; falls back to `(unavailable)` in manifest");
    }

    // ── ngspice ─────────────────────────────────────────────────
    println!("\nngspice");
    match Command::new("ngspice").arg("--version").output() {
        Ok(o) if o.status.success() => {
            let line = String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("(empty)")
                .to_string();
            println!("  ✓ on PATH: {line}");
        }
        _ => {
            println!("  ! optional; needed only for L4 mixed-signal sim and noise calibration");
        }
    }

    // ── docker (for ORFS backend) ────────────────────────────────
    println!("\ndocker");
    match Command::new("docker").arg("--version").output() {
        Ok(o) if o.status.success() => {
            let line = String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("(empty)")
                .to_string();
            println!("  ✓ on PATH: {line}");
        }
        _ => {
            println!("  ! optional; needed only for `--features bench-orfs`");
        }
    }

    // ── ORFS image ──────────────────────────────────────────────
    println!("\nORFS docker image");
    match Command::new("docker")
        .args(["image", "inspect", "openroad/orfs:latest", "--format", "{{.Id}}"])
        .output()
    {
        Ok(o) if o.status.success() => {
            let id = String::from_utf8_lossy(&o.stdout).trim().to_string();
            println!("  ✓ pulled: {id}");
            println!("  Pin via: ./crates/eda-bench-tinyconv/docker/pin.sh");
        }
        _ => {
            println!("  ! optional. Pull via: docker pull openroad/orfs:latest");
        }
    }

    // ── Summary ─────────────────────────────────────────────────
    println!();
    if all_ok {
        println!("All required components present. Demo binary should work:");
        println!("  cargo run -p eda-bench-tinyconv --features demo-bin --bin demo_report");
        std::process::exit(0);
    } else {
        println!("One or more required components missing — see ✗ above.");
        std::process::exit(1);
    }
}
