//! Lightweight sky130 synthesis backend — Yosys + ABC against the
//! foundry `sky130_fd_sc_hd` Liberty (`tt_025C_1v80` corner).
//!
//! ## What this is for
//!
//! "Measure it in silicon." Verilator gives cycle counts; this gives
//! the **gate-level area + a worst-path delay estimate** for the same
//! SV, so the FPGA-vs-ASIC matrix can be filled with real sky130
//! numbers without paying the full ORFS PnR cost (which is minutes
//! per design — Yosys-only is seconds).
//!
//! Not a replacement for [`super::orfs::OrfsBackend`]: ABC's reported
//! delay is a synthesis-time mapping estimate, not closed STA. For
//! sign-off-grade timing, run the ORFS flow.
//!
//! ## What it produces
//!
//! ```text
//! cells:        N standard-cell instances after tech mapping
//! area_um2:     Σ cell_area  (from `stat -liberty`)
//! abc_delay_ps: ABC's worst-path estimate during mapping
//! ```
//!
//! Gated by `bench-rtl-sim` (same feature as the Verilator path) so
//! the silicon-measurement pipeline lives behind one toggle.

#![cfg(feature = "bench-rtl-sim")]

use std::path::{Path, PathBuf};

use eda_container::DockerRun;

/// We use the ORFS image's Yosys (0.64) instead of the local
/// `rlx-yosys:local` (0.23) — the older parser doesn't accept the
/// modern SystemVerilog our codegen emits (multi-dim unpacked
/// localparams, `string` parameters in `block_rom`).
///
/// ORFS also bundles every sky130 Liberty corner under
/// `/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/` so we don't
/// need to mount the host PDK.
pub const YOSYS_IMAGE: &str = "openroad/orfs:latest";

/// Liberty path inside the ORFS image — typical-typical 25°C, 1.8 V.
pub const SKY130_LIB_IN_IMAGE: &str =
    "/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib";

#[derive(Debug, Clone, Copy)]
pub struct SynthMetrics {
    pub cells: u64,
    pub area_um2: f64,
    pub abc_delay_ps: Option<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum SynthError {
    #[error("yosys docker: {0}")]
    Docker(String),
    #[error("yosys parse: {0}\n--- stdout ---\n{1}")]
    Parse(String, String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Synthesize every `*.sv` file under `design_dir` (excluding any
/// file whose name starts with `tb`) against sky130_fd_sc_hd, with
/// `top_module` as the design top.
///
/// Mounts:
/// - host `design_dir` → `/design`
///
/// Uses `abc -fast` so per-design wall time stays manageable on
/// x86-emulated docker (the full `abc` script iterates several
/// times over the same netlist for closure; `-fast` is one pass
/// and finishes in seconds for thousand-gate designs).
pub fn synth_sky130(design_dir: &Path, top_module: &str) -> Result<SynthMetrics, SynthError> {
    let design_dir = design_dir
        .canonicalize()
        .map_err(|e| SynthError::Docker(format!("canonicalize {}: {e}", design_dir.display())))?;

    // Yosys is finicky about `parameter string ...` in modules
    // synthesized for ASIC. The rlx-fpga `block_rom` / `block_ram`
    // primitives use it for `INIT_FILE`. Strip the `string` keyword
    // on the in-container copy — it parses then as an untyped
    // parameter (which Yosys handles fine).
    sanitize_for_yosys(&design_dir)?;

    // Yosys script. Read every .sv except tb*; map to sky130_fd_sc_hd
    // with `abc -fast` (single mapping pass instead of the iterative
    // closure script — fast enough that x86 emulation completes in
    // under a minute even for the unrolled-Dense design).
    let ys_script = format!(
        r#"
foreach f [glob -nocomplain /design/*.sv /design/primitives/*.sv \
                            /design/layers/*.sv] {{
    if {{[string match -nocase "tb*" [file tail $f]]}} continue
    yosys read_verilog -sv $f
}}
yosys hierarchy -check -top {top}
yosys synth -top {top} -flatten
yosys dfflibmap -liberty {lib}
yosys abc -fast -liberty {lib}
yosys clean -purge
yosys stat -liberty {lib}
"#,
        top = top_module,
        lib = SKY130_LIB_IN_IMAGE,
    );

    let bash = format!(
        // `yosys -c <tcl-file>` runs the script. Heredoc keeps quoting
        // simple; `-q` cuts the per-pass banners but ABC's "Delay = ..."
        // line still survives so we can scrape it.
        "set -e; cat > /tmp/synth.tcl <<'EOF'\n{ys_script}\nEOF\n\
         yosys -q -c /tmp/synth.tcl",
    );

    let stdout = DockerRun::new(YOSYS_IMAGE)
        .platform("linux/amd64")
        .entrypoint("bash")
        .workdir("/design")
        .mount(design_dir.clone(), PathBuf::from("/design"))
        .arg("-c")
        .arg(bash)
        .run_with_stdin(b"")
        .map_err(|e| SynthError::Docker(format!("{e}")))?;

    parse_yosys_stat(&stdout)
}

/// Parse `stat -liberty` output. The block looks like:
///
/// ```text
/// === top ===
///    Number of wires:               12345
///    Number of cells:                4567
///       sky130_fd_sc_hd__nand2_1     100
///       ...
///    Chip area for module '\top': 12345.678
/// ```
///
/// ABC's delay estimate appears earlier in the log, line shaped
/// roughly `ABC: + WireLoad = ... Gates = ... Cap = ... Area = ... Delay = N ps`.
fn parse_yosys_stat(stdout: &str) -> Result<SynthMetrics, SynthError> {
    let mut cells: Option<u64> = None;
    let mut area: Option<f64> = None;
    let mut abc_delay_ps: Option<f64> = None;

    for line in stdout.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("Number of cells:") {
            // Yosys 0.23 format
            cells = rest.trim().parse().ok();
        } else if l.ends_with(" cells") && l.split_whitespace().count() == 3 {
            // Yosys 0.64 format: "<count> <area> cells"
            let mut it = l.split_whitespace();
            cells = it.next().and_then(|s| s.parse().ok());
        } else if let Some(idx) = l.find("Chip area for module") {
            // "Chip area for module '\top': 12345.678"
            let after_colon = &l[idx..];
            if let Some(colon) = after_colon.find(':') {
                area = after_colon[colon + 1..].trim().parse().ok();
            }
        } else if l.starts_with("ABC:") && l.contains("Delay =") {
            // "ABC: + ... Area = 1234.56 Delay = 7890.12 ps"
            if let Some(after) = l.split("Delay =").nth(1) {
                let tok = after.trim().split_whitespace().next().unwrap_or("");
                if let Ok(v) = tok.parse::<f64>() {
                    abc_delay_ps = Some(v);
                }
            }
        }
    }

    let cells = cells.ok_or_else(|| {
        SynthError::Parse(
            "missing `Number of cells:` line in yosys stdout".into(),
            stdout.to_string(),
        )
    })?;
    let area_um2 = area.ok_or_else(|| {
        SynthError::Parse(
            "missing `Chip area for module` line in yosys stdout".into(),
            stdout.to_string(),
        )
    })?;
    Ok(SynthMetrics {
        cells,
        area_um2,
        abc_delay_ps,
    })
}

/// Strip Yosys-unfriendly SystemVerilog tokens from every .sv file
/// under `design_dir`, in place. We only target `parameter string` →
/// `parameter`: rlx-fpga's `block_rom`/`block_ram` primitives use
/// `string INIT_FILE = ""` which Yosys 0.64 rejects despite
/// `read_verilog -sv`. The untyped form parses fine and yields the
/// same elaboration since the parameter is only ever assigned a
/// string literal that gets passed straight through to `$readmemh`.
fn sanitize_for_yosys(design_dir: &Path) -> Result<(), SynthError> {
    fn visit(dir: &Path) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit(&path)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("sv") {
                let content = std::fs::read_to_string(&path)?;
                let patched = content.replace("parameter string ", "parameter ");
                if patched != content {
                    std::fs::write(&path, patched)?;
                }
            }
        }
        Ok(())
    }
    visit(design_dir).map_err(SynthError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sample_yosys_stat() {
        let sample = r#"
ABC: + WireLoad = "none"  Gates = 100  Cap = 5.0 ff  Area = 1234.56  Delay = 4321.0 ps  Lags = 0
=== top ===
   Number of wires:                123
   Number of cells:                456
      sky130_fd_sc_hd__nand2_1     100
      sky130_fd_sc_hd__buf_2        50
   Chip area for module '\top': 9876.543
"#;
        let m = parse_yosys_stat(sample).unwrap();
        assert_eq!(m.cells, 456);
        assert!((m.area_um2 - 9876.543).abs() < 1e-3);
        assert_eq!(m.abc_delay_ps, Some(4321.0));
    }
}
