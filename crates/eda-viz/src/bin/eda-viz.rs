//! `eda-viz` — CLI for rendering GDS / OASIS layouts.
//!
//! Usage:
//!
//!     eda-viz <input.gds | input.oas> [-o out.svg | out.png] [--cell <name>]
//!
//! Format is inferred from the input extension (`.gds` → GDS, `.oas` /
//! `.oasis` → OASIS). Output format is inferred from the `-o` extension
//! (`.svg` or `.png`). If `-o` is omitted, writes `<input>.svg` next to
//! the source file.
//!
//! `--cell <name>` selects a top cell by name; default is the first
//! top-level cell `Library::top_cells()` returns.

use std::path::PathBuf;
use std::process::ExitCode;

use eda_viz::{layout, png::svg_to_png, Style};
use klayout_io::{gds::read::read_gds_path, oasis::read::read_oasis_path};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("eda-viz: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut cell_name: Option<String> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" | "--output" => {
                output = Some(PathBuf::from(args.next().ok_or("-o needs a value")?));
            }
            "--cell" => {
                cell_name = Some(args.next().ok_or("--cell needs a value")?);
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other if !other.starts_with('-') && input.is_none() => {
                input = Some(PathBuf::from(other));
            }
            other => return Err(format!("unexpected argument: {other}")),
        }
    }

    let input = input.ok_or("missing input file (try --help)")?;
    let lib = read_layout(&input)?;

    // Pick top cell.
    let top = match &cell_name {
        Some(name) => lib
            .by_name(name)
            .ok_or_else(|| format!("no cell named '{name}' in {}", input.display()))?,
        None => *lib
            .top_cells()
            .first()
            .ok_or("library has no top cells")?,
    };

    // Resolve output path / format.
    let out = output.unwrap_or_else(|| input.with_extension("svg"));
    let svg = layout::render_to_svg(&lib, top, &Style::default());

    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("svg")
        .to_ascii_lowercase();
    match ext.as_str() {
        "svg" => {
            std::fs::write(&out, &svg).map_err(|e| format!("write {}: {e}", out.display()))?;
        }
        "png" => {
            let bytes = svg_to_png(&svg, 2.0)
                .map_err(|e| format!("rasterize: {e}"))?;
            std::fs::write(&out, bytes).map_err(|e| format!("write {}: {e}", out.display()))?;
        }
        other => return Err(format!("unknown output extension '{other}' (use .svg or .png)")),
    }

    println!("wrote {}", out.display());
    Ok(())
}

fn read_layout(path: &std::path::Path) -> Result<klayout_core::Library, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "gds" | "gdsii" => read_gds_path(path).map_err(|e| format!("read GDS: {e}")),
        "oas" | "oasis" => read_oasis_path(path).map_err(|e| format!("read OASIS: {e}")),
        other => Err(format!(
            "unknown input extension '{other}' (expected .gds or .oas)"
        )),
    }
}

fn print_help() {
    print!(
        "\
eda-viz — render a GDS / OASIS layout to SVG or PNG.

USAGE:
    eda-viz <INPUT> [-o OUTPUT] [--cell NAME]

ARGS:
    <INPUT>          .gds or .oas file
    -o, --output     Output path (defaults to <INPUT>.svg). Format chosen
                     from .svg / .png extension.
    --cell NAME      Top cell to render (default: first top cell)
    -h, --help       Print this help
"
    );
}
