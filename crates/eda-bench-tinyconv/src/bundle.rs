//! Bundle (weights + bitstream/GDS) export.
//!
//! Two modes:
//!
//! - **Separate (default)**: writes the bitstream / GDS and the
//!   weight blob to distinct files. Standard FPGA workflow —
//!   `top.sv` + `weights/*.mem` consumed independently by yosys at
//!   synthesis time. Easy to diff in version control.
//!
//! - **Merged**: writes a single tarball containing both, plus a
//!   `manifest.toml` recording the bundle's checksums and the
//!   `BenchConfig::run.seed` that produced it. ASIC mask-ROM
//!   convention: ship one artifact, not two. Reproducibility
//!   manifest from `crate::manifest` rides along inside.
//!
//! Future work: when `merge_weights = true`, generate a single
//! SystemVerilog file with weights inlined as `localparam` arrays
//! (instead of `$readmemh`). That's the literal "bake into mask
//! ROM" form. v1 ships the tarball variant — same-effort,
//! conventional, doesn't require modifying `rlx-fpga`'s emit pass.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Bundle configuration. Lives in `BenchConfig::bundle`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BundleConfig {
    /// When `true`, write a single bundled artifact at
    /// `output_path`. When `false`, do nothing (caller manages
    /// `top.sv` + `weights/*.mem` separately).
    pub merge_weights: bool,
    /// Where the bundle lands on disk. Relative paths resolve
    /// against the workspace root. Convention: under `target/bench/`
    /// so it's git-ignored by default.
    pub output_path: String,
    /// Bundle format. Currently only `Tarball` is implemented;
    /// `InlineSv` (weights baked into SystemVerilog `localparam`s
    /// for mask-ROM-style single-file deliverables) lands in v1.5
    /// once `rlx-fpga::codegen` supports the inline emit path.
    pub format: BundleFormat,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            merge_weights: false,
            output_path: "target/bench/demo/bundle.tar".into(),
            format: BundleFormat::Tarball,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BundleFormat {
    /// POSIX `ustar`-format tarball: one file containing all bundle
    /// entries plus a `manifest.toml`. No compression — bench
    /// artifacts are typically already-compressed `.bit` / `.gds`.
    Tarball,
    /// SystemVerilog with weights inlined as `localparam` arrays.
    /// Stub — emit support pending.
    InlineSv,
}

/// Per-entry record for the bundle's `manifest.toml`. One per file
/// included in the tarball.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleEntry {
    pub name: String,
    pub byte_len: u64,
    /// SHA-256 hex of the file's contents. Same hash function used
    /// by `crate::manifest::sha256_file` so manifest checksums
    /// round-trip.
    pub sha256: String,
}

#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bundle format `{0:?}` not yet implemented")]
    Unimplemented(BundleFormat),
}

/// Write the bundle to `cfg.output_path`. Returns the list of
/// included entries (for inclusion in the bench Report).
///
/// `entries` is `(filename_in_bundle, file_contents)`. Caller
/// supplies whatever it wants bundled — typically the GDS / SV +
/// weight blob + the bench Report itself.
pub fn write_bundle(
    cfg: &BundleConfig,
    entries: &[(&str, &[u8])],
) -> Result<Vec<BundleEntry>, BundleError> {
    if !cfg.merge_weights {
        // Toggle is off — caller writes files separately. We still
        // return the would-be entry list for reporting parity.
        return Ok(entries
            .iter()
            .map(|(name, body)| BundleEntry {
                name: (*name).to_string(),
                byte_len: body.len() as u64,
                sha256: sha256_hex(body),
            })
            .collect());
    }

    match cfg.format {
        BundleFormat::Tarball => write_tarball(&cfg.output_path, entries),
        BundleFormat::InlineSv => Err(BundleError::Unimplemented(cfg.format)),
    }
}

fn write_tarball(
    output_path: &str,
    entries: &[(&str, &[u8])],
) -> Result<Vec<BundleEntry>, BundleError> {
    let path = PathBuf::from(output_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut out = std::fs::File::create(&path)?;

    let mut manifest_entries: Vec<BundleEntry> = Vec::with_capacity(entries.len());
    for (name, body) in entries {
        write_ustar_entry(&mut out, name, body)?;
        manifest_entries.push(BundleEntry {
            name: (*name).to_string(),
            byte_len: body.len() as u64,
            sha256: sha256_hex(body),
        });
    }

    // Manifest at the end so consumers can find it by walking
    // entries (and so its checksum doesn't include itself).
    let manifest = toml::to_string_pretty(&BundleManifest {
        entries: manifest_entries.clone(),
    })
    .unwrap_or_default();
    write_ustar_entry(&mut out, "manifest.toml", manifest.as_bytes())?;

    // Two zero blocks terminate a ustar archive.
    out.write_all(&[0u8; 1024])?;

    Ok(manifest_entries)
}

#[derive(Serialize, Deserialize)]
struct BundleManifest {
    entries: Vec<BundleEntry>,
}

/// Minimal POSIX `ustar` entry writer. No symlinks, no long-name
/// extension, no permissions beyond mode 0644 — bundle entries are
/// always regular files written by the bench harness, so the
/// reduced format suffices.
fn write_ustar_entry<W: Write>(out: &mut W, name: &str, body: &[u8]) -> std::io::Result<()> {
    let mut header = [0u8; 512];
    // name: bytes 0..100
    let nb = name.as_bytes();
    let n = nb.len().min(100);
    header[..n].copy_from_slice(&nb[..n]);
    // mode: octal "0000644 \0" at bytes 100..108
    let mode = b"0000644\0";
    header[100..108].copy_from_slice(mode);
    // uid / gid: octal "0000000 \0" at 108..116, 116..124
    header[108..116].copy_from_slice(b"0000000\0");
    header[116..124].copy_from_slice(b"0000000\0");
    // size: octal at 124..136 (11 chars + NUL)
    let size_str = format!("{:011o}\0", body.len());
    header[124..136].copy_from_slice(size_str.as_bytes());
    // mtime: zeros (deterministic bundle reproducibility — the bench
    // manifest carries the actual run timestamp).
    header[136..148].copy_from_slice(b"00000000000\0");
    // typeflag: '0' = regular file
    header[156] = b'0';
    // ustar magic + version: "ustar\0" + "00"
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    // checksum: spaces while computing, then octal
    header[148..156].copy_from_slice(b"        ");
    let checksum: u32 = header.iter().map(|b| *b as u32).sum();
    let cksum_str = format!("{:06o}\0 ", checksum);
    header[148..156].copy_from_slice(cksum_str.as_bytes());

    out.write_all(&header)?;
    out.write_all(body)?;
    // Pad to 512-byte boundary.
    let pad = (512 - body.len() % 512) % 512;
    if pad > 0 {
        out.write_all(&vec![0u8; pad])?;
    }
    Ok(())
}

/// SHA-256 of a byte slice, lowercase hex. Same shape as
/// `crate::manifest::sha256_file`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(out.len() * 2);
    for b in out {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Read back a bundle's entries (without unpacking the file
/// contents) — useful for tests + bench-report parity checks.
pub fn read_bundle_entries(path: &Path) -> Result<Vec<BundleEntry>, BundleError> {
    let bytes = std::fs::read(path)?;
    let mut out = Vec::new();
    let mut i = 0;
    while i + 512 <= bytes.len() {
        if bytes[i..i + 100].iter().all(|&b| b == 0) {
            break; // end-of-archive
        }
        let name_end = bytes[i..i + 100]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(100);
        let name = std::str::from_utf8(&bytes[i..i + name_end])
            .unwrap_or("(invalid utf8)")
            .to_string();
        let size_str = std::str::from_utf8(&bytes[i + 124..i + 135]).unwrap_or("0");
        let size = u64::from_str_radix(size_str.trim(), 8).unwrap_or(0);
        let body_start = i + 512;
        let body_end = body_start + size as usize;
        if body_end > bytes.len() {
            break;
        }
        let body_sha = sha256_hex(&bytes[body_start..body_end]);
        out.push(BundleEntry {
            name,
            byte_len: size,
            sha256: body_sha,
        });
        let pad = (512 - (size as usize) % 512) % 512;
        i = body_end + pad;
    }
    Ok(out)
}
