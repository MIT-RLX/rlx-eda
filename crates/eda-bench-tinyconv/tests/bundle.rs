//! Bundle (weights + bitstream) export tests.

use eda_bench_tinyconv::bundle::{
    read_bundle_entries, write_bundle, BundleConfig, BundleError, BundleFormat,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

fn fresh_path(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let mut p = std::env::temp_dir();
    p.push(format!("bench-bundle-{}-{tag}-{n}.tar", std::process::id()));
    p
}

#[test]
fn merge_disabled_returns_entry_summary_without_writing_file() {
    let cfg = BundleConfig {
        merge_weights: false,
        output_path: "/tmp/should-not-be-created.tar".into(),
        format: BundleFormat::Tarball,
    };
    let entries = write_bundle(
        &cfg,
        &[("top.sv", b"module top; endmodule"[..].as_ref())],
    )
    .expect("merge=false should not error");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "top.sv");
    assert_eq!(entries[0].byte_len, 21); // "module top; endmodule" no NL
    assert!(!std::path::Path::new(&cfg.output_path).exists());
}

#[test]
fn merge_enabled_writes_tarball_with_correct_entries() {
    let path = fresh_path("merge-on");
    let cfg = BundleConfig {
        merge_weights: true,
        output_path: path.to_string_lossy().into_owned(),
        format: BundleFormat::Tarball,
    };
    let weights = b"\x01\x02\x03\x04";
    let top = b"module top; endmodule";
    let entries = write_bundle(
        &cfg,
        &[
            ("top.sv", top.as_ref()),
            ("weights/conv1.mem", weights.as_ref()),
        ],
    )
    .unwrap();

    assert_eq!(entries.len(), 2);
    assert!(path.exists());

    // Read it back via the public reader.
    let read_back = read_bundle_entries(&path).expect("readable");
    // 2 user entries + manifest.toml.
    assert!(read_back.len() >= 3, "got: {read_back:?}");
    let names: Vec<&str> = read_back.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"top.sv"));
    assert!(names.contains(&"weights/conv1.mem"));
    assert!(names.contains(&"manifest.toml"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn bundle_entry_sha256_is_deterministic_and_matches_writeback() {
    // Two writes of the same payload → same sha256 in both
    // round-trips. Catches non-deterministic ordering / mtime
    // leakage in the tar headers.
    let path1 = fresh_path("sha-check-a");
    let path2 = fresh_path("sha-check-b");
    let cfg1 = BundleConfig {
        merge_weights: true,
        output_path: path1.to_string_lossy().into_owned(),
        format: BundleFormat::Tarball,
    };
    let cfg2 = BundleConfig {
        output_path: path2.to_string_lossy().into_owned(),
        ..cfg1.clone()
    };
    let payload = b"hello bundle";
    write_bundle(&cfg1, &[("only.bin", payload.as_ref())]).unwrap();
    write_bundle(&cfg2, &[("only.bin", payload.as_ref())]).unwrap();

    let r1 = read_bundle_entries(&path1).unwrap();
    let r2 = read_bundle_entries(&path2).unwrap();
    let h1 = r1.iter().find(|e| e.name == "only.bin").unwrap();
    let h2 = r2.iter().find(|e| e.name == "only.bin").unwrap();
    assert_eq!(h1.sha256, h2.sha256, "sha256 should be deterministic");
    assert_eq!(h1.sha256.len(), 64, "sha256 hex is 64 chars");
    assert_eq!(h1.byte_len, 12); // "hello bundle" is 12 bytes

    let _ = std::fs::remove_file(&path1);
    let _ = std::fs::remove_file(&path2);
}

#[test]
fn inline_sv_format_returns_unimplemented() {
    let cfg = BundleConfig {
        merge_weights: true,
        output_path: fresh_path("inline-sv").to_string_lossy().into_owned(),
        format: BundleFormat::InlineSv,
    };
    match write_bundle(&cfg, &[("dummy", b"x".as_ref())]) {
        Err(BundleError::Unimplemented(BundleFormat::InlineSv)) => {}
        other => panic!("expected Unimplemented(InlineSv), got {other:?}"),
    }
}

#[test]
fn tarball_round_trips_multiple_entries() {
    let path = fresh_path("multi");
    let cfg = BundleConfig {
        merge_weights: true,
        output_path: path.to_string_lossy().into_owned(),
        format: BundleFormat::Tarball,
    };
    let entries: Vec<(&str, &[u8])> = vec![
        ("a.txt", b"alpha"),
        ("b.bin", b"\x00\x01\x02"),
        ("nested/c.mem", b"\xde\xad\xbe\xef"),
    ];
    write_bundle(&cfg, &entries).unwrap();
    let read_back = read_bundle_entries(&path).unwrap();
    for (name, body) in &entries {
        let entry = read_back
            .iter()
            .find(|e| e.name == *name)
            .unwrap_or_else(|| panic!("{name} missing from tarball"));
        assert_eq!(entry.byte_len, body.len() as u64);
    }
    let _ = std::fs::remove_file(&path);
}
