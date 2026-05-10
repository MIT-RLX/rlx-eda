//! Manifest tests — exercise pure helpers without depending on git,
//! docker, ngspice, or any specific PDK checkout being present.

use eda_bench_tinyconv::manifest::{sha256_file, Manifest, ManifestError, ManifestInputs};
use std::io::Write;
use std::path::PathBuf;

fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("eda-bench-tinyconv-test-{name}-{}", std::process::id()));
    let mut f = std::fs::File::create(&p).expect("create temp");
    f.write_all(bytes).expect("write temp");
    p
}

#[test]
fn sha256_file_is_stable() {
    let p = write_temp("sha-stable", b"hello, sky130");
    let h1 = sha256_file(&p).expect("hash");
    let h2 = sha256_file(&p).expect("hash again");
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 64, "hex SHA-256 is 64 chars");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn sha256_file_changes_with_content() {
    let a = write_temp("sha-a", b"alpha");
    let b = write_temp("sha-b", b"beta");
    assert_ne!(sha256_file(&a).unwrap(), sha256_file(&b).unwrap());
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
fn sha256_file_errors_on_missing() {
    let p = PathBuf::from("/nonexistent/path/that/does/not/exist");
    assert!(sha256_file(&p).is_err());
}

#[test]
fn manifest_captures_with_only_cargo_lock_required() {
    // Use this crate's own test source as a "Cargo.lock surrogate" —
    // we just need a real, readable file. Optional inputs are all
    // None so the manifest soft-fills them with `(unavailable)`.
    let cargo_lock = write_temp("fake-cargo-lock", b"[[package]]\nname = \"x\"\n");

    let inputs = ManifestInputs {
        sky130_repo: None,
        orfs_image: None,
        weights: None,
        cargo_lock: &cargo_lock,
        seed: 42,
    };
    let m = Manifest::capture(inputs).expect("capture");
    assert_eq!(m.optimizer_seed, 42);
    assert_eq!(m.sky130_commit, "(unavailable)");
    assert_eq!(m.orfs_image, "(unavailable)");
    assert_eq!(m.weights_sha256, "(unavailable)");
    assert_eq!(m.cargo_lock_sha256.len(), 64);
    // ngspice may or may not be on PATH; either way capture succeeds.
    assert!(!m.ngspice_version.is_empty());

    let _ = std::fs::remove_file(&cargo_lock);
}

#[test]
fn manifest_fails_when_cargo_lock_missing() {
    let inputs = ManifestInputs {
        sky130_repo: None,
        orfs_image: None,
        weights: None,
        cargo_lock: std::path::Path::new("/nonexistent/Cargo.lock"),
        seed: 0,
    };
    match Manifest::capture(inputs) {
        Err(ManifestError::NoCargoLock) => {}
        other => panic!("expected NoCargoLock, got {other:?}"),
    }
}

#[test]
fn manifest_round_trips_through_json() {
    let cargo_lock = write_temp("rt-cargo-lock", b"x");
    let m = Manifest::capture(ManifestInputs {
        sky130_repo: None,
        orfs_image: None,
        weights: None,
        cargo_lock: &cargo_lock,
        seed: 7,
    })
    .expect("capture");

    let json = serde_json::to_string(&m).expect("ser");
    let m2: Manifest = serde_json::from_str(&json).expect("de");
    assert_eq!(m.optimizer_seed, m2.optimizer_seed);
    assert_eq!(m.cargo_lock_sha256, m2.cargo_lock_sha256);
    let _ = std::fs::remove_file(&cargo_lock);
}
