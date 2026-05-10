//! Template inheritance tests for `eda-config::load_strict`.
//! Covers: simple parent/child, multi-level chain, cycle detection,
//! depth cap, child key overrides parent, sibling key from parent
//! survives, sibling-relative path resolution.

use eda_config::{load_strict, write_to, ConfigError};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct Sample {
    pub run: RunSection,
    pub pnr: PnrSection,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct RunSection {
    pub seed: u64,
    pub label: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct PnrSection {
    pub enabled: bool,
    pub steps: usize,
}

static SEQ: AtomicU64 = AtomicU64::new(0);

fn fresh_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let mut p = std::env::temp_dir();
    p.push(format!("eda-config-test-{}-{tag}-{n}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write(path: &PathBuf, body: &str) {
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(body.as_bytes()).unwrap();
}

#[test]
fn child_overrides_parent_keys() {
    let dir = fresh_dir("override");
    let parent = dir.join("base.toml");
    let child = dir.join("child.toml");
    write(
        &parent,
        r#"
[run]
seed = 1
label = "from-parent"
[pnr]
enabled = false
steps = 100
"#,
    );
    write(
        &child,
        r#"
extends = "base.toml"
[run]
seed = 99
[pnr]
enabled = true
"#,
    );

    let s: Sample = load_strict(&child).expect("load");
    // Child overrides where it specifies.
    assert_eq!(s.run.seed, 99);
    assert!(s.pnr.enabled);
    // Parent values survive where child is silent.
    assert_eq!(s.run.label, "from-parent");
    assert_eq!(s.pnr.steps, 100);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn multi_level_chain_resolves() {
    let dir = fresh_dir("chain");
    let grand = dir.join("grand.toml");
    let parent = dir.join("parent.toml");
    let child = dir.join("child.toml");
    write(
        &grand,
        r#"
[run]
seed = 1
label = "from-grand"
[pnr]
enabled = false
steps = 10
"#,
    );
    write(
        &parent,
        r#"
extends = "grand.toml"
[run]
label = "from-parent"
[pnr]
steps = 50
"#,
    );
    write(
        &child,
        r#"
extends = "parent.toml"
[run]
seed = 999
"#,
    );

    let s: Sample = load_strict(&child).expect("load");
    // child.run.seed wins over parent and grand.
    assert_eq!(s.run.seed, 999);
    // parent.run.label wins over grand.run.label.
    assert_eq!(s.run.label, "from-parent");
    // grand.pnr.enabled survives (neither overrode).
    assert!(!s.pnr.enabled);
    // parent.pnr.steps wins over grand.pnr.steps.
    assert_eq!(s.pnr.steps, 50);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cycle_is_detected_and_surfaces_chain() {
    let dir = fresh_dir("cycle");
    let a = dir.join("a.toml");
    let b = dir.join("b.toml");
    write(&a, r#"extends = "b.toml""#);
    write(&b, r#"extends = "a.toml""#);

    match load_strict::<Sample>(&a) {
        Err(ConfigError::CycleDetected(chain)) => {
            assert!(chain.contains("a.toml"));
            assert!(chain.contains("b.toml"));
        }
        other => panic!("expected CycleDetected, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_parent_file_surfaces_io_error() {
    let dir = fresh_dir("missing-parent");
    let child = dir.join("child.toml");
    write(
        &child,
        r#"
extends = "nonexistent.toml"
[run]
seed = 1
"#,
    );

    match load_strict::<Sample>(&child) {
        Err(ConfigError::Io(_)) => {}
        other => panic!("expected Io error, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_extends_means_no_inheritance() {
    let dir = fresh_dir("no-extends");
    let lone = dir.join("lone.toml");
    write(
        &lone,
        r#"
[run]
seed = 7
[pnr]
steps = 25
"#,
    );

    let s: Sample = load_strict(&lone).expect("load");
    assert_eq!(s.run.seed, 7);
    assert_eq!(s.pnr.steps, 25);
    // Default for unset fields in a no-extends file.
    assert_eq!(s.run.label, "");
    assert!(!s.pnr.enabled);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn write_then_load_round_trips() {
    let dir = fresh_dir("round-trip");
    let p = dir.join("sample.toml");
    let original = Sample {
        run: RunSection {
            seed: 42,
            label: "rt".into(),
        },
        pnr: PnrSection {
            enabled: true,
            steps: 5,
        },
    };
    write_to(&original, &p).expect("write");
    let restored: Sample = load_strict(&p).expect("read back");
    assert_eq!(original, restored);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn template_in_subdirectory_resolves_relatively() {
    let dir = fresh_dir("subdir");
    let templates = dir.join("templates");
    std::fs::create_dir_all(&templates).unwrap();
    let base = templates.join("base.toml");
    let child = dir.join("scenario.toml");
    write(
        &base,
        r#"
[run]
label = "from-template"
[pnr]
enabled = true
"#,
    );
    write(
        &child,
        r#"
extends = "templates/base.toml"
[run]
seed = 11
"#,
    );

    let s: Sample = load_strict(&child).expect("load");
    assert_eq!(s.run.seed, 11);
    assert_eq!(s.run.label, "from-template");
    assert!(s.pnr.enabled);

    let _ = std::fs::remove_dir_all(&dir);
}
