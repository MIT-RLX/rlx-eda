//! `BenchConfig` round-trip + per-section override tests.
//! Loading machinery now lives in `eda-config`; consumer-side
//! tests cover the `Configurable` trait wiring + per-section
//! defaults.

use eda_bench_tinyconv::config::{BenchConfig, RunConfig};
use eda_bench_tinyconv::pnr::PnrMode;
use eda_config::{from_toml, load_or_default, to_toml, write_to, Configurable};

#[test]
fn default_config_round_trips_through_toml() {
    let original = BenchConfig::default();
    let s = to_toml(&original).expect("serialize");
    let restored: BenchConfig = from_toml(&s).expect("toml parses");
    assert_eq!(original.run.seed, restored.run.seed);
    assert_eq!(original.loss_weights.alpha_energy, restored.loss_weights.alpha_energy);
    assert_eq!(original.inner.max_steps, restored.inner.max_steps);
    assert_eq!(original.outer.orfs_cadence, restored.outer.orfs_cadence);
    assert_eq!(original.pnr.enabled, restored.pnr.enabled);
    assert_eq!(original.noise.vdd_nominal, restored.noise.vdd_nominal);
}

#[test]
fn empty_toml_loads_as_default() {
    let cfg: BenchConfig = from_toml("").expect("empty toml ok");
    let dflt = BenchConfig::default();
    assert_eq!(cfg.run.seed, dflt.run.seed);
    assert_eq!(cfg.inner.max_steps, dflt.inner.max_steps);
}

#[test]
fn partial_toml_overrides_only_named_sections() {
    let toml = r#"
[run]
seed = 99

[pnr]
enabled = true
"#;
    let cfg: BenchConfig = from_toml(toml).expect("partial parses");
    assert_eq!(cfg.run.seed, 99);
    assert!(cfg.pnr.enabled);
    assert_eq!(cfg.inner.max_steps, BenchConfig::default().inner.max_steps);
    assert_eq!(cfg.loss_weights.alpha_energy, 1.0);
}

#[test]
fn pnr_mode_projection_respects_enabled_flag() {
    let mut cfg = BenchConfig::default();
    cfg.pnr.enabled = false;
    assert!(matches!(cfg.pnr_mode(), PnrMode::Disabled));

    cfg.pnr.enabled = true;
    cfg.pnr.adam.max_steps = 50;
    match cfg.pnr_mode() {
        PnrMode::AdamHpwl(adam) => assert_eq!(adam.max_steps, 50),
        _ => panic!("expected AdamHpwl"),
    }
}

#[test]
fn load_or_default_returns_default_for_missing_path() {
    let cfg: BenchConfig = load_or_default(std::path::Path::new(
        "/nonexistent/never/exists/bench.toml",
    ));
    assert_eq!(cfg.run.seed, 0);
}

#[test]
fn load_or_default_reads_real_file() {
    use std::io::Write;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut p = std::env::temp_dir();
    p.push(format!("bench-config-load-{}-{n}", std::process::id()));
    std::fs::File::create(&p)
        .unwrap()
        .write_all(b"[run]\nseed = 7\n")
        .unwrap();

    let cfg: BenchConfig = load_or_default(&p);
    assert_eq!(cfg.run.seed, 7);
    let _ = std::fs::remove_file(&p);
}

#[test]
fn write_to_creates_parent_directories() {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut p = std::env::temp_dir();
    p.push(format!("bench-config-write-{}-{n}", std::process::id()));
    p.push("nested");
    p.push("bench.toml");

    let cfg = BenchConfig {
        run: RunConfig {
            seed: 42,
            output_path: "out.md".into(),
        },
        ..BenchConfig::default()
    };
    write_to(&cfg, &p).expect("write succeeds");

    let restored: BenchConfig = load_or_default(&p);
    assert_eq!(restored.run.seed, 42);
    assert_eq!(restored.run.output_path, "out.md");

    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_dir(p.parent().unwrap());
}

#[test]
fn example_toml_in_crate_root_parses_cleanly() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench.example.toml");
    let text = std::fs::read_to_string(&path).expect("example file readable");
    let _cfg: BenchConfig = from_toml(&text).expect("example toml parses");
}

#[test]
fn configurable_trait_filename_and_env_var_are_correct() {
    assert_eq!(BenchConfig::FILENAME, "bench.toml");
    assert_eq!(BenchConfig::env_var_name(), "EDA_CONFIG_BENCH");
}

#[test]
fn bench_config_inherits_from_template() {
    use eda_config::load_strict;
    use std::io::Write;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut dir = std::env::temp_dir();
    dir.push(format!("bench-config-inherit-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Template = "tt corner" defaults.
    let template = dir.join("tt.toml");
    std::fs::File::create(&template)
        .unwrap()
        .write_all(
            br#"
[run]
seed = 1
output_path = "tt-report.md"

[loss_weights]
alpha_energy = 2.0
beta_delay = 1.0

[pnr]
enabled = true
"#,
        )
        .unwrap();

    // Scenario file overlays just the seed.
    let scenario = dir.join("scenario.toml");
    std::fs::File::create(&scenario)
        .unwrap()
        .write_all(
            br#"
extends = "tt.toml"
[run]
seed = 42
"#,
        )
        .unwrap();

    let cfg: BenchConfig = load_strict(&scenario).expect("inherit + parse");
    assert_eq!(cfg.run.seed, 42, "scenario overrides template seed");
    assert_eq!(cfg.run.output_path, "tt-report.md", "template path survives");
    assert_eq!(cfg.loss_weights.alpha_energy, 2.0, "template loss_weights survive");
    assert_eq!(cfg.loss_weights.beta_delay, 1.0);
    assert!(cfg.pnr.enabled, "template pnr.enabled survives");
    // Default fields untouched by both.
    assert_eq!(cfg.inner.max_steps, BenchConfig::default().inner.max_steps);

    let _ = std::fs::remove_dir_all(&dir);
}
