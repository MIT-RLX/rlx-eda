//! End-to-end CLI test: register a synthetic PDK, list it, show it,
//! forget it. Auto-section discovery exercised against a tempdir
//! `.lib` file so we don't depend on any installed PDK.
//!
//! Drives the actual `rlx-eda` binary via `Command`, with the registry
//! redirected via `RLX_EDA_PDK_CONFIG` so we don't clobber the user's
//! real `~/.config/rlx-eda/pdks.toml`.

use std::process::Command;

fn rlx_eda_bin() -> std::path::PathBuf {
    // Cargo sets CARGO_BIN_EXE_<name> for binary crates' own integration
    // tests.
    let p = env!("CARGO_BIN_EXE_rlx-eda");
    std::path::PathBuf::from(p)
}

fn run(bin: &std::path::Path, cfg: &std::path::Path, args: &[&str]) -> (std::process::ExitStatus, String, String) {
    let out = Command::new(bin)
        .env("RLX_EDA_PDK_CONFIG", cfg)
        // Make sure the test doesn't cross-contaminate against the
        // user's real ciel/volare installs. Empty PDK_ROOT + HOME
        // pointing into a tempdir means the ciel scanner finds nothing.
        .env("PDK_ROOT", "/nonexistent-pdk-root-for-test")
        .env_remove("HOME")
        .args(args)
        .output()
        .expect("rlx-eda exec");
    (
        out.status,
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn register_list_show_forget_roundtrip() {
    let bin = rlx_eda_bin();
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("pdks.toml");
    let lib = dir.path().join("fake.lib.spice");
    std::fs::write(&lib, r#"
* synthetic test PDK
.lib tt
.param mc=0
.endl tt
.lib ff
.endl ff
.lib ss
.endl ss
"#).unwrap();

    // 1. register
    let (st, stdout, stderr) = run(&bin, &cfg, &[
        "pdk", "register",
        "fake_180",
        "--lib", lib.to_str().unwrap(),
        "--vdd", "1.8",
    ]);
    assert!(st.success(), "register failed: stderr={stderr}\nstdout={stdout}");
    assert!(stdout.contains("registered fake_180"), "stdout={stdout}");
    assert!(stdout.contains("tt, ff, ss"), "auto-detected sections missing in stdout={stdout}");

    // 2. list — sees the registered entry; ciel scan returns nothing
    //    because PDK_ROOT/HOME are bogus.
    let (st, stdout, _) = run(&bin, &cfg, &["pdk", "list"]);
    assert!(st.success());
    assert!(stdout.contains("fake_180"), "list missing entry: {stdout}");
    assert!(stdout.contains("user"), "user source not labeled: {stdout}");

    // 3. show — verify all fields round-trip.
    let (st, stdout, _) = run(&bin, &cfg, &["pdk", "show", "fake_180"]);
    assert!(st.success());
    assert!(stdout.contains("name:      fake_180"));
    assert!(stdout.contains("source:    user-registered"));
    assert!(stdout.contains("vdd_nom:   1.8 V"));
    assert!(stdout.contains("- tt"));
    assert!(stdout.contains("- ff"));
    assert!(stdout.contains("- ss"));

    // 4. show against an unknown name fails cleanly.
    let (st, _, stderr) = run(&bin, &cfg, &["pdk", "show", "nope_pdk"]);
    assert!(!st.success());
    assert!(stderr.contains("not found"), "stderr={stderr}");

    // 5. forget — entry vanishes; list shows nothing.
    let (st, stdout, _) = run(&bin, &cfg, &["pdk", "forget", "fake_180"]);
    assert!(st.success(), "forget failed: {stdout}");
    let (st, stdout, _) = run(&bin, &cfg, &["pdk", "list"]);
    assert!(st.success());
    assert!(!stdout.contains("fake_180"), "fake_180 still listed after forget: {stdout}");

    // 6. forget on already-removed PDK errors.
    let (st, _, stderr) = run(&bin, &cfg, &["pdk", "forget", "fake_180"]);
    assert!(!st.success());
    assert!(stderr.contains("not registered"), "stderr={stderr}");
}

#[test]
fn register_with_explicit_sections_overrides_autodetect() {
    let bin = rlx_eda_bin();
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("pdks.toml");
    let lib = dir.path().join("fake.lib.spice");
    // No `.lib` headers in the file — auto-detect would yield zero
    // sections. The explicit --sections must override.
    std::fs::write(&lib, "* no sections in here\n.model NMOS NMOS LEVEL=1\n").unwrap();

    let (st, stdout, stderr) = run(&bin, &cfg, &[
        "pdk", "register",
        "vendor_x",
        "--lib", lib.to_str().unwrap(),
        "--sections", "nom, hot, cold",
        "--vdd", "3.3",
    ]);
    assert!(st.success(), "register failed: stderr={stderr}\nstdout={stdout}");

    let (st, stdout, _) = run(&bin, &cfg, &["pdk", "show", "vendor_x"]);
    assert!(st.success());
    assert!(stdout.contains("- nom"));
    assert!(stdout.contains("- hot"));
    assert!(stdout.contains("- cold"));
    assert!(stdout.contains("vdd_nom:   3.3 V"));
}

#[test]
fn register_without_sections_and_no_headers_errors() {
    let bin = rlx_eda_bin();
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("pdks.toml");
    let lib = dir.path().join("flat.lib");
    std::fs::write(&lib, "* nothing here\n").unwrap();

    let (st, _, stderr) = run(&bin, &cfg, &[
        "pdk", "register",
        "empty_pdk",
        "--lib", lib.to_str().unwrap(),
    ]);
    assert!(!st.success(), "register should have errored");
    assert!(stderr.contains("no .lib sections"), "expected sections error, got: {stderr}");
}
