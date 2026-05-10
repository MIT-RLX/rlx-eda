//! External-validation tier — differential check against KLayout's own
//! `klayout.db` Python bindings.
//!
//! ## Pattern (mirroring klayout-rs `validation/klayout-validate`)
//!
//! 1. Build the divider Library via the trait-driven flow.
//! 2. Compute our canonical JSON dump via `klayout_validate::canonical_dump`.
//! 3. Write GDS via `klayout_io::write_gds_path` to a temp file.
//! 4. Invoke the oracle script (`oracle.py verify <gds>`) which uses
//!    Python `klayout.db` to read the same GDS and emit *its* canonical
//!    JSON dump.
//! 5. Compare the two dumps for structural equality.
//!
//! ## Soft skip when no KLayout Python
//!
//! `klayout_validate::klayout_python()` returns `None` if neither the
//! `KLAYOUT_PYTHON` env var nor `/tmp/klayout-venv/bin/python` resolves.
//! In that case the test prints a `skipping` line and returns `Ok` —
//! CI installs klayout's Python bindings; local dev can run the rest of
//! the suite without them.

use klayout_io::write_gds_path;
use klayout_validate::{canonical_dump, klayout_canonical_dump, klayout_python};
use serde_json::Value;
use spike_divider_block::*;

/// Make the cell-array order canonical (sort by `name`) before comparing.
/// klayout-rs and klayout.db happen to emit cells in different orders — a
/// real upstream parity gap, but not one we should let block our LVS
/// tier from catching genuine geometry / property differences.
fn sort_cells_by_name(v: &mut Value) {
    if let Value::Object(obj) = v {
        if let Some(Value::Array(cells)) = obj.get_mut("cells") {
            cells.sort_by(|a, b| {
                a.get("name").and_then(|n| n.as_str()).unwrap_or("")
                    .cmp(b.get("name").and_then(|n| n.as_str()).unwrap_or(""))
            });
        }
    }
}

#[test]
fn divider_layout_matches_klayout_python_oracle() {
    if klayout_python().is_none() {
        eprintln!(
            "skipping: no KLayout-equipped python \
             (set KLAYOUT_PYTHON or install /tmp/klayout-venv with klayout pip pkg)"
        );
        return;
    }

    let (lib, _pdk, _top) = make_divider_layout(10_000, 30_000);

    let tmp = tempfile::tempdir().expect("mktempdir");
    let gds_path = tmp.path().join("divider_oracle.gds");
    write_gds_path(&lib, &gds_path).expect("write GDS");

    let mut ours   = canonical_dump(&lib);
    let mut theirs = klayout_canonical_dump(&gds_path)
        .expect("klayout python should be available — checked above");
    sort_cells_by_name(&mut ours);
    sort_cells_by_name(&mut theirs);

    if ours != theirs {
        let o = serde_json::to_string_pretty(&ours).unwrap();
        let t = serde_json::to_string_pretty(&theirs).unwrap();
        panic!(
            "klayout-rs ↔ klayout.db oracle disagreement:\n\n--- ours ---\n{o}\n\n--- klayout.db ---\n{t}\n"
        );
    }
}

#[test]
fn resistor_cell_alone_matches_oracle() {
    // Smaller failure surface: just the primitive resistor, no routing.
    // If both this and the divider test pass, we know both the leaf
    // geometry AND the hierarchical instances + ports survive the
    // GDS roundtrip identically.
    if klayout_python().is_none() {
        eprintln!("skipping: no KLayout python");
        return;
    }

    use eda_hir::Layout;
    let lib = RcDemo::new_library("oracle_leaf");
    let pdk = RcDemo::register(&lib);
    let r = Resistor { length: 5_000, id: "test".into() };
    let _id = r.layout(&lib, &pdk);

    let tmp = tempfile::tempdir().expect("mktempdir");
    let gds_path = tmp.path().join("resistor_oracle.gds");
    write_gds_path(&lib, &gds_path).expect("write GDS");

    let mut ours   = canonical_dump(&lib);
    let mut theirs = klayout_canonical_dump(&gds_path).expect("klayout python");
    sort_cells_by_name(&mut ours);
    sort_cells_by_name(&mut theirs);
    assert_eq!(ours, theirs,
        "resistor cell oracle mismatch:\nours: {}\ntheirs: {}",
        serde_json::to_string_pretty(&ours).unwrap(),
        serde_json::to_string_pretty(&theirs).unwrap(),
    );
}
