//! End-to-end Phase-1 validation: divider deck → ngspice + LTspice
//! → diff report.
//!
//! Each backend is gated by both:
//!   1. Cargo feature flag (`--features ngspice` / `--features ltspice`),
//!      so the workspace builds clean without either simulator installed.
//!   2. Runtime soft-skip via `LocalBinary::from_env_optional`, so the
//!      same test code runs on a developer machine that *is* missing
//!      one of the two simulators.
//!
//! ## Running
//!
//! ```sh
//! cargo test -p spike-triangulate                              # only-Rust
//! cargo test -p spike-triangulate --features ngspice           # + ngspice
//! cargo test -p spike-triangulate --features ngspice,ltspice   # both
//! ```

use spike_triangulate::Divider;

#[test]
fn deck_builds() {
    // Always-on smoke: deck construction has no SPICE dependency.
    let _ = Divider::default().deck();
}

#[cfg(feature = "ngspice")]
#[test]
fn ngspice_dc_matches_closed_form() {
    use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};

    let Some(()) = ngspice_present() else { return; };
    let inv = LocalBinary::from_env().unwrap();
    let d = Divider::default();
    let res = inv
        .run_dc(
            &d.deck().deck(),
            &[OutputRequest::NodeVoltage("mid".into())],
        )
        .expect("ngspice .op");
    let v = res.node_voltages["mid"];
    let env = 1e-3 + 1e-6 * d.mid_voltage().abs();
    assert!(
        (v - d.mid_voltage()).abs() < env,
        "ngspice mid={v} vs analytic {} (envelope {env})",
        d.mid_voltage(),
    );
}

#[cfg(feature = "ltspice")]
#[test]
fn ltspice_dc_matches_closed_form() {
    use eda_extern_ltspice::{Invoker, LocalBinary, OutputRequest};

    let Some(inv) = LocalBinary::from_env_optional() else {
        eprintln!("LTspice not installed; skipping ltspice_dc_matches_closed_form");
        return;
    };
    let d = Divider::default();
    let res = inv
        .run_dc(
            &d.deck().deck(),
            &[OutputRequest::NodeVoltage("mid".into())],
        )
        .expect("LTspice .op");
    let v = res.node_voltages["mid"];
    let env = 1e-3 + 1e-6 * d.mid_voltage().abs();
    assert!(
        (v - d.mid_voltage()).abs() < env,
        "LTspice mid={v} vs analytic {} (envelope {env})",
        d.mid_voltage(),
    );
}

/// The Phase-1 headline: same deck, two simulators, one diff report.
#[cfg(all(feature = "ngspice", feature = "ltspice"))]
#[test]
fn ngspice_and_ltspice_agree_on_divider() {
    use std::collections::HashMap;

    use eda_extern_ngspice::{Invoker as NgInvoker, LocalBinary as NgLocal, OutputRequest as NgReq};
    use eda_extern_ltspice::{Invoker as LtInvoker, LocalBinary as LtLocal, OutputRequest as LtReq};
    use eda_validate::compare_dc_voltages;

    let Some(()) = ngspice_present() else { return; };
    let Some(lt) = LtLocal::from_env_optional() else {
        eprintln!("LTspice not installed; skipping triangulation");
        return;
    };

    let d = Divider::default();
    let deck = d.deck().deck();
    let nodes = ["vin", "mid"];

    let ng = NgLocal::from_env().unwrap();
    let ng_res = ng
        .run_dc(&deck, &nodes.iter().map(|n| NgReq::NodeVoltage((*n).into())).collect::<Vec<_>>())
        .expect("ngspice");
    let lt_res = lt
        .run_dc(&deck, &nodes.iter().map(|n| LtReq::NodeVoltage((*n).into())).collect::<Vec<_>>())
        .expect("ltspice");

    // Collect into shape-compatible HashMaps for the validator.
    let to_map = |hm: &HashMap<String, f64>| -> HashMap<String, f64> {
        nodes.iter().map(|n| ((*n).to_string(), hm[*n])).collect()
    };
    let report = compare_dc_voltages(&to_map(&ng_res.node_voltages), &to_map(&lt_res.node_voltages));

    eprintln!("triangulation report: {report:#?}");
    // 1mV absolute envelope is comfortably above SPICE solver tolerance
    // for a linear divider; both simulators converge to ~µV agreement
    // in practice. If this ever fails, something fundamental moved.
    report.assert_within(1e-6, 1e-3, "ngspice vs LTspice on divider");
}

#[cfg(feature = "ngspice")]
fn ngspice_present() -> Option<()> {
    use eda_extern_ngspice::LocalBinary;
    LocalBinary::from_env().ok().map(|_| ()).or_else(|| {
        eprintln!("ngspice not installed; skipping");
        None
    })
}
