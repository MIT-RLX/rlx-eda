//! Truth-table validation: enumerate every input combination, run
//! `.op` through ngspice (and LTspice when present), assert the output
//! is at the correct rail.
//!
//! ## What "correct rail" means
//!
//! `.op` solves DC steady state, so a passing gate puts Vout within
//! `RAIL_TOL` of either 0 V (logic 0) or Vdd (logic 1). The expected
//! rail per input combo follows from the gate's truth table.
//!
//! For LEVEL=1 with Vdd=1.8 V, Vto=±0.5 V, and matched βn/βp the
//! steady-state outputs land within ~1 mV of the rails — `RAIL_TOL =
//! 50 mV` is comfortably above that and below the noise margin
//! (Vdd/2 - max(|Vtn|, |Vtp|) ≈ 400 mV), so any failure means a real
//! topology bug, not analog tolerance creep.
//!
//! ## Soft-skip
//!
//! Each backend gated by Cargo feature **and** runtime presence check.
//! On a machine without LTspice, only ngspice runs; the test still
//! validates correctness for that backend.

#![cfg(feature = "ngspice")]

const VDD: f64 = 1.8;
const RAIL_TOL: f64 = 0.05; // 50 mV

use eda_extern_ngspice::{Invoker as NgInvoker, LocalBinary as NgLocal, OutputRequest as NgReq};
use spike_cmos_gates::{deck_for_levels, And2, Inverter, Nand2, Nand3, Nor2, Or2};

/// Run one operating-point through ngspice and return Vout.
fn ngspice_vout(ng: &NgLocal, deck: &str) -> f64 {
    let res = ng
        .run_dc(deck, &[NgReq::NodeVoltage("out".into())])
        .expect("ngspice .op");
    res.node_voltages["out"]
}

/// Same shape, LTspice. Soft-skipped at the call site.
#[cfg(feature = "ltspice")]
fn ltspice_vout(lt: &eda_extern_ltspice::LocalBinary, deck: &str) -> f64 {
    use eda_extern_ltspice::{Invoker, OutputRequest};
    let res = lt
        .run_dc(deck, &[OutputRequest::NodeVoltage("out".into())])
        .expect("LTspice .op");
    res.node_voltages["out"]
}

fn assert_rail(actual: f64, expected_bit: u8, label: &str) {
    let target = if expected_bit == 1 { VDD } else { 0.0 };
    let env = RAIL_TOL;
    assert!(
        (actual - target).abs() < env,
        "{label}: got Vout = {actual:.4} V, expected ≈ {target:.1} V (env {env})",
    );
}

#[test]
fn inverter_truth_table_ngspice() {
    let Ok(ng) = NgLocal::from_env() else { eprintln!("ngspice missing"); return; };
    let inv = Inverter::default();
    for (in_bit, out_bit) in [(0u8, 1u8), (1, 0)] {
        let deck = deck_for_levels(&inv, &["a"], &[in_bit], "out", VDD, "u1").deck();
        let v = ngspice_vout(&ng, &deck);
        assert_rail(v, out_bit, &format!("ngspice INV in={in_bit}"));
    }
}

#[test]
fn nand2_truth_table_ngspice() {
    let Ok(ng) = NgLocal::from_env() else { eprintln!("ngspice missing"); return; };
    let g = Nand2::default();
    // truth table: out = !(a & b) — 1 unless both inputs are 1.
    for (a, b, expected) in [(0u8, 0u8, 1u8), (0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        let deck = deck_for_levels(&g, &["a", "b"], &[a, b], "out", VDD, "u1").deck();
        let v = ngspice_vout(&ng, &deck);
        assert_rail(v, expected, &format!("ngspice NAND2 (a={a}, b={b})"));
    }
}

#[test]
fn nand3_truth_table_ngspice() {
    let Ok(ng) = NgLocal::from_env() else { eprintln!("ngspice missing"); return; };
    let g = Nand3::default();
    // out = !(a & b & c) — 1 unless all three inputs are 1.
    for combo in 0u8..8 {
        let a = (combo >> 0) & 1;
        let b = (combo >> 1) & 1;
        let c = (combo >> 2) & 1;
        let expected = if a == 1 && b == 1 && c == 1 { 0 } else { 1 };
        let deck = deck_for_levels(&g, &["a", "b", "c"], &[a, b, c], "out", VDD, "u1").deck();
        let v = ngspice_vout(&ng, &deck);
        assert_rail(v, expected, &format!("ngspice NAND3 (a={a},b={b},c={c})"));
    }
}

#[test]
fn and2_truth_table_ngspice() {
    let Ok(ng) = NgLocal::from_env() else { eprintln!("ngspice missing"); return; };
    let g = And2::default();
    // out = a & b — 1 only when both inputs are 1.
    for (a, b, expected) in [(0u8, 0u8, 0u8), (0, 1, 0), (1, 0, 0), (1, 1, 1)] {
        let deck = deck_for_levels(&g, &["a", "b"], &[a, b], "out", VDD, "u1").deck();
        let v = ngspice_vout(&ng, &deck);
        assert_rail(v, expected, &format!("ngspice AND2 (a={a}, b={b})"));
    }
}

#[test]
fn nor2_truth_table_ngspice() {
    let Ok(ng) = NgLocal::from_env() else { eprintln!("ngspice missing"); return; };
    let g = Nor2::default();
    // out = !(a | b) — 1 only when both inputs are 0.
    for (a, b, expected) in [(0u8, 0u8, 1u8), (0, 1, 0), (1, 0, 0), (1, 1, 0)] {
        let deck = deck_for_levels(&g, &["a", "b"], &[a, b], "out", VDD, "u1").deck();
        let v = ngspice_vout(&ng, &deck);
        assert_rail(v, expected, &format!("ngspice NOR2 (a={a}, b={b})"));
    }
}

#[test]
fn or2_truth_table_ngspice() {
    let Ok(ng) = NgLocal::from_env() else { eprintln!("ngspice missing"); return; };
    let g = Or2::default();
    // out = a | b — 1 unless both inputs are 0.
    for (a, b, expected) in [(0u8, 0u8, 0u8), (0, 1, 1), (1, 0, 1), (1, 1, 1)] {
        let deck = deck_for_levels(&g, &["a", "b"], &[a, b], "out", VDD, "u1").deck();
        let v = ngspice_vout(&ng, &deck);
        assert_rail(v, expected, &format!("ngspice OR2 (a={a}, b={b})"));
    }
}

/// Triangulate every gate's truth table across both backends. When
/// LTspice is missing the test soft-skips the LTspice arm but the
/// ngspice arm still runs (covered by the per-gate tests above too).
#[cfg(feature = "ltspice")]
#[test]
fn all_gates_triangulate_ngspice_vs_ltspice() {
    use eda_extern_ltspice::LocalBinary as LtLocal;

    let Ok(ng) = NgLocal::from_env() else { eprintln!("ngspice missing"); return; };
    let Some(lt) = LtLocal::from_env_optional() else {
        eprintln!("LTspice missing; skipping all_gates_triangulate_ngspice_vs_ltspice");
        return;
    };

    type GateRow = (&'static str, Vec<(Vec<&'static str>, Vec<u8>, u8)>);
    let inverter_rows: GateRow = (
        "Inverter",
        vec![(vec!["a"], vec![0], 1), (vec!["a"], vec![1], 0)],
    );
    let nand2_rows: GateRow = (
        "Nand2",
        vec![
            (vec!["a", "b"], vec![0, 0], 1),
            (vec!["a", "b"], vec![0, 1], 1),
            (vec!["a", "b"], vec![1, 0], 1),
            (vec!["a", "b"], vec![1, 1], 0),
        ],
    );

    for (label, rows) in [inverter_rows, nand2_rows] {
        for (input_nets, levels, expected) in rows {
            let deck = match label {
                "Inverter" => deck_for_levels(&Inverter::default(), &input_nets, &levels, "out", VDD, "u1").deck(),
                "Nand2" => deck_for_levels(&Nand2::default(), &input_nets, &levels, "out", VDD, "u1").deck(),
                _ => unreachable!(),
            };
            let ng_v = ngspice_vout(&ng, &deck);
            let lt_v = ltspice_vout(&lt, &deck);
            // 1) both at expected rail.
            assert_rail(ng_v, expected, &format!("{label} ng inputs={levels:?}"));
            assert_rail(lt_v, expected, &format!("{label} lt inputs={levels:?}"));
            // 2) and they agree with each other.
            assert!(
                (ng_v - lt_v).abs() < RAIL_TOL,
                "{label} inputs={levels:?}: ngspice={ng_v:.4} vs LTspice={lt_v:.4}",
            );
        }
    }
}
