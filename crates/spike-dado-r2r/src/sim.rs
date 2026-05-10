//! ngspice cross-validation for a perturbed R-2R DAC.
//!
//! Builds a netlist that mirrors `spike_dac_r2r::R2RDac::emit_spice` but
//! with a *per-resistor* ohm value taken from a `Design` — so each of
//! the 16 resistors carries its actual perturbed resistance, not the
//! nominal R2RDac single `r_ohms` parameter. We then sweep all 256
//! codes through ngspice, collecting `vout`, and compare against the
//! analytical `solve_r2r` evaluator. The point is to confirm that our
//! linear nodal-analysis solver agrees with an independent simulator on
//! the same perturbed network.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, LocalBinary, NgspiceError, OutputRequest};
use eda_spice_emit::{Netlist, SpiceEmit, R};

use crate::{
    r_in_idx, r_sp_idx, r_term_idx, r_value, solve_r2r, Design, N_BITS, N_CODES, N_NODES,
};

const VREF: f64 = 1.0;
const VLOW: f64 = 0.0;

/// Build a SPICE deck for one input code with the resistor values
/// implied by `design`.
pub fn deck_for_design_and_code(design: &Design, code: u32) -> String {
    let mut net = Netlist::new("DADO R-2R DAC (perturbed)");

    // DC sources: vref, vlow, and one per input bit.
    net.add_dc_source("ref", "vref", "0", VREF);
    net.add_dc_source("low", "vlow", "0", VLOW);
    for i in 0..N_BITS {
        let v = if (code >> i) & 1 == 1 { VREF } else { VLOW };
        net.add_dc_source(&format!("b{i}"), &format!("in{i}"), "0", v);
    }

    // Internal node names — n_0..n_{N-1}, with n_{N-1} aliased as `vout`.
    let node_name = |i: usize| -> String {
        if i == N_NODES - 1 { "vout".to_string() } else { format!("n{i}") }
    };

    // r_term: vlow -- 2R(perturbed) -- n_0
    R { ohms: r_value(design, r_term_idx()) }
        .emit_spice(&mut net, &["vlow", &node_name(0)], "term").unwrap();

    // r_in[i]: in_i -- 2R(perturbed) -- n_i
    for i in 0..N_BITS {
        let r = R { ohms: r_value(design, r_in_idx(i)) };
        let in_name = format!("in{i}");
        let n_i = node_name(i);
        r.emit_spice(&mut net, &[&in_name, &n_i], &format!("in{i}")).unwrap();
    }

    // r_sp[s]: n_s -- R(perturbed) -- n_{s+1}
    for s in 0..(N_NODES - 1) {
        let r = R { ohms: r_value(design, r_sp_idx(s)) };
        let n_a = node_name(s);
        let n_b = node_name(s + 1);
        r.emit_spice(&mut net, &[&n_a, &n_b], &format!("sp{s}")).unwrap();
    }

    net.deck()
}

/// One row of the cross-validation report.
#[derive(Clone, Copy, Debug)]
pub struct CrossvalRow {
    pub code: u32,
    pub analytical_vout: f64,
    pub ngspice_vout: f64,
    pub ideal_vout: f64,
}

/// Sweep all 256 codes through ngspice for the perturbed `design`,
/// returning per-code analytical/ngspice/ideal voltages plus the max
/// |analytical - ngspice| residual (which should be ≪ 1 µV if the two
/// solvers agree to numerical precision).
pub fn cross_validate(
    ng: &LocalBinary,
    design: &Design,
) -> Result<(Vec<CrossvalRow>, f64), NgspiceError> {
    use spike_dac_r2r::ideal_vout;
    let mut rows = Vec::with_capacity(N_CODES);
    let mut max_resid = 0.0_f64;
    for code in 0..N_CODES as u32 {
        let deck = deck_for_design_and_code(design, code);
        let res = ng.run_dc(&deck, &[OutputRequest::NodeVoltage("vout".into())])?;
        let ng_v = res.node_voltages["vout"];
        let an_v = solve_r2r(design, code, VREF, VLOW);
        let id_v = ideal_vout(code, N_BITS as u32, VREF, VLOW);
        max_resid = max_resid.max((ng_v - an_v).abs());
        rows.push(CrossvalRow {
            code,
            analytical_vout: an_v,
            ngspice_vout: ng_v,
            ideal_vout: id_v,
        });
    }
    Ok((rows, max_resid))
}

/// Pretty-print a crossval report (max residual + a few code samples)
/// for inclusion in a `validation.txt` artifact.
pub fn report_text(design: &Design, rows: &[CrossvalRow], max_resid: f64) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "ngspice ↔ analytical-MNA cross-validation");
    let _ = writeln!(s, "------------------------------------------");
    let _ = writeln!(s, "design: {design:?}");
    let _ = writeln!(s, "max |ngspice - analytical| over all 256 codes: {max_resid:.3e} V");
    let _ = writeln!(s);
    let _ = writeln!(s, "  code |    analytical |       ngspice |        ideal | err vs ideal");
    for &c in &[0u32, 1, 64, 127, 128, 192, 255] {
        if let Some(r) = rows.iter().find(|r| r.code == c) {
            let _ = writeln!(
                s, "  {:>4} | {:>12.7} | {:>12.7} | {:>11.7} | {:>+11.4e}",
                r.code, r.analytical_vout, r.ngspice_vout, r.ideal_vout,
                r.analytical_vout - r.ideal_vout,
            );
        }
    }
    s
}
