//! ngspice-driven SPICE evaluator.
//!
//! For each design, instantiates `SarAdc<4>` with catalog values plugged
//! into the matching sub-block fields, drives a small set of `vin`
//! levels through ngspice, samples the digital codes, and scores by
//! mean squared (digital − ideal) error.
//!
//! ## Per-clique decomposition (approximate)
//!
//! Static-input MSE on a SAR ADC is mostly comparator + DAC limited.
//! Sample-hold and SAR-logic effects show up in dynamic behaviour we
//! don't exercise here. We attribute each conversion's squared error
//! to the comparator/DAC cliques in equal portions, leaving SH + SAR
//! at zero. This is a documented approximation — it's the SPICE-side
//! analogue of the per-bit max-INL attribution that didn't help DADO
//! in the prior `spike-dado-r2r` experiment, and we expect a similar
//! "no real DADO advantage" outcome on B.
//!
//! ## Per-design cost
//!
//! `score_spice` runs `n_vins` separate ngspice transients (one per
//! input level) so each transient is the existing single-conversion
//! pattern from `spike-sar-adc/src/bin/conversion_trace.rs`. Default
//! `n_vins = 4` keeps a design eval to ~1 ngspice-second × 4.

#![cfg(feature = "ngspice")]

use eda_extern_ngspice::{Invoker, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pwl, SpiceEmit};
use spike_sar_adc::{ideal_sar_code, SarAdc};

use crate::catalog::{
    self, Design, CL_COMP, CL_DAC, CL_SAR, CL_SH, N_CLIQUES,
};

const N_BITS: usize = 4;
/// Fixed supply rail used by the SPICE deck. The catalog `I_VREF`
/// variable is analytical-only — varying the SAR ADC's actual supply
/// in SPICE would push CMOS gate threshold margins around (`Vt = 0.5
/// V`, so `vdd ≤ 1.0 V` makes SAR latches unreliable). Pin to the
/// supply that `conversion_trace.rs` validates against and treat the
/// catalog `vref` as a separate analytical parameter, alongside
/// `dac_match_pct`.
const VDD: f64 = 1.8;

/// Build a `SarAdc<4>` with sub-block parameters set from `design`.
fn build_adc(design: &Design) -> SarAdc<4> {
    let mut adc: SarAdc<4> = SarAdc::default();

    adc.sh.c_hold     = catalog::c_hold(design[catalog::I_C_HOLD]);
    adc.sh.nmos.w     = catalog::sh_nmos_w(design[catalog::I_SH_NMOS_W]);
    adc.sh.pmos.w     = catalog::sh_pmos_w(design[catalog::I_SH_PMOS_W]);
    adc.sh.nmos.l     = catalog::sh_l(design[catalog::I_SH_L]);
    adc.sh.pmos.l     = catalog::sh_l(design[catalog::I_SH_L]);

    adc.comp.k        = catalog::comp_k(design[catalog::I_COMP_K]);
    adc.comp.voh      = catalog::comp_voh(design[catalog::I_COMP_VOH]);
    adc.comp.vol      = catalog::comp_vol(design[catalog::I_COMP_VOL]);

    adc.dac.r_ohms    = catalog::dac_r_ohms(design[catalog::I_DAC_R_OHMS]);

    let nand_w = catalog::sar_nand_w(design[catalog::I_SAR_NAND_W]);
    let inv_w  = catalog::sar_inv_w(design[catalog::I_SAR_INV_W]);
    adc.sar.stage.master.nand2.nmos.w = nand_w;
    adc.sar.stage.master.nand2.pmos.w = 2.0 * nand_w;
    adc.sar.stage.master.nand3.nmos.w = nand_w;
    adc.sar.stage.master.nand3.pmos.w = 2.0 * nand_w;
    adc.sar.stage.slave.nand2.nmos.w  = nand_w;
    adc.sar.stage.slave.nand2.pmos.w  = 2.0 * nand_w;
    adc.sar.stage.slave.nand3.nmos.w  = nand_w;
    adc.sar.stage.slave.nand3.pmos.w  = 2.0 * nand_w;
    adc.sar.inv.nmos.w = inv_w;
    adc.sar.inv.pmos.w = 2.0 * inv_w;

    adc
}

/// Build a single-conversion deck — same shape as
/// `spike-sar-adc/src/bin/conversion_trace.rs` but parameterised over
/// the ADC instance, vin, and supply rail (`vdd_v`, used as both the
/// vdd source and the ADC's effective vref since the SAR DAC swings
/// rail-to-rail).
fn deck_for_conversion(adc: &SarAdc<4>, vin: f64, vdd_v: f64) -> String {
    // Phase + capture window timing — match conversion_trace defaults.
    let bit_starts = [1.1e-6, 2.2e-6, 3.3e-6, 4.4e-6];
    let phase_w = 0.8e-6;
    let cap_off = 0.85e-6;
    let cap_w   = 0.10e-6;

    let mut net = Netlist::new("DADO SAR ADC eval");
    net.add_dc_source("dd", "vdd", "0", vdd_v);
    net.add_dc_source("in", "vin", "0", vin);
    net.add_pwl_source("rb", "reset_b", "0", &Pwl { points: vec![
        (0.0, 0.0), (0.4e-6 - 5e-9, 0.0), (0.4e-6, vdd_v), (10.0, vdd_v),
    ]});
    net.add_pwl_source("clk", "clk_sh", "0", &Pwl { points: vec![
        (0.0, 0.0), (0.5e-6 - 5e-9, 0.0), (0.5e-6, vdd_v),
        (1.0e-6 - 5e-9, vdd_v), (1.0e-6, 0.0), (10.0, 0.0),
    ]});
    let phase_pwl = |start: f64| Pwl { points: vec![
        (0.0, 0.0), (start - 5e-9, 0.0), (start, vdd_v),
        (start + phase_w - 5e-9, vdd_v), (start + phase_w, 0.0), (10.0, 0.0),
    ]};
    let cap_pwl = |start: f64| {
        let cs = start + cap_off; let ce = cs + cap_w;
        Pwl { points: vec![
            (0.0, 0.0), (cs - 5e-9, 0.0), (cs, vdd_v),
            (ce - 5e-9, vdd_v), (ce, 0.0), (10.0, 0.0),
        ]}
    };
    net.add_pwl_source("p3", "p3", "0", &phase_pwl(bit_starts[0]));
    net.add_pwl_source("p2", "p2", "0", &phase_pwl(bit_starts[1]));
    net.add_pwl_source("p1", "p1", "0", &phase_pwl(bit_starts[2]));
    net.add_pwl_source("p0", "p0", "0", &phase_pwl(bit_starts[3]));
    net.add_pwl_source("c3", "c3", "0", &cap_pwl(bit_starts[0]));
    net.add_pwl_source("c2", "c2", "0", &cap_pwl(bit_starts[1]));
    net.add_pwl_source("c1", "c1", "0", &cap_pwl(bit_starts[2]));
    net.add_pwl_source("c0", "c0", "0", &cap_pwl(bit_starts[3]));

    adc.emit_spice(&mut net, &[
        "vin",
        "p0", "p1", "p2", "p3",
        "c0", "c1", "c2", "c3",
        "clk_sh", "reset_b",
        "b0", "b1", "b2", "b3",
        "vdd", "0",
    ], "u1").expect("emit_spice");
    net.deck()
}

fn lerp(xs: &[f64], ys: &[f64], xq: f64) -> f64 {
    if xq <= xs[0] { return ys[0]; }
    if xq >= xs[xs.len() - 1] { return ys[ys.len() - 1]; }
    let i = match xs.binary_search_by(|x| x.partial_cmp(&xq).unwrap()) {
        Ok(j) => return ys[j],
        Err(j) => j - 1,
    };
    let t = (xq - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + t * (ys[i + 1] - ys[i])
}

/// Sample digital code at the end of the conversion (after the LSB
/// capture pulse). Threshold is half-rail. Returns `Some(code)` on
/// success, `None` if any bit node was missing from the trace.
fn extract_code(trace: &eda_extern_ngspice::TransientTrace, vdd_v: f64) -> Option<u32> {
    let bit_starts = [1.1e-6, 2.2e-6, 3.3e-6, 4.4e-6];
    let cap_off = 0.85e-6; let cap_w = 0.10e-6;
    let t_after_lsb_cap = bit_starts[3] + cap_off + cap_w + 50e-9;
    let mut code = 0u32;
    for (i, k) in ["b0", "b1", "b2", "b3"].iter().enumerate() {
        let v = lerp(&trace.time, trace.node_voltages.get(*k)?, t_after_lsb_cap);
        if v >= vdd_v / 2.0 { code |= 1 << i; }
    }
    Some(code)
}

/// Vin levels to evaluate at. Spaced to land *between* code boundaries so
/// the ideal code is unambiguous, and to span the bit positions (so an
/// MSB error vs LSB error are both possible).
fn vin_grid(vref: f64, n_vins: usize) -> Vec<f64> {
    let lsb = vref / (1u32 << N_BITS) as f64;
    (0..n_vins)
        .map(|i| {
            let frac = (2 * i + 1) as f64 / (2 * n_vins) as f64;   // (1/2N, 3/2N, ...)
            (frac * vref).clamp(lsb * 0.5, vref - lsb * 0.5)
        })
        .collect()
}

/// SPICE score: mean squared (digital − ideal) over `n_vins` levels.
/// Per-clique decomposition splits each squared error 50/50 between the
/// comparator and DAC cliques (see module docs).
pub fn score_spice<I: Invoker + ?Sized>(
    invoker: &I,
    design: &Design,
    n_vins: usize,
) -> (f64, [f64; N_CLIQUES]) {
    let adc = build_adc(design);
    // SPICE always uses VDD as the effective vref — see the const docs.
    let vref = VDD;
    let levels = vin_grid(vref, n_vins);

    let analysis = TransientAnalysis::new(5e-9, 6.0e-6).with_t_max(5e-9);
    let requests = vec![
        OutputRequest::NodeVoltage("u1_vhold".into()),
        OutputRequest::NodeVoltage("u1_vdac".into()),
        OutputRequest::NodeVoltage("b0".into()),
        OutputRequest::NodeVoltage("b1".into()),
        OutputRequest::NodeVoltage("b2".into()),
        OutputRequest::NodeVoltage("b3".into()),
    ];

    let mut total_sq = 0.0_f64;
    let mut comps = [0.0_f64; N_CLIQUES];
    let mut n_ok = 0_usize;

    for &vin in &levels {
        let deck = deck_for_conversion(&adc, vin, vref);
        let trace = match invoker.run_transient_trace(&deck, &analysis, &requests) {
            Ok(t) => t,
            Err(_) => continue, // bad design: ngspice convergence failure
        };
        let Some(code) = extract_code(&trace, vref) else { continue; };
        let ideal = ideal_sar_code(vin, vref, N_BITS);
        let err = (code as f64 - ideal as f64).abs();
        let err_sq = err * err;
        total_sq += err_sq;
        comps[CL_COMP] -= err_sq * 0.5;
        comps[CL_DAC]  -= err_sq * 0.5;
        // SH and SAR remain 0 — see module docs.
        n_ok += 1;
        let _ = (CL_SH, CL_SAR); // silence unused warn until decomposition is richer
    }

    if n_ok == 0 {
        // Every transient failed — return worst-possible score so the
        // optimizer learns to avoid this region.
        let big = -1e6;
        return (big, [big / N_CLIQUES as f64; N_CLIQUES]);
    }
    let score = -total_sq / n_ok as f64;
    // Normalise components by n_ok too so total ≈ score.
    for c in comps.iter_mut() { *c /= n_ok as f64; }
    (score, comps)
}

/// Build (or reuse) a Docker invoker if `RLX_NGSPICE_BACKEND=docker`,
/// else fall back to the local binary.
pub fn invoker_from_env()
    -> Result<Box<dyn Invoker>, eda_extern_ngspice::NgspiceError>
{
    use eda_extern_ngspice::{DockerInvoker, LocalBinary};
    match std::env::var("NGSPICE_BACKEND").as_deref() {
        Ok("docker") => {
            let d = DockerInvoker::from_env()?;
            d.ensure_image()?;
            Ok(Box::new(d))
        }
        _ => Ok(Box::new(LocalBinary::from_env()?)),
    }
}
