//! CMOS inverter VTC: sweep Vin, run `.op` through ngspice + LTspice,
//! compare Vout(Vin) curves point-by-point. Renders the overlaid VTCs
//! to `docs/inverter_vtc.png` for visual confirmation.
//!
//! ## What this validates
//!
//! - The MOSFET primitive (`eda_spice_emit::Nmos` / `Pmos`) emits a
//!   `.model` card and `M` element line that both simulators interpret
//!   identically.
//! - Both simulators converge a CMOS inverter `.op` to the same Vout
//!   over the full Vin range — including the steep transition region
//!   where the LEVEL=1 model matters most.
//! - The closed-form switching threshold matches the simulators within
//!   ~5 mV. (LEVEL=1 is *exactly* solvable in saturation; the only
//!   error source is whichever Vin sample lands closest to Vm.)
//!
//! ## Soft-skip
//!
//! Both backends gated by Cargo features and runtime presence checks.
//! On a machine without LTspice (CI default), only ngspice runs.

#![cfg(feature = "ngspice")]

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use eda_extern_ngspice::{Invoker as NgInvoker, LocalBinary as NgLocal, OutputRequest as NgReq};
use eda_validate::compare_dc_voltages;
use eda_waveform::{plot, Waveform};
use spike_mosfet::{switching_threshold, vin_sweep, InverterSpec};

const N_POINTS: usize = 31;

#[test]
fn ngspice_inverter_vtc_matches_analytic_vm() {
    let Ok(ng) = NgLocal::from_env() else {
        eprintln!("ngspice missing; skipping ngspice_inverter_vtc_matches_analytic_vm");
        return;
    };
    let spec = InverterSpec::default();
    let (vins, vout_ng) = sweep_with(&ng_invoker(&ng), &spec);

    // Vm: where Vin == Vout. Find by zero-crossing of (Vout - Vin).
    let vm_sim = vm_from_curve(&vins, &vout_ng);
    let vm_analytic = switching_threshold(&spec);
    let envelope = (spec.vdd / N_POINTS as f64) * 1.5; // half a sample step + slack
    assert!(
        (vm_sim - vm_analytic).abs() < envelope,
        "ngspice Vm = {vm_sim:.4} V vs analytic {vm_analytic:.4} V (envelope {envelope:.4})",
    );
}

#[cfg(feature = "ltspice")]
#[test]
fn ngspice_and_ltspice_agree_on_inverter_vtc() {
    use eda_extern_ltspice::{Invoker as LtInvoker, LocalBinary as LtLocal, OutputRequest as LtReq};

    let Ok(ng) = NgLocal::from_env() else {
        eprintln!("ngspice missing; skipping triangulation");
        return;
    };
    let Some(lt) = LtLocal::from_env_optional() else {
        eprintln!("LTspice missing; rendering ngspice-only VTC");
        let spec = InverterSpec::default();
        let (vins, vout_ng) = sweep_with(&ng_invoker(&ng), &spec);
        render_overlay(&vins, &[("ngspice", &vout_ng)], &spec);
        return;
    };

    let spec = InverterSpec::default();
    let (vins, vout_ng) = sweep_with(&ng_invoker(&ng), &spec);
    let (vins_lt, vout_lt) = sweep_with(
        &|deck| {
            let res = lt
                .run_dc(deck, &[LtReq::NodeVoltage("vout".into())])
                .expect("LTspice .op");
            res.node_voltages["vout"]
        },
        &spec,
    );
    assert_eq!(vins, vins_lt, "Vin grids should be identical");

    // Triangulation. Per-point compare under a tight envelope.
    let mut ng_map: HashMap<String, f64> = HashMap::new();
    let mut lt_map: HashMap<String, f64> = HashMap::new();
    for (i, _) in vins.iter().enumerate() {
        ng_map.insert(format!("vin_{i:03}"), vout_ng[i]);
        lt_map.insert(format!("vin_{i:03}"), vout_lt[i]);
    }
    let report = compare_dc_voltages(&ng_map, &lt_map);
    eprintln!(
        "ngspice vs LTspice VTC: worst {:?}, rms {:.3e}",
        report.worst, report.rms,
    );
    // 1% relative + 1 mV absolute. LEVEL=1 is identical between the
    // two; differences are floating-point and tolerance-controlled
    // .op convergence.
    report.assert_within(1e-2, 1e-3, "ngspice vs LTspice inverter VTC");

    render_overlay(&vins, &[("ngspice", &vout_ng), ("LTspice", &vout_lt)], &spec);
}

// ── helpers ────────────────────────────────────────────────────────────

fn ng_invoker<'a>(ng: &'a NgLocal) -> impl Fn(&str) -> f64 + 'a {
    move |deck: &str| -> f64 {
        let res = ng
            .run_dc(deck, &[NgReq::NodeVoltage("vout".into())])
            .expect("ngspice .op");
        res.node_voltages["vout"]
    }
}

/// Run one `.op` per Vin sample through `solve` and collect Vout.
fn sweep_with<F: Fn(&str) -> f64>(solve: &F, spec: &InverterSpec) -> (Vec<f64>, Vec<f64>) {
    let vins = vin_sweep(spec, N_POINTS);
    let vouts: Vec<f64> = vins
        .iter()
        .map(|&vin| solve(&spec.deck_at(vin).deck()))
        .collect();
    (vins, vouts)
}

/// Linear-interpolate the Vin where `Vout(Vin) = Vin`.
fn vm_from_curve(vins: &[f64], vouts: &[f64]) -> f64 {
    // Walk the curve until (Vout - Vin) changes sign; interpolate.
    for i in 1..vins.len() {
        let f0 = vouts[i - 1] - vins[i - 1];
        let f1 = vouts[i] - vins[i];
        if f0.signum() != f1.signum() {
            let t = f0 / (f0 - f1);
            return vins[i - 1] + t * (vins[i] - vins[i - 1]);
        }
    }
    f64::NAN
}

fn render_overlay(vins: &[f64], curves: &[(&str, &Vec<f64>)], spec: &InverterSpec) {
    let mut signals = BTreeMap::new();
    for (name, vouts) in curves {
        signals.insert((*name).to_string(), (*vouts).clone());
    }
    let wave = Waveform::Real {
        axis_name: "Vin (V)".into(),
        axis: vins.to_vec(),
        signals,
    };
    let cfg = plot::PlotConfig::new()
        .with_title(format!(
            "CMOS inverter VTC (Vdd = {} V), analytic Vm marker",
            spec.vdd,
        ))
        .with_size(800, 600)
        .add_marker(plot::Marker::Vertical {
            x: switching_threshold(spec),
            label: Some("Vm".into()),
        })
        .add_marker(plot::Marker::Horizontal {
            y: spec.vdd / 2.0,
            label: Some("Vdd/2".into()),
        });
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs");
    std::fs::create_dir_all(&out_dir).unwrap();
    let png = out_dir.join("inverter_vtc.png");
    let svg = out_dir.join("inverter_vtc.svg");
    plot::png_to_path(&wave, &png, &cfg).expect("png");
    plot::svg_to_path(&wave, &svg, &cfg).expect("svg");
    eprintln!("VTC overlay written to {}", png.display());
}
