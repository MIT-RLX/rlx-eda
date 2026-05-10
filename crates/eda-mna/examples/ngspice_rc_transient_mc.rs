//! ngspice transient parity bench for the RC discharge.
//!
//! Same circuit `batched_transient.rs` validates the batched BE
//! integrator on, run through both eda-mna and ngspice for N draws
//! of distinct initial v_C. Compares per-draw final v_C.
//!
//! Run with: `cargo run --example ngspice_rc_transient_mc -p eda-mna --release`
//!
//! Requires `ngspice` on PATH.

use std::collections::HashMap;
use std::time::Instant;

use eda_hir::Block;
use eda_mna::{batched_transient_from, transient_from, Circuit, NetId, NewtonOptions};
use eda_extern_ngspice::{
    Invoker, LocalBinary, OutputRequest, TransientAnalysis,
};
use spike_divider_block::{Capacitor, Resistor};

const R_OHMS:   f32 = 1_000.0;
const C_FARADS: f32 = 1e-9;

fn make_deck(v_init: f32, t_stop: f32) -> String {
    // Backward-Euler integration via `method=gear maxord=1` so the
    // ngspice trajectory matches our BE step shape — otherwise
    // ngspice's default trapezoidal integrator would drift from our
    // BE in opposite directions and inflate the apparent disagreement.
    format!(
        "* RC discharge MC parity\n\
         .options noecho method=gear maxord=1 abstol=1e-14 reltol=1e-7 vntol=1e-9\n\
         .ic v(vmid)={v_init}\n\
         R1 vmid 0 {r}\n\
         C1 vmid 0 {c}\n\
         .end\n",
        r = R_OHMS as f64, c = C_FARADS as f64,
    ).replace(".end\n", &format!(".tran {} {} uic\n.end\n", t_stop / 50.0, t_stop))
}

fn run_one_n(n_draws: usize) {
    let mut c = Circuit::new();
    let vmid = c.alloc_unknown_net();
    let r   = Resistor { length: 10_000, id: "R1".into() };
    let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
    c.add_device(r.clone(),    &[vmid, NetId::GND]);
    c.add_storage(cap.clone(), [vmid, NetId::GND]);

    let mut params = HashMap::new();
    params.insert(Block::name(&r), R_OHMS);
    params.insert(format!("{}_C", Block::name(&cap)), C_FARADS);

    // N distinct initial v_C values spanning a 20× range.
    let v_init: Vec<f32> = (0..n_draws).map(|i| {
        let t = i as f32 / (n_draws.max(2) - 1) as f32;
        0.05 + 0.95 * (1.0 - t)
    }).collect();
    let mut ic: HashMap<NetId, Vec<f32>> = HashMap::new();
    ic.insert(vmid, v_init.clone());

    let tau     = R_OHMS * C_FARADS;
    let dt      = tau / 50.0;
    let n_steps = 250;        // → t_end = 5 τ
    let t_end   = n_steps as f32 * dt;

    let boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
    let mc_params: HashMap<String, Vec<f32>> = HashMap::new();
    let opt = NewtonOptions::default();

    // ── eda-mna run ──
    let t0 = Instant::now();
    let waveform = batched_transient_from(
        &c, n_draws, &params, &mc_params, &boundary, &ic,
        dt, n_steps, opt,
    );
    let t_eda = t0.elapsed();
    let final_step = waveform.last().unwrap();
    let n_converged = final_step.converged.iter().filter(|&&b| b).count();
    println!("eda-mna   : {n_converged}/{n_draws} converged at t=5τ in {:.2} ms ({} steps × N draws)",
        t_eda.as_secs_f64() * 1e3, n_steps);

    // ── ngspice per-draw run ──
    let ng = match LocalBinary::from_env() {
        Ok(ng) => ng,
        Err(e) => { eprintln!("ngspice unavailable: {e}"); return; }
    };
    let analysis = TransientAnalysis::new(dt as f64, t_end as f64);

    let t0 = Instant::now();
    let mut ng_final: Vec<f64> = Vec::with_capacity(n_draws);
    let mut ng_failures = 0;
    for d in 0..n_draws {
        let deck = make_deck(v_init[d], t_end);
        match ng.run_transient_trace(
            &deck, &analysis,
            &[OutputRequest::NodeVoltage("vmid".into())],
        ) {
            Ok(trace) => {
                // ngspice's adaptive grid may not land on t_end exactly
                // — pick the last sample, or interpolate. For this test
                // the final sample is close enough (ngspice will output
                // at t_end if the deck reaches it).
                let v = trace.node_voltages.get("vmid")
                    .and_then(|vs| vs.last().copied())
                    .unwrap_or(f64::NAN);
                ng_final.push(v);
            }
            Err(e) => { eprintln!("draw {d}: {e}"); ng_failures += 1; ng_final.push(f64::NAN); }
        }
    }
    let t_ng = t0.elapsed();
    println!("ngspice   : {}/{n_draws} succeeded in {:.2} ms ({:.1} ms/draw)",
        n_draws - ng_failures, t_ng.as_secs_f64() * 1e3,
        t_ng.as_secs_f64() * 1e3 / n_draws as f64);

    // ── Drift analysis ──
    let mut max_abs = 0.0_f64;
    let mut sum_sq  = 0.0_f64;
    let mut n_compared = 0usize;
    for d in 0..n_draws {
        if !final_step.converged[d] || ng_final[d].is_nan() { continue; }
        let v_eda = final_step.voltages[&vmid][d] as f64;
        let v_ng  = ng_final[d];
        let delta = (v_eda - v_ng).abs();
        max_abs = max_abs.max(delta);
        sum_sq += delta * delta;
        n_compared += 1;
    }
    let rms = (sum_sq / n_compared as f64).sqrt();
    println!("Per-draw v(5τ) drift over {n_compared} comparable draws:");
    println!("  max |Δ|  = {:.3e} V    RMS Δ = {:.3e} V", max_abs, rms);

    // Per-draw scalar reference (transient_from per draw) — gives a
    // pure eda-mna parity check, separate from the ngspice cross-check.
    let mut max_scalar_drift = 0.0_f64;
    for d in 0..n_draws {
        let mut ic_s = HashMap::new();
        ic_s.insert(vmid, v_init[d]);
        let bnd_s: HashMap<NetId, f32> = HashMap::new();
        let scalar_wf = transient_from(&c, &params, &bnd_s, &ic_s, dt, n_steps, opt);
        let v_scalar = scalar_wf[n_steps].voltages[&vmid] as f64;
        let v_batched = final_step.voltages[&vmid][d] as f64;
        max_scalar_drift = max_scalar_drift.max((v_batched - v_scalar).abs());
    }
    println!("  max |Δ vs scalar transient_from| = {:.3e} V", max_scalar_drift);
    println!();
}

fn main() {
    println!("== Single-N detailed run (N=8) ==\n");
    run_one_n(8);

    println!("== N-sweep ==\n");
    println!("{:>6}  {:>14}  {:>14}  {:>14}  {:>10}",
        "N", "eda-mna (ms)", "ngspice (ms)", "ms/draw (ng)", "speedup");

    let mut c = Circuit::new();
    let vmid = c.alloc_unknown_net();
    let r   = Resistor { length: 10_000, id: "R1".into() };
    let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
    c.add_device(r.clone(),    &[vmid, NetId::GND]);
    c.add_storage(cap.clone(), [vmid, NetId::GND]);
    let mut params = HashMap::new();
    params.insert(Block::name(&r), R_OHMS);
    params.insert(format!("{}_C", Block::name(&cap)), C_FARADS);
    let tau     = R_OHMS * C_FARADS;
    let dt      = tau / 50.0;
    let n_steps = 250;
    let t_end   = n_steps as f32 * dt;
    let opt = NewtonOptions::default();

    for &n in &[1usize, 4, 16, 64, 256] {
        let v_init: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / (n.max(2) - 1) as f32;
            0.05 + 0.95 * (1.0 - t)
        }).collect();
        let mut ic: HashMap<NetId, Vec<f32>> = HashMap::new();
        ic.insert(vmid, v_init.clone());
        let boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
        let mc_params: HashMap<String, Vec<f32>> = HashMap::new();

        let t0 = Instant::now();
        let _ = batched_transient_from(
            &c, n, &params, &mc_params, &boundary, &ic, dt, n_steps, opt,
        );
        let t_eda = t0.elapsed().as_secs_f64() * 1e3;

        let ng = LocalBinary::from_env().expect("ngspice");
        let analysis = TransientAnalysis::new(dt as f64, t_end as f64);
        let t0 = Instant::now();
        for d in 0..n {
            let deck = make_deck(v_init[d], t_end);
            let _ = ng.run_transient_trace(
                &deck, &analysis,
                &[OutputRequest::NodeVoltage("vmid".into())],
            ).expect("ngspice");
        }
        let t_ng = t0.elapsed().as_secs_f64() * 1e3;

        println!("{:>6}  {:>14.2}  {:>14.2}  {:>14.2}  {:>9.1}×",
            n, t_eda, t_ng, t_ng / n as f64, t_ng / t_eda.max(1e-6));
    }
}
