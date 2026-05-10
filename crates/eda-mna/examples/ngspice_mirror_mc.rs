//! ngspice MC parity bench for the 1:4 NMOS current mirror.
//!
//! Runs the same circuit through two paths:
//!   1. eda-mna `batched_solve_dc` — one batched call, all N draws on
//!      Apple GPU via the Metal LU+solve kernel.
//!   2. ngspice — N independent forks of the binary, each with a
//!      per-draw `.model … VTO=…` deck.
//!
//! Reports per-draw vgs drift + wall-clock for each path.
//!
//! Run with: `cargo run --example ngspice_mirror_mc -p eda-mna --release`
//!
//! Requires `ngspice` on PATH (or `NGSPICE_BIN` env var).

use std::collections::HashMap;
use std::time::Instant;

use eda_hir::{Block, NonlinearDcBehavioral};
use eda_mna::{batched_solve_dc, Circuit, NetId, NewtonOptions};
use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};
use rlx_ir::{op::BinaryOp, DType, Graph, NodeId, Op, Shape};
use spike_divider_block::Mosfet;

// ── Inline current source (same as the existing batched mirror test) ──

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct ConstantCurrentSource { id: String }
impl Block for ConstantCurrentSource {
    fn name(&self) -> String { format!("Iref_{}", self.id) }
}
impl NonlinearDcBehavioral for ConstantCurrentSource {
    fn name(&self) -> String { <Self as Block>::name(self) }
    fn n_terminals(&self) -> usize { 1 }
    fn currents(&self, voltages: &[NodeId], g: &mut Graph) -> Vec<NodeId> {
        let s = Shape::new(&[1], DType::F32);
        let i_ref = g.param(format!("{}_I", <Self as Block>::name(self)), s.clone());
        let zero = g.add_node(Op::Constant { data: 0.0_f32.to_le_bytes().to_vec() }, vec![], s.clone());
        let zero_v = g.binary(BinaryOp::Mul, zero, voltages[0], s.clone());
        let i_with_phantom = g.binary(BinaryOp::Add, i_ref, zero_v, s);
        vec![i_with_phantom]
    }
}

// ── Bench ─────────────────────────────────────────────────────────────

const I_REF: f32  = 5e-6;
const V_BIAS: f32 = 0.9;
const VTH_NOM: f32 = 0.5;
const KP_F32: f32 = 100e-6;

/// Generate Pelgrom-like Vth draws via deterministic Box-Muller LCG —
/// reproducibility matters for a parity bench. σ_M1 = 5 mV (1×1 µm²),
/// σ_M2 = 2.5 mV (4× area).
fn pelgrom_draws(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut next = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (s >> 32) as u32;
        (bits as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    };
    let mut m1 = Vec::with_capacity(n);
    let mut m2 = Vec::with_capacity(n);
    while m1.len() < n {
        let u1 = next().max(1e-7); let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let z1 = r * (2.0_f32 * std::f32::consts::PI * u2).cos();
        let z2 = r * (2.0_f32 * std::f32::consts::PI * u2).sin();
        m1.push(VTH_NOM + 0.005 * z1);
        if m1.len() < n { m1.push(VTH_NOM + 0.005 * z2); }
    }
    while m2.len() < n {
        let u1 = next().max(1e-7); let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let z1 = r * (2.0_f32 * std::f32::consts::PI * u2).cos();
        let z2 = r * (2.0_f32 * std::f32::consts::PI * u2).sin();
        m2.push(VTH_NOM + 0.0025 * z1);
        if m2.len() < n { m2.push(VTH_NOM + 0.0025 * z2); }
    }
    (m1, m2)
}

fn make_deck(vth_m1: f32, vth_m2: f32) -> String {
    format!(
        "* 1:4 NMOS mirror MC parity\n\
         .options noecho gmin=1e-13 abstol=1e-12 reltol=1e-7\n\
         .model nch_m1 nmos LEVEL=1 KP={kp:e} VTO={vth_m1:.6} LAMBDA=0 GAMMA=0\n\
         .model nch_m2 nmos LEVEL=1 KP={kp:e} VTO={vth_m2:.6} LAMBDA=0 GAMMA=0\n\
         M1 vgs vgs 0 0 nch_m1 W=1u L=1u\n\
         M2 vbias vgs 0 0 nch_m2 W=4u L=1u\n\
         Iref 0 vgs DC {iref:e}\n\
         Vbias vbias 0 DC {vbias}\n\
         .op\n\
         .end\n",
        kp = KP_F32 as f64, iref = I_REF as f64, vbias = V_BIAS,
    )
}

fn run_one_n(n_draws: usize) {
    let (vth_m1_draws, vth_m2_draws) = pelgrom_draws(n_draws, 0xC1C50001);

    // ── Build eda-mna circuit (mirrors the existing test setup). ──
    let mut c = Circuit::new();
    let v_bias_net = c.alloc_boundary_net();
    let vgs_net    = c.alloc_unknown_net();
    let iref_dev = ConstantCurrentSource { id: "ref".into() };
    c.add_device(iref_dev.clone(), &[vgs_net]);
    let m1 = Mosfet::nmos(1_000, 1_000, "M1");
    c.add_device(m1.clone(), &[vgs_net, vgs_net, NetId::GND, NetId::GND]);
    let m2 = Mosfet::nmos(4_000, 1_000, "M2");
    c.add_device(m2.clone(), &[v_bias_net, vgs_net, NetId::GND, NetId::GND]);

    let mut params: HashMap<String, f32> = m1.default_params();
    params.extend(m2.default_params());
    let m1_vth_key = format!("{}_Vth", Block::name(&m1));
    let m2_vth_key = format!("{}_Vth", Block::name(&m2));
    let _ = params.remove(&m1_vth_key);
    let _ = params.remove(&m2_vth_key);
    params.insert(format!("{}_I", Block::name(&iref_dev)), I_REF);

    let mut mc_params: HashMap<String, Vec<f32>> = HashMap::new();
    mc_params.insert(m1_vth_key.clone(), vth_m1_draws.clone());
    mc_params.insert(m2_vth_key.clone(), vth_m2_draws.clone());

    let mut boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
    boundary.insert(v_bias_net, vec![V_BIAS; n_draws]);

    // Default opts now include a vntol guard that forces ≥1 Newton
    // iter and rejects "residual already small at the init guess"
    // false-converge — see NewtonOptions docs. No manual tol override
    // needed; default 1e-7 abstol + 1e-6 V vntol gets the right answer.
    let opt = NewtonOptions { init: 0.82, ..NewtonOptions::default() };

    // ── eda-mna run ──
    let t0 = Instant::now();
    let batched = batched_solve_dc(&c, n_draws, &params, &mc_params, &boundary, opt);
    let t_eda = t0.elapsed();
    let n_converged = batched.converged.iter().filter(|&&b| b).count();
    println!("eda-mna   : {n_converged}/{n_draws} converged in {} iters, {:.2} ms",
        batched.iters, t_eda.as_secs_f64() * 1e3);

    // ── ngspice run (per-draw fork) ──
    let ng = match LocalBinary::from_env() {
        Ok(ng) => ng,
        Err(e) => {
            eprintln!("ngspice unavailable: {e}");
            std::process::exit(1);
        }
    };

    let t0 = Instant::now();
    let mut ng_vgs: Vec<f64> = Vec::with_capacity(n_draws);
    let mut ng_failures = 0usize;
    for d in 0..n_draws {
        let deck = make_deck(vth_m1_draws[d], vth_m2_draws[d]);
        match ng.run_dc(&deck, &[OutputRequest::NodeVoltage("vgs".into())]) {
            Ok(r) => ng_vgs.push(*r.node_voltages.get("vgs").unwrap_or(&f64::NAN)),
            Err(e) => {
                eprintln!("draw {d}: ngspice error: {e}");
                ng_failures += 1;
                ng_vgs.push(f64::NAN);
            }
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
        if !batched.converged[d] || ng_vgs[d].is_nan() { continue; }
        let v_eda = batched.voltages[&vgs_net][d] as f64;
        let v_ng  = ng_vgs[d];
        let delta = (v_eda - v_ng).abs();
        max_abs = max_abs.max(delta);
        sum_sq += delta * delta;
        n_compared += 1;
    }
    let rms = (sum_sq / n_compared as f64).sqrt();
    println!();
    println!("Per-draw vgs drift over {n_compared} comparable draws:");
    println!("  max |Δvgs|  = {:.3e} V", max_abs);
    println!("  RMS  Δvgs   = {:.3e} V", rms);

    // Spot check: print the first 5 draws side by side.
    println!();
    println!("{:>4}  {:>10}  {:>10}  {:>12}  {:>12}  {:>12}",
        "draw", "Vth_M1", "Vth_M2", "vgs (eda-mna)", "vgs (ngspice)", "Δ");
    for d in 0..n_draws.min(8) {
        let v_eda = batched.voltages[&vgs_net][d];
        let v_ng = ng_vgs[d];
        println!("{d:>4}  {:>10.6}  {:>10.6}  {:>12.6}  {:>12.6}  {:>+12.3e}",
            vth_m1_draws[d], vth_m2_draws[d], v_eda, v_ng, v_eda as f64 - v_ng);
    }

    println!();
    println!("Speedup eda-mna vs ngspice (N={n_draws}): {:.1}×",
        t_ng.as_secs_f64() / t_eda.as_secs_f64().max(1e-9));
    println!();
}

fn main() {
    println!("== Single-N detailed run (N=32) ==\n");
    run_one_n(32);

    println!("== N-sweep (drift omitted for brevity) ==\n");
    println!("{:>6}  {:>14}  {:>14}  {:>14}  {:>10}",
        "N", "eda-mna (ms)", "ngspice (ms)", "ms/draw (ng)", "speedup");
    for &n in &[1usize, 4, 16, 64, 256] {
        let (vth_m1, vth_m2) = pelgrom_draws(n, 0xC1C50001);

        // ── Build circuit (per-iteration so n changes correctly) ──
        let mut c = Circuit::new();
        let v_bias_net = c.alloc_boundary_net();
        let vgs_net    = c.alloc_unknown_net();
        let iref_dev = ConstantCurrentSource { id: "ref".into() };
        c.add_device(iref_dev.clone(), &[vgs_net]);
        let m1 = Mosfet::nmos(1_000, 1_000, "M1");
        c.add_device(m1.clone(), &[vgs_net, vgs_net, NetId::GND, NetId::GND]);
        let m2 = Mosfet::nmos(4_000, 1_000, "M2");
        c.add_device(m2.clone(), &[v_bias_net, vgs_net, NetId::GND, NetId::GND]);

        let mut params: HashMap<String, f32> = m1.default_params();
        params.extend(m2.default_params());
        let m1_vth_key = format!("{}_Vth", Block::name(&m1));
        let m2_vth_key = format!("{}_Vth", Block::name(&m2));
        let _ = params.remove(&m1_vth_key);
        let _ = params.remove(&m2_vth_key);
        params.insert(format!("{}_I", Block::name(&iref_dev)), I_REF);
        let mut mc_params: HashMap<String, Vec<f32>> = HashMap::new();
        mc_params.insert(m1_vth_key, vth_m1.clone());
        mc_params.insert(m2_vth_key, vth_m2.clone());
        let mut boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
        boundary.insert(v_bias_net, vec![V_BIAS; n]);
        let opt = NewtonOptions { init: 0.82, ..NewtonOptions::default() };

        let t0 = Instant::now();
        let _ = batched_solve_dc(&c, n, &params, &mc_params, &boundary, opt);
        let t_eda = t0.elapsed().as_secs_f64() * 1e3;

        let ng = LocalBinary::from_env().expect("ngspice");
        let t0 = Instant::now();
        for d in 0..n {
            let deck = make_deck(vth_m1[d], vth_m2[d]);
            let _ = ng.run_dc(&deck, &[OutputRequest::NodeVoltage("vgs".into())])
                .expect("ngspice");
        }
        let t_ng = t0.elapsed().as_secs_f64() * 1e3;

        println!("{:>6}  {:>14.2}  {:>14.2}  {:>14.2}  {:>9.1}×",
            n, t_eda, t_ng, t_ng / n as f64, t_ng / t_eda.max(1e-6));
    }
}
