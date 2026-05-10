//! Mach-Zehnder inverse-design demo.
//!
//! Builds a 100 µm / 110 µm asymmetric MZI, sweeps |T_through(λ)|² across
//! the C-band, then runs Adam on `n_eff_A` (the photonic stand-in for a
//! thermo-optic phase shifter) to drop a notch onto λ = 1550 nm. Prints
//! the spectrum before and after as a tiny ASCII bar chart so the
//! mode-shift is visible without a plotting backend.
//!
//! Run:    cargo run -p spike-waveguide-block --bin mzi_demo

use rlx_ir::{Op, NodeId};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_waveguide_block::Mzi;

const TARGET_LAMBDA_NM: f32 = 1550.0;
const NEFF_INIT: f32 = 2.35;
const NEFF_FROZEN: f32 = 2.4;
const ADAM_STEPS: usize = 4000;
const LR: f32 = 1e-3;

fn main() {
    let mzi = Mzi::new(500, 100_000, 110_000, "demo");

    // ── Forward sweep before optimization ───────────────────────────
    println!("Mach-Zehnder demo: arms {} nm vs {} nm, λ-target = {} nm",
        mzi.arm_a.length, mzi.arm_b.length, TARGET_LAMBDA_NM);
    println!();
    let before = sweep(&mzi, NEFF_INIT, NEFF_FROZEN);
    println!("Before optimization (n_eff_A = {NEFF_INIT}):");
    print_spectrum(&before);

    // ── Adam on n_eff_A to put a notch at the target wavelength ─────
    let (neff_final, history) = adam_optimize(&mzi);
    println!();
    println!("Optimization (Adam, lr={LR}, {} steps):", ADAM_STEPS);
    println!("  step       loss        n_eff_A");
    for (step, loss, neff) in history {
        println!("  {step:5}   {loss:.6e}   {neff:.6}");
    }

    // ── Forward sweep after optimization ────────────────────────────
    let after = sweep(&mzi, neff_final, NEFF_FROZEN);
    println!();
    println!("After optimization (n_eff_A = {neff_final:.6}):");
    print_spectrum(&after);

    // Highlight the notch depth at λ_target.
    let target_idx = after.iter().position(|(wl, _)| (wl - TARGET_LAMBDA_NM).abs() < 0.5).unwrap();
    let (wl, t) = after[target_idx];
    let extinction_db = if t > 0.0 { -10.0 * t.log10() } else { f32::INFINITY };
    println!();
    println!("|T_through|² at λ = {wl} nm: {t:.3e}  ({extinction_db:.1} dB extinction)");
}

fn sweep(mzi: &Mzi, neff_a: f32, neff_b: f32) -> Vec<(f32, f32)> {
    let g = mzi.build_intensity_graph();
    let mut sess = Session::new(Device::Cpu).compile(g);
    sess.set_param(&mzi.arm_a.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_b.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_a.neff_param_name(), &[neff_a]);
    sess.set_param(&mzi.arm_b.neff_param_name(), &[neff_b]);

    let mut out = Vec::new();
    for k in 0..=40 {
        let wl = 1500.0 + (k as f32) * 2.5; // 1500..1600 nm, 2.5 nm step
        let o = sess.run(&[("wavelength_nm", &[wl])]);
        out.push((wl, o[0][0]));
    }
    out
}

fn adam_optimize(mzi: &Mzi) -> (f32, Vec<(usize, f32, f32)>) {
    let fwd = mzi.build_notch_loss_graph();
    let neff_a_id = find_param(&fwd, &mzi.arm_a.neff_param_name());
    let bwd = grad_with_loss(&fwd, &[neff_a_id]);
    let mut sess = Session::new(Device::Cpu).compile(bwd);
    sess.set_param(&mzi.arm_a.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_b.loss_param_name(), &[0.0]);
    sess.set_param(&mzi.arm_b.neff_param_name(), &[NEFF_FROZEN]);

    let (mut neff, mut m, mut v) = (NEFF_INIT, 0.0_f32, 0.0_f32);
    let (b1, b2, eps) = (0.9_f32, 0.999_f32, 1e-8_f32);
    let mut history = Vec::new();
    for t in 1..=ADAM_STEPS {
        sess.set_param(&mzi.arm_a.neff_param_name(), &[neff]);
        let o = sess.run(&[("wavelength_nm", &[TARGET_LAMBDA_NM]), ("d_output", &[1.0])]);
        let loss = o[0][0];
        let g = o[1][0];
        if t == 1 || t.is_power_of_two() || t == ADAM_STEPS {
            history.push((t, loss, neff));
        }
        m = b1 * m + (1.0 - b1) * g;
        v = b2 * v + (1.0 - b2) * g * g;
        let m_hat = m / (1.0 - b1.powi(t as i32));
        let v_hat = v / (1.0 - b2.powi(t as i32));
        neff -= LR * m_hat / (v_hat.sqrt() + eps);
    }
    (neff, history)
}

fn print_spectrum(samples: &[(f32, f32)]) {
    const COLS: usize = 50;
    for &(wl, t) in samples {
        let bars = (t.clamp(0.0, 1.0) * COLS as f32).round() as usize;
        let bar: String = std::iter::repeat('█').take(bars).collect();
        let mark = if (wl - TARGET_LAMBDA_NM).abs() < 0.5 { " ←" } else { "" };
        println!("  λ={wl:6.1} nm  {t:.3}  {bar}{mark}");
    }
}

fn find_param(g: &rlx_ir::Graph, name: &str) -> NodeId {
    g.nodes()
        .iter()
        .enumerate()
        .find_map(|(i, n)| match &n.op {
            Op::Param { name: pn, .. } if pn == name => Some(NodeId(i as u32)),
            _ => None,
        })
        .expect("param missing")
}
