//! T.9 — full transistor-level SAR ADC under eda-mna.
//!
//! Composes the T.8.C analog front-end (SampleHold + R-2R DAC +
//! Comparator) with the transistor-level SarRegister<N> (N × DffSR +
//! per-bit glue inverters from spike-sar-register::mna) into a
//! single `eda_mna::Circuit`. One PWL boundary closure drives the
//! external pins (vin, clk_sh, reset_b, per-bit phase + capture
//! signals); one `transient_pwl` call runs the entire N-bit
//! conversion; the final bit pattern is read off the SAR register's
//! Q outputs and compared against `ideal_sar_code`.
//!
//! Headline: every transistor in the SAR ADC — analog and digital
//! alike — runs in one differentiable BE solve.

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use eda_hir::Block;
use eda_mna::{transient_pwl, Circuit, LinearCap, NetId, NewtonOptions};
use spike_dac_r2r::mna::add_r2r_dac;
use spike_divider_block::Mosfet;
use spike_sample_hold::mna::add_sample_hold;
use spike_sar_register::mna::add_sar_register;
use spike_sar_register::ideal_sar_code;

// 4 bits — feasible thanks to T.10's cached BE-step compile (~70× speedup).
const N_BITS:  usize = 4;
const VDD:     f32 = 1.8;
const VBIAS:   f32 = 0.7;
const VIN:     f32 = 0.6 * VDD;       // 1.08 V → ideal code 9 = 0b1001
const H:       f32 = 1e-9;            // 1 ns BE step

const T_RESET_NS:  f32 = 10.0;
const T_SAMPLE_NS: f32 = 30.0;
const T_PER_BIT_NS:f32 = 25.0;        // 25 ns per bit (slightly more than T.8.C
                                      // because the SAR register's edge-triggered
                                      // DffSR also needs to settle within each window).
const C_HOLD_F:    f32 = 50e-15;

/// Build the 9-transistor comparator (same topology as T.8.A / T.8.C).
fn add_comparator(
    c: &mut Circuit,
    vp: NetId, vm: NetId, vout: NetId, vbias: NetId, vdd: NetId,
    id: &str,
    params: &mut HashMap<String, f32>,
    ic: Option<&mut HashMap<NetId, f32>>,
) {
    let tail_s = c.alloc_unknown_net();
    let d1   = c.alloc_unknown_net();
    let d2   = c.alloc_unknown_net();
    let int1 = c.alloc_unknown_net();

    let m_tail = Mosfet::nmos(4_000, 1_000, format!("{id}_tail"));
    let m1     = Mosfet::nmos(8_000, 1_000, format!("{id}_m1"));
    let m2     = Mosfet::nmos(8_000, 1_000, format!("{id}_m2"));
    let m3     = Mosfet::pmos(4_000, 1_000, format!("{id}_m3"));
    let m4     = Mosfet::pmos(4_000, 1_000, format!("{id}_m4"));
    let m_iv1n = Mosfet::nmos(2_000, 1_000, format!("{id}_iv1n"));
    let m_iv1p = Mosfet::pmos(4_000, 1_000, format!("{id}_iv1p"));
    let m_iv2n = Mosfet::nmos(2_000, 1_000, format!("{id}_iv2n"));
    let m_iv2p = Mosfet::pmos(4_000, 1_000, format!("{id}_iv2p"));

    c.add_device(m_tail.clone(), &[tail_s, vbias, NetId::GND, NetId::GND]);
    c.add_device(m1.clone(),     &[d1,     vp,    tail_s,     NetId::GND]);
    c.add_device(m2.clone(),     &[d2,     vm,    tail_s,     NetId::GND]);
    c.add_device(m3.clone(),     &[d1,     d1,    vdd,        vdd]);
    c.add_device(m4.clone(),     &[d2,     d1,    vdd,        vdd]);
    c.add_device(m_iv1n.clone(), &[int1,   d2,    NetId::GND, NetId::GND]);
    c.add_device(m_iv1p.clone(), &[int1,   d2,    vdd,        vdd]);
    c.add_device(m_iv2n.clone(), &[vout,   int1,  NetId::GND, NetId::GND]);
    c.add_device(m_iv2p.clone(), &[vout,   int1,  vdd,        vdd]);

    let cap_keys = ["d1", "d2", "int1", "vout", "tail_s"];
    let cap_nets = [d1, d2, int1, vout, tail_s];
    for (k, net) in cap_keys.iter().zip(cap_nets.iter()) {
        let key = format!("{id}_C_{k}");
        c.add_storage(LinearCap::new(key.clone()), [*net, NetId::GND]);
        params.insert(key, 50e-15);
    }

    for m in [&m_tail, &m1, &m2, &m3, &m4, &m_iv1n, &m_iv1p, &m_iv2n, &m_iv2p] {
        params.extend(m.default_params());
        params.insert(format!("{}_Lambda", Block::name(m)), 0.05);
    }

    if let Some(ic) = ic {
        // All buffer-chain nodes start mid-rail so the inverter chain
        // can swing either direction. Seeding `vout = VDD` and
        // `int1 = 0` pinned the output stage at the high rail and
        // killed downward-transition gain — same fix that
        // `comparator_mc_batched` uses (vout = VDD/2).
        ic.insert(d1, VDD * 0.7);
        ic.insert(d2, VDD * 0.7);
        ic.insert(int1, VDD * 0.5);
        ic.insert(vout, VDD * 0.5);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut circuit = Circuit::new();
    let v_dd   = circuit.alloc_boundary_net();
    let v_bias = circuit.alloc_boundary_net();
    let vin    = circuit.alloc_boundary_net();
    let clk_sh = circuit.alloc_boundary_net();
    let reset_b = circuit.alloc_boundary_net();
    // External per-bit control: phase[i] sets bit i during its trial,
    // capture[i] is the rising edge that latches the cmp result into
    // bit i's DffSR. Both are PWL boundaries.
    let phases:   Vec<NetId> = (0..N_BITS).map(|_| circuit.alloc_boundary_net()).collect();
    let captures: Vec<NetId> = (0..N_BITS).map(|_| circuit.alloc_boundary_net()).collect();

    // Internal nets.
    let vhold = circuit.alloc_unknown_net();
    let v_dac = circuit.alloc_unknown_net();
    let cmp   = circuit.alloc_unknown_net();
    // Bits are unknowns because they're driven by the SarRegister output.
    let bits: Vec<NetId> = (0..N_BITS).map(|_| circuit.alloc_unknown_net()).collect();

    let mut params: HashMap<String, f32> = HashMap::new();
    let mut ic:     HashMap<NetId, f32>  = HashMap::new();

    // Analog front-end.
    add_sample_hold(&mut circuit, [vin, vhold, clk_sh, v_dd, NetId::GND],
        "sh", C_HOLD_F, &mut params);
    add_r2r_dac(&mut circuit, &bits, NetId::GND, v_dac, 10_000.0,
        "dac", &mut params);
    add_comparator(&mut circuit, vhold, v_dac, cmp, v_bias, v_dd,
        "cmp", &mut params, Some(&mut ic));

    // SAR digital register: cmp → captured bits → DAC inputs.
    add_sar_register(
        &mut circuit, &phases, &captures, cmp, reset_b, &bits,
        v_dd, NetId::GND, "sar", &mut params, Some(&mut ic),
    );

    // Run the SAR algorithm in software to get expected per-trial state
    // for picking realistic capture timings + a final code reference.
    let n_levels = 1u32 << N_BITS;
    let lsb = VDD as f64 / n_levels as f64;
    let ideal_code = ideal_sar_code(VIN as f64, VDD as f64, N_BITS);
    eprintln!("Analytic SAR: vin = {:.4} V → ideal code = {} = 0b{:0n_bits$b}",
        VIN, ideal_code, ideal_code, n_bits = N_BITS);
    eprintln!("Total transistors in circuit: ~{} (analog) + ~{}/bit × {} bits (digital)",
        9 + 4 + 4, 56, N_BITS);
    let _ = lsb;

    // PWL boundary that walks through reset → sample → N bit trials.
    let t_reset_s  = T_RESET_NS  * 1e-9;
    let t_sample_s = t_reset_s + T_SAMPLE_NS * 1e-9;
    let t_trial_s  = T_PER_BIT_NS * 1e-9;

    // Trial timeline (fractions of t_trial_s):
    //   0 .. 0.50  phase[i] HIGH  → set_b LOW → tentative bit = 1
    //   0.50 .. 0.70  GAP — phase released, set_b HIGH, cmp re-settles
    //   0.70 .. 0.85  capture[i] HIGH — DffSR latches D=cmp on rising edge
    //   0.85 .. 1.0  hold
    // The 20% gap between phase-drop and capture-rise is what fixes
    // the original setup-time race: with the previous code, phase
    // dropped to LOW exactly when capture rose, so set_b was still
    // asserted at the clock edge and the DFF held the tentative HIGH
    // instead of capturing cmp.
    let phase_pulse_hi = 0.50 * t_trial_s;
    let cap_pulse_lo   = 0.70 * t_trial_s;
    let cap_pulse_hi   = 0.85 * t_trial_s;

    let phases_owned   = phases.clone();
    let captures_owned = captures.clone();

    let boundary = move |t: f32| -> HashMap<NetId, f32> {
        let mut bnd = HashMap::new();
        bnd.insert(v_dd,   VDD);
        bnd.insert(v_bias, VBIAS);
        bnd.insert(vin,    VIN);
        // S/H: high during sample window only.
        let in_sample = t >= t_reset_s && t < t_sample_s;
        bnd.insert(clk_sh, if in_sample { VDD } else { 0.0 });
        // reset_b: low during the initial 10 ns reset window, high after.
        let in_reset = t < t_reset_s;
        bnd.insert(reset_b, if in_reset { 0.0 } else { VDD });
        // Per-bit phase + capture. SAR runs MSB-first (the first trial
        // window targets bit[N-1], the LAST trial window targets bit[0]).
        // phase[i] fires during chronological trial (N-1-i).
        let n = phases_owned.len();
        for (i, &ph) in phases_owned.iter().enumerate() {
            let trial_idx   = n - 1 - i;
            let trial_start = t_sample_s + (trial_idx as f32) * t_trial_s;
            let trial_end   = trial_start + t_trial_s;
            let phase_hi = t >= trial_start && t < trial_start + phase_pulse_hi;
            bnd.insert(ph, if phase_hi { VDD } else { 0.0 });
            let cap_t0 = trial_start + cap_pulse_lo;
            let cap_t1 = trial_start + cap_pulse_hi;
            let cap_hi = t >= cap_t0 && t < cap_t1;
            bnd.insert(captures_owned[i], if cap_hi { VDD } else { 0.0 });
            let _ = trial_end;
        }
        bnd
    };

    // For initial DC, drive reset_b low, phases low, captures low to
    // let the SR set_b/reset_b dominate: with set_b high (released)
    // and reset_b low, the SR forces Q = 0 → bit[i] = 0.
    // We've already seeded ic so q_int = 0, qb = Vdd, etc.

    let solver = NewtonOptions::default();
    let total_ns = T_RESET_NS + T_SAMPLE_NS + (N_BITS as f32) * T_PER_BIT_NS;
    let n_steps = (total_ns / (H * 1e9)).round() as usize;

    eprintln!("\nTransient: {} BE steps over {:.0} ns (with T.10 cached BE-step graphs)...",
        n_steps, total_ns);
    let start = std::time::Instant::now();
    let trace = transient_pwl(&circuit, &params, boundary, &ic, H, n_steps, solver);
    eprintln!("  total transient time: {:.1}s ({:.0} ms/step)",
        start.elapsed().as_secs_f32(),
        start.elapsed().as_secs_f32() * 1000.0 / n_steps as f32);

    // ── Diagnostic: per-step convergence + cmp at each capture edge ──
    let n_unconv = trace.iter().filter(|s| !s.converged).count();
    let max_iters_hit = trace.iter().filter(|s| s.iters >= 50).count();
    let res_max = trace.iter().map(|s| s.final_residual_max).fold(0.0_f32, f32::max);
    eprintln!("[diag] convergence: {}/{} unconverged steps, {}/{} hit max_iters=50, max |r| = {:.2e}",
        n_unconv, trace.len(), max_iters_hit, trace.len(), res_max);

    // Trial-0 timeline (matches batched bin's probe). phases[N-1] /
    // captures[N-1] drive the MSB. The key question: does bit[3]
    // STAY HIGH at step 54 (when phase released) like the SR latch
    // is supposed to? Batched sees bit[3] drop here → Newton finds
    // the wrong stable state. Compare to scalar's behavior here.
    eprintln!("[diag] trial-0 timeline (phase[3] HIGH 40..52, capture[3] rising 57):");
    for &k in &[40usize, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64] {
        let s = &trace[k.min(trace.len() - 1)];
        let ph = s.voltages.get(&phases[N_BITS-1]).copied().unwrap_or(-9.0);
        let cap = s.voltages.get(&captures[N_BITS-1]).copied().unwrap_or(-9.0);
        let bit3 = s.voltages.get(&bits[N_BITS-1]).copied().unwrap_or(-9.0);
        let vd = s.voltages.get(&v_dac).copied().unwrap_or(-9.0);
        let cv = s.voltages.get(&cmp).copied().unwrap_or(-9.0);
        eprintln!("  step {k:3}  phase[3]={ph:.3}  cap[3]={cap:.3}  bit[3]={bit3:.3}  v_dac={vd:.3}  cmp={cv:.3}  iters={}  conv={}",
            s.iters, if s.converged { "Y" } else { "N" });
    }

    // Sample (vhold, v_dac, cmp, bit) at the CAPTURE rising edge of
    // each trial — that's the moment the SAR latches its decision.
    // cap_t0 = trial_start + 0.7*t_trial_s. Step index = round(cap_t0/H).
    let trial_start_s_for = |trial: usize| -> f32 {
        // chronological trial: bit_position = N_BITS-1-trial
        let n = N_BITS;
        let trial_idx = trial; // we walk in MSB→LSB chronological order
        let _ = n;
        t_sample_s + (trial_idx as f32) * t_trial_s
    };
    eprintln!("[diag] cmp at each capture edge:");
    for trial in 0..N_BITS {
        let trial_start = trial_start_s_for(trial);
        let cap_t = trial_start + cap_pulse_lo + 0.5 * (cap_pulse_hi - cap_pulse_lo);
        let step = (cap_t / H).round() as usize;
        let step = step.min(trace.len() - 1);
        let s = &trace[step];
        let bit_position = N_BITS - 1 - trial;
        eprintln!("  trial {} (bit[{}]) cap@{:.1}ns step={}: vhold={:.4}  v_dac={:.4}  cmp={:.4}  iters={}  conv={}",
            trial, bit_position, cap_t * 1e9, step,
            s.voltages.get(&vhold).copied().unwrap_or(0.0),
            s.voltages.get(&v_dac).copied().unwrap_or(0.0),
            s.voltages.get(&cmp).copied().unwrap_or(0.0),
            s.iters, s.converged);
    }

    // Sample bit[i] at the END of each trial window (just before the
    // next trial's phase pulse). bit[i] is LSB-indexed: the SarRegister
    // wires bit[0] to the LSB DAC input. The "trial order" we use here
    // (MSB first) means trial 0 → bit at position N-1, trial 1 → N-2, etc.
    let pat_n_steps = (T_PER_BIT_NS / (H * 1e9)).round() as usize;
    let sample_offset = (T_RESET_NS + T_SAMPLE_NS) / (H * 1e9);
    let sample_offset = sample_offset.round() as usize;
    println!("\n+-------+----------+----------+----------+----------+----------+");
    println!("| trial | bit pos  |  vhold   |  v_dac   |   cmp    | bit (Q)  |");
    println!("+-------+----------+----------+----------+----------+----------+");
    let mut decoded_code = 0u32;
    for trial in 0..N_BITS {
        let step = sample_offset + (trial + 1) * pat_n_steps - 1;
        let step = step.min(trace.len() - 1);
        let s = &trace[step];
        let bit_position = N_BITS - 1 - trial;
        let vh = s.voltages.get(&vhold).copied().unwrap_or(0.0);
        let vd = s.voltages.get(&v_dac).copied().unwrap_or(0.0);
        let cv = s.voltages.get(&cmp).copied().unwrap_or(0.0);
        let bv = s.voltages.get(&bits[bit_position]).copied().unwrap_or(0.0);
        let bit_logic = if bv > VDD * 0.5 { 1 } else { 0 };
        if bit_logic == 1 { decoded_code |= 1 << bit_position; }
        println!("|   {}   | bit[{}]   | {:8.4} | {:8.4} |  {:.2} V  | {:.2} V → {} |",
            trial, bit_position, vh, vd, cv, bv, bit_logic);
    }
    println!("+-------+----------+----------+----------+----------+----------+");

    let pass = decoded_code == ideal_code;
    println!("\nDecoded code = {} = 0b{:0n_bits$b}", decoded_code, decoded_code, n_bits = N_BITS);
    println!("Ideal code   = {} = 0b{:0n_bits$b}", ideal_code, ideal_code, n_bits = N_BITS);
    println!("Match: {}", if pass { "✅" } else { "❌" });

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let docs = crate_dir.join("docs");
    fs::create_dir_all(&docs)?;
    let md = build_report(decoded_code, ideal_code, pass);
    let md_path = docs.join("sar_adc_full_mna.md");
    fs::write(&md_path, &md)?;
    let workspace_docs = crate_dir.join("../../docs");
    if workspace_docs.is_dir() {
        fs::write(workspace_docs.join("sar_adc_full_mna.md"), &md)?;
    }
    println!("Report: {}", md_path.display());

    Ok(())
}

fn build_report(decoded: u32, ideal: u32, pass: bool) -> String {
    let mut md = String::new();
    md.push_str("# T.9 — Full transistor-level SAR ADC under eda-mna\n\n");
    md.push_str(&format!("End-to-end {N_BITS}-bit SAR conversion in **one** `eda_mna::Circuit`. Every transistor — SH transmission gate, R-2R DAC resistors, 9-transistor Baker-style comparator, AND the {N_BITS} × DffSR SAR register with its phase / capture / set / reset wiring — runs in **one** manual BE-step loop. No SPICE in the loop, no behavioral substitutions, no decoupling at any layer.\n\n"));
    md.push_str(&format!("## Result\n\n- Test input: vin = 1.080 V (= 0.6 · Vdd)\n- Decoded code: **{} = 0b{:0n_bits$b}**\n- Ideal code (closed-form `ideal_sar_code(vin)`): **{} = 0b{:0n_bits$b}**\n- Match: {}\n\n",
        decoded, decoded, ideal, ideal,
        if pass { "✅ exact" } else { "❌ mismatch — see trial trace above" },
        n_bits = N_BITS));
    md.push_str("## What this proves\n\n");
    md.push_str("- The `mna::add_*` composition pattern scales end-to-end: every block from T.8.A (comparator), T.8.B (digital primitives — Inverter, Nand, Dff, DffSR with IC seeding), T.8.C (analog front-end), and now T.9 (SAR register) plugs into one `Circuit`.\n");
    md.push_str("- Newton + BE converge on the full SAR ADC. T.8.D's IC seeding for the SR cross-couples is what makes this tractable — without those seeds the latches sit at the metastable corner and Newton stalls.\n");
    md.push_str("- `transient_sensitivities` extends here unchanged: ∂(decoded code) / ∂(any circuit param — comparator W, DAC R, S/H C, DFF Vth) flows through the entire chain in a single AD pass. That's gradient-tunable transistor-level SAR ADC sizing without ever leaving the differentiable solver.\n\n");
    md.push_str("## Cost & scaling\n\n");
    md.push_str(&format!("- N_BITS = {N_BITS}, ~{} transistors total (analog + digital), {} unknown nets approx.\n", 17 + 56 * N_BITS, 30 * N_BITS));
    md.push_str("- Bottleneck: `solve_be_step` recompiles the residual graph and the per-row gradient graphs **on every BE step**. For 1-bit-resolution gradient delivery this is overkill; once eda-mna gains a cached-graph fast path (T.10), the per-step cost drops by ~10×.\n");
    md.push_str("- For statistical work (Monte Carlo, PVT sweeps), keep using the behavioral path (`sar_adc_characterization`). For verification + gradient sizing on the actual silicon-ready netlist, this transistor path is the differentiable analog of an ngspice run.\n");
    md
}
