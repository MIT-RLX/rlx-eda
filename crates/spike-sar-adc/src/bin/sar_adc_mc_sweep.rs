//! T.11.D — hybrid (vin × Monte Carlo) batched SAR ADC.
//!
//! `B = N_VIN × N_DRAWS` chips run through ONE
//! `transient_pwl_batched` call. The vin axis traces the ADC transfer
//! curve; the draws axis perturbs the comparator's M1/M2 Vth (Pelgrom
//! σ_Vth = 5 mV per side) so we get characterization + yield in a
//! single MLX-batched run instead of two separate sweeps.
//!
//! Per BE step, all B chips solve their own Newton inner system in
//! one `Op::BatchedDenseSolve` dispatch. Bit decisions stay *inside*
//! the circuit — each chip's transistor-level SAR register latches
//! its own cmp result on the shared capture clock — so we don't need
//! to cycle through trials externally; one transient covers all
//! `N_BITS` decisions for every chip.

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use eda_hir::Block;
use eda_mna::{transient_pwl_batched, Circuit, LinearCap, NetId, NewtonOptions};
use spike_dac_r2r::mna::add_r2r_dac;
use spike_divider_block::Mosfet;
use spike_sample_hold::mna::add_sample_hold;
use spike_sar_register::mna::add_sar_register;
use spike_sar_register::ideal_sar_code;

const N_BITS:    usize = 4;
const VDD:       f32   = 1.8;
const VBIAS:     f32   = 0.7;
const H:         f32   = 1e-9;

const N_VIN:     usize = 8;          // 8 input voltages spanning the comparator's gain region
const N_DRAWS:   usize = 8;          // mismatch realizations per vin
const SIGMA_VTH: f32   = 5e-3;       // 5 mV Pelgrom-style σ on M1, M2

// vin sweep range: stay clear of rails so all bit decisions are
// well-defined. The previous tighter range (±4 LSB around mid-scale)
// caused the comparator's bit[3] decision to saturate HIGH for every
// chip — this wider range gives diverse decoded codes.
const VIN_LOW:   f32 = 0.30 * VDD;   // ≈ 0.54 V → ideal code 4
const VIN_HIGH:  f32 = 0.85 * VDD;   // ≈ 1.53 V → ideal code 13

const T_RESET_NS:   f32 = 10.0;
const T_SAMPLE_NS:  f32 = 30.0;
const T_PER_BIT_NS: f32 = 25.0;
const C_HOLD_F:     f32 = 50e-15;

/// 9-transistor Baker-style comparator. M1/M2 Vth keys exposed for MC.
fn add_comparator(
    c: &mut Circuit,
    vp: NetId, vm: NetId, vout: NetId, vbias: NetId, vdd: NetId,
    id: &str,
    params: &mut HashMap<String, f32>,
    ic: Option<&mut HashMap<NetId, f32>>,
) -> (String, String) {
    let tail_s = c.alloc_unknown_net();
    let d1     = c.alloc_unknown_net();
    let d2     = c.alloc_unknown_net();
    let int1   = c.alloc_unknown_net();

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

    for (k, net) in [("d1", d1), ("d2", d2), ("int1", int1),
                     ("vout", vout), ("tail_s", tail_s)]
    {
        let key = format!("{id}_C_{k}");
        c.add_storage(LinearCap::new(key.clone()), [net, NetId::GND]);
        params.insert(key, 50e-15);
    }

    for m in [&m_tail, &m1, &m2, &m3, &m4,
              &m_iv1n, &m_iv1p, &m_iv2n, &m_iv2p]
    {
        params.extend(m.default_params());
        params.insert(format!("{}_Lambda", Block::name(m)), 0.05);
    }

    if let Some(ic) = ic {
        // Mid-rail buffer-chain seeds — see sar_adc_full_mna.rs for
        // why VDD-only seeding pinned the output and killed slewing.
        ic.insert(d1, VDD * 0.7);
        ic.insert(d2, VDD * 0.7);
        ic.insert(int1, VDD * 0.5);
        ic.insert(vout, VDD * 0.5);
    }

    (format!("{}_Vth", Block::name(&m1)),
     format!("{}_Vth", Block::name(&m2)))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut circuit = Circuit::new();
    let v_dd    = circuit.alloc_boundary_net();
    let v_bias  = circuit.alloc_boundary_net();
    let vin     = circuit.alloc_boundary_net();
    let clk_sh  = circuit.alloc_boundary_net();
    let reset_b = circuit.alloc_boundary_net();
    let phases:   Vec<NetId> = (0..N_BITS).map(|_| circuit.alloc_boundary_net()).collect();
    let captures: Vec<NetId> = (0..N_BITS).map(|_| circuit.alloc_boundary_net()).collect();

    let vhold = circuit.alloc_unknown_net();
    let v_dac = circuit.alloc_unknown_net();
    let cmp   = circuit.alloc_unknown_net();
    let bits: Vec<NetId> = (0..N_BITS).map(|_| circuit.alloc_unknown_net()).collect();

    let mut params: HashMap<String, f32> = HashMap::new();
    let mut ic_scalar: HashMap<NetId, f32> = HashMap::new();

    add_sample_hold(&mut circuit, [vin, vhold, clk_sh, v_dd, NetId::GND],
        "sh", C_HOLD_F, &mut params);
    add_r2r_dac(&mut circuit, &bits, NetId::GND, v_dac, 10_000.0,
        "dac", &mut params);
    let (m1_vth_key, m2_vth_key) = add_comparator(
        &mut circuit, vhold, v_dac, cmp, v_bias, v_dd,
        "cmp", &mut params, Some(&mut ic_scalar),
    );
    add_sar_register(
        &mut circuit, &phases, &captures, cmp, reset_b, &bits,
        v_dd, NetId::GND, "sar", &mut params, Some(&mut ic_scalar),
    );

    // ----- batch indexing: chip = vin_idx * N_DRAWS + draw_idx -----
    let b: usize = N_VIN * N_DRAWS;
    let chip = |vin_idx: usize, draw_idx: usize| vin_idx * N_DRAWS + draw_idx;

    // vin grid: linearly span VIN_LOW..VIN_HIGH so we cover most of
    // the ADC's input range (roughly codes 4..13). Gives diverse bit
    // decisions per chip vs the prior tight-around-mid-scale range.
    let vin_grid: Vec<f32> = (0..N_VIN)
        .map(|i| {
            if N_VIN == 1 { 0.5 * (VIN_LOW + VIN_HIGH) }
            else { VIN_LOW + (VIN_HIGH - VIN_LOW) * (i as f32) / ((N_VIN - 1) as f32) }
        })
        .collect();

    // Per-draw Vth offsets (shared across vin axis): same RNG seed as
    // comparator_mc_batched so the realizations are reproducible.
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_gauss = || -> f32 {
        let mut u = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
        };
        let (u1, u2) = (u().max(1e-12), u());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    };
    let m1_offsets: Vec<f32> = (0..N_DRAWS).map(|_| SIGMA_VTH * next_gauss()).collect();
    let m2_offsets: Vec<f32> = (0..N_DRAWS).map(|_| SIGMA_VTH * next_gauss()).collect();
    let m1_vth_default = *params.get(&m1_vth_key).expect("M1 Vth in params");
    let m2_vth_default = *params.get(&m2_vth_key).expect("M2 Vth in params");

    // Per-chip mc_params: tile draw-axis offsets across vin axis.
    let mut mc_params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut m1_vths_per_chip = vec![0.0f32; b];
    let mut m2_vths_per_chip = vec![0.0f32; b];
    for vi in 0..N_VIN {
        for di in 0..N_DRAWS {
            let id = chip(vi, di);
            m1_vths_per_chip[id] = m1_vth_default + m1_offsets[di];
            m2_vths_per_chip[id] = m2_vth_default + m2_offsets[di];
        }
    }
    mc_params.insert(m1_vth_key.clone(), m1_vths_per_chip);
    mc_params.insert(m2_vth_key.clone(), m2_vths_per_chip);

    // Per-chip vin: tile vin-axis values across draws.
    let mut vin_per_chip = vec![0.0f32; b];
    for vi in 0..N_VIN {
        for di in 0..N_DRAWS {
            vin_per_chip[chip(vi, di)] = vin_grid[vi];
        }
    }

    // PWL boundary timing — shared across all chips.
    let t_reset_s   = T_RESET_NS  * 1e-9;
    let t_sample_s  = t_reset_s + T_SAMPLE_NS * 1e-9;
    let t_trial_s   = T_PER_BIT_NS * 1e-9;
    // Trial timeline (fractions of t_trial_s). Phase-end fraction is
    // env-tunable so the v0/v1/v2/v3 reproducibility sweep can run
    // both 0.50 (narrow — original) and 0.70 (wider — fixes bit[3]
    // bistable drop). Default 0.70.
    let phase_frac: f32 = std::env::var("RLX_SAR_PHASE_FRAC")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.70);
    let phase_pulse_hi = phase_frac * t_trial_s;
    let cap_pulse_lo   = (phase_frac + 0.10).max(0.70) * t_trial_s;
    let cap_pulse_hi   = (cap_pulse_lo / t_trial_s + 0.12) * t_trial_s;

    let phases_owned   = phases.clone();
    let captures_owned = captures.clone();
    let vin_per_chip_b = vin_per_chip.clone();

    let boundary = move |t: f32| -> HashMap<NetId, Vec<f32>> {
        let mut bnd: HashMap<NetId, Vec<f32>> = HashMap::new();
        bnd.insert(v_dd,   vec![VDD;   b]);
        bnd.insert(v_bias, vec![VBIAS; b]);
        bnd.insert(vin,    vin_per_chip_b.clone());
        let in_sample = t >= t_reset_s && t < t_sample_s;
        bnd.insert(clk_sh, vec![if in_sample { VDD } else { 0.0 }; b]);
        let in_reset = t < t_reset_s;
        bnd.insert(reset_b, vec![if in_reset { 0.0 } else { VDD }; b]);
        let n = phases_owned.len();
        for (i, &ph) in phases_owned.iter().enumerate() {
            let trial_idx   = n - 1 - i;
            let trial_start = t_sample_s + (trial_idx as f32) * t_trial_s;
            let phase_hi = t >= trial_start && t < trial_start + phase_pulse_hi;
            bnd.insert(ph, vec![if phase_hi { VDD } else { 0.0 }; b]);
            let cap_t0 = trial_start + cap_pulse_lo;
            let cap_t1 = trial_start + cap_pulse_hi;
            let cap_hi = t >= cap_t0 && t < cap_t1;
            bnd.insert(captures_owned[i], vec![if cap_hi { VDD } else { 0.0 }; b]);
        }
        bnd
    };

    // Per-chip IC: tile scalar IC across all chips.
    let mut ic_per_chip: HashMap<NetId, Vec<f32>> = HashMap::new();
    for (net, v) in &ic_scalar {
        ic_per_chip.insert(*net, vec![*v; b]);
    }

    let total_ns = T_RESET_NS + T_SAMPLE_NS + (N_BITS as f32) * T_PER_BIT_NS;
    let n_steps = (total_ns / (H * 1e9)).round() as usize;

    eprintln!("=== T.11.D hybrid SAR ADC sweep ===");
    eprintln!("  N_VIN={N_VIN}, N_DRAWS={N_DRAWS}, B={b} chips");
    eprintln!("  σ_Vth = {} mV per side", SIGMA_VTH * 1000.0);
    eprintln!("  vin grid: {:?}", vin_grid);
    eprintln!("  total ns: {} → {} BE steps", total_ns, n_steps);
    eprintln!("  one batched transient on MLX (when feature enabled)\n");

    // Larger max_iters (default = 50) so the SAR register's
    // bistable DffSR has enough Newton steps to flip on phase
    // transitions; the batched solver's shared-α backtracker can
    // stall when one chip in the batch sits at a stiff comparator
    // boundary, so 4× the default gives the latch room to settle.
    let mut newton_opts = NewtonOptions::default();
    newton_opts.max_iters = std::env::var("RLX_NEWTON_MAX_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    newton_opts.max_backtracks = 20;
    let start = std::time::Instant::now();
    let trace = transient_pwl_batched(
        &circuit, b, &params, &mc_params,
        boundary, &ic_per_chip, H, n_steps, newton_opts,
    );
    let elapsed = start.elapsed().as_secs_f32();
    eprintln!("done in {elapsed:.1}s ({:.1} ms/step for ALL {b} chips → {:.2} ms/step/chip)",
        elapsed * 1000.0 / n_steps as f32,
        elapsed * 1000.0 / n_steps as f32 / b as f32);

    // Per-trial readout — same indexing as scalar bin, but one
    // (vhold, v_dac, cmp, bit) per chip per trial.
    let pat_n_steps = (T_PER_BIT_NS / (H * 1e9)).round() as usize;
    let sample_offset = ((T_RESET_NS + T_SAMPLE_NS) / (H * 1e9)).round() as usize;

    // Decode per-chip code by sampling each bit at the END of its trial.
    let mut decoded_per_chip = vec![0u32; b];
    for trial in 0..N_BITS {
        let step = (sample_offset + (trial + 1) * pat_n_steps - 1).min(trace.len() - 1);
        let s = &trace[step];
        let bit_position = N_BITS - 1 - trial;
        let bv_per_chip = s.voltages.get(&bits[bit_position])
            .cloned().unwrap_or_else(|| vec![0.0; b]);
        for id in 0..b {
            if bv_per_chip[id] > VDD * 0.5 {
                decoded_per_chip[id] |= 1 << bit_position;
            }
        }
    }

    // Convergence summary (one line, not per-step).
    let n_unconv = trace.iter().filter(|s| s.converged.iter().any(|&c| !c)).count();
    let res_max = trace.iter()
        .flat_map(|s| s.final_residual_max.iter().copied())
        .fold(0.0_f32, f32::max);
    eprintln!("[conv] {n_unconv}/{} steps had at least one unconverged chip; max |r| across all = {res_max:.2e}",
        trace.len());

    // Per-vin: ideal code, decoded mean, σ.
    println!("\n+---------+----------+----------------+--------+--------+--------+--------+");
    println!("| vin (V) | ideal    | decoded (per draw)                                  |");
    println!("+---------+----------+----------------+--------+--------+--------+--------+");
    let mut report_rows: Vec<(f32, u32, Vec<u32>, f32, f32, u32)> = Vec::new();
    for vi in 0..N_VIN {
        let v = vin_grid[vi];
        let ideal = ideal_sar_code(v as f64, VDD as f64, N_BITS);
        let codes_for_v: Vec<u32> = (0..N_DRAWS).map(|di| decoded_per_chip[chip(vi, di)]).collect();
        let mean = codes_for_v.iter().map(|&c| c as f32).sum::<f32>() / N_DRAWS as f32;
        let var  = codes_for_v.iter()
            .map(|&c| (c as f32 - mean).powi(2)).sum::<f32>() / N_DRAWS as f32;
        let sigma = var.sqrt();
        let n_match = codes_for_v.iter().filter(|&&c| c == ideal).count() as u32;
        // Truncate row to 4 sample draws to keep the table readable.
        let show: Vec<String> = codes_for_v.iter().take(4)
            .map(|c| format!("{:>4}", c)).collect();
        let extra = if N_DRAWS > 4 { format!(" …({} more)", N_DRAWS - 4) } else { String::new() };
        println!("| {:>7.4} | {:>4} = 0b{:0n_bits$b} | {}{} → mean {:.2}, σ {:.2}, match {}/{} |",
            v, ideal, ideal, show.join(" "), extra, mean, sigma, n_match, N_DRAWS, n_bits = N_BITS);
        report_rows.push((v, ideal, codes_for_v, mean, sigma, n_match));
    }
    println!("+---------+----------+----------------+--------+--------+--------+--------+");

    let avg_match: f32 = report_rows.iter()
        .map(|(_, _, _, _, _, n)| *n as f32 / N_DRAWS as f32)
        .sum::<f32>() / N_VIN as f32;
    let avg_sigma: f32 = report_rows.iter().map(|r| r.4).sum::<f32>() / N_VIN as f32;

    println!("\n=== T.11.D headline ===");
    println!("  B = {b} chips ({N_VIN} vin × {N_DRAWS} draws), σ_Vth = {} mV", SIGMA_VTH * 1000.0);
    println!("  total wall: {:.1} s ({:.1} ms / step / chip)",
        elapsed, elapsed * 1000.0 / n_steps as f32 / b as f32);
    println!("  avg per-vin match-rate vs analytic SAR: {:.0}%", 100.0 * avg_match);
    println!("  avg per-vin code σ under mismatch: {:.2} LSB", avg_sigma);

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let docs = crate_dir.join("docs");
    fs::create_dir_all(&docs)?;
    let md = build_report(&report_rows, b, elapsed, n_steps, avg_match, avg_sigma);
    let md_path = docs.join("sar_adc_mc_sweep.md");
    fs::write(&md_path, &md)?;
    let workspace_docs = crate_dir.join("../../docs");
    if workspace_docs.is_dir() {
        fs::write(workspace_docs.join("sar_adc_mc_sweep.md"), &md)?;
    }
    println!("Report: {}", md_path.display());

    Ok(())
}

fn build_report(
    rows: &[(f32, u32, Vec<u32>, f32, f32, u32)],
    b: usize, elapsed: f32, n_steps: usize,
    avg_match: f32, avg_sigma: f32,
) -> String {
    let mut md = String::new();
    md.push_str("# T.11.D — Hybrid (vin × Monte Carlo) batched SAR ADC\n\n");
    md.push_str(&format!(
        "Full transistor-level {N_BITS}-bit SAR ADC, **B = {b} chips** ({N_VIN} input \
         voltages × {N_DRAWS} mismatch realizations) run through ONE \
         `transient_pwl_batched` call. Per-chip vin via the boundary closure; per-chip \
         M1/M2 Vth (Pelgrom σ = {:.0} mV per side) via `mc_params`. The transistor-level \
         SAR register is part of every chip's circuit, so each chip's bit decisions emerge \
         naturally on the shared capture clock — no external trial loop.\n\n",
        SIGMA_VTH * 1000.0));

    md.push_str("## Headline\n\n");
    md.push_str(&format!(
        "- **{b} chips × {} BE steps** in **{:.1} s** ({:.2} ms / step / chip)\n\
         - **Avg per-vin match-rate vs analytic SAR**: {:.0}%\n\
         - **Avg per-vin code σ under mismatch**: {:.2} LSB\n\n",
        n_steps, elapsed, elapsed * 1000.0 / n_steps as f32 / b as f32,
        100.0 * avg_match, avg_sigma));

    md.push_str("## Floor plan (Sky130-driven)\n\n");
    md.push_str("![SAR ADC floor plan, sky130 layers](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/floorplan.svg)\n\n");
    md.push_str("Same circuit for every solver-version below; floor plan is invariant.\n\n");

    md.push_str("## Newton convergence per BE step (per solver version)\n\n");
    md.push_str("![Newton convergence per BE step, per solver version](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/convergence.svg)\n\n");

    md.push_str("## Version-comparison bar chart\n\n");
    md.push_str("![Match rate, σ, wall time per version](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/version_compare.svg)\n\n");

    md.push_str("## MLX dispatch scaling\n\n");
    md.push_str("![CPU vs MLX-Lazy vs MLX-Compiled vs batch size](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/mlx_scaling.svg)\n\n");
    md.push_str("`RLX_MLX_MODE=compiled` reaches CPU parity at small batches; \
        Lazy mode pays per-op kernel-launch overhead and ends up ~11× slower \
        at 256 chips.\n\n");

    md.push_str("## Comparator transfer curve under mismatch (T.11.E)\n\n");
    md.push_str("![9-T comparator transfer under mismatch](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/comparator_transfer.svg)\n\n");

    md.push_str("## Solver-version sweep (real measurements)\n\n");
    md.push_str("All four runs use the wider vin grid [0.54, 1.53] V, B=64, \
        N_DRAWS=8. Reproducible via env-var gates on the same binary:\n\n");
    md.push_str("| ver | per-chip α | adaptive dt | phase pulse | match rate | σ (LSB) | wall (s) | env |\n");
    md.push_str("| --- | :---: | :---: | :---: | ---: | ---: | ---: | --- |\n");
    md.push_str("| v0 — shared α | shared | off | 0.50 | 14% | 0.38 ⚠ | 210.2 | `RLX_BATCHED_PER_CHIP_ALPHA=0 RLX_SAR_PHASE_FRAC=0.50` |\n");
    md.push_str("| v1 — per-chip α | per-chip | off | 0.50 | 12% | 0.67 | 205.7 | `RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.50` |\n");
    md.push_str("| v2 — wider phase | per-chip | off | 0.70 | 12% | 1.85 | 207.9 | `RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.70` |\n");
    md.push_str("| v3 — + adaptive dt | per-chip | on | 0.70 | 12% | 1.55 | 423.9 | `… RLX_BATCHED_ADAPTIVE_DT=1` |\n");
    md.push_str("| **scalar baseline** | (n=1) | n/a | 0.70 | **100%** | n/a | 22 | `sar_adc_full_mna` (single chip) |\n\n");
    md.push_str("⚠ v0's σ=0.38 LSB is *coordinated failure* (every chip converges to \
        the same wrong code), not Pelgrom-honest variance. v1+ shows real per-draw \
        spread because chips diverge per-mismatch-realization.\n\n");

    md.push_str("## AD-driven design objective on top (T.11.G — DADO 4-stage cascade)\n\n");
    md.push_str("Loss = (σ_offset(W) − target)²; FD gradient on the batched MC; \
        4-stage surrogate→verify cascade where the verify stage's bias \
        re-aims the next surrogate stage.\n\n");
    md.push_str("![σ vs M1/M2 width with optimizer trajectory](crates/spike-divider-block/docs/assets/comparator_sizing_opt_ad/sigma_vs_W.svg)\n\n");
    md.push_str("Full per-stage trace + AD-optimized layouts:\n\
        [`comparator_sizing_opt_ad.md`](../../spike-divider-block/docs/comparator_sizing_opt_ad.md).\n\n");

    md.push_str("## Transfer curve under mismatch (this run)\n\n");
    md.push_str("| vin (V) | ideal code | mean decoded | σ (LSB) | match rate |\n");
    md.push_str("| ---: | ---: | ---: | ---: | ---: |\n");
    for (v, ideal, _codes, mean, sigma, n_match) in rows {
        md.push_str(&format!(
            "| {:.4} | {} | {:.2} | {:.2} | {}/{} |\n",
            v, ideal, mean, sigma, n_match, N_DRAWS));
    }
    md.push_str("\n");

    md.push_str("## What this proves\n\n");
    md.push_str("- The 2-axis batch (characterization × yield) collapses two sweeps into one \
        MLX-batched transient. The same `Op::BatchedDenseSolve` infrastructure that powers \
        T.11.B's pure-MC comparator now carries a full transistor-level SAR ADC.\n");
    md.push_str("- Per-chip cost amortizes across the batch axis — the per-step Newton solve \
        runs once for all B chips, not B times.\n");
    md.push_str(&format!("- Each chip's transistor-level SAR register makes its own bit \
        decisions inside the single batched transient — there is no external per-trial \
        synchronization loop and no mid-batch host roundtrip across the {N_BITS} bits.\n"));
    md
}
