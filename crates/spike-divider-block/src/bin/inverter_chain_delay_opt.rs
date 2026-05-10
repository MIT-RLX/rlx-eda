//! Differentiable optimization of an inverter chain's propagation delay.
//!
//! Mirror of `inverter_multiparam_opt.rs`, but operating in the
//! **time domain** instead of DC: the loss is the chain's output
//! voltage at a target sample time, and the gradients flow through
//! `eda_mna::transient_sensitivities` (T.1's headline primitive).
//!
//! Topology: 3-stage CMOS inverter chain.
//!
//! ```text
//!   vin → INV1 → n1 → INV2 → n2 → INV3 → vout
//! ```
//!
//! Stimulus: `vin` rises from 0 to Vdd at t = 10 ns (rectangular
//! pulse). Internal state initialized so the chain is at the proper
//! pre-edge steady state (vout = high, since 3 inversions of low = high).
//! After the rising edge, vout falls toward 0 — the time it takes is
//! the chain's propagation delay.
//!
//! Loss = `(vout(t_target) − vout_target)²` at one sample time.
//! Picking `t_target = 60 ns` and `vout_target = 0.0 V` says "the
//! chain should be done falling by 60 ns". If vout(60ns) is too HIGH,
//! the gradient pushes Vth_n's DOWN (NMOSs turn on more easily, chain
//! gets faster).
//!
//! Parameters: per-stage NMOS Vth (3 of them). Optimize via Adam.
//!
//! Run:
//!   cargo run -p spike-divider-block --bin inverter_chain_delay_opt

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use eda_mna::{
    pulse_boundary, transient_pwl, transient_sensitivities,
    Circuit, LinearCap, NetId, NewtonOptions,
};
use spike_divider_block::Resistor;
use eda_hir::{Block, Layout as _};
use spike_divider_block::{Adam, Mosfet, Optimizer};
use spike_divider_block::pdks_foundry::{Sky130, HAS_SKY130};
use klayout_connect::{extract_hierarchical, Conductor, ExtractConfig, Via};
use klayout_core::{Bbox, CellBuilder, CellId, Instance, LayerIndex, Library, Point, Rect, Trans, Vec2};
use klayout_geom::Region;
use spike_divider_block::MosfetPdk;

const VDD: f32 = 1.0;
const H: f32 = 0.5e-9;          // 0.5 ns BE step
const N_STEPS: usize = 80;      // 40 ns transient window
const T_TARGET_S: f32 = 15e-9;  // sample loss at t = 15 ns (mid-fall)
const VOUT_TARGET: f32 = 0.0;   // by 15 ns, chain should be fully fallen
const T_RISE_S: f32 = 5e-9;     // vin rising edge at 5 ns
const C_LOAD: f32 = 200e-15;    // 200 fF ground-tied load on each internal
                                // node — gives clean RC delay, no Miller
                                // kickback (no gate↔drain Cgd path)
const VTH_INIT: f32 = 0.45;     // start with slowed chain (above default Vth=0.5
                                // would shut everything off; below 0.5 gives
                                // a transition-region trace at our timescale)

#[derive(Clone, Copy, Debug)]
struct StepRow {
    step: usize,
    vth_1: f32, vth_2: f32, vth_3: f32,
    vout_at_t_target: f32,
    loss: f32,
    g1: f32, g2: f32, g3: f32,
}

struct ChainBuild {
    circuit: Circuit,
    v_dd: NetId,
    vin: NetId,
    #[allow(dead_code)] n1: NetId,
    #[allow(dead_code)] n2: NetId,
    vout: NetId,
    params: HashMap<String, f32>,
    ic: HashMap<NetId, f32>,
    vth_param_names: [String; 3],
    pex: Option<Pex>,
}

#[derive(Clone, Debug)]
struct Pex {
    r_gate_path: f32,   // Ω, from drain bus → poly gate bridge per stage
    c_per_node:  f32,   // F, additional substrate cap per internal node
}

fn build_chain(with_pex: bool) -> ChainBuild {
    let mut circuit = Circuit::new();
    let v_dd = circuit.alloc_boundary_net();
    let vin  = circuit.alloc_boundary_net();
    let n1   = circuit.alloc_unknown_net();
    let n2   = circuit.alloc_unknown_net();
    let vout = circuit.alloc_unknown_net();

    // Optional gate-input nets so we can put the parasitic R between the
    // previous stage's drain and this stage's gate input.
    let (g1_net, g2_net, g3_net, pex) = if with_pex {
        let pex = estimate_pex();
        (vin, circuit.alloc_unknown_net(), circuit.alloc_unknown_net(), Some(pex))
    } else {
        (vin, n1, n2, None)
    };

    // 3 inverter stages.
    let nmos1 = Mosfet::nmos(2_000, 1_000, "Mn1");
    let pmos1 = Mosfet::pmos(4_000, 1_000, "Mp1");
    let nmos2 = Mosfet::nmos(2_000, 1_000, "Mn2");
    let pmos2 = Mosfet::pmos(4_000, 1_000, "Mp2");
    let nmos3 = Mosfet::nmos(2_000, 1_000, "Mn3");
    let pmos3 = Mosfet::pmos(4_000, 1_000, "Mp3");
    // Stage 1: gate=g1 (= vin), drain=n1
    circuit.add_device(nmos1.clone(), &[n1, g1_net, NetId::GND, NetId::GND]);
    circuit.add_device(pmos1.clone(), &[n1, g1_net, v_dd, v_dd]);
    // Stage 2: gate=g2, drain=n2 — g2 is fed from n1 via a parasitic R
    circuit.add_device(nmos2.clone(), &[n2, g2_net, NetId::GND, NetId::GND]);
    circuit.add_device(pmos2.clone(), &[n2, g2_net, v_dd, v_dd]);
    // Stage 3: gate=g3, drain=vout — g3 is fed from n2 via a parasitic R
    circuit.add_device(nmos3.clone(), &[vout, g3_net, NetId::GND, NetId::GND]);
    circuit.add_device(pmos3.clone(), &[vout, g3_net, v_dd, v_dd]);

    // Cload to ground on each drain node — sets the dominant RC delay.
    circuit.add_storage(LinearCap::new("Cload_n1"),   [n1,   NetId::GND]);
    circuit.add_storage(LinearCap::new("Cload_n2"),   [n2,   NetId::GND]);
    circuit.add_storage(LinearCap::new("Cload_vout"), [vout, NetId::GND]);

    let mut params: HashMap<String, f32> = HashMap::new();
    for m in [&nmos1, &nmos2, &nmos3] { params.extend(m.default_params()); }
    for m in [&pmos1, &pmos2, &pmos3] { params.extend(m.default_params()); }
    params.insert("Cload_n1".into(),   C_LOAD);
    params.insert("Cload_n2".into(),   C_LOAD);
    params.insert("Cload_vout".into(), C_LOAD);

    // Wire in extracted parasitics: R in series at the gate input of
    // stages 2 and 3, plus an additional substrate cap on each gate
    // input node and each drain node.
    if let Some(p) = &pex {
        let rg2 = Resistor { length: 0, id: "Rg2".into() };
        let rg3 = Resistor { length: 0, id: "Rg3".into() };
        circuit.add_device(rg2.clone(), &[n1, g2_net]);
        circuit.add_device(rg3.clone(), &[n2, g3_net]);
        params.insert(Block::name(&rg2), p.r_gate_path);
        params.insert(Block::name(&rg3), p.r_gate_path);
        // Extra to-ground caps from M1/poly area (in addition to Cload).
        circuit.add_storage(LinearCap::new("Cpex_g2"), [g2_net, NetId::GND]);
        circuit.add_storage(LinearCap::new("Cpex_g3"), [g3_net, NetId::GND]);
        circuit.add_storage(LinearCap::new("Cpex_n1"), [n1,     NetId::GND]);
        circuit.add_storage(LinearCap::new("Cpex_n2"), [n2,     NetId::GND]);
        circuit.add_storage(LinearCap::new("Cpex_vo"), [vout,   NetId::GND]);
        params.insert("Cpex_g2".into(), p.c_per_node);
        params.insert("Cpex_g3".into(), p.c_per_node);
        params.insert("Cpex_n1".into(), p.c_per_node);
        params.insert("Cpex_n2".into(), p.c_per_node);
        params.insert("Cpex_vo".into(), p.c_per_node);
    }

    let n1_name = Block::name(&nmos1);
    let n2_name = Block::name(&nmos2);
    let n3_name = Block::name(&nmos3);
    let vth_param_names = [
        format!("{n1_name}_Vth"),
        format!("{n2_name}_Vth"),
        format!("{n3_name}_Vth"),
    ];

    // Override default Vth's with our slow-chain starting point.
    for name in &vth_param_names { params.insert(name.clone(), VTH_INIT); }

    // IC: with vin=0, chain steady state is n1=Vdd, n2=0, vout=Vdd. Gate-
    // input nets settle to the same state as their driving drain.
    let mut ic = HashMap::new();
    ic.insert(n1, VDD);
    ic.insert(n2, 0.0);
    ic.insert(vout, VDD);
    if with_pex {
        ic.insert(g2_net, VDD);
        ic.insert(g3_net, 0.0);
    }

    ChainBuild { circuit, v_dd, vin, n1, n2, vout, params, ic, vth_param_names, pex }
}

/// Per-net wire R + parasitic C estimated from the floorplan's actual
/// routing geometry (M1 + POLY) using published Sky130A sheet values.
fn estimate_pex() -> Pex {
    // Sky130 typical sheet R / area C (process docs).
    const R_M1_PER_SQ:    f32 = 0.125;       // Ω/square
    const R_POLY_PER_SQ:  f32 = 48.0;        // Ω/square
    // 38 aF/µm² = 38e-18 F per 1e6 nm² = 38e-24 F/nm². Same shift for POLY.
    const C_M1_PER_NM2:   f32 = 38.0e-24;    // F/nm² (= 38 aF/µm²)
    const C_POLY_PER_NM2: f32 = 110.0e-24;   // F/nm²

    // Per-stage signal-wire dimensions (DBU = nm) — read off the
    // floorplan's known routing geometry.
    const M1_INTER_LEN: f32 =   8_750.0;  // inter-stage M1 horizontal
    const M1_DRAIN_LEN: f32 =  15_000.0;  // drain bus M1 vertical
    const M1_WIDTH:     f32 =     600.0;
    const POLY_LEN:     f32 =  12_000.0;  // poly gate U-bridge total path
    const POLY_W:       f32 =   1_000.0;

    let r_inter = (M1_INTER_LEN / M1_WIDTH) * R_M1_PER_SQ;
    let r_poly  = (POLY_LEN / POLY_W) * R_POLY_PER_SQ;
    // Drain-bus M1 R is small and lumped into the drain node Cload — we
    // attribute the full series gate path (inter-stage M1 + poly bridge)
    // to the gate input of the next stage.
    let r_gate_path = r_inter + r_poly;

    // Lump cap = M1 inter-stage + drain bus + poly bridge → to substrate.
    let c_per_node = M1_INTER_LEN * M1_WIDTH * C_M1_PER_NM2
        + M1_DRAIN_LEN * M1_WIDTH * C_M1_PER_NM2
        + POLY_LEN     * POLY_W   * C_POLY_PER_NM2;

    Pex { r_gate_path, c_per_node }
}

fn run_optimization(build: &mut ChainBuild) -> Vec<StepRow> {
    let target_param_names = build.vth_param_names.to_vec();
    let mut p_vec = [
        *build.params.get(&build.vth_param_names[0]).unwrap(),
        *build.params.get(&build.vth_param_names[1]).unwrap(),
        *build.params.get(&build.vth_param_names[2]).unwrap(),
    ];

    let max_iters = 60_usize;
    let tol = 1e-4_f32;
    let solver = NewtonOptions::default();
    let mut opt = Adam::new(0.02_f32, 3);

    // Pre-edge static boundary (vdd = Vdd, vin = 0).
    let mut static_b = HashMap::new();
    static_b.insert(build.v_dd, VDD);
    let bnd_pulse = pulse_boundary(static_b, build.vin, 0.0, VDD, T_RISE_S, 1e9);

    let target_step = (T_TARGET_S / H).round() as usize;
    assert!(target_step < N_STEPS, "T_TARGET out of run");

    // Post-pulse boundary for sensitivities (vin = VDD after the rising edge).
    let mut bnd_post = HashMap::new();
    bnd_post.insert(build.v_dd, VDD);
    bnd_post.insert(build.vin, VDD);

    // Find vout's index inside the unknowns vector. Unknowns are
    // allocated in order: n1, n2, vout, [g2, g3] (PEX only). vout
    // is index 2 either way.
    let vout_idx = 2;

    let mut rows: Vec<StepRow> = Vec::with_capacity(max_iters + 1);
    for step in 0..=max_iters {
        for (i, name) in build.vth_param_names.iter().enumerate() {
            build.params.insert(name.clone(), p_vec[i]);
        }
        let trace = transient_pwl(
            &build.circuit, &build.params, &bnd_pulse, &build.ic, H, N_STEPS, solver,
        );
        let vout_t = trace[target_step].voltages.get(&build.vout).copied().unwrap_or(0.0);
        let err = vout_t - VOUT_TARGET;
        let loss = err * err;

        let sens = transient_sensitivities(
            &build.circuit, &build.params, &bnd_post, &trace, H, &target_param_names,
        );
        let g_vth = |i: usize| -> f32 {
            sens.get(&build.vth_param_names[i])
                .and_then(|t| t.get(target_step))
                .map(|v| v[vout_idx])
                .unwrap_or(0.0) * 2.0 * err
        };
        let g1 = g_vth(0); let g2 = g_vth(1); let g3 = g_vth(2);

        rows.push(StepRow {
            step,
            vth_1: p_vec[0], vth_2: p_vec[1], vth_3: p_vec[2],
            vout_at_t_target: vout_t,
            loss,
            g1, g2, g3,
        });
        if loss < tol { break; }
        opt.step(&mut p_vec, &[g1, g2, g3]);
        for x in &mut p_vec { *x = x.clamp(0.05, 0.95); }
    }
    rows
}

fn main() -> Result<(), Box<dyn Error>> {
    // Baseline: ideal chain (no extracted parasitics).
    let mut build_ideal = build_chain(false);
    let rows = run_optimization(&mut build_ideal);

    // PEX-aware: same chain + parasitic R + parasitic Cs from the layout.
    let mut build_pex = build_chain(true);
    let rows_pex = run_optimization(&mut build_pex);
    let pex = build_pex.pex.clone().expect("pex set");

    let final_row = *rows.last().unwrap();
    let final_pex_row = *rows_pex.last().unwrap();
    println!("Inverter-chain delay optimization (Adam, transient gradients)");
    println!("  vdd = {VDD} V, t_target = {} ns, vout_target = {VOUT_TARGET} V",
        (T_TARGET_S * 1e9).round());
    println!("  ideal:    Vth_n = ({:.3}, {:.3}, {:.3}) V, vout(t*) = {:.4} V, loss = {:.3e}, steps = {}",
        final_row.vth_1, final_row.vth_2, final_row.vth_3,
        final_row.vout_at_t_target, final_row.loss, final_row.step);
    println!("  pex-aware:Vth_n = ({:.3}, {:.3}, {:.3}) V, vout(t*) = {:.4} V, loss = {:.3e}, steps = {}",
        final_pex_row.vth_1, final_pex_row.vth_2, final_pex_row.vth_3,
        final_pex_row.vout_at_t_target, final_pex_row.loss, final_pex_row.step);
    println!("  pex: R_gate_path = {:.1} Ω, C_per_node = {:.3} fF",
        pex.r_gate_path, pex.c_per_node * 1e15);

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let assets = crate_dir.join("docs/assets/inverter_chain_delay_opt");
    fs::create_dir_all(&assets)?;
    write_rendered_svgs(&rows, &assets)?;
    fs::write(assets.join("schematic.svg"), schematic_svg(&rows[0], &final_row))?;
    let (drc_summary, lvs_summary) = build_floorplan_and_run_checks(&assets)?;
    let csv = "/tmp/rlx_eda_inverter_chain_delay_opt.csv";
    fs::write(csv, build_csv(&rows))?;
    let md_path = crate_dir.join("docs/inverter_chain_delay_opt_trace.md");
    let md = build_report(&rows, &rows_pex, &pex, &drc_summary, &lvs_summary);
    fs::write(&md_path, &md)?;
    println!("\nwrote CSV report : {csv}\nwrote MD report  : {}\nwrote SVG charts : {}/",
        md_path.display(), assets.display());

    let workspace_docs = crate_dir.join("../../docs");
    if workspace_docs.is_dir() {
        let workspace_md = workspace_docs.join("inverter_chain_delay_opt_trace.md");
        let workspace_assets = workspace_docs.join("assets/inverter_chain_delay_opt");
        fs::create_dir_all(&workspace_assets)?;
        for name in ["loss.svg", "params.svg", "output.svg", "grads.svg",
                     "schematic.svg", "floorplan.svg", "floorplan.png"] {
            let src = assets.join(name);
            if src.exists() {
                fs::copy(src, workspace_assets.join(name))?;
            }
        }
        fs::write(&workspace_md, &md)?;
        println!("mirrored to      : {}", workspace_md.display());
    }
    Ok(())
}

fn build_csv(rows: &[StepRow]) -> String {
    let mut s = String::from("step,vth_1,vth_2,vth_3,vout_at_t_target,loss,g1,g2,g3\n");
    for r in rows {
        s.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6e},{:.6e},{:.6e},{:.6e}\n",
            r.step, r.vth_1, r.vth_2, r.vth_3, r.vout_at_t_target,
            r.loss, r.g1, r.g2, r.g3,
        ));
    }
    s
}

fn build_report(rows: &[StepRow], rows_pex: &[StepRow], pex: &Pex, drc: &DrcSummary, lvs: &LvsSummary) -> String {
    let first = rows.first().unwrap();
    let last = rows.last().unwrap();

    let mut md = String::new();
    md.push_str("# rlx-eda inverter-chain delay optimization (Adam, transient gradients)\n\n");
    md.push_str("Circuit: 3-stage CMOS inverter chain (NMOS + PMOS per stage), with a 200 fF ground-tied load cap on each internal node setting the RC propagation delay. `vin` steps from 0 → Vdd at t = 5 ns; the output `vout` falls 3 inversions later, on a delay set by the per-stage NMOS Vth and the load caps.\n\n");
    md.push_str(&format!(
        "Stimulus: `Vdd = {VDD} V`, `t_target = {} ns`, `vout_target = {VOUT_TARGET} V`.\n\n",
        (T_TARGET_S * 1e9).round(),
    ));
    md.push_str("Loss:\n\n");
    md.push_str("$$L = (V_{out}(t_{\\text{target}}) - V_{out}^*)^2$$\n\n");
    md.push_str("Per-parameter gradient via reverse-mode AD on the BE-step residual at each timestep, propagated forward through the cap history coupling. See `eda_mna::transient_sensitivities` for the IFT recurrence.\n\n");

    md.push_str("## Optimization outcome\n\n");
    md.push_str(&format!(
        "- initial: `Vth_n = ({:.3}, {:.3}, {:.3}) V`, `vout({} ns) = {:.4} V`, `loss = {:.3e}`\n",
        first.vth_1, first.vth_2, first.vth_3,
        (T_TARGET_S * 1e9).round(), first.vout_at_t_target, first.loss,
    ));
    md.push_str(&format!(
        "- final:   `Vth_n = ({:.3}, {:.3}, {:.3}) V`, `vout({} ns) = {:.4} V`, `loss = {:.3e}`, `steps = {}`\n",
        last.vth_1, last.vth_2, last.vth_3,
        (T_TARGET_S * 1e9).round(), last.vout_at_t_target, last.loss, last.step,
    ));
    md.push_str("\nAll gradients computed by reverse-mode AD on the BE residual graph + per-step IFT recurrence — no SPICE oracle, no finite differences.\n\n");

    md.push_str("## Schematic\n\n");
    md.push_str("Three CMOS inverter stages in series, with a 200 fF ground-tied load cap on each internal node setting the per-stage RC delay. The three NMOS Vth values (`Vth_n1`, `Vth_n2`, `Vth_n3`) are the optimization parameters; PMOS Vth's and W/L are held fixed.\n\n");
    md.push_str("![schematic](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/schematic.svg)\n\n");

    md.push_str("## Floorplan (Sky130)\n\n");
    md.push_str("Real PDK-driven layout — 6 `Mosfet` cells (3 NMOS bottom, 3 PMOS top) placed into a Sky130 `Library`, with full electrical routing on metal1 + poly. Layers + colors come from the Sky130 `.lyp`; rendered via `eda_viz::layout::render_to_svg`.\n\n");
    md.push_str("**Routing scheme** (every net is electrically distinct — no stray shorts):\n\n");
    md.push_str("- **Power**: wide horizontal M1 Vdd / GND rails top + bottom, with short vertical source straps from each transistor's source port.\n");
    md.push_str("- **Body bias**: NMOS body-tap drops directly to GND (clear column south of the device). PMOS body-tap routes UP, jogs east through the narrow M1 channel between the PMOS drain pad and gate pad (y ≈ 17.2 µm), then up to Vdd at x = +4.5 µm — clear of both the drain bus and the gate bus.\n");
    md.push_str("- **Gate bus**: each stage's PMOS gate ↔ NMOS gate is on POLY (RES layer), filling the gap between the existing per-cell poly sticks. POLY-only gate routing keeps M1 clear of the body-tap pad column entirely (the original M1-only attempt had the gate net merging with the PMOS body pad — fatal short).\n");
    md.push_str("- **Drain bus**: vertical M1 per stage shorting PMOS drain to NMOS drain — this *is* the stage output net.\n");
    md.push_str("- **Inter-stage**: horizontal M1 in the routing channel (y ≈ 7 µm) carries each stage's output to the next stage's gate, where a VIA1 (M1↔poly contact) drops the signal onto the poly gate bus.\n");
    md.push_str("- **External**: vin pad on the left feeds stage-0 gate via the same M1+VIA1 path; vout pad on the right is driven by stage-2's drain bus.\n\n");
    md.push_str("> **Project rule**: every floorplan in this repo must be rendered against a real PDK (Sky130, Gf180mcu, …) — no stylized hand-drawn floorplans. Layer geometry has to match a real foundry stack so reviewers can spot routing/DRC issues by eye.\n\n");
    md.push_str("![floorplan](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/floorplan.png)\n\n");
    md.push_str("[Open as SVG](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/floorplan.svg) (vector, zoomable).\n\n");

    md.push_str("## DRC (Sky130)\n\n");
    if drc.skipped {
        md.push_str("Skipped — Sky130 `.lyp` not present at build time.\n\n");
    } else {
        let total = drc.total();
        let banner = if total == 0 {
            "✅ **DRC clean** — 0 violations across the rule set below."
        } else {
            "❌ **DRC violations detected** — see counts below."
        };
        md.push_str(banner); md.push_str("\n\n");
        md.push_str("Selected min-rule subset of the published Sky130A deck (real foundry decks have hundreds of rules; this is the load-bearing slice for hand-routed analog floorplans):\n\n");
        md.push_str("| Rule | Min (DBU) | Violations |\n");
        md.push_str("| --- | ---: | ---: |\n");
        for r in &drc.rules {
            md.push_str(&format!("| `{}` | {} | {} |\n", r.name, r.min_dbu, r.violations));
        }
        md.push_str("\nRun via `klayout_drc::{width, space, enclosing}` over a `klayout_geom::Region` extracted from the floorplan's top cell, per layer. Each rule returns a region of failing geometry; the violation count is the polygon count of that region.\n\n");
    }

    md.push_str("## LVS (layout-vs-schematic)\n\n");
    if lvs.skipped {
        md.push_str("Skipped — Sky130 `.lyp` not present at build time.\n\n");
    } else {
        let banner = if lvs.pass() {
            "✅ **LVS pass** — extracted net count matches the schematic, every expected probe lands in its own distinct net."
        } else {
            "❌ **LVS mismatch** — see per-net detail below."
        };
        md.push_str(banner); md.push_str("\n\n");
        md.push_str(&format!("Extracted **{}** nets ({}M1 + POLY merged via VIA1); schematic expects **{}**.\n\n",
            lvs.extracted_net_count, "", lvs.expected_net_count));
        md.push_str("Probe-to-net match (each row asserts that a known coordinate inside the named net's wire actually lands inside an extracted net):\n\n");
        md.push_str("| Net | Probe (DBU) | Matched extracted net | Result |\n");
        md.push_str("| --- | --- | ---: | :---: |\n");
        for n in &lvs.nets {
            let p = n.expected.probe;
            let (idx_str, ok) = match n.matched_net_idx {
                Some(i) => (format!("net_{i}"), "✅"),
                None    => ("(none)".into(),    "❌"),
            };
            md.push_str(&format!("| **{}** | ({}, {}) | {} | {} |\n",
                n.expected.name, p.x, p.y, idx_str, ok));
        }
        md.push_str(&format!("\nDistinct extracted nets matched: **{}** of {} expected.\n\n",
            lvs.distinct_count(), lvs.nets.len()));
        md.push_str("Run via `klayout_connect::extract_hierarchical` with M1 + POLY conductors and a VIA1 join rule. The extractor walks every shape on the conductor layers, merges polygons that touch (per layer), then joins layers across via cuts.\n\n");
    }

    md.push_str("## PEX (parasitic-aware re-optimization)\n\n");
    md.push_str("Per-net wire R + per-node parasitic C estimated from the floorplan's actual routing geometry, using published Sky130A sheet values:\n\n");
    md.push_str("- M1 sheet R: 125 mΩ/sq, M1 area C: 38 aF/µm²\n");
    md.push_str("- POLY sheet R: 48 Ω/sq, POLY area C: 110 aF/µm²\n");
    md.push_str("- inter-stage M1 horizontal: 8.75 µm × 0.6 µm\n");
    md.push_str("- drain bus M1 vertical: 15 µm × 0.6 µm\n");
    md.push_str("- POLY gate U-bridge: 12 µm × 1 µm\n\n");
    md.push_str(&format!(
        "→ **R_gate_path = {:.1} Ω** (in series at each stage's gate input — drain-bus M1 + inter-stage M1 + poly gate bridge)\n",
        pex.r_gate_path));
    md.push_str(&format!(
        "→ **C_per_node = {:.3} fF** (parasitic cap from M1 + poly area to substrate, attached on each gate-input net AND each drain net in addition to the 200 fF Cload)\n\n",
        pex.c_per_node * 1e15));
    md.push_str("These values are wired into the simulated circuit as new `Resistor` + `LinearCap` devices. The differentiable solver doesn't know which nets came from \"the layout\" vs \"the schematic\" — `transient_sensitivities` propagates ∂L/∂Vth through the *augmented* circuit just as it did through the ideal one. **No code path in eda-mna changes** — extracted parasitics are first-class circuit elements.\n\n");

    let f0 = rows.first().unwrap();
    let l0 = rows.last().unwrap();
    let lp = rows_pex.last().unwrap();
    md.push_str("Comparison of converged Adam state:\n\n");
    md.push_str("| Variant | Vth_n1 | Vth_n2 | Vth_n3 | vout(t*) | loss | steps |\n");
    md.push_str("| --- | --- | --- | --- | --- | --- | ---: |\n");
    md.push_str(&format!(
        "| initial   | {:.3} | {:.3} | {:.3} | {:.4} | {:.3e} | — |\n",
        f0.vth_1, f0.vth_2, f0.vth_3, f0.vout_at_t_target, f0.loss));
    md.push_str(&format!(
        "| ideal     | {:.3} | {:.3} | {:.3} | {:.4} | {:.3e} | {} |\n",
        l0.vth_1, l0.vth_2, l0.vth_3, l0.vout_at_t_target, l0.loss, l0.step));
    md.push_str(&format!(
        "| pex-aware | {:.3} | {:.3} | {:.3} | {:.4} | {:.3e} | {} |\n",
        lp.vth_1, lp.vth_2, lp.vth_3, lp.vout_at_t_target, lp.loss, lp.step));
    let dvth1 = lp.vth_1 - l0.vth_1;
    let dvth2 = lp.vth_2 - l0.vth_2;
    let dvth3 = lp.vth_3 - l0.vth_3;
    md.push_str(&format!(
        "\nVth shift attributable to parasitics: ΔVth = ({:+.4}, {:+.4}, {:+.4}) V. The chain has to compensate for the added gate-side RC delay; whichever Vth's the optimizer pulls further down is the gradient telling us where the parasitics bite hardest.\n\n",
        dvth1, dvth2, dvth3));


    md.push_str("## Rendered charts\n\n");
    md.push_str("| Loss over steps | Per-stage Vth trajectories |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![loss](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/loss.svg) | ![params](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/params.svg) |\n\n");
    md.push_str("| vout(t_target) tracking | Per-parameter gradient |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![output](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/output.svg) | ![grads](crates/spike-divider-block/docs/assets/inverter_chain_delay_opt/grads.svg) |\n\n");

    // (Mermaid xychart-beta blocks were tried here but the project's
    // markdown renderer doesn't support them — the SVGs above already
    // visualize the same data and render anywhere. The trace table
    // below carries the exact numbers.)


    md.push_str("## Step-by-step trace\n\n");
    md.push_str("| step | Vth_n1 | Vth_n2 | Vth_n3 | vout(t*) | loss | g1 | g2 | g3 |\n");
    md.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3e} | {:.3e} | {:.3e} | {:.3e} |\n",
            r.step, r.vth_1, r.vth_2, r.vth_3, r.vout_at_t_target,
            r.loss, r.g1, r.g2, r.g3,
        ));
    }
    md
}

// SVG rendering — same style as ml_trace / inverter_multiparam_opt.
struct LineSeries<'a> { name: &'a str, color: &'a str, values: &'a [f32] }

fn write_rendered_svgs(rows: &[StepRow], dir: &PathBuf) -> Result<(), Box<dyn Error>> {
    let steps: Vec<f32> = rows.iter().map(|r| r.step as f32).collect();
    let loss: Vec<f32> = rows.iter().map(|r| r.loss).collect();
    let v1: Vec<f32> = rows.iter().map(|r| r.vth_1).collect();
    let v2: Vec<f32> = rows.iter().map(|r| r.vth_2).collect();
    let v3: Vec<f32> = rows.iter().map(|r| r.vth_3).collect();
    let vo: Vec<f32> = rows.iter().map(|r| r.vout_at_t_target).collect();
    let g1: Vec<f32> = rows.iter().map(|r| r.g1).collect();
    let g2: Vec<f32> = rows.iter().map(|r| r.g2).collect();
    let g3: Vec<f32> = rows.iter().map(|r| r.g3).collect();

    fs::write(dir.join("loss.svg"), line_chart_svg(
        "Inverter-chain delay loss", "step", "loss", &steps,
        &[LineSeries { name: "loss", color: "#2563eb", values: &loss }]))?;
    fs::write(dir.join("params.svg"), line_chart_svg(
        "Per-stage NMOS Vth_n", "step", "Vth (V)", &steps, &[
            LineSeries { name: "Vth_n1", color: "#0f766e", values: &v1 },
            LineSeries { name: "Vth_n2", color: "#b45309", values: &v2 },
            LineSeries { name: "Vth_n3", color: "#7c3aed", values: &v3 },
        ]))?;
    fs::write(dir.join("output.svg"), line_chart_svg(
        "vout(t_target) tracking", "step", "voltage (V)", &steps,
        &[LineSeries { name: "vout(t*)", color: "#1d4ed8", values: &vo }]))?;
    fs::write(dir.join("grads.svg"), line_chart_svg(
        "Per-parameter gradients", "step", "gradient", &steps, &[
            LineSeries { name: "g1", color: "#0f766e", values: &g1 },
            LineSeries { name: "g2", color: "#b45309", values: &g2 },
            LineSeries { name: "g3", color: "#7c3aed", values: &g3 },
        ]))?;
    Ok(())
}

fn line_chart_svg(title: &str, x_label: &str, y_label: &str, x: &[f32], series: &[LineSeries<'_>]) -> String {
    let width = 920.0_f32; let height = 480.0_f32;
    let left = 78.0_f32; let right = 26.0_f32; let top = 56.0_f32; let bottom = 62.0_f32;
    let plot_w = width - left - right; let plot_h = height - top - bottom;
    let min_x = *x.first().unwrap_or(&0.0); let max_x = *x.last().unwrap_or(&1.0);
    let dx = (max_x - min_x).max(1.0);
    let mut min_y = f32::INFINITY; let mut max_y = f32::NEG_INFINITY;
    for s in series { for &v in s.values { min_y = min_y.min(v); max_y = max_y.max(v); }}
    if !min_y.is_finite() || !max_y.is_finite() { min_y = -1.0; max_y = 1.0; }
    if (max_y - min_y).abs() < 1e-12 { max_y += 1.0; min_y -= 1.0; }
    let y_pad = 0.08 * (max_y - min_y); min_y -= y_pad; max_y += y_pad;
    let dy = (max_y - min_y).max(1e-9);
    let map_x = |v: f32| left + ((v - min_x) / dx) * plot_w;
    let map_y = |v: f32| top + (1.0 - (v - min_y) / dy) * plot_h;
    let mut svg = String::new();
    svg.push_str(&format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        width as i32, height as i32, width as i32, height as i32));
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n");
    for i in 0..=6 {
        let t = i as f32 / 6.0;
        let yv = min_y + t * dy; let py = map_y(yv);
        svg.push_str(&format!("<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>\n", left, py, left + plot_w, py));
        svg.push_str(&format!("<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\" font-size=\"12\" fill=\"#374151\">{:.3e}</text>\n", left - 8.0, py + 4.0, yv));
    }
    for i in 0..=6 {
        let t = i as f32 / 6.0;
        let xv = min_x + t * dx; let px = map_x(xv);
        svg.push_str(&format!("<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>\n", px, top, px, top + plot_h));
        svg.push_str(&format!("<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#374151\">{:.0}</text>\n", px, top + plot_h + 20.0, xv));
    }
    svg.push_str(&format!("<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#111827\" stroke-width=\"1.5\"/>\n", left, top + plot_h, left + plot_w, top + plot_h));
    svg.push_str(&format!("<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#111827\" stroke-width=\"1.5\"/>\n", left, top, left, top + plot_h));
    svg.push_str(&format!("<text x=\"{:.2}\" y=\"24\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#111827\">{}</text>\n", width / 2.0, title));
    svg.push_str(&format!("<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"middle\" font-size=\"13\" fill=\"#111827\">{}</text>\n", left + plot_w / 2.0, height - 16.0, x_label));
    svg.push_str(&format!("<text x=\"22\" y=\"{:.2}\" transform=\"rotate(-90 22 {:.2})\" text-anchor=\"middle\" font-size=\"13\" fill=\"#111827\">{}</text>\n", top + plot_h / 2.0, top + plot_h / 2.0, y_label));
    for s in series {
        let mut pts = String::new();
        for (i, &yv) in s.values.iter().enumerate() {
            let xv = x.get(i).copied().unwrap_or(i as f32);
            pts.push_str(&format!("{:.2},{:.2} ", map_x(xv), map_y(yv)));
        }
        svg.push_str(&format!("<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2.2\"/>\n", pts.trim_end(), s.color));
    }
    let lx = left + plot_w - 170.0; let ly = top + 10.0;
    let lh = 26.0 + series.len() as f32 * 22.0;
    svg.push_str(&format!("<rect x=\"{:.2}\" y=\"{:.2}\" width=\"160\" height=\"{:.2}\" rx=\"8\" fill=\"#f9fafb\" stroke=\"#d1d5db\"/>\n", lx, ly, lh));
    svg.push_str(&format!("<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"12\" font-weight=\"700\" fill=\"#111827\">Legend</text>\n", lx + 10.0, ly + 16.0));
    for (i, s) in series.iter().enumerate() {
        let y = ly + 32.0 + i as f32 * 22.0;
        svg.push_str(&format!("<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"3\"/>\n", lx + 10.0, y, lx + 36.0, y, s.color));
        svg.push_str(&format!("<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"12\" fill=\"#111827\">{}</text>\n", lx + 44.0, y + 4.0, s.name));
    }
    svg.push_str("</svg>\n");
    svg
}

/// Schematic with proper IEEE-style MOSFET symbols (gate stick + dashed
/// channel + S/D stubs, PMOS bubble on gate). Three inverter stages
/// share Vdd / GND rails; each output node sees a 200 fF load cap to
/// ground. Per-stage `Vth_n` is annotated initial → trained.
fn schematic_svg(initial: &StepRow, final_row: &StepRow) -> String {
    let w = 1180.0_f32; let h = 420.0_f32;
    let mut s = String::new();
    s.push_str(&format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        w as i32, h as i32, w as i32, h as i32));
    s.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n");
    s.push_str(&format!("<text x=\"{:.0}\" y=\"24\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#111827\">3-stage CMOS inverter chain — schematic</text>\n", w / 2.0));

    let vdd_y = 60.0; let gnd_y = 360.0;
    s.push_str(&format!("<line x1=\"40\" y1=\"{vdd_y}\" x2=\"{:.1}\" y2=\"{vdd_y}\" stroke=\"#dc2626\" stroke-width=\"2.2\"/>\n", w - 40.0));
    s.push_str(&format!("<text x=\"50\" y=\"{:.0}\" font-size=\"13\" fill=\"#dc2626\" font-weight=\"700\">Vdd = 1 V</text>\n", vdd_y - 8.0));
    s.push_str(&format!("<line x1=\"40\" y1=\"{gnd_y}\" x2=\"{:.1}\" y2=\"{gnd_y}\" stroke=\"#1f2937\" stroke-width=\"2.2\"/>\n", w - 40.0));
    s.push_str(&format!("<text x=\"50\" y=\"{:.0}\" font-size=\"13\" fill=\"#1f2937\" font-weight=\"700\">GND</text>\n", gnd_y + 18.0));

    let stage_w = 320.0; let stage_x0 = 100.0;
    let labels = ["vin", "n1", "n2", "vout"];
    let initial_vths = [initial.vth_1, initial.vth_2, initial.vth_3];
    let final_vths   = [final_row.vth_1, final_row.vth_2, final_row.vth_3];

    for stage in 0..3 {
        let cx = stage_x0 + stage_w * stage as f32 + stage_w * 0.5;
        let in_x = stage_x0 + stage_w * stage as f32;
        let out_x = stage_x0 + stage_w * (stage as f32 + 1.0);

        // Input net at gate-bus column (gate_bus_x). Vertical bus connects
        // PMOS gate (top) and NMOS gate (bottom) and continues from the
        // previous stage's output wire.
        let gate_bus_x = cx - 28.0;
        let p_cy = vdd_y + 80.0;   // PMOS symbol vertical center
        let n_cy = gnd_y - 80.0;   // NMOS symbol vertical center
        let drain_x = cx + 28.0;   // shared drain-net column

        // input wire (from previous stage / vin) coming in horizontally
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n",
            in_x, p_cy, gate_bus_x, p_cy));
        // input net label (above gate-bus join with PMOS gate level)
        s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"13\" fill=\"#1f2937\" font-weight=\"700\">{}</text>\n",
            (in_x + gate_bus_x) / 2.0, p_cy - 8.0, labels[stage]));
        // gate-bus vertical
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n",
            gate_bus_x, p_cy, gate_bus_x, n_cy));

        // ── PMOS symbol at (cx, p_cy) ──
        // Source rail (top of channel) → Vdd
        let ch_top = p_cy - 20.0; let ch_bot = p_cy + 20.0;
        s.push_str(&format!("<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{vdd_y:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", ch_top));
        // Drain rail (bottom of channel) → drain bus
        s.push_str(&format!("<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", ch_bot, ch_bot + 14.0));
        // Channel (dashed vertical bar = enhancement-mode)
        s.push_str(&format!("<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"3\" stroke-dasharray=\"4,3\"/>\n", ch_top, ch_bot));
        // S/D bars
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"2\"/>\n", cx - 8.0, ch_top, cx + 8.0, ch_top));
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"2\"/>\n", cx - 8.0, ch_bot, cx + 8.0, ch_bot));
        // Gate stick + lead + bubble (PMOS)
        let gate_x = cx - 12.0;
        s.push_str(&format!("<line x1=\"{gate_x:.1}\" y1=\"{:.1}\" x2=\"{gate_x:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"2\"/>\n", ch_top, ch_bot));
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{p_cy:.1}\" x2=\"{:.1}\" y2=\"{p_cy:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", gate_bus_x, gate_x - 4.0));
        s.push_str(&format!("<circle cx=\"{:.1}\" cy=\"{p_cy:.1}\" r=\"3\" fill=\"#ffffff\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", gate_x - 4.0));
        // PMOS label
        s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"start\" font-size=\"12\" font-weight=\"700\" fill=\"#7f1d1d\">Mp{}</text>\n",
            cx + 14.0, p_cy + 4.0, stage + 1));
        s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"start\" font-size=\"10\" fill=\"#7f1d1d\">PMOS · 4µ/1µ</text>\n",
            cx + 14.0, p_cy + 20.0));

        // ── NMOS symbol at (cx, n_cy) ──
        let n_top = n_cy - 20.0; let n_bot = n_cy + 20.0;
        // Source (bottom of channel) → GND
        s.push_str(&format!("<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{gnd_y:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", n_bot));
        // Drain (top of channel) → drain bus
        s.push_str(&format!("<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", n_top - 14.0, n_top));
        // Channel
        s.push_str(&format!("<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"3\" stroke-dasharray=\"4,3\"/>\n", n_top, n_bot));
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"2\"/>\n", cx - 8.0, n_top, cx + 8.0, n_top));
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"2\"/>\n", cx - 8.0, n_bot, cx + 8.0, n_bot));
        // Gate (no bubble)
        s.push_str(&format!("<line x1=\"{gate_x:.1}\" y1=\"{:.1}\" x2=\"{gate_x:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"2\"/>\n", n_top, n_bot));
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{n_cy:.1}\" x2=\"{:.1}\" y2=\"{n_cy:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", gate_bus_x, gate_x));
        s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"start\" font-size=\"12\" font-weight=\"700\" fill=\"#1e3a8a\">Mn{}</text>\n",
            cx + 14.0, n_cy + 4.0, stage + 1));
        s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"start\" font-size=\"10\" fill=\"#1e3a8a\">NMOS · 2µ/1µ</text>\n",
            cx + 14.0, n_cy + 20.0));

        // ── Drain interconnect: shared drain net ──
        // PMOS drain stub goes RIGHT from (cx, ch_bot+14) to drain_x
        let pdrain_y = ch_bot + 14.0;
        let ndrain_y = n_top - 14.0;
        s.push_str(&format!("<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{drain_x:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", pdrain_y, pdrain_y));
        s.push_str(&format!("<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{drain_x:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", ndrain_y, ndrain_y));
        s.push_str(&format!("<line x1=\"{drain_x:.1}\" y1=\"{:.1}\" x2=\"{drain_x:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", pdrain_y, ndrain_y));

        // ── Output net: tap mid-rail, route to next stage's input ──
        let mid_y = (p_cy + n_cy) / 2.0;
        s.push_str(&format!("<line x1=\"{drain_x:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n",
            mid_y, out_x, mid_y));
        // Vertical wire from mid_y to gate-bus level (so next stage sees it at p_cy)
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n",
            out_x, mid_y, out_x, p_cy));
        s.push_str(&format!("<circle cx=\"{drain_x:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"#374151\"/>\n", mid_y));

        // ── Load cap on output: between mid_y and GND ──
        let cap_x = drain_x + 60.0;
        s.push_str(&format!("<line x1=\"{drain_x:.1}\" y1=\"{:.1}\" x2=\"{cap_x:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", mid_y, mid_y));
        s.push_str(&format!("<line x1=\"{cap_x:.1}\" y1=\"{:.1}\" x2=\"{cap_x:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", mid_y, mid_y + 22.0));
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"2.4\"/>\n", cap_x - 14.0, mid_y + 22.0, cap_x + 14.0, mid_y + 22.0));
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#374151\" stroke-width=\"2.4\"/>\n", cap_x - 14.0, mid_y + 30.0, cap_x + 14.0, mid_y + 30.0));
        s.push_str(&format!("<line x1=\"{cap_x:.1}\" y1=\"{:.1}\" x2=\"{cap_x:.1}\" y2=\"{gnd_y:.1}\" stroke=\"#374151\" stroke-width=\"1.6\"/>\n", mid_y + 30.0));
        s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"start\" font-size=\"11\" fill=\"#1f2937\">200 fF</text>\n",
            cap_x + 22.0, mid_y + 28.0));

        // Per-stage Vth annotation underneath everything
        s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"12\" font-weight=\"700\" fill=\"#1e3a8a\">Vth_n{}: {:.3} V → {:.3} V</text>\n",
            cx, gnd_y + 36.0, stage + 1, initial_vths[stage], final_vths[stage]));
    }
    // Final output label at the rightmost column
    let cx_last = stage_x0 + stage_w * 3.0 + 40.0;
    let mid_y = ((vdd_y + 80.0) + (gnd_y - 80.0)) / 2.0;
    s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"start\" font-size=\"13\" fill=\"#1f2937\" font-weight=\"700\">vout</text>\n",
        cx_last, mid_y + 4.0));

    s.push_str("</svg>\n");
    s
}

/// PDK-driven floorplan: builds a Sky130 `Library`, places 6 mosfet
/// cells (3 PMOS top row, 3 NMOS bottom row), and renders the layout
/// to SVG via `eda_viz::layout::render_to_svg`. Layers + colors come
/// from the foundry `.lyp` so this matches the real Sky130 stack.
///
/// Falls back to a "PDK unavailable" placeholder SVG when the
/// `HAS_SKY130` flag is false (foundry `.lyp` not checked out).
///
/// **Floorplan-rendering rule for this repo: never ship a stylized
/// hand-drawn floorplan in a demo report. Always pick a target PDK
/// (Sky130 / Gf180mcu / etc.) so dimensions, layer colors, and
/// implant/well overhangs match a real foundry stack.**
fn build_floorplan_and_run_checks(assets: &PathBuf)
    -> Result<(DrcSummary, LvsSummary), Box<dyn Error>>
{
    if !HAS_SKY130 {
        let svg = placeholder_svg("Sky130 floorplan unavailable",
            "Foundry .lyp not checked out at build time (HAS_SKY130 = false). \
             Build with the sky130 lyp present to get a real PDK-driven floorplan.");
        fs::write(assets.join("floorplan.svg"), svg)?;
        return Ok((DrcSummary::skipped(), LvsSummary::skipped()));
    }
    let (lib, top_id, pdk) = build_floorplan_layout();
    let style = eda_viz::Style { units_per_dbu: 0.022, ..Default::default() };
    let svg = eda_viz::layout::render_to_svg(&lib, top_id, &style);
    fs::write(assets.join("floorplan.svg"), &svg)?;
    // Many markdown renderers don't display inline SVG (or strip
    // <?xml> headers). Emit a 3× PNG fallback so the floorplan
    // shows up regardless of viewer.
    if let Ok(png_bytes) = eda_viz::png::svg_to_png(&svg, 3.0) {
        fs::write(assets.join("floorplan.png"), png_bytes)?;
    }
    let drc = run_sky130_drc(&lib, top_id, &pdk);
    let lvs = run_lvs_check(&lib, top_id, &pdk);
    Ok((drc, lvs))
}

fn build_floorplan_layout() -> (Library, CellId, Sky130) {
    let lib = Sky130::new_library("inv_chain_floorplan");
    let pdk = Sky130::register(&lib);

    // Per-cell layouts.
    let nmos = Mosfet::nmos(2_000, 1_000, "Mn_chain");
    let pmos = Mosfet::pmos(4_000, 1_000, "Mp_chain");
    let nmos_id = nmos.layout(&lib, &pdk);
    let pmos_id = pmos.layout(&lib, &pdk);

    // Placement: NMOS row at y=0, PMOS row at y=Y_PMOS. Stage pitch
    // chosen so cells (incl. nwell margin) don't overlap and give a
    // routing channel between rows.
    const STAGE_PITCH: i64 = 10_000;  // 10 µm pitch (DBU = nm)
    const Y_PMOS: i64      = 14_000;  // 14 µm above NMOS row baseline

    // Mosfet port coords (mirrors the constants in spike-divider-block::Mosfet::layout):
    // local s @ (   750, w/2)   d @ (3250, w/2)   g @ (2000, w + POLY_OVERHANG/2)
    // POLY_OVERHANG = 1500, so the poly stick extends to y = w + 1500.
    const S_LX: i64 = 750;
    const D_LX: i64 = 3_250;
    const G_LX: i64 = 2_000;
    let n_s_y: i64 = 1_000;            // w/2 for NMOS w=2000
    let n_d_y: i64 = 1_000;
    let _n_g_y: i64 = 2_000 + 750;     // w + POLY_OVERHANG/2 = 2750 (informational)
    let p_s_y: i64 = Y_PMOS + 2_000;   // PMOS w=4000 → cy_diff=2000
    let p_d_y: i64 = Y_PMOS + 2_000;
    let _p_g_y: i64 = Y_PMOS + 4_000 + 750;  // PMOS gate y

    // Vdd above PMOS nwell margin (Y_PMOS + W_pmos + NWELL_MARGIN).
    const VDD_Y: i64 = 22_000;
    // GND below NMOS body tap (NMOS body tap at y = -2500).
    const GND_Y: i64 = -5_000;
    let m1 = pdk.metal1();
    const RAIL_W: i64 = 1_400;  // 1.4 µm power rail
    const SIG_W:  i64 =   600;  // 0.6 µm signal wire
    let chain_x_min: i64 = -2_000;
    let chain_x_max: i64 = 3 * STAGE_PITCH + 1_500;
    // Mid-channel y for inter-stage signal routing + gate-bus M1↔poly tap.
    const ROUTING_Y: i64 = 7_000;

    let mut top = CellBuilder::new("inv_chain_floorplan_top");

    // Place all 6 transistors.
    for stage in 0..3 {
        let x = stage as i64 * STAGE_PITCH;
        top.add_instance(Instance::new(nmos_id, Trans::translate(Vec2::new(x, 0))));
        top.add_instance(Instance::new(pmos_id, Trans::translate(Vec2::new(x, Y_PMOS))));
    }

    // Vdd / GND rails on metal1, spanning the full chain width.
    add_h_rail(&mut top, m1, chain_x_min, chain_x_max, VDD_Y, RAIL_W);
    add_h_rail(&mut top, m1, chain_x_min, chain_x_max, GND_Y, RAIL_W);

    // Per-stage routing.
    for stage in 0..3 {
        let x = stage as i64 * STAGE_PITCH;
        let s_x = x + S_LX; let d_x = x + D_LX; let g_x = x + G_LX;

        // PMOS source up to Vdd rail.
        add_v_wire(&mut top, m1, s_x, p_s_y, VDD_Y, SIG_W);
        // NMOS source down to GND rail.
        add_v_wire(&mut top, m1, s_x, GND_Y, n_s_y, SIG_W);
        // NMOS body-tap (south at y = -2500) → GND. Body-tap pad is
        // BELOW the NMOS device, so this routing column doesn't
        // intersect any other M1 net at this stage.
        add_v_wire(&mut top, m1, x + 2_000, GND_Y, -2_500, SIG_W);

        // PMOS body-tap → Vdd via a jog over the drain pads.
        // Body-tap pad sits at (x+1250..x+2750, y≈10750..12250) and
        // shares the drain column on its way up. Route:
        //   1) vertical M1 from body-tap pad up to y = 17_200 (the
        //      narrow gap between drain pad top y=16750 and gate pad
        //      bottom y=17750)
        //   2) horizontal M1 across to a clear column (x + 4_500)
        //      that's east of the drain bus (x+2950..3550)
        //   3) vertical M1 up to Vdd rail.
        let p_btap_top: i64 = Y_PMOS - 1_750;     // 12_250
        let jog_y:      i64 = Y_PMOS + 3_200;     // 17_200
        let jog_x:      i64 = x + 4_500;
        add_v_wire(&mut top, m1, x + 2_000, p_btap_top, jog_y, SIG_W);
        add_h_wire(&mut top, m1, x + 2_000, jog_x,      jog_y, SIG_W);
        add_v_wire(&mut top, m1, jog_x,     jog_y,      VDD_Y, SIG_W);

        // Gate bus is on POLY (RES layer), not M1 — frees the M1 gate
        // column from any conflict with the body-tap pad. Each Mosfet's
        // poly stick already extends past the diff (NMOS to y≈3000,
        // PMOS down to y≈13000); we fill the y=3000..13000 gap.
        //
        // BUT: the PMOS body-tap VIA1 at (x+2000, y+11500) sits inside
        // the gate column, and its shape footprint (1750..2250 ×
        // 11250..11750) would join the M1 body-tap pad to the poly
        // gate bus — a fatal short. We detour the bridge east, around
        // the body-tap pad, with a 4-piece U:
        //
        //          poly stick   ───┐
        //   PMOS  ──────────── (top jog) ────┐
        //                           │        │
        //   body-tap M1 pad         │        │   (east jog vertical)
        //                           │        │
        //   NMOS  ──────────── (bot jog) ────┘
        //          poly stick   ───┘
        //
        // Each detour piece clears the body-tap pad / VIA1 footprint.
        // Bridge boundaries:
        //   NMOS poly stick top = w_n + POLY_OVERHANG = 2000 + 1500 = 3500
        //   PMOS poly stick bot = Y_PMOS - POLY_OVERHANG = 14000 - 1500 = 12500
        // Body-tap M1 pad spans y = 10750..12250 — bridge has to detour east of
        // x=2750 (pad east edge) for any y in that range.
        let n_poly_top:      i64 = 2_000 + 1_500;        // 3_500
        let p_poly_bot:      i64 = Y_PMOS - 1_500;       // 12_500
        let p_btap_clear_lo: i64 = 10_500;               // just below pad bot 10_750
        let p_btap_clear_hi: i64 = p_poly_bot;           // 12_500 — bridge top edge
        // bottom vertical (gate column, NMOS-poly top up to bot-jog level)
        top.add_shape(pdk.poly(), Rect::new(Bbox::new(
            Point::new(x + 1_500, n_poly_top),       Point::new(x + 2_500, p_btap_clear_lo),
        )));
        // bottom horizontal jog (east past body-tap pad east edge x=2750)
        top.add_shape(pdk.poly(), Rect::new(Bbox::new(
            Point::new(x + 1_500, p_btap_clear_lo),  Point::new(x + 3_500, p_btap_clear_lo + 500),
        )));
        // east vertical (clear of body-tap pad)
        top.add_shape(pdk.poly(), Rect::new(Bbox::new(
            Point::new(x + 3_000, p_btap_clear_lo + 500), Point::new(x + 3_500, p_btap_clear_hi),
        )));
        // top horizontal jog (back west to gate column, joins PMOS poly stick)
        top.add_shape(pdk.poly(), Rect::new(Bbox::new(
            Point::new(x + 1_500, p_btap_clear_hi - 500), Point::new(x + 3_500, p_btap_clear_hi),
        )));

        // VIA1 (M1↔poly) at (g_x, ROUTING_Y) lets the inter-stage M1
        // signal drive the poly gate bus. The inter-stage M1 already
        // covers this point (added below).
        const VIA_SZ: i64 = 500;
        top.add_shape(pdk.via1(), Rect::new(Bbox::new(
            Point::new(g_x - VIA_SZ / 2, ROUTING_Y - VIA_SZ / 2),
            Point::new(g_x + VIA_SZ / 2, ROUTING_Y + VIA_SZ / 2),
        )));
        // Small M1 landing pad around the via to satisfy enclosure.
        const PAD: i64 = 700;
        top.add_shape(m1, Rect::new(Bbox::new(
            Point::new(g_x - PAD / 2, ROUTING_Y - PAD / 2),
            Point::new(g_x + PAD / 2, ROUTING_Y + PAD / 2),
        )));

        // Drain bus: vertical M1 from NMOS drain up to PMOS drain.
        // This is the stage's output net.
        add_v_wire(&mut top, m1, d_x, n_d_y, p_d_y, SIG_W);
    }

    // Inter-stage signal wires: drain bus (x = stage*PITCH + D_LX) routes
    // horizontally in the channel between PMOS and NMOS rows over to the
    // next stage's gate bus (x = (stage+1)*PITCH + G_LX) at ROUTING_Y.
    for stage in 0..2 {
        let x_out  = stage as i64 * STAGE_PITCH + D_LX;
        let x_gate = (stage as i64 + 1) * STAGE_PITCH + G_LX;
        // Vertical drop from drain bus into routing channel (already on
        // bus from n_d_y up to p_d_y, but we need a tap into ROUTING_Y;
        // ROUTING_Y is between n_d_y=1000 and p_d_y=Y_PMOS+2000=16000,
        // so it's already covered by the existing drain bus).
        // Horizontal wire across the channel.
        add_h_wire(&mut top, m1, x_out, x_gate, ROUTING_Y, SIG_W);
        // Vertical drop into next stage's gate bus (ROUTING_Y is between
        // n_g_y=2500 and p_g_y=Y_PMOS+4500=18500, so it's also already
        // covered by next stage's existing gate bus).
    }

    // External: vin pad on the left (drives stage-0 gate).
    add_h_wire(&mut top, m1, chain_x_min, G_LX, ROUTING_Y, SIG_W);
    // External: vout pad on the right (driven by stage-2 drain).
    let stage2_d_x = 2 * STAGE_PITCH + D_LX;
    add_h_wire(&mut top, m1, stage2_d_x, chain_x_max, ROUTING_Y, SIG_W);

    let top_id = lib.insert(top);
    (lib, top_id, pdk)
}

/// Per-rule violation count from a single DRC pass on the floorplan.
#[derive(Debug, Clone)]
struct DrcRuleResult {
    name: &'static str,
    min_dbu: i64,
    violations: usize,
}

#[derive(Debug, Clone)]
struct DrcSummary {
    rules: Vec<DrcRuleResult>,
    skipped: bool,
}

impl DrcSummary {
    fn skipped() -> Self { Self { rules: vec![], skipped: true } }
    fn total(&self) -> usize { self.rules.iter().map(|r| r.violations).sum() }
}

/// Run a representative slice of Sky130 DRC rules against the floorplan.
/// Real foundry decks have hundreds of rules; this hits the high-value
/// ones for hand-routed analog floorplans: M1 width/space, POLY width/
/// space, DIFF width/space, NWELL width/space, NWELL→PMOS-DIFF
/// enclosure, M1→VIA1 enclosure. Numbers come from the published
/// SkyWater Sky130 process node (sky130A) min rules.
fn run_sky130_drc(lib: &Library, top: CellId, pdk: &Sky130) -> DrcSummary {
    let m1    = Region::from_cell_layer(lib, top, pdk.METAL1);
    let poly  = Region::from_cell_layer(lib, top, pdk.RES);
    let diff  = Region::from_cell_layer(lib, top, pdk.DIFF);
    let nwell = Region::from_cell_layer(lib, top, pdk.NWELL);
    let pplus = Region::from_cell_layer(lib, top, pdk.PPLUS);
    let via1  = Region::from_cell_layer(lib, top, pdk.VIA1);

    let mut rules = Vec::new();
    let mut check = |name: &'static str, min_dbu: i64, region: klayout_geom::Region| {
        rules.push(DrcRuleResult { name, min_dbu, violations: region.polygons().len() });
    };

    // Sky130A min rules (selected — DBU = nm).
    check("M1 min width   (≥ 140 nm)", 140, klayout_drc::width(&m1, 140));
    check("M1 min space   (≥ 140 nm)", 140, klayout_drc::space(&m1, 140));
    check("POLY min width (≥ 150 nm)", 150, klayout_drc::width(&poly, 150));
    check("POLY min space (≥ 210 nm)", 210, klayout_drc::space(&poly, 210));
    check("DIFF min width (≥ 150 nm)", 150, klayout_drc::width(&diff, 150));
    check("DIFF min space (≥ 270 nm)", 270, klayout_drc::space(&diff, 270));
    check("NWELL min width (≥ 840 nm)", 840, klayout_drc::width(&nwell, 840));
    check("NWELL min space (≥ 1270 nm)", 1270, klayout_drc::space(&nwell, 1270));
    check("VIA1 min width  (≥ 170 nm)", 170, klayout_drc::width(&via1, 170));
    check("VIA1 min space  (≥ 170 nm)", 170, klayout_drc::space(&via1, 170));
    // Cross-layer enclosures: NWELL must enclose all PMOS DIFF (pplus
    // marks PMOS) by ≥ 180 nm, M1 must enclose every VIA1 by ≥ 30 nm.
    let pmos_diff = klayout_geom::boolean::intersection(&diff, &pplus);
    check("NWELL enc PMOS-DIFF (≥ 180 nm)", 180,
        klayout_drc::enclosing(&nwell, &pmos_diff, 180));
    check("M1 enc VIA1 (≥ 30 nm)", 30,
        klayout_drc::enclosing(&m1, &via1, 30));

    DrcSummary { rules, skipped: false }
}

#[derive(Debug, Clone)]
struct LvsExpectation {
    name: &'static str,
    probe: Point,        // a point we expect to land in this net
}

#[derive(Debug, Clone)]
struct LvsNetCheck {
    expected: LvsExpectation,
    matched_net_idx: Option<usize>,
}

#[derive(Debug, Clone)]
struct LvsSummary {
    extracted_net_count: usize,
    expected_net_count: usize,
    nets: Vec<LvsNetCheck>,
    skipped: bool,
}

impl LvsSummary {
    fn skipped() -> Self { Self {
        extracted_net_count: 0, expected_net_count: 0, nets: vec![], skipped: true,
    }}
    fn pass(&self) -> bool {
        if self.skipped { return false; }
        self.extracted_net_count == self.expected_net_count
            && self.nets.iter().all(|n| n.matched_net_idx.is_some())
            && self.distinct_count() == self.nets.len()
    }
    fn distinct_count(&self) -> usize {
        let mut s = std::collections::BTreeSet::new();
        for n in &self.nets {
            if let Some(i) = n.matched_net_idx { s.insert(i); }
        }
        s.len()
    }
}

/// Extract connectivity from the floorplan and check against the
/// expected schematic net structure. M1 + POLY are the conductors;
/// VIA1 joins them where it lands on poly. The expected net set
/// is { vin, n1, n2, vout, Vdd, GND } — six distinct nets, with one
/// known probe point per net (a coordinate inside that net's wire).
fn run_lvs_check(lib: &Library, top: CellId, pdk: &Sky130) -> LvsSummary {
    const STAGE_PITCH: i64 = 10_000;
    const Y_PMOS: i64 = 14_000;
    const G_LX: i64 = 2_000;
    const D_LX: i64 = 3_250;
    const ROUTING_Y: i64 = 7_000;
    const VDD_Y: i64 = 22_000;
    const GND_Y: i64 = -5_000;

    let cfg = ExtractConfig {
        conductors: vec![
            Conductor { layer: pdk.METAL1, label_layer: pdk.METAL1 },
            Conductor { layer: pdk.RES,    label_layer: pdk.RES },
        ],
        // VIA1 is licon1 in sky130 — connects M1 to whatever's beneath.
        // Our gate-bus M1↔poly hand-off uses VIA1 over poly.
        vias: vec![
            Via { layer: pdk.VIA1, a: pdk.METAL1, b: pdk.RES },
        ],
    };
    let nl = extract_hierarchical(lib, top, &cfg);

    // One probe point per expected net. Each probe sits well inside a
    // known wire AND outside all other nets' bboxes — so the
    // bbox-containment check below is unambiguous.
    let expected = vec![
        LvsExpectation { name: "vin",  probe: Point::new(-1_500, ROUTING_Y) },                       // left external M1
        LvsExpectation { name: "n1",   probe: Point::new(STAGE_PITCH / 2,         8_000) },          // mid stage 0→1
        LvsExpectation { name: "n2",   probe: Point::new(STAGE_PITCH + STAGE_PITCH / 2,  8_000) },   // mid stage 1→2
        LvsExpectation { name: "vout", probe: Point::new(2 * STAGE_PITCH + D_LX + 4_000, ROUTING_Y) },// right external M1
        LvsExpectation { name: "Vdd",  probe: Point::new(STAGE_PITCH + G_LX, VDD_Y) },               // mid Vdd rail
        LvsExpectation { name: "GND",  probe: Point::new(STAGE_PITCH + G_LX, GND_Y) },               // mid GND rail
    ];
    let _ = (Y_PMOS, D_LX, G_LX); // referenced via probes

    let nets = expected.into_iter().map(|exp| {
        let matched = nl.nets().iter().enumerate().find_map(|(i, n)| {
            // Bbox-containment is a coarse net match; sufficient because
            // each expected probe point sits well inside one wire.
            if n.bbox.min.x <= exp.probe.x && exp.probe.x <= n.bbox.max.x
                && n.bbox.min.y <= exp.probe.y && exp.probe.y <= n.bbox.max.y
            {
                Some(i)
            } else { None }
        });
        LvsNetCheck { expected: exp, matched_net_idx: matched }
    }).collect::<Vec<_>>();

    LvsSummary {
        extracted_net_count: nl.nets().len(),
        expected_net_count: 6,
        nets,
        skipped: false,
    }
}

fn add_h_rail(top: &mut CellBuilder, layer: LayerIndex, x0: i64, x1: i64, y: i64, w: i64) {
    top.add_shape(layer, Rect::new(Bbox::new(
        Point::new(x0, y - w / 2), Point::new(x1, y + w / 2),
    )));
}

fn add_h_wire(top: &mut CellBuilder, layer: LayerIndex, x0: i64, x1: i64, y: i64, w: i64) {
    let (x0, x1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    top.add_shape(layer, Rect::new(Bbox::new(
        Point::new(x0, y - w / 2), Point::new(x1, y + w / 2),
    )));
}

fn add_v_wire(top: &mut CellBuilder, layer: LayerIndex, x: i64, y0: i64, y1: i64, w: i64) {
    let (y0, y1) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
    top.add_shape(layer, Rect::new(Bbox::new(
        Point::new(x - w / 2, y0), Point::new(x + w / 2, y1),
    )));
}

fn placeholder_svg(title: &str, body: &str) -> String {
    let w = 1000.0_f32; let h = 220.0_f32;
    let mut s = String::new();
    s.push_str(&format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        w as i32, h as i32, w as i32, h as i32));
    s.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#fef3c7\" stroke=\"#b45309\" stroke-width=\"2\"/>\n");
    s.push_str(&format!("<text x=\"{:.0}\" y=\"60\" text-anchor=\"middle\" font-size=\"22\" font-weight=\"700\" fill=\"#92400e\">{}</text>\n", w / 2.0, title));
    s.push_str(&format!("<text x=\"{:.0}\" y=\"110\" text-anchor=\"middle\" font-size=\"14\" fill=\"#78350f\">{}</text>\n", w / 2.0, body));
    s.push_str("</svg>\n");
    s
}
