//! Multi-parameter Adam optimization of a CMOS inverter.
//!
//! Scales the single-parameter `inverter_vm_opt` example up to **two
//! coupled parameters**: NMOS and PMOS thresholds (`Vth_n`, `|Vth_p|`),
//! optimized via Adam against a regularized loss:
//!
//! $$L = (V_m - V_m^*)^2 + \lambda \cdot (V_{th,n} - |V_{th,p}|)^2$$
//!
//! The first term targets the switching threshold; the second is a
//! standard analog-design regularization (matched-threshold inverters
//! have symmetric noise margins). Without it, the (Vth_n, Vth_p) →
//! V_m mapping is underdetermined — many threshold pairs give the same
//! V_m. The regularization picks the matched-threshold corner.
//!
//! Demonstrates:
//! - Multi-parameter sensitivities through `eda-mna::sensitivities`
//! - Adam optimizer on top of those gradients
//! - Composite loss (data term + regularization term, both differentiable)
//!
//! Same DC framework as `inverter_vm_opt` — no transient, no SPICE.
//!
//! Run:
//!   cargo run -p spike-divider-block --bin inverter_multiparam_opt

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use eda_mna::{sensitivities, solve_dc, Circuit, NetId, NewtonOptions};
use spike_divider_block::{Adam, Mosfet, Optimizer};

#[derive(Clone, Copy, Debug)]
struct StepRow {
    step: usize,
    vth_n: f32,
    /// PMOS threshold magnitude. (The Mosfet model stores Vth as the
    /// POSITIVE magnitude even for PMOS — its `currents()` impl applies
    /// the polarity sign internally via `sign_c`. So both Vth_n and
    /// Vth_p are positive numbers in `[0, 1]`-ish range here.)
    vth_p_abs: f32,
    vm: f32,
    /// Total loss = data + reg.
    loss: f32,
    /// Data-term loss alone — `(V_m − V_m*)^2`.
    loss_data: f32,
    /// Regularization term alone — `λ·(Vth_n − |Vth_p|)^2`.
    loss_reg: f32,
    /// Gradients of total loss WRT each parameter.
    dloss_dvth_n: f32,
    dloss_dvth_p: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let vdd = 1.0_f32;
    let vm_target = 0.62_f32; // shift more aggressively from the initial 0.525 V
    // Keep the regularizer present but mild — λ=0.5 lets the data term
    // dominate while the reg still biases toward matched thresholds
    // when there's flexibility. Larger λ would over-constrain.
    let lambda_reg = 0.5_f32;
    let lr = 0.04_f32;
    let max_iters = 200_usize;
    let tol = 5e-5_f32;

    // Inverter as self-consistent DC: Vin shorted to Vout = Vm.
    let mut circuit = Circuit::new();
    let v_dd = circuit.alloc_boundary_net();
    let vmid = circuit.alloc_unknown_net();
    let nmos = Mosfet::nmos(2_000, 1_000, "Mn");
    let pmos = Mosfet::pmos(4_000, 1_000, "Mp");
    circuit.add_device(nmos.clone(), &[vmid, vmid, NetId::GND, NetId::GND]);
    circuit.add_device(pmos.clone(), &[vmid, vmid, v_dd, v_dd]);

    let n_name = eda_hir::Block::name(&nmos);
    let p_name = eda_hir::Block::name(&pmos);
    let vth_n_param = format!("{n_name}_Vth");
    let vth_p_param = format!("{p_name}_Vth");

    let mut params: HashMap<String, f32> = nmos.default_params();
    params.extend(pmos.default_params());

    let mut boundary = HashMap::new();
    boundary.insert(v_dd, vdd);

    let solver = NewtonOptions::default();
    let target_param_names = vec![vth_n_param.clone(), vth_p_param.clone()];

    let mut opt = Adam::new(lr, 2);
    let mut p_vec: [f32; 2] = [
        *params.get(&vth_n_param).unwrap(),
        *params.get(&vth_p_param).unwrap(),
    ];

    let mut rows: Vec<StepRow> = Vec::with_capacity(max_iters + 1);
    for step in 0..=max_iters {
        params.insert(vth_n_param.clone(), p_vec[0]);
        params.insert(vth_p_param.clone(), p_vec[1]);

        // Forward: solve DC, read Vm, compute losses.
        let op = solve_dc(&circuit, &params, &boundary, solver);
        let vm = op.voltages.get(&vmid).copied().unwrap_or(0.0);
        let err = vm - vm_target;
        // Both Vth values stored as POSITIVE magnitudes in the Mosfet
        // model — see StepRow doc. So mismatch is just a subtraction.
        let mismatch = p_vec[0] - p_vec[1];
        let loss_data = err * err;
        let loss_reg = lambda_reg * mismatch * mismatch;
        let loss = loss_data + loss_reg;

        // Sensitivities: ∂Vm/∂Vth_n, ∂Vm/∂Vth_p.
        let sens = sensitivities(&circuit, &params, &boundary, &op, &target_param_names);
        let target_idx = 0; // vmid is the only unknown net
        let dvm_dvthn = sens.get(&vth_n_param).map(|v| v[target_idx]).unwrap_or(0.0);
        let dvm_dvthp = sens.get(&vth_p_param).map(|v| v[target_idx]).unwrap_or(0.0);

        // Compose total-loss gradients.
        // ∂L/∂Vth_n = 2·err·∂Vm/∂Vth_n + 2·λ·mismatch
        // ∂L/∂Vth_p = 2·err·∂Vm/∂Vth_p − 2·λ·mismatch
        let dl_dvthn = 2.0 * err * dvm_dvthn + 2.0 * lambda_reg * mismatch;
        let dl_dvthp = 2.0 * err * dvm_dvthp - 2.0 * lambda_reg * mismatch;

        rows.push(StepRow {
            step,
            vth_n: p_vec[0],
            vth_p_abs: p_vec[1],
            vm,
            loss,
            loss_data,
            loss_reg,
            dloss_dvth_n: dl_dvthn,
            dloss_dvth_p: dl_dvthp,
        });

        if loss < tol { break; }

        let g = [dl_dvthn, dl_dvthp];
        opt.step(&mut p_vec, &g);
        // Both Vth params are positive magnitudes; clamp to [0.05, 0.95].
        p_vec[0] = p_vec[0].clamp(0.05, 0.95);
        p_vec[1] = p_vec[1].clamp(0.05, 0.95);
    }

    let final_row = *rows.last().expect("rows non-empty");
    println!("Multi-param differentiable inverter optimization (Adam)");
    println!(
        "  vdd = {vdd:.3} V, target Vm = {vm_target:.3} V, λ_reg = {lambda_reg}, lr = {lr}"
    );
    println!(
        "  initial: Vth_n = {:.4} V, |Vth_p| = {:.4} V, Vm = {:.6}, loss = {:.4e}",
        rows[0].vth_n, rows[0].vth_p_abs, rows[0].vm, rows[0].loss
    );
    println!(
        "  final:   Vth_n = {:.4} V, |Vth_p| = {:.4} V, Vm = {:.6}, loss = {:.4e}, steps = {}",
        final_row.vth_n, final_row.vth_p_abs, final_row.vm, final_row.loss, final_row.step
    );
    println!(
        "  data loss = {:.4e}, reg loss = {:.4e}, threshold mismatch = {:.4} V",
        final_row.loss_data,
        final_row.loss_reg,
        (final_row.vth_n - final_row.vth_p_abs).abs()
    );

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let assets = crate_dir.join("docs/assets/inverter_multiparam_opt");
    fs::create_dir_all(&assets)?;
    write_rendered_svgs(&rows, vm_target, &assets)?;

    let csv = "/tmp/rlx_eda_inverter_multiparam_opt.csv";
    fs::write(csv, build_csv(&rows))?;
    let md_path = crate_dir.join("docs/inverter_multiparam_opt_trace.md");
    let md = build_report(&rows, vdd, vm_target, lambda_reg, lr);
    fs::write(&md_path, &md)?;
    println!();
    println!("wrote CSV report : {csv}");
    println!("wrote MD report  : {}", md_path.display());
    println!("wrote SVG charts : {}/", assets.display());

    let workspace_docs = crate_dir.join("../../docs");
    if workspace_docs.is_dir() {
        let workspace_md = workspace_docs.join("inverter_multiparam_opt_trace.md");
        let workspace_assets = workspace_docs.join("assets/inverter_multiparam_opt");
        fs::create_dir_all(&workspace_assets)?;
        for name in ["loss.svg", "params.svg", "output.svg", "grads.svg"] {
            fs::copy(assets.join(name), workspace_assets.join(name))?;
        }
        fs::write(&workspace_md, &md)?;
        println!("mirrored to      : {}", workspace_md.display());
    }
    Ok(())
}

fn build_csv(rows: &[StepRow]) -> String {
    let mut s = String::from(
        "step,vth_n,vth_p_abs,vm,loss,loss_data,loss_reg,dl_dvth_n,dl_dvth_p\n"
    );
    for r in rows {
        s.push_str(&format!(
            "{},{:.6},{:.6},{:.8},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
            r.step, r.vth_n, r.vth_p_abs, r.vm, r.loss, r.loss_data, r.loss_reg,
            r.dloss_dvth_n, r.dloss_dvth_p,
        ));
    }
    s
}

fn build_report(rows: &[StepRow], vdd: f32, vm_target: f32, lambda: f32, lr: f32) -> String {
    let first = rows.first().unwrap();
    let last = rows.last().unwrap();
    let xs = rows.iter().map(|r| r.step.to_string()).collect::<Vec<_>>().join(", ");
    let y_loss = rows.iter().map(|r| format!("{:.6}", r.loss)).collect::<Vec<_>>().join(", ");
    let y_data = rows.iter().map(|r| format!("{:.6}", r.loss_data)).collect::<Vec<_>>().join(", ");
    let y_reg  = rows.iter().map(|r| format!("{:.6}", r.loss_reg)).collect::<Vec<_>>().join(", ");
    let y_vthn = rows.iter().map(|r| format!("{:.4}", r.vth_n)).collect::<Vec<_>>().join(", ");
    let y_vthp = rows.iter().map(|r| format!("{:.4}", r.vth_p_abs)).collect::<Vec<_>>().join(", ");
    let y_vm = rows.iter().map(|r| format!("{:.4}", r.vm)).collect::<Vec<_>>().join(", ");
    let y_err = rows.iter().map(|r| format!("{:.4}", r.vm - vm_target)).collect::<Vec<_>>().join(", ");
    let y_gn = rows.iter().map(|r| format!("{:.6}", r.dloss_dvth_n)).collect::<Vec<_>>().join(", ");
    let y_gp = rows.iter().map(|r| format!("{:.6}", r.dloss_dvth_p)).collect::<Vec<_>>().join(", ");

    let mut md = String::new();
    md.push_str("# rlx-eda multi-parameter CMOS inverter optimization (Adam)\n\n");
    md.push_str("Circuit: CMOS inverter (NMOS + PMOS via `spike_divider_block::Mosfet`), Vin shorted to Vout. ");
    md.push_str("**Two parameters** under simultaneous gradient descent:\n");
    md.push_str("- `Vth_n` — NMOS threshold voltage\n");
    md.push_str("- `Vth_p` — PMOS threshold voltage (negative; we log magnitude)\n\n");
    md.push_str(&format!(
        "Stimulus: `Vdd = {vdd:.3} V`, `V_m* = {vm_target:.3} V`, `λ_reg = {lambda}`, Adam `lr = {lr}`\n\n"
    ));
    md.push_str("Composite loss:\n\n");
    md.push_str("$$L = (V_m - V_m^*)^2 + \\lambda \\cdot (V_{th,n} - |V_{th,p}|)^2$$\n\n");
    md.push_str("First term targets the switching threshold; second is a matched-threshold regularizer (a standard analog design rule for symmetric noise margins). Without it, the (Vth_n, Vth_p) → V_m mapping is underdetermined.\n\n");
    md.push_str("Both gradients flow through the rlx graph that stamps the LEVEL-1 MOSFET equations into the MNA residual; per-parameter gradients arrive via `eda_mna::sensitivities` (reverse-mode AD).\n\n");

    md.push_str("## Optimization outcome\n\n");
    md.push_str(&format!(
        "- initial: `Vth_n = {:.4} V`, `|Vth_p| = {:.4} V`, `Vm = {:.6}`, `loss = {:.4e}`\n",
        first.vth_n, first.vth_p_abs, first.vm, first.loss,
    ));
    md.push_str(&format!(
        "- final:   `Vth_n = {:.4} V`, `|Vth_p| = {:.4} V`, `Vm = {:.6}`, `loss = {:.4e}`, `steps = {}`\n",
        last.vth_n, last.vth_p_abs, last.vm, last.loss, last.step,
    ));
    md.push_str(&format!(
        "- data-term loss: `{:.4e}`, regularizer loss: `{:.4e}`, threshold mismatch: `{:.4} V`\n\n",
        last.loss_data, last.loss_reg, (last.vth_n - last.vth_p_abs).abs(),
    ));

    md.push_str("## Rendered charts\n\n");
    md.push_str("| Loss components | Parameter trajectories |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![Rendered loss chart](crates/spike-divider-block/docs/assets/inverter_multiparam_opt/loss.svg) | ![Rendered parameter chart](crates/spike-divider-block/docs/assets/inverter_multiparam_opt/params.svg) |\n\n");
    md.push_str("| Output and error | Per-parameter gradients |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![Rendered output chart](crates/spike-divider-block/docs/assets/inverter_multiparam_opt/output.svg) | ![Rendered gradient chart](crates/spike-divider-block/docs/assets/inverter_multiparam_opt/grads.svg) |\n\n");

    md.push_str("## Chart grid\n\n");
    md.push_str("| Row | Left panel | Right panel |\n");
    md.push_str("| --- | --- | --- |\n");
    md.push_str("| 1 | A. Total / data / reg loss | B. Vth_n and \\|Vth_p\\| trajectories |\n");
    md.push_str("| 2 | C. V_m tracking vs target  | D. ∂L/∂Vth_n and ∂L/∂Vth_p |\n\n");

    md.push_str("## A) Loss decomposition\n\n```mermaid\nxychart-beta\n");
    md.push_str("  title \"Total loss decomposed into data + regularization\"\n");
    md.push_str(&format!("  x-axis \"step\" [{xs}]\n"));
    md.push_str("  y-axis \"loss\"\n");
    md.push_str(&format!("  line [{y_loss}]\n"));
    md.push_str(&format!("  line [{y_data}]\n"));
    md.push_str(&format!("  line [{y_reg}]\n"));
    md.push_str("```\n\nLegend:\n\n- line 1: total $L$\n- line 2: data $(V_m - V_m^*)^2$\n- line 3: reg $\\lambda(V_{th,n} - |V_{th,p}|)^2$\n\n");

    md.push_str("## B) Threshold trajectories\n\n```mermaid\nxychart-beta\n");
    md.push_str("  title \"Vth_n and |Vth_p| converging toward each other\"\n");
    md.push_str(&format!("  x-axis \"step\" [{xs}]\n"));
    md.push_str("  y-axis \"Vth (V)\"\n");
    md.push_str(&format!("  line [{y_vthn}]\n"));
    md.push_str(&format!("  line [{y_vthp}]\n"));
    md.push_str("```\n\nLegend:\n\n- line 1: `Vth_n`\n- line 2: `|Vth_p|`\n\n");

    md.push_str("## C) V_m tracking vs target\n\n```mermaid\nxychart-beta\n");
    md.push_str("  title \"V_m and V_m - V_m_target signed error\"\n");
    md.push_str(&format!("  x-axis \"step\" [{xs}]\n"));
    md.push_str("  y-axis \"voltage (V)\"\n");
    md.push_str(&format!("  line [{y_vm}]\n"));
    md.push_str(&format!("  line [{y_err}]\n"));
    md.push_str("```\n\nLegend:\n\n- line 1: `V_m`\n- line 2: `V_m - V_m_target`\n\n");

    md.push_str("## D) Per-parameter gradients\n\n```mermaid\nxychart-beta\n");
    md.push_str("  title \"∂L/∂Vth_n and ∂L/∂Vth_p driving Adam updates\"\n");
    md.push_str(&format!("  x-axis \"step\" [{xs}]\n"));
    md.push_str("  y-axis \"gradient\"\n");
    md.push_str(&format!("  line [{y_gn}]\n"));
    md.push_str(&format!("  line [{y_gp}]\n"));
    md.push_str("```\n\nLegend:\n\n- line 1: $\\partial L / \\partial V_{th,n}$\n- line 2: $\\partial L / \\partial V_{th,p}$\n\n");

    md.push_str("## Step-by-step trace\n\n");
    md.push_str("| step | Vth_n | \\|Vth_p\\| | V_m | loss | data | reg | ∂L/∂Vth_n | ∂L/∂Vth_p |\n");
    md.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.6} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} |\n",
            r.step, r.vth_n, r.vth_p_abs, r.vm, r.loss, r.loss_data, r.loss_reg,
            r.dloss_dvth_n, r.dloss_dvth_p,
        ));
    }
    md
}

// SVG renderer (same style as ml_trace.rs / inverter_vm_opt.rs).
struct LineSeries<'a> {
    name: &'a str,
    color: &'a str,
    values: &'a [f32],
}

fn write_rendered_svgs(rows: &[StepRow], vm_target: f32, dir: &PathBuf) -> Result<(), Box<dyn Error>> {
    let steps: Vec<f32> = rows.iter().map(|r| r.step as f32).collect();
    let total: Vec<f32> = rows.iter().map(|r| r.loss).collect();
    let data:  Vec<f32> = rows.iter().map(|r| r.loss_data).collect();
    let reg:   Vec<f32> = rows.iter().map(|r| r.loss_reg).collect();
    let vthn:  Vec<f32> = rows.iter().map(|r| r.vth_n).collect();
    let vthp:  Vec<f32> = rows.iter().map(|r| r.vth_p_abs).collect();
    let vm:    Vec<f32> = rows.iter().map(|r| r.vm).collect();
    let err:   Vec<f32> = rows.iter().map(|r| r.vm - vm_target).collect();
    let gn:    Vec<f32> = rows.iter().map(|r| r.dloss_dvth_n).collect();
    let gp:    Vec<f32> = rows.iter().map(|r| r.dloss_dvth_p).collect();

    fs::write(dir.join("loss.svg"), line_chart_svg(
        "Loss decomposition (Adam)", "step", "loss", &steps,
        &[
            LineSeries { name: "total",  color: "#2563eb", values: &total },
            LineSeries { name: "data",   color: "#0891b2", values: &data },
            LineSeries { name: "reg",    color: "#a16207", values: &reg },
        ],
    ))?;
    fs::write(dir.join("params.svg"), line_chart_svg(
        "Vth_n and |Vth_p| trajectories", "step", "Vth (V)", &steps,
        &[
            LineSeries { name: "Vth_n",   color: "#0f766e", values: &vthn },
            LineSeries { name: "|Vth_p|", color: "#b45309", values: &vthp },
        ],
    ))?;
    fs::write(dir.join("output.svg"), line_chart_svg(
        "V_m and signed error", "step", "voltage (V)", &steps,
        &[
            LineSeries { name: "V_m",       color: "#1d4ed8", values: &vm },
            LineSeries { name: "V_m - V_m*", color: "#dc2626", values: &err },
        ],
    ))?;
    fs::write(dir.join("grads.svg"), line_chart_svg(
        "Per-parameter gradients", "step", "gradient", &steps,
        &[
            LineSeries { name: "dL/dVth_n", color: "#7c3aed", values: &gn },
            LineSeries { name: "dL/dVth_p", color: "#db2777", values: &gp },
        ],
    ))?;
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
