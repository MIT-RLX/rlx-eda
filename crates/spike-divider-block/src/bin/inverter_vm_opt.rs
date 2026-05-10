//! Differentiable optimization of a CMOS inverter's switching threshold.
//!
//! Mirror of `ml_trace.rs` for an analog block. The "circuit" is a
//! CMOS inverter (NMOS + PMOS); the "parameter" is the NMOS threshold `Vth_n`;
//! the "target" is the inverter's switching threshold V_m. Loss =
//! (V_m − V_target)². The framework's `eda-mna` solver computes V_m
//! via a self-consistent DC operating point with Vin shorted to Vout,
//! and the rlx graph provides ∂V_m / ∂Vth_n via sensitivities.
//!
//! Headline: this proves rlx-eda's differentiable solver runs through
//! transistor-level analog blocks and produces gradients the optimizer
//! can act on — same loop shape as the divider's `ml_trace`, just
//! scaled up to two MOSFETs.
//!
//! Run:
//!   cargo run -p spike-divider-block --bin inverter_vm_opt
//!
//! Outputs:
//! - `crates/spike-divider-block/docs/inverter_vm_opt_trace.md`
//! - `crates/spike-divider-block/docs/assets/inverter_vm_opt/{loss,params,output,grads}.svg`
//! - workspace mirror under `docs/`
//! - `/tmp/rlx_eda_inverter_vm_opt.csv`

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use eda_mna::{sensitivities, solve_dc, Circuit, NetId, NewtonOptions};
use spike_divider_block::Mosfet;

#[derive(Clone, Copy, Debug)]
struct StepRow {
    step: usize,
    /// NMOS threshold voltage Vth0 — the optimization variable.
    /// (Width W is baked into the rlx graph as a const at circuit
    /// construction; threshold is a true graph Param so we can flow
    /// gradients through it without rebuilding the circuit each step.)
    vth_n: f32,
    /// Current switching threshold from solve_dc.
    vm: f32,
    loss: f32,
    /// ∂Vm / ∂Vth_n via sensitivities (rlx reverse-mode AD).
    dvm_dvthn: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let vdd = 1.0_f32;
    let vm_target = 0.60_f32; // skewed away from Vdd/2 to require optimization

    // Build the inverter as a self-consistent DC problem: Vin shorted
    // to Vout. The solved V at this shared "vmid" net IS Vm.
    let mut circuit = Circuit::new();
    let v_dd = circuit.alloc_boundary_net();
    let vmid = circuit.alloc_unknown_net(); // Vin = Vout = Vm

    // Default sizes: PMOS twice as wide as NMOS to nominally hit Vdd/2.
    // We hold L = 1 µm constant; Vth_n is the optimization variable.
    let nmos = Mosfet::nmos(2_000, 1_000, "Mn");
    let pmos = Mosfet::pmos(4_000, 1_000, "Mp");

    // NMOS: D = vmid, G = vmid, S = B = gnd
    circuit.add_device(nmos.clone(), &[vmid, vmid, NetId::GND, NetId::GND]);
    // PMOS: D = vmid, G = vmid, S = B = vdd
    circuit.add_device(pmos.clone(), &[vmid, vmid, v_dd, v_dd]);

    let n_name = eda_hir::Block::name(&nmos);
    let _p_name = eda_hir::Block::name(&pmos);

    // Default model params (Kp, Vth0, Lambda, Gamma, TwoPhiF, N) for
    // both transistors. We optimize the NMOS Vth0 — same first-order
    // effect on V_m as W (both shift the comparator's switching point);
    // Vth happens to be a true graph Param while W is baked in as a
    // const at circuit construction.
    let mut params: HashMap<String, f32> = nmos.default_params();
    params.extend(pmos.default_params());
    let vth_param = format!("{n_name}_Vth");
    assert!(params.contains_key(&vth_param),
        "expected NMOS to expose {vth_param}; got {:?}", params.keys().collect::<Vec<_>>());

    let mut boundary = HashMap::new();
    boundary.insert(v_dd, vdd);

    // Outer-Newton optimization loop, with per-step logging so we can
    // emit a trace report.
    let max_iters = 60_usize;
    let tol = 1e-4_f32;
    let damping = 0.5_f32;
    let solver = NewtonOptions::default();
    let target_param_names = vec![vth_param.clone()];

    let mut rows: Vec<StepRow> = Vec::with_capacity(max_iters + 1);

    for step in 0..=max_iters {
        let op = solve_dc(&circuit, &params, &boundary, solver);
        let vm = op.voltages.get(&vmid).copied().unwrap_or(0.0);
        let err = vm - vm_target;
        let loss = err * err;

        let sens = sensitivities(&circuit, &params, &boundary, &op, &target_param_names);
        let target_idx = 0; // vmid is the only unknown net
        let dvm_dvthn = sens.get(&vth_param).map(|v| v[target_idx]).unwrap_or(0.0);

        rows.push(StepRow {
            step,
            vth_n: *params.get(&vth_param).unwrap_or(&0.0),
            vm,
            loss,
            dvm_dvthn,
        });

        if err.abs() < tol { break; }
        if dvm_dvthn.abs() < 1e-30 { break; }

        // Damped outer-Newton step on V_m(Vth_n) − target = 0
        let raw_step = -err / dvm_dvthn;
        let p = params.get_mut(&vth_param).expect("vth_param missing");
        // Vth typically lives in [0.1 V, 0.9 V]; cap each step at 0.1 V.
        let step_size = (damping * raw_step).clamp(-0.1, 0.1);
        *p += step_size;
        // Keep Vth in a sensible window so we stay in saturation-like
        // operation through the sweep.
        if *p < 0.05 { *p = 0.05; }
        if *p > 0.95 { *p = 0.95; }
    }

    let final_row = *rows.last().expect("rows non-empty");
    println!("Differentiable inverter Vm optimization");
    println!("  vdd = {vdd:.3} V, target Vm = {vm_target:.3} V");
    println!(
        "  initial: Vth_n = {:.4} V, Vm = {:.6}, loss = {:.3e}",
        rows[0].vth_n, rows[0].vm, rows[0].loss
    );
    println!(
        "  final:   Vth_n = {:.4} V, Vm = {:.6}, loss = {:.3e}, steps = {}",
        final_row.vth_n, final_row.vm, final_row.loss, final_row.step
    );

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let assets = crate_dir.join("docs/assets/inverter_vm_opt");
    fs::create_dir_all(&assets)?;
    write_rendered_svgs(&rows, vm_target, &assets)?;

    let csv = "/tmp/rlx_eda_inverter_vm_opt.csv";
    fs::write(csv, build_csv(&rows))?;
    let md_path = crate_dir.join("docs/inverter_vm_opt_trace.md");
    let md = build_report(&rows, vdd, vm_target);
    fs::write(&md_path, &md)?;
    println!();
    println!("wrote CSV report : {csv}");
    println!("wrote MD report  : {}", md_path.display());
    println!("wrote SVG charts : {}/", assets.display());

    let workspace_docs = crate_dir.join("../../docs");
    if workspace_docs.is_dir() {
        let workspace_md = workspace_docs.join("inverter_vm_opt_trace.md");
        let workspace_assets = workspace_docs.join("assets/inverter_vm_opt");
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
    let mut s = String::from("step,vth_n,vm,loss,dvm_dvthn\n");
    for r in rows {
        s.push_str(&format!(
            "{},{:.6},{:.8},{:.10e},{:.10e}\n",
            r.step, r.vth_n, r.vm, r.loss, r.dvm_dvthn
        ));
    }
    s
}

fn build_report(rows: &[StepRow], vdd: f32, vm_target: f32) -> String {
    let first = rows.first().unwrap();
    let last = rows.last().unwrap();
    let xs = rows.iter().map(|r| r.step.to_string()).collect::<Vec<_>>().join(", ");
    let y_loss = rows.iter().map(|r| format!("{:.8}", r.loss)).collect::<Vec<_>>().join(", ");
    let y_w = rows.iter().map(|r| format!("{:.4}", r.vth_n)).collect::<Vec<_>>().join(", ");
    let y_vm = rows.iter().map(|r| format!("{:.4}", r.vm)).collect::<Vec<_>>().join(", ");
    let y_err = rows.iter().map(|r| format!("{:.4}", r.vm - vm_target)).collect::<Vec<_>>().join(", ");
    let y_grad = rows.iter().map(|r| format!("{:.6}", r.dvm_dvthn)).collect::<Vec<_>>().join(", ");

    let mut md = String::new();
    md.push_str("# rlx-eda differentiable CMOS inverter Vm optimization\n\n");
    md.push_str("Circuit: CMOS inverter (NMOS + PMOS via `spike_divider_block::Mosfet`), with input shorted to output so the solved DC operating point IS the switching threshold V_m.\n\n");
    md.push_str(&format!("Stimulus: `Vdd = {vdd:.3} V`, `V_m* = {vm_target:.3} V`\n\n"));
    md.push_str("Loss definition:\n\n");
    md.push_str("$$L = (V_m - V_{m,\\text{target}})^2$$\n\n");
    md.push_str("Gradient-driven parameter update (outer Newton on $V_m(Vth_n) - V_m^* = 0$):\n\n");
    md.push_str("$$Vth_n \\leftarrow Vth_n - \\eta \\cdot \\frac{V_m - V_m^*}{\\partial V_m / \\partial Vth_n}$$\n\n");

    md.push_str("## Optimization outcome\n\n");
    md.push_str(&format!(
        "- initial: `Vth_n (V) = {:.3}`, `V_m = {:.6}`, `loss = {:.3e}`\n",
        first.vth_n, first.vm, first.loss
    ));
    md.push_str(&format!(
        "- final:   `Vth_n (V) = {:.3}`, `V_m = {:.6}`, `loss = {:.3e}`, `steps = {}`\n\n",
        last.vth_n, last.vm, last.loss, last.step
    ));
    md.push_str("All gradients computed via reverse-mode AD on the rlx graph that stamps the MOSFET LEVEL-1 equations into the MNA residual. No SPICE oracle.\n\n");

    md.push_str("## Rendered charts\n\n");
    md.push_str("| Loss and objective | Parameter evolution |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![Rendered loss chart](crates/spike-divider-block/docs/assets/inverter_vm_opt/loss.svg) | ![Rendered parameter chart](crates/spike-divider-block/docs/assets/inverter_vm_opt/params.svg) |\n\n");
    md.push_str("| Output and error | Gradient signals |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![Rendered output chart](crates/spike-divider-block/docs/assets/inverter_vm_opt/output.svg) | ![Rendered gradient chart](crates/spike-divider-block/docs/assets/inverter_vm_opt/grads.svg) |\n\n");

    md.push_str("## Chart grid\n\n");
    md.push_str("| Row | Left panel | Right panel |\n");
    md.push_str("| --- | --- | --- |\n");
    md.push_str("| 1 | A. Loss over steps | B. NMOS Vth trajectory |\n");
    md.push_str("| 2 | C. V_m tracking vs target | D. ∂V_m / ∂Vth_n evolution |\n\n");

    md.push_str("## A) Loss over steps\n\n```mermaid\nxychart-beta\n");
    md.push_str("  title \"Inverter Vm optimization loss trajectory\"\n");
    md.push_str(&format!("  x-axis \"step\" [{xs}]\n"));
    md.push_str("  y-axis \"loss\"\n");
    md.push_str(&format!("  line [{y_loss}]\n"));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n- line 1: optimization loss $L = (V_m - V_m^*)^2$\n\n");

    md.push_str("## B) NMOS Vth trajectory\n\n```mermaid\nxychart-beta\n");
    md.push_str("  title \"NMOS threshold voltage by step\"\n");
    md.push_str(&format!("  x-axis \"step\" [{xs}]\n"));
    md.push_str("  y-axis \"Vth_n (V)\"\n");
    md.push_str(&format!("  line [{y_w}]\n"));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n- line 1: `Vth_n` (NMOS threshold voltage in V)\n\n");

    md.push_str("## C) V_m tracking vs target\n\n```mermaid\nxychart-beta\n");
    md.push_str("  title \"V_m and V_m - V_m_target signed error\"\n");
    md.push_str(&format!("  x-axis \"step\" [{xs}]\n"));
    md.push_str("  y-axis \"voltage (V)\"\n");
    md.push_str(&format!("  line [{y_vm}]\n"));
    md.push_str(&format!("  line [{y_err}]\n"));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n- line 1: `V_m`\n- line 2: `V_m - V_m_target` (signed error)\n\n");

    md.push_str("## D) Gradient evolution\n\n```mermaid\nxychart-beta\n");
    md.push_str("  title \"∂V_m / ∂Vth_n driving the parameter updates\"\n");
    md.push_str(&format!("  x-axis \"step\" [{xs}]\n"));
    md.push_str("  y-axis \"sensitivity\"\n");
    md.push_str(&format!("  line [{y_grad}]\n"));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n- line 1: $\\partial V_m / \\partial Vth_n$ (V per unit width-multiplier)\n\n");

    md.push_str("## Step-by-step trace\n\n");
    md.push_str("| step | Vth_n (V) | V_m (V) | loss | dV_m/dVth_n |\n");
    md.push_str("| --- | --- | --- | --- | --- |\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | {:.4} | {:.6} | {:.4e} | {:.4e} |\n",
            r.step, r.vth_n, r.vm, r.loss, r.dvm_dvthn
        ));
    }

    md
}

// SVG chart code lifted from `ml_trace.rs` (same style, same layout).
struct LineSeries<'a> {
    name: &'a str,
    color: &'a str,
    values: &'a [f32],
}

fn write_rendered_svgs(rows: &[StepRow], vm_target: f32, dir: &PathBuf) -> Result<(), Box<dyn Error>> {
    let steps: Vec<f32> = rows.iter().map(|r| r.step as f32).collect();
    let loss: Vec<f32> = rows.iter().map(|r| r.loss).collect();
    let w: Vec<f32> = rows.iter().map(|r| r.vth_n).collect();
    let vm: Vec<f32> = rows.iter().map(|r| r.vm).collect();
    let err: Vec<f32> = rows.iter().map(|r| r.vm - vm_target).collect();
    let g: Vec<f32> = rows.iter().map(|r| r.dvm_dvthn).collect();

    fs::write(dir.join("loss.svg"), line_chart_svg(
        "Inverter Vm optimization loss", "step", "loss", &steps,
        &[LineSeries { name: "loss", color: "#2563eb", values: &loss }],
    ))?;
    fs::write(dir.join("params.svg"), line_chart_svg(
        "NMOS threshold Vth_n", "step", "Vth_n (V)", &steps,
        &[LineSeries { name: "Vth_n", color: "#0f766e", values: &w }],
    ))?;
    fs::write(dir.join("output.svg"), line_chart_svg(
        "V_m and signed error", "step", "voltage (V)", &steps,
        &[
            LineSeries { name: "V_m",       color: "#1d4ed8", values: &vm },
            LineSeries { name: "V_m - V_m*", color: "#dc2626", values: &err },
        ],
    ))?;
    fs::write(dir.join("grads.svg"), line_chart_svg(
        "Sensitivity dV_m/dVth_n", "step", "sensitivity", &steps,
        &[LineSeries { name: "dV_m/dVth_n", color: "#7c3aed", values: &g }],
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
