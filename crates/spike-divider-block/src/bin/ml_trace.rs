//! Single-circuit ML optimization trace for the RC divider.
//!
//! This binary logs every optimization step (loss, gradients, params, Vout)
//! and writes two artifacts:
//! - CSV trace (all steps)
//! - Markdown report with Mermaid charts
//!
//! Run:
//!   cargo run -p spike-divider-block --bin ml_trace

use std::error::Error;
use std::fs;

use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_divider_block::{Adam, Optimizer, RcDivider, Resistor};

#[derive(Clone, Copy, Debug)]
struct StepRow {
    step: usize,
    r1: f32,
    r2: f32,
    vout: f32,
    loss: f32,
    dloss_dr1: f32,
    dloss_dr2: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    // One fixed circuit and one fixed inverse-design target.
    let v_in = 1.0_f32;
    let target_vout = 0.4_f32;

    // Divider block whose param names are used to seed rlx params.
    let divider = RcDivider::new(
        Resistor {
            length: 10_000,
            id: "R1".into(),
        },
        Resistor {
            length: 30_000,
            id: "R2".into(),
        },
    );

    // Build a differentiable loss graph: L = (Vout - target)^2.
    let (fwd, r1_id, r2_id) = divider.build_loss_graph();
    let bwd = grad_with_loss(&fwd, &[r1_id, r2_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);

    let [name_r1, name_r2] = divider.dc_param_names();

    // Initial design (same as the main demo).
    let mut params = [1000.0_f32, 3000.0_f32];

    // Adam optimizer over the two circuit parameters.
    let mut opt = Adam::new(80.0, 2);

    let max_iters = 250usize;
    let tol = 1e-4_f32;
    let r_min = 1.0_f32;

    let mut rows: Vec<StepRow> = Vec::with_capacity(max_iters + 1);

    for step in 0..=max_iters {
        compiled.set_param(&name_r1, &[params[0]]);
        compiled.set_param(&name_r2, &[params[1]]);

        let outs = compiled.run(&[
            ("V", &[v_in][..]),
            ("target", &[target_vout][..]),
            ("d_output", &[1.0_f32][..]),
        ]);

        let loss = outs[0][0];
        let grad_r1 = outs[1][0];
        let grad_r2 = outs[2][0];
        let vout = v_in * params[1] / (params[0] + params[1]);

        rows.push(StepRow {
            step,
            r1: params[0],
            r2: params[1],
            vout,
            loss,
            dloss_dr1: grad_r1,
            dloss_dr2: grad_r2,
        });

        if loss.sqrt() < tol {
            break;
        }

        opt.step(&mut params, &[grad_r1, grad_r2]);
        params[0] = params[0].max(r_min);
        params[1] = params[1].max(r_min);
    }

    let final_row = *rows.last().expect("rows should be non-empty");

    // Console summary plus full per-step table.
    println!("Single-circuit ML optimization trace (RcDivider)");
    println!("  objective : L = (Vout - target)^2");
    println!("  target    : Vin = {:.3}, Vout* = {:.3}", v_in, target_vout);
    println!(
        "  initial   : R1 = {:.3} ohm, R2 = {:.3} ohm",
        rows[0].r1, rows[0].r2
    );
    println!(
        "  converged : R1 = {:.3} ohm, R2 = {:.3} ohm, Vout = {:.6}, loss = {:.3e}, steps = {}",
        final_row.r1,
        final_row.r2,
        final_row.vout,
        final_row.loss,
        final_row.step
    );
    println!();
    println!("step,r1_ohm,r2_ohm,vout,loss,dloss_dr1,dloss_dr2");
    for row in &rows {
        println!(
            "{},{:.6},{:.6},{:.8},{:.8e},{:.8e},{:.8e}",
            row.step, row.r1, row.r2, row.vout, row.loss, row.dloss_dr1, row.dloss_dr2
        );
    }

    let csv_path = "/tmp/rlx_eda_divider_ml_trace.csv";
    let report_path = "/tmp/rlx_eda_divider_ml_trace.md";
    let rendered_charts_dir = "docs/assets/ml_trace";

    fs::create_dir_all(rendered_charts_dir)?;
    write_rendered_svgs(&rows, rendered_charts_dir)?;

    fs::write(csv_path, build_csv(&rows))?;
    fs::write(report_path, build_report(&rows, v_in, target_vout))?;

    println!();
    println!("wrote CSV report : {}", csv_path);
    println!("wrote MD report  : {}", report_path);
    println!("wrote SVG charts : {}/", rendered_charts_dir);

    Ok(())
}

fn build_csv(rows: &[StepRow]) -> String {
    let mut out = String::from("step,r1_ohm,r2_ohm,vout,loss,dloss_dr1,dloss_dr2\n");
    for row in rows {
        out.push_str(&format!(
            "{},{:.8},{:.8},{:.8},{:.10e},{:.10e},{:.10e}\n",
            row.step, row.r1, row.r2, row.vout, row.loss, row.dloss_dr1, row.dloss_dr2
        ));
    }
    out
}

fn build_report(rows: &[StepRow], v_in: f32, target_vout: f32) -> String {
    let first = rows.first().expect("rows should be non-empty");
    let last = rows.last().expect("rows should be non-empty");

    let sampled = sample_rows(rows, 24);

    let x_steps = sampled
        .iter()
        .map(|r| r.step.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let y_loss = sampled
        .iter()
        .map(|r| format!("{:.8}", r.loss))
        .collect::<Vec<_>>()
        .join(", ");

    let y_r1 = sampled
        .iter()
        .map(|r| format!("{:.4}", r.r1))
        .collect::<Vec<_>>()
        .join(", ");

    let y_r2 = sampled
        .iter()
        .map(|r| format!("{:.4}", r.r2))
        .collect::<Vec<_>>()
        .join(", ");

    let y_vout = sampled
        .iter()
        .map(|r| format!("{:.8}", r.vout))
        .collect::<Vec<_>>()
        .join(", ");

    let y_err = sampled
        .iter()
        .map(|r| format!("{:.8}", r.vout - target_vout))
        .collect::<Vec<_>>()
        .join(", ");

    let y_grad_r1 = sampled
        .iter()
        .map(|r| format!("{:.10}", r.dloss_dr1))
        .collect::<Vec<_>>()
        .join(", ");

    let y_grad_r2 = sampled
        .iter()
        .map(|r| format!("{:.10}", r.dloss_dr2))
        .collect::<Vec<_>>()
        .join(", ");

    let mut md = String::new();
    md.push_str("# rlx-eda single-circuit ML optimization trace\n\n");
    md.push_str("Circuit: `RcDivider` (`spike-divider-block`)\n\n");
    md.push_str(&format!(
        "Target: `Vin = {:.3}`, `Vout* = {:.3}`\n\n",
        v_in, target_vout
    ));

    md.push_str("Loss definition:\n\n");
    md.push_str("$$L = (V_{out} - V_{target})^2$$\n\n");
    md.push_str("Gradient-driven parameter updates:\n\n");
    md.push_str(r"$$R_1 \leftarrow R_1 - \eta \frac{\partial L}{\partial R_1}, \quad R_2 \leftarrow R_2 - \eta \frac{\partial L}{\partial R_2}$$");
    md.push_str("\n\n");

    md.push_str("## Optimization outcome\n\n");
    md.push_str(&format!(
        "- initial: `R1={:.3} ohm`, `R2={:.3} ohm`, `Vout={:.6}`, `loss={:.3e}`\n",
        first.r1, first.r2, first.vout, first.loss
    ));
    md.push_str(&format!(
        "- final: `R1={:.3} ohm`, `R2={:.3} ohm`, `Vout={:.6}`, `loss={:.3e}`, `steps={}`\n\n",
        last.r1, last.r2, last.vout, last.loss, last.step
    ));

    md.push_str("## Rendered charts\n\n");
    md.push_str("| Loss and objective | Parameter evolution |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![Rendered loss chart](crates/spike-divider-block/docs/assets/ml_trace/loss.svg) | ![Rendered parameter chart](crates/spike-divider-block/docs/assets/ml_trace/params.svg) |\n\n");
    md.push_str("| Output and error | Gradient signals |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![Rendered output chart](crates/spike-divider-block/docs/assets/ml_trace/output.svg) | ![Rendered gradient chart](crates/spike-divider-block/docs/assets/ml_trace/grads.svg) |\n\n");

    md.push_str("## Chart grid\n\n");
    md.push_str("| Row | Left panel | Right panel |\n");
    md.push_str("| --- | --- | --- |\n");
    md.push_str("| 1 | A. Loss over steps | B. Parameter trajectory |\n");
    md.push_str("| 2 | C. Output tracking vs target | D. Gradient evolution |\n\n");

    md.push_str("## A) Loss over steps (sampled)\n\n");
    md.push_str("```mermaid\n");
    md.push_str("xychart-beta\n");
    md.push_str("  title \"RcDivider optimization loss trajectory\"\n");
    md.push_str(&format!("  x-axis \"step\" [{}]\n", x_steps));
    md.push_str("  y-axis \"loss\"\n");
    md.push_str(&format!("  line [{}]\n", y_loss));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n");
    md.push_str("- line 1: optimization loss $L = (V_{out} - V_{target})^2$\n\n");

    md.push_str("## B) Parameter trajectory (sampled)\n\n");
    md.push_str("```mermaid\n");
    md.push_str("xychart-beta\n");
    md.push_str("  title \"RcDivider parameters by optimization step\"\n");
    md.push_str(&format!("  x-axis \"step\" [{}]\n", x_steps));
    md.push_str("  y-axis \"resistance (ohm)\"\n");
    md.push_str(&format!("  line [{}]\n", y_r1));
    md.push_str(&format!("  line [{}]\n", y_r2));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n");
    md.push_str("- line 1: `R1` (ohm)\n");
    md.push_str("- line 2: `R2` (ohm)\n\n");

    md.push_str("## C) Output tracking vs target (sampled)\n\n");
    md.push_str("```mermaid\n");
    md.push_str("xychart-beta\n");
    md.push_str("  title \"Vout and Vout-target error by step\"\n");
    md.push_str(&format!("  x-axis \"step\" [{}]\n", x_steps));
    md.push_str("  y-axis \"voltage\"\n");
    md.push_str(&format!("  line [{}]\n", y_vout));
    md.push_str(&format!("  line [{}]\n", y_err));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n");
    md.push_str("- line 1: `Vout`\n");
    md.push_str("- line 2: `Vout - Vtarget` (signed error)\n\n");

    md.push_str("## D) Gradient evolution (sampled)\n\n");
    md.push_str("```mermaid\n");
    md.push_str("xychart-beta\n");
    md.push_str("  title \"Loss-gradient signals driving parameter updates\"\n");
    md.push_str(&format!("  x-axis \"step\" [{}]\n", x_steps));
    md.push_str("  y-axis \"gradient\"\n");
    md.push_str(&format!("  line [{}]\n", y_grad_r1));
    md.push_str(&format!("  line [{}]\n", y_grad_r2));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n");
    md.push_str("- line 1: $\\partial L / \\partial R_1$\n");
    md.push_str("- line 2: $\\partial L / \\partial R_2$\n\n");

    md.push_str("## Step-by-step trace (all steps)\n\n");
    md.push_str("| step | R1 (ohm) | R2 (ohm) | Vout | loss | dL/dR1 | dL/dR2 |\n");
    md.push_str("| --- | --- | --- | --- | --- | --- | --- |\n");
    for row in rows {
        md.push_str(&format!(
            "| {} | {:.6} | {:.6} | {:.8} | {:.8e} | {:.8e} | {:.8e} |\n",
            row.step, row.r1, row.r2, row.vout, row.loss, row.dloss_dr1, row.dloss_dr2
        ));
    }

    md
}

fn sample_rows(rows: &[StepRow], target_count: usize) -> Vec<StepRow> {
    if rows.len() <= target_count {
        return rows.to_vec();
    }

    let mut out = Vec::with_capacity(target_count);
    let last = rows.len() - 1;
    for i in 0..target_count {
        let idx = i * last / (target_count - 1);
        out.push(rows[idx]);
    }
    out
}

fn write_rendered_svgs(rows: &[StepRow], out_dir: &str) -> Result<(), Box<dyn Error>> {
    let steps: Vec<f32> = rows.iter().map(|r| r.step as f32).collect();

    let loss: Vec<f32> = rows.iter().map(|r| r.loss).collect();
    let r1: Vec<f32> = rows.iter().map(|r| r.r1).collect();
    let r2: Vec<f32> = rows.iter().map(|r| r.r2).collect();
    let vout: Vec<f32> = rows.iter().map(|r| r.vout).collect();
    let err: Vec<f32> = rows.iter().map(|r| r.vout - 0.4).collect();
    let g1: Vec<f32> = rows.iter().map(|r| r.dloss_dr1).collect();
    let g2: Vec<f32> = rows.iter().map(|r| r.dloss_dr2).collect();

    let loss_svg = line_chart_svg(
        "RcDivider optimization loss trajectory",
        "step",
        "loss",
        &steps,
        &[LineSeries { name: "loss", color: "#2563eb", values: &loss }],
    );
    fs::write(format!("{}/loss.svg", out_dir), loss_svg)?;

    let params_svg = line_chart_svg(
        "RcDivider parameter trajectory",
        "step",
        "resistance (ohm)",
        &steps,
        &[
            LineSeries { name: "R1", color: "#0f766e", values: &r1 },
            LineSeries { name: "R2", color: "#b45309", values: &r2 },
        ],
    );
    fs::write(format!("{}/params.svg", out_dir), params_svg)?;

    let output_svg = line_chart_svg(
        "Vout and signed error",
        "step",
        "voltage",
        &steps,
        &[
            LineSeries { name: "Vout", color: "#1d4ed8", values: &vout },
            LineSeries { name: "Vout - Vtarget", color: "#dc2626", values: &err },
        ],
    );
    fs::write(format!("{}/output.svg", out_dir), output_svg)?;

    let grads_svg = line_chart_svg(
        "Gradient evolution",
        "step",
        "gradient",
        &steps,
        &[
            LineSeries { name: "dL/dR1", color: "#7c3aed", values: &g1 },
            LineSeries { name: "dL/dR2", color: "#db2777", values: &g2 },
        ],
    );
    fs::write(format!("{}/grads.svg", out_dir), grads_svg)?;

    Ok(())
}

struct LineSeries<'a> {
    name: &'a str,
    color: &'a str,
    values: &'a [f32],
}

fn line_chart_svg(
    title: &str,
    x_label: &str,
    y_label: &str,
    x: &[f32],
    series: &[LineSeries<'_>],
) -> String {
    let width = 920.0_f32;
    let height = 480.0_f32;
    let left = 78.0_f32;
    let right = 26.0_f32;
    let top = 56.0_f32;
    let bottom = 62.0_f32;

    let plot_w = width - left - right;
    let plot_h = height - top - bottom;

    let min_x = *x.first().unwrap_or(&0.0);
    let max_x = *x.last().unwrap_or(&1.0);
    let dx = (max_x - min_x).max(1.0);

    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for s in series {
        for &v in s.values {
            min_y = min_y.min(v);
            max_y = max_y.max(v);
        }
    }
    if !min_y.is_finite() || !max_y.is_finite() {
        min_y = -1.0;
        max_y = 1.0;
    }
    if (max_y - min_y).abs() < 1e-12 {
        max_y += 1.0;
        min_y -= 1.0;
    }
    let y_pad = 0.08 * (max_y - min_y);
    min_y -= y_pad;
    max_y += y_pad;
    let dy = (max_y - min_y).max(1e-9);

    let map_x = |vx: f32| left + ((vx - min_x) / dx) * plot_w;
    let map_y = |vy: f32| top + (1.0 - (vy - min_y) / dy) * plot_h;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        width as i32, height as i32, width as i32, height as i32
    ));
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n");

    for i in 0..=6 {
        let t = i as f32 / 6.0;
        let yv = min_y + t * dy;
        let py = map_y(yv);
        svg.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>\n",
            left, py, left + plot_w, py
        ));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\" font-size=\"12\" fill=\"#374151\">{:.3e}</text>\n",
            left - 8.0,
            py + 4.0,
            yv
        ));
    }

    for i in 0..=6 {
        let t = i as f32 / 6.0;
        let xv = min_x + t * dx;
        let px = map_x(xv);
        svg.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>\n",
            px,
            top,
            px,
            top + plot_h
        ));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#374151\">{:.0}</text>\n",
            px,
            top + plot_h + 20.0,
            xv
        ));
    }

    svg.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#111827\" stroke-width=\"1.5\"/>\n",
        left,
        top + plot_h,
        left + plot_w,
        top + plot_h
    ));
    svg.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#111827\" stroke-width=\"1.5\"/>\n",
        left,
        top,
        left,
        top + plot_h
    ));

    svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"24\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#111827\">{}</text>\n",
        width / 2.0,
        title
    ));
    svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"middle\" font-size=\"13\" fill=\"#111827\">{}</text>\n",
        left + plot_w / 2.0,
        height - 16.0,
        x_label
    ));
    svg.push_str(&format!(
        "<text x=\"22\" y=\"{:.2}\" transform=\"rotate(-90 22 {:.2})\" text-anchor=\"middle\" font-size=\"13\" fill=\"#111827\">{}</text>\n",
        top + plot_h / 2.0,
        top + plot_h / 2.0,
        y_label
    ));

    for s in series {
        let mut pts = String::new();
        for (i, &yv) in s.values.iter().enumerate() {
            let xv = x.get(i).copied().unwrap_or(i as f32);
            let px = map_x(xv);
            let py = map_y(yv);
            pts.push_str(&format!("{:.2},{:.2} ", px, py));
        }
        svg.push_str(&format!(
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2.2\"/>\n",
            pts.trim_end(),
            s.color
        ));
    }

    let legend_x = left + plot_w - 170.0;
    let legend_y = top + 10.0;
    let legend_h = 26.0 + series.len() as f32 * 22.0;
    svg.push_str(&format!(
        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"160\" height=\"{:.2}\" rx=\"8\" fill=\"#f9fafb\" stroke=\"#d1d5db\"/>\n",
        legend_x, legend_y, legend_h
    ));
    svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"12\" font-weight=\"700\" fill=\"#111827\">Legend</text>\n",
        legend_x + 10.0,
        legend_y + 16.0
    ));
    for (i, s) in series.iter().enumerate() {
        let y = legend_y + 32.0 + i as f32 * 22.0;
        svg.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"3\"/>\n",
            legend_x + 10.0,
            y,
            legend_x + 36.0,
            y,
            s.color
        ));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"12\" fill=\"#111827\">{}</text>\n",
            legend_x + 44.0,
            y + 4.0,
            s.name
        ));
    }

    svg.push_str("</svg>\n");
    svg
}
