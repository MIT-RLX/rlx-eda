//! Single-conversion trace for the closed-loop SAR ADC.
//!
//! Mirror of `spike-divider-block::bin::ml_trace` — instead of an ML
//! optimization trajectory, this captures a SAR conversion trajectory:
//! one DC `vin` → one digital code, with per-cycle DAC/comparator state
//! sampled from a real ngspice transient.
//!
//! Run:
//!   cargo run -p spike-sar-adc --bin conversion_trace --features ngspice
//!
//! Outputs:
//! - Markdown report at `docs/sar_conversion_trace_example.md`
//! - SVG charts at `docs/assets/sar_trace/{dac,bits,comp,error}.svg`
//! - Mirror copies under `../../docs/` (workspace-level)
//! - CSV trace at `/tmp/rlx_eda_sar_conversion_trace.csv`

#![cfg(feature = "ngspice")]

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
use eda_spice_emit::{Netlist, Pwl, SpiceEmit};
use spike_sar_adc::{ideal_sar_code, SarAdc};

const VDD: f64 = 1.8;
const VREF: f64 = VDD;

/// One SAR cycle's snapshot.
#[derive(Clone, Copy, Debug)]
struct CycleRow {
    /// 1-indexed cycle number (1 = MSB trial, N = LSB trial).
    cycle: usize,
    /// Bit index being trialled this cycle (N-1 = MSB, 0 = LSB).
    bit_index: usize,
    /// DAC trial value at this cycle (with the trial bit set to 1 plus
    /// previously-decided bits).
    v_dac: f64,
    /// Sample/Hold output (vin captured at the start of conversion).
    v_hold: f64,
    /// Comparator output during this cycle. > vdd/2 ⇒ "vhold > vdac".
    cmp: f64,
    /// Bit decision after this cycle's capture pulse.
    bit_value: u32,
    /// Quantization error AFTER this cycle's decision: |vhold − vdac_committed|
    /// where vdac_committed is the DAC value computed from the bits
    /// decided so far (with the rest 0).
    err: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    // vin = 0.5 V picked to exercise all four MSB→LSB decisions clearly
    // (binary search visibly halves at each step) AND land away from a
    // quantization boundary so the ngspice + analytic agreement is
    // unambiguous. With vref = 1.8 V the expected code is
    // floor(0.5 / (1.8/16)) = 4 = 0b0100.
    let vin = 0.5_f64;
    let n_bits = 4_usize;

    // Phase + capture window timing — same as the end-to-end test.
    let bit_starts = [1.1e-6, 2.2e-6, 3.3e-6, 4.4e-6];
    let phase_w = 0.8e-6;
    let cap_off = 0.85e-6;
    let cap_w = 0.10e-6;

    let mut net = Netlist::new("SAR ADC conversion trace");
    net.add_dc_source("dd", "vdd", "0", VDD);
    net.add_dc_source("in", "vin", "0", vin);
    net.add_pwl_source("rb", "reset_b", "0", &Pwl { points: vec![
        (0.0,             0.0),
        (0.4e-6 - 5e-9,   0.0),
        (0.4e-6,          VDD),
        (10.0,            VDD),
    ]});
    net.add_pwl_source("clk", "clk_sh", "0", &Pwl { points: vec![
        (0.0,             0.0),
        (0.5e-6 - 5e-9,   0.0),
        (0.5e-6,          VDD),
        (1.0e-6 - 5e-9,   VDD),
        (1.0e-6,          0.0),
        (10.0,            0.0),
    ]});
    let phase_pwl = |start: f64| Pwl { points: vec![
        (0.0, 0.0), (start - 5e-9, 0.0), (start, VDD),
        (start + phase_w - 5e-9, VDD), (start + phase_w, 0.0), (10.0, 0.0)
    ]};
    let cap_pwl = |start: f64| {
        let cs = start + cap_off;
        let ce = cs + cap_w;
        Pwl { points: vec![
            (0.0, 0.0), (cs - 5e-9, 0.0), (cs, VDD),
            (ce - 5e-9, VDD), (ce, 0.0), (10.0, 0.0),
        ]}
    };
    // bit_starts[0] → MSB (p3); bit_starts[3] → LSB (p0)
    net.add_pwl_source("p3", "p3", "0", &phase_pwl(bit_starts[0]));
    net.add_pwl_source("p2", "p2", "0", &phase_pwl(bit_starts[1]));
    net.add_pwl_source("p1", "p1", "0", &phase_pwl(bit_starts[2]));
    net.add_pwl_source("p0", "p0", "0", &phase_pwl(bit_starts[3]));
    net.add_pwl_source("c3", "c3", "0", &cap_pwl(bit_starts[0]));
    net.add_pwl_source("c2", "c2", "0", &cap_pwl(bit_starts[1]));
    net.add_pwl_source("c1", "c1", "0", &cap_pwl(bit_starts[2]));
    net.add_pwl_source("c0", "c0", "0", &cap_pwl(bit_starts[3]));

    let adc: SarAdc<4> = SarAdc::default();
    adc.emit_spice(&mut net, &[
        "vin",
        "p0", "p1", "p2", "p3",
        "c0", "c1", "c2", "c3",
        "clk_sh", "reset_b",
        "b0", "b1", "b2", "b3",
        "vdd", "0",
    ], "u1")?;

    let ng = LocalBinary::from_env().expect("ngspice not on PATH; install or set NGSPICE_BIN");
    let h = 5e-9;
    let t_stop = 6.0e-6;
    let analysis = TransientAnalysis::new(h, t_stop).with_t_max(h);
    let trace = ng.run_transient_trace(
        &net.deck(),
        &analysis,
        &[
            OutputRequest::NodeVoltage("u1_vhold".into()),
            OutputRequest::NodeVoltage("u1_vdac".into()),
            OutputRequest::NodeVoltage("u1_cmp".into()),
            OutputRequest::NodeVoltage("b0".into()),
            OutputRequest::NodeVoltage("b1".into()),
            OutputRequest::NodeVoltage("b2".into()),
            OutputRequest::NodeVoltage("b3".into()),
        ],
    ).expect("ngspice transient");

    // Sample at "mid-phase, after settling": each phase fires for 800ns,
    // so sample 600ns into the phase (200ns before phase end). At this
    // point the trial DAC has settled and the comparator output reflects
    // the trial decision; the bit value is whatever was latched by the
    // PREVIOUS phase's capture (or the initial state for the first phase).
    let t = &trace.time;
    let mut rows: Vec<CycleRow> = Vec::with_capacity(n_bits);
    let lsb = (VREF - 0.0) / (1u32 << n_bits) as f64;

    let bits_keys = ["b0", "b1", "b2", "b3"];

    let mut committed_bits: u32 = 0;
    for (cycle_idx, &start) in bit_starts.iter().enumerate() {
        let trial_bit = n_bits - 1 - cycle_idx; // cycle 0 → bit 3 (MSB)
        let t_sample_mid = start + 0.6e-6;
        // Sample bit value AFTER this phase's capture window
        // (capture pulse ends at start + cap_off + cap_w):
        let t_sample_after_cap = start + cap_off + cap_w + 50e-9;

        let v_hold = lerp(t, &trace.node_voltages["u1_vhold"], t_sample_mid);
        let v_dac = lerp(t, &trace.node_voltages["u1_vdac"], t_sample_mid);
        let cmp = lerp(t, &trace.node_voltages["u1_cmp"], t_sample_mid);

        // Read the bit value AFTER this cycle's capture pulse.
        let mut bit_value_u32 = 0u32;
        for (i, k) in bits_keys.iter().enumerate() {
            let v = lerp(t, &trace.node_voltages[*k], t_sample_after_cap);
            if v >= VDD / 2.0 { bit_value_u32 |= 1 << i; }
        }
        let this_bit = (bit_value_u32 >> trial_bit) & 1;
        committed_bits = (committed_bits & !(1u32 << trial_bit)) | (this_bit << trial_bit);

        // Quantization error after committing this bit (with all
        // not-yet-decided lower bits assumed 0).
        let v_dac_committed = (committed_bits as f64) * lsb;
        let err = (v_hold - v_dac_committed).abs();

        rows.push(CycleRow {
            cycle: cycle_idx + 1,
            bit_index: trial_bit,
            v_dac,
            v_hold,
            cmp,
            bit_value: this_bit,
            err,
        });
    }

    let final_code = committed_bits;
    let ideal = ideal_sar_code(vin, VREF, n_bits);

    // Console summary
    println!("Single-conversion SAR ADC trace");
    println!("  vin = {:.3} V, vref = {:.3} V, N = {}", vin, VREF, n_bits);
    println!(
        "  ngspice converged code: {} = 0b{:0>4b}; ideal: {} = 0b{:0>4b}",
        final_code, final_code, ideal, ideal
    );
    println!();
    println!("cycle,bit,v_dac_trial,v_hold,cmp,bit_value,err");
    for row in &rows {
        println!(
            "{},{},{:.4},{:.4},{:.4},{},{:.4}",
            row.cycle, row.bit_index, row.v_dac, row.v_hold, row.cmp, row.bit_value, row.err
        );
    }

    // Output paths.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crate_assets = crate_dir.join("docs/assets/sar_trace");
    let crate_md = crate_dir.join("docs/sar_conversion_trace_example.md");
    fs::create_dir_all(&crate_assets)?;
    write_rendered_svgs(&rows, &crate_assets)?;

    let csv = "/tmp/rlx_eda_sar_conversion_trace.csv";
    fs::write(csv, build_csv(&rows))?;

    let md = build_report(&rows, vin, VREF, n_bits, final_code, ideal);
    fs::write(&crate_md, &md)?;
    println!();
    println!("wrote CSV report : {}", csv);
    println!("wrote MD report  : {}", crate_md.display());
    println!("wrote SVG charts : {}/", crate_assets.display());

    // Mirror to workspace docs/ if it exists (matches ml_trace pattern).
    let workspace_docs = crate_dir.join("../../docs");
    if workspace_docs.is_dir() {
        let workspace_md = workspace_docs.join("sar_conversion_trace_example.md");
        let workspace_assets = workspace_docs.join("assets/sar_trace");
        fs::create_dir_all(&workspace_assets)?;
        for name in ["dac.svg", "bits.svg", "comp.svg", "error.svg"] {
            fs::copy(crate_assets.join(name), workspace_assets.join(name))?;
        }
        fs::write(&workspace_md, &md)?;
        println!("mirrored to      : {}", workspace_md.display());
    }

    Ok(())
}

fn lerp(xs: &[f64], ys: &[f64], xq: f64) -> f64 {
    if xq <= xs[0] { return ys[0]; }
    if xq >= xs[xs.len() - 1] { return ys[ys.len() - 1]; }
    let i = match xs.binary_search_by(|x| x.partial_cmp(&xq).unwrap()) {
        Ok(j) => return ys[j],
        Err(j) => j - 1,
    };
    let t = (xq - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + t * (ys[i + 1] - ys[i])
}

fn build_csv(rows: &[CycleRow]) -> String {
    let mut out = String::from("cycle,bit_index,v_dac_trial,v_hold,cmp,bit_value,err\n");
    for r in rows {
        out.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{},{:.6}\n",
            r.cycle, r.bit_index, r.v_dac, r.v_hold, r.cmp, r.bit_value, r.err
        ));
    }
    out
}

fn build_report(
    rows: &[CycleRow],
    vin: f64,
    vref: f64,
    n_bits: usize,
    final_code: u32,
    ideal_code: u32,
) -> String {
    let xs = rows.iter().map(|r| r.cycle.to_string()).collect::<Vec<_>>().join(", ");
    let y_vdac = rows.iter().map(|r| format!("{:.4}", r.v_dac)).collect::<Vec<_>>().join(", ");
    let y_vhold = rows.iter().map(|r| format!("{:.4}", r.v_hold)).collect::<Vec<_>>().join(", ");
    let y_bit = rows.iter().map(|r| r.bit_value.to_string()).collect::<Vec<_>>().join(", ");
    let y_cmp_bit: Vec<String> = rows.iter()
        .map(|r| if r.cmp >= vref / 2.0 { "1".into() } else { "0".into() })
        .collect();
    let y_cmp = y_cmp_bit.join(", ");
    let y_err = rows.iter().map(|r| format!("{:.5}", r.err)).collect::<Vec<_>>().join(", ");

    let mut md = String::new();
    md.push_str("# rlx-eda single-conversion SAR ADC trace\n\n");
    md.push_str("Circuit: `SarAdc<4>` (`spike-sar-adc`) — Sample/Hold + SarRegister + R-2R DAC + behavioral Comparator.\n\n");
    md.push_str(&format!(
        "Stimulus: `Vin = {:.3} V`, `Vref = {:.3} V`, N = {} bits.\n\n",
        vin, vref, n_bits
    ));
    md.push_str("Algorithm:\n\n");
    md.push_str("$$\\text{bit}_i \\leftarrow \\begin{cases} 1 & \\text{if } V_{hold} > V_{dac}(\\text{trial} = 1) \\\\ 0 & \\text{otherwise} \\end{cases}$$\n\n");
    md.push_str("Each cycle the SAR sets `bit_i = 1`, the DAC produces a trial voltage from the partial result, the comparator decides, and the capture pulse latches the decision. The trial bit walks from MSB down to LSB.\n\n");

    md.push_str("## Conversion outcome\n\n");
    md.push_str(&format!(
        "- ngspice transient: code = {} = `0b{:0>4b}`\n",
        final_code, final_code
    ));
    md.push_str(&format!(
        "- analytic ideal  : code = {} = `0b{:0>4b}`\n",
        ideal_code, ideal_code
    ));
    md.push_str(&format!(
        "- match: {}\n\n",
        if final_code == ideal_code { "✅" } else { "❌" }
    ));

    md.push_str("## Rendered charts\n\n");
    md.push_str("| DAC trajectory & V_hold | Per-bit decisions |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![Rendered DAC chart](assets/sar_trace/dac.svg) | ![Rendered bits chart](assets/sar_trace/bits.svg) |\n\n");
    md.push_str("| Comparator output | Quantization error |\n");
    md.push_str("| --- | --- |\n");
    md.push_str("| ![Rendered comparator chart](assets/sar_trace/comp.svg) | ![Rendered error chart](assets/sar_trace/error.svg) |\n\n");

    md.push_str("## Chart grid\n\n");
    md.push_str("| Row | Left panel | Right panel |\n");
    md.push_str("| --- | --- | --- |\n");
    md.push_str("| 1 | A. DAC trial voltage vs V_hold | B. Per-bit decision (kept/cleared) |\n");
    md.push_str("| 2 | C. Comparator output by cycle | D. Quantization error halving |\n\n");

    md.push_str("## A) DAC trajectory (binary search)\n\n");
    md.push_str("```mermaid\n");
    md.push_str("xychart-beta\n");
    md.push_str("  title \"DAC trial voltage vs sampled V_hold by SAR cycle\"\n");
    md.push_str(&format!("  x-axis \"cycle\" [{}]\n", xs));
    md.push_str("  y-axis \"voltage (V)\"\n");
    md.push_str(&format!("  line [{}]\n", y_vdac));
    md.push_str(&format!("  line [{}]\n", y_vhold));
    md.push_str("```\n\n");
    md.push_str("Legend:\n\n");
    md.push_str("- line 1: `V_dac` trial value (with current bit set to 1)\n");
    md.push_str("- line 2: `V_hold` (sampled `V_in`) — the comparator's reference\n\n");

    md.push_str("## B) Per-bit decisions\n\n");
    md.push_str("```mermaid\n");
    md.push_str("xychart-beta\n");
    md.push_str("  title \"Bit value latched after each cycle's capture pulse\"\n");
    md.push_str(&format!("  x-axis \"cycle\" [{}]\n", xs));
    md.push_str("  y-axis \"bit value\"\n");
    md.push_str(&format!("  bar [{}]\n", y_bit));
    md.push_str("```\n\n");
    md.push_str("Each bar is the latched decision for the bit being trialled in that cycle:\n\n");
    let bit_labels = rows.iter()
        .map(|r| format!("  - cycle {}: bit[{}] = {}", r.cycle, r.bit_index, r.bit_value))
        .collect::<Vec<_>>().join("\n");
    md.push_str(&bit_labels);
    md.push_str("\n\n");

    md.push_str("## C) Comparator output sequence\n\n");
    md.push_str("```mermaid\n");
    md.push_str("xychart-beta\n");
    md.push_str("  title \"Comparator output during each trial (1 = V_hold > V_dac)\"\n");
    md.push_str(&format!("  x-axis \"cycle\" [{}]\n", xs));
    md.push_str("  y-axis \"comp\"\n");
    md.push_str(&format!("  bar [{}]\n", y_cmp));
    md.push_str("```\n\n");
    md.push_str("The comparator output drives each bit's latched value: `cmp = 1` keeps the trial bit, `cmp = 0` clears it.\n\n");

    md.push_str("## D) Quantization error halving\n\n");
    md.push_str("```mermaid\n");
    md.push_str("xychart-beta\n");
    md.push_str("  title \"|V_hold − V_dac_committed| after each cycle (binary search → exponential decay)\"\n");
    md.push_str(&format!("  x-axis \"cycle\" [{}]\n", xs));
    md.push_str("  y-axis \"error (V)\"\n");
    md.push_str(&format!("  line [{}]\n", y_err));
    md.push_str("```\n\n");
    md.push_str("Binary search halves the residual at each cycle (in expectation). After N cycles the residual is bounded by 1 LSB = `Vref / 2^N`.\n\n");

    md.push_str("## Per-cycle trace\n\n");
    md.push_str("| cycle | bit | V_dac trial (V) | V_hold (V) | cmp (V) | bit value | err (V) |\n");
    md.push_str("| --- | --- | --- | --- | --- | --- | --- |\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | bit[{}] | {:.4} | {:.4} | {:.4} | {} | {:.4} |\n",
            r.cycle, r.bit_index, r.v_dac, r.v_hold, r.cmp, r.bit_value, r.err
        ));
    }

    md
}

// ── Custom SVG renderer (mirrors ml_trace's style) ────────────────────

struct LineSeries<'a> {
    name: &'a str,
    color: &'a str,
    values: &'a [f64],
}

fn write_rendered_svgs(rows: &[CycleRow], out_dir: &PathBuf) -> Result<(), Box<dyn Error>> {
    let xs: Vec<f64> = rows.iter().map(|r| r.cycle as f64).collect();
    let vdac: Vec<f64> = rows.iter().map(|r| r.v_dac).collect();
    let vhold: Vec<f64> = rows.iter().map(|r| r.v_hold).collect();
    let bit: Vec<f64> = rows.iter().map(|r| r.bit_value as f64).collect();
    let cmp_bit: Vec<f64> = rows.iter()
        .map(|r| if r.cmp >= 0.9 { 1.0 } else { 0.0 })
        .collect();
    let err: Vec<f64> = rows.iter().map(|r| r.err).collect();

    fs::write(
        out_dir.join("dac.svg"),
        line_chart_svg(
            "DAC trial voltage vs V_hold",
            "cycle",
            "voltage (V)",
            &xs,
            &[
                LineSeries { name: "V_dac (trial)", color: "#1d4ed8", values: &vdac },
                LineSeries { name: "V_hold",        color: "#dc2626", values: &vhold },
            ],
        ),
    )?;
    fs::write(
        out_dir.join("bits.svg"),
        line_chart_svg(
            "Per-bit decision",
            "cycle (MSB → LSB)",
            "bit value",
            &xs,
            &[LineSeries { name: "latched bit", color: "#0f766e", values: &bit }],
        ),
    )?;
    fs::write(
        out_dir.join("comp.svg"),
        line_chart_svg(
            "Comparator output sequence",
            "cycle",
            "cmp",
            &xs,
            &[LineSeries { name: "comparator", color: "#7c3aed", values: &cmp_bit }],
        ),
    )?;
    fs::write(
        out_dir.join("error.svg"),
        line_chart_svg(
            "Quantization error halving",
            "cycle",
            "|V_hold − V_dac_committed| (V)",
            &xs,
            &[LineSeries { name: "|err|", color: "#b45309", values: &err }],
        ),
    )?;
    Ok(())
}

fn line_chart_svg(
    title: &str,
    x_label: &str,
    y_label: &str,
    x: &[f64],
    series: &[LineSeries<'_>],
) -> String {
    let width = 920.0_f64;
    let height = 480.0_f64;
    let left = 78.0_f64;
    let right = 26.0_f64;
    let top = 56.0_f64;
    let bottom = 62.0_f64;

    let plot_w = width - left - right;
    let plot_h = height - top - bottom;

    let min_x = *x.first().unwrap_or(&0.0);
    let max_x = *x.last().unwrap_or(&1.0);
    let dx = (max_x - min_x).max(1.0);

    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for s in series {
        for &v in s.values {
            if v < min_y { min_y = v; }
            if v > max_y { max_y = v; }
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
    let dy = (max_y - min_y).max(1e-12);

    let map_x = |vx: f64| left + ((vx - min_x) / dx) * plot_w;
    let map_y = |vy: f64| top + (1.0 - (vy - min_y) / dy) * plot_h;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        width as i32, height as i32, width as i32, height as i32
    ));
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n");

    // Y grid + labels
    for i in 0..=6 {
        let t = i as f64 / 6.0;
        let yv = min_y + t * dy;
        let py = map_y(yv);
        svg.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>\n",
            left, py, left + plot_w, py
        ));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\" font-size=\"12\" fill=\"#374151\">{:.3}</text>\n",
            left - 8.0, py + 4.0, yv
        ));
    }

    // X grid + labels
    for i in 0..=6 {
        let t = i as f64 / 6.0;
        let xv = min_x + t * dx;
        let px = map_x(xv);
        svg.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>\n",
            px, top, px, top + plot_h
        ));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#374151\">{:.0}</text>\n",
            px, top + plot_h + 20.0, xv
        ));
    }

    // Axes
    svg.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#111827\" stroke-width=\"1.5\"/>\n",
        left, top + plot_h, left + plot_w, top + plot_h
    ));
    svg.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#111827\" stroke-width=\"1.5\"/>\n",
        left, top, left, top + plot_h
    ));

    // Title and axis labels
    svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"24\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#111827\">{}</text>\n",
        width / 2.0, title
    ));
    svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"middle\" font-size=\"13\" fill=\"#111827\">{}</text>\n",
        left + plot_w / 2.0, height - 16.0, x_label
    ));
    svg.push_str(&format!(
        "<text x=\"22\" y=\"{:.2}\" transform=\"rotate(-90 22 {:.2})\" text-anchor=\"middle\" font-size=\"13\" fill=\"#111827\">{}</text>\n",
        top + plot_h / 2.0, top + plot_h / 2.0, y_label
    ));

    // Series polylines + markers
    for s in series {
        let mut pts = String::new();
        for (i, &yv) in s.values.iter().enumerate() {
            let xv = x.get(i).copied().unwrap_or(i as f64);
            let px = map_x(xv);
            let py = map_y(yv);
            pts.push_str(&format!("{:.2},{:.2} ", px, py));
        }
        svg.push_str(&format!(
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2.2\"/>\n",
            pts.trim_end(), s.color
        ));
        for (i, &yv) in s.values.iter().enumerate() {
            let xv = x.get(i).copied().unwrap_or(i as f64);
            let px = map_x(xv);
            let py = map_y(yv);
            svg.push_str(&format!(
                "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"3.5\" fill=\"{}\" stroke=\"#ffffff\" stroke-width=\"1.5\"/>\n",
                px, py, s.color
            ));
        }
    }

    // Legend (top-right)
    let legend_x = left + plot_w - 200.0;
    let legend_y = top + 10.0;
    let legend_h = 26.0 + series.len() as f64 * 22.0;
    svg.push_str(&format!(
        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"190\" height=\"{:.2}\" rx=\"8\" fill=\"#f9fafb\" stroke=\"#d1d5db\"/>\n",
        legend_x, legend_y, legend_h
    ));
    svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"12\" font-weight=\"700\" fill=\"#111827\">Legend</text>\n",
        legend_x + 10.0, legend_y + 16.0
    ));
    for (i, s) in series.iter().enumerate() {
        let y = legend_y + 32.0 + i as f64 * 22.0;
        svg.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"3\"/>\n",
            legend_x + 10.0, y, legend_x + 36.0, y, s.color
        ));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"12\" fill=\"#111827\">{}</text>\n",
            legend_x + 44.0, y + 4.0, s.name
        ));
    }

    svg.push_str("</svg>\n");
    svg
}
