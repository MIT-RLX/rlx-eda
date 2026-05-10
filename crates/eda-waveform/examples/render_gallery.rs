//! Renders a small gallery of representative waveforms as PNG + SVG.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p eda-waveform --example render_gallery
//! ```
//!
//! Output goes to `crates/eda-waveform/docs/gallery/` next to the
//! crate's README, which embeds the same images. Re-run after
//! changing `plot.rs` to refresh the README's screenshots. Each
//! example produces both `<name>.png` and `<name>.svg` so you can
//! compare bitmap vs vector rendering side-by-side.
//!
//! The four examples cover the rendering surface:
//!   1. Single sine — simplest real waveform.
//!   2. Clock + sample/hold — multi-signal real, mimics the LTspice
//!      paper's S/H figure shape.
//!   3. SAR-style digital bit lines — multi-signal real with mostly-
//!      flat segments and step transitions; tests palette + legend.
//!   4. RC low-pass Bode magnitude — complex waveform on log-frequency.

use std::collections::BTreeMap;
use std::path::PathBuf;

use eda_waveform::plot::{png_to_path, svg_to_path, Layout, Marker, PlotConfig};
use eda_waveform::Waveform;

fn main() {
    // Resolve to <crate>/docs/gallery so the example writes alongside
    // the README that embeds the images. CARGO_MANIFEST_DIR is set by
    // cargo at build time and is the absolute path to the crate root.
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/gallery");
    std::fs::create_dir_all(&out_dir).unwrap();

    let sh_with_clock_markers = PlotConfig::new()
        .with_title("clock + sample/hold reconstruction (single pane)")
        .add_marker(Marker::Vertical { x: 20e-6, label: Some("sample".into()) })
        .add_marker(Marker::Vertical { x: 40e-6, label: Some("sample".into()) })
        .add_marker(Marker::Vertical { x: 60e-6, label: Some("sample".into()) })
        .add_marker(Marker::Vertical { x: 80e-6, label: Some("sample".into()) });
    let sh_stacked = PlotConfig::new()
        .with_title("clock + sample/hold reconstruction (stacked)")
        .with_layout(Layout::Stacked)
        .with_size(800, 720);
    let sar_with_threshold = PlotConfig::new()
        .with_title("SAR-style 4-bit output trace (with comparator threshold)")
        .add_marker(Marker::Horizontal { y: 0.5, label: Some("Vth".into()) });
    let bode_with_cutoff = PlotConfig::new()
        .with_title("RC low-pass Bode (fc = 1 kHz, mag + phase)")
        .with_size(800, 720)
        .add_marker(Marker::Vertical { x: 1000.0, label: Some("fc".into()) });

    let cases: Vec<(&str, Waveform, PlotConfig)> = vec![
        ("01_sine_1khz", sine_1khz(), titled("1 kHz sine, 1 V peak")),
        ("02_clock_and_sh", clock_and_sample_hold(), sh_with_clock_markers),
        ("03_clock_and_sh_stacked", clock_and_sample_hold(), sh_stacked),
        ("04_sar_bits", sar_bit_lines(), sar_with_threshold),
        ("05_rc_bode", rc_lowpass_bode(), bode_with_cutoff),
    ];

    println!("Writing gallery to {}", out_dir.display());
    for (name, wave, cfg) in &cases {
        let png = out_dir.join(format!("{name}.png"));
        let svg = out_dir.join(format!("{name}.svg"));
        png_to_path(wave, &png, cfg).expect("png");
        svg_to_path(wave, &svg, cfg).expect("svg");
        println!("  {}", png.display());
        println!("  {}", svg.display());
    }
    println!("\nAll plots written to {}", out_dir.display());
}

fn titled(s: &str) -> PlotConfig {
    PlotConfig { title: Some(s.into()), ..Default::default() }
}

fn sine_1khz() -> Waveform {
    let n = 500;
    let t_stop = 5e-3; // 5 ms ⇒ 5 cycles
    let axis: Vec<f64> = (0..=n).map(|i| i as f64 * t_stop / n as f64).collect();
    let y: Vec<f64> = axis
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * 1000.0 * t).sin())
        .collect();
    let mut signals = BTreeMap::new();
    signals.insert("v(in)".into(), y);
    Waveform::Real { axis_name: "time (s)".into(), axis, signals }
}

fn clock_and_sample_hold() -> Waveform {
    // 100 µs window. 50 kHz square clock; sampled signal = 100 Hz cosine
    // tracked at clock rising edges, held flat between.
    let n = 1000;
    let t_stop = 100e-6;
    let axis: Vec<f64> = (0..=n).map(|i| i as f64 * t_stop / n as f64).collect();
    let f_clk = 50e3;
    let f_in = 10e3;

    let clk: Vec<f64> = axis
        .iter()
        .map(|t| if (t * f_clk).fract() < 0.5 { 1.0 } else { 0.0 })
        .collect();
    let v_in: Vec<f64> = axis
        .iter()
        .map(|t| 0.5 + 0.45 * (2.0 * std::f64::consts::PI * f_in * t).sin())
        .collect();
    let mut held = 0.5_f64;
    let mut last_clk = 0.0_f64;
    let v_sh: Vec<f64> = axis
        .iter()
        .zip(v_in.iter())
        .zip(clk.iter())
        .map(|((_, &v), &c)| {
            if c > 0.5 && last_clk <= 0.5 {
                held = v;
            }
            last_clk = c;
            held
        })
        .collect();

    let mut signals = BTreeMap::new();
    signals.insert("clk".into(), clk);
    signals.insert("v(in)".into(), v_in);
    signals.insert("v(sh)".into(), v_sh);
    Waveform::Real { axis_name: "time (s)".into(), axis, signals }
}

fn sar_bit_lines() -> Waveform {
    // 8 µs window, 4 SAR-output bits. Each bit transitions on
    // successive 1-µs ticks like a binary search settling: b3 first,
    // then b2, b1, b0. The pattern resolves to "1011" by t = 5 µs.
    let n = 800;
    let t_stop = 8e-6;
    let axis: Vec<f64> = (0..=n).map(|i| i as f64 * t_stop / n as f64).collect();

    let bit_at = |t: f64, settle_at: f64, value: u8| -> f64 {
        if t >= settle_at && value == 1 { 1.0 } else { 0.0 }
    };
    let b3: Vec<f64> = axis.iter().map(|t| bit_at(*t, 1e-6, 1)).collect();
    let b2: Vec<f64> = axis.iter().map(|t| bit_at(*t, 2e-6, 0)).collect();
    let b1: Vec<f64> = axis.iter().map(|t| bit_at(*t, 3e-6, 1)).collect();
    let b0: Vec<f64> = axis.iter().map(|t| bit_at(*t, 4e-6, 1)).collect();

    let mut signals = BTreeMap::new();
    signals.insert("out3".into(), b3);
    signals.insert("out2".into(), b2);
    signals.insert("out1".into(), b1);
    signals.insert("out0".into(), b0);
    Waveform::Real { axis_name: "time (s)".into(), axis, signals }
}

fn rc_lowpass_bode() -> Waveform {
    // RC low-pass: H(s) = 1 / (1 + s · R · C). With R=1k, C=159nF the
    // -3 dB cutoff is f_c = 1 / (2π R C) ≈ 1 kHz. Sweep 10 Hz .. 1 MHz.
    let r = 1.0e3;
    let c = 159.0e-9;
    let n = 200;
    let f0 = 10.0_f64;
    let f1 = 1.0e6_f64;
    let log_f0 = f0.log10();
    let log_f1 = f1.log10();
    let axis: Vec<f64> = (0..=n)
        .map(|i| 10f64.powf(log_f0 + (log_f1 - log_f0) * i as f64 / n as f64))
        .collect();
    // H(jω) = 1 / (1 + j ω R C)
    let resp: Vec<(f64, f64)> = axis
        .iter()
        .map(|f| {
            let w = 2.0 * std::f64::consts::PI * f;
            let denom_re = 1.0;
            let denom_im = w * r * c;
            let mag2 = denom_re * denom_re + denom_im * denom_im;
            // 1 / (a + jb) = (a - jb) / (a² + b²)
            (denom_re / mag2, -denom_im / mag2)
        })
        .collect();

    let mut signals = BTreeMap::new();
    signals.insert("v(out)".into(), resp);
    Waveform::Complex { axis_name: "frequency (Hz)".into(), axis, signals }
}

