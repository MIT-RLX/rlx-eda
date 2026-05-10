//! Batched AC sweep on Apple GPU.
//!
//! Build the auto-derived AC response graph for an RC low-pass,
//! vmap with `omega` as the batched input, dispatch in one MLX
//! call to compute the entire Bode plot.
//!
//! Run: `cargo run --example ac_sweep_mlx -p eda-mna --release`

#[cfg(target_os = "macos")]
mod bench {
    use std::collections::HashMap;
    use std::time::Instant;

    use eda_hir::Block;
    use eda_mna::{build_ac_response_graph, Circuit, NetId};
    use rlx_mlx::{MlxExecutable, MlxMode};
    use spike_divider_block::{Capacitor, Resistor};

    pub fn run() {
        let r_ohms   = 1_000.0_f32;
        let c_farads = 1e-9_f32;

        let mut c = Circuit::new();
        let v_in = c.alloc_boundary_net();
        let vmid = c.alloc_unknown_net();
        let r = Resistor { length: 10_000, id: "R".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C".into() };
        c.add_device(r.clone(), &[v_in, vmid]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);

        let mut params = HashMap::new();
        params.insert(Block::name(&r), r_ohms);
        params.insert(format!("{}_C", Block::name(&cap)), c_farads);

        let ac = build_ac_response_graph(&c, &params, v_in)
            .expect("AC graph");
        println!("Extracted: G = {:.3e} S, C = {:.3e} F, b_ac = {:.3e} A",
            ac.g, ac.c, ac.b_ac);

        // Vmap over `omega`. The graph has one input and two outputs;
        // vmap'd version takes [N] omega → [N] V_re, [N] V_im.
        println!();
        println!("== Batched AC sweep ==\n");
        println!("{:>6}  {:>14}  {:>14}  {:>14}",
            "N",  "scan (ms)", "ms / point",  "max rel err");

        let f0 = 1.0_f32 / (2.0 * std::f32::consts::PI * r_ohms * c_farads);
        for &n in &[8usize, 32, 128, 512, 2048, 8192] {
            // Logarithmic frequency sweep across 4 decades centered on f0.
            let omegas: Vec<f32> = (0..n).map(|i| {
                let log_decade = -2.0 + 4.0 * i as f32 / (n.max(2) - 1) as f32;
                let f = f0 * 10f32.powf(log_decade);
                2.0 * std::f32::consts::PI * f
            }).collect();

            let g_b = rlx_opt::vmap::vmap(&ac.graph, &["omega"], n);
            let mut exe = MlxExecutable::compile_with_mode(g_b, MlxMode::Lazy);
            // Warmup.
            let _ = exe.run(&[("omega", omegas.as_slice())]);
            let t0 = Instant::now();
            let outs = exe.run(&[("omega", omegas.as_slice())]);
            let t_ms = t0.elapsed().as_secs_f64() * 1e3;
            let v_re = &outs[0];
            let v_im = &outs[1];

            // Compare each point to analytic 1/sqrt(1+(ωRC)²).
            let mut max_rel = 0.0_f32;
            for i in 0..n {
                let mag = (v_re[i] * v_re[i] + v_im[i] * v_im[i]).sqrt();
                let analytic = 1.0_f32 / (1.0 + (omegas[i] * r_ohms * c_farads).powi(2)).sqrt();
                let rel = (mag - analytic).abs() / analytic.max(1e-12);
                if rel > max_rel { max_rel = rel; }
            }
            println!("{:>6}  {:>14.3}  {:>14.5}  {:>14.3e}",
                n, t_ms, t_ms / n as f64, max_rel);
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    bench::run();
    #[cfg(not(target_os = "macos"))]
    eprintln!("requires macOS (MLX backend)");
}
