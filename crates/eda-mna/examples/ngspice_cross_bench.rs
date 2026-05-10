//! ngspice cross-bench: headline numbers for every analysis type
//! eda-mna currently supports vs ngspice on the same circuits.
//!
//! Run with: `cargo run --example ngspice_cross_bench -p eda-mna --release`

#[cfg(target_os = "macos")]
mod bench {
    use std::collections::HashMap;
    use std::time::Instant;

    use eda_hir::Block;
    use eda_extern_ngspice::{
        AcAnalysis, Invoker, LocalBinary, OutputRequest, TransientAnalysis,
    };
    use eda_mna::{
        batched_solve_dc, batched_transient_from, build_ac_response_graph,
        build_linear_be_step_body, Circuit, NetId, NewtonOptions,
    };
    use rlx_ir::{DType, Graph, Shape};
    use rlx_mlx::{MlxExecutable, MlxMode};
    use spike_divider_block::{Capacitor, Resistor};

    // ── DC bench: linear divider MC over per-draw R values ──
    fn bench_dc(ng: &LocalBinary) {
        println!("== DC: linear divider with per-draw R ==");
        let v_dd = 1.8_f32;
        let r2_o = 3_000.0_f32;

        for &n in &[16usize, 64, 256] {
            let r1_draws: Vec<f32> = (0..n).map(|i| {
                500.0 + 1500.0 * i as f32 / (n.max(2) - 1) as f32
            }).collect();

            // eda-mna setup: divider with R1, R2 between v_dd and gnd, vmid as unknown.
            // R1 baked as mc-able via ParamSweep is harder; here MC is just over per-draw R1.
            // Use solve_dc per-draw for both paths to keep them apples-to-apples for DC.
            let mut c = Circuit::new();
            let v_in_net = c.alloc_boundary_net();
            let vmid     = c.alloc_unknown_net();
            let r1 = Resistor { length: 10_000, id: "R1".into() };
            let r2 = Resistor { length: 30_000, id: "R2".into() };
            c.add_device(r1.clone(), &[v_in_net, vmid]);
            c.add_device(r2.clone(), &[vmid, NetId::GND]);

            // batched_solve_dc with R1 as mc_param. R1's value lives in the
            // "Resistor_R1_L10000" param key; MC bypasses set_param and
            // routes through mc_params Vec.
            let r1_key = Block::name(&r1);
            let r2_key = Block::name(&r2);
            let mut params: HashMap<String, f32> = HashMap::new();
            params.insert(r2_key, r2_o);
            let mut mc_params: HashMap<String, Vec<f32>> = HashMap::new();
            mc_params.insert(r1_key.clone(), r1_draws.clone());
            let mut boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
            boundary.insert(v_in_net, vec![v_dd; n]);
            let opt = NewtonOptions { init: 0.5 * v_dd, ..NewtonOptions::default() };

            let t0 = Instant::now();
            let _ = batched_solve_dc(&c, n, &params, &mc_params, &boundary, opt);
            let t_eda = t0.elapsed().as_secs_f64() * 1e3;

            // ngspice per-draw fork.
            let t0 = Instant::now();
            for d in 0..n {
                let deck = format!(
                    "* divider DC sweep\n\
                     .options noecho\n\
                     V1 vin 0 DC {v_dd}\n\
                     R1 vin vmid {r1}\n\
                     R2 vmid 0 {r2}\n\
                     .op\n\
                     .end\n",
                    r1 = r1_draws[d] as f64, r2 = r2_o as f64,
                );
                let _ = ng.run_dc(&deck, &[OutputRequest::NodeVoltage("vmid".into())])
                    .expect("ngspice DC");
            }
            let t_ng = t0.elapsed().as_secs_f64() * 1e3;
            println!("  N={:>4}  eda-mna {:>8.2} ms   ngspice {:>9.2} ms   speedup {:>6.1}×",
                n, t_eda, t_ng, t_ng / t_eda.max(1e-6));
        }
    }

    // ── Transient (BE Newton): RC discharge per-draw IC ──
    fn bench_transient_newton(ng: &LocalBinary) {
        println!();
        println!("== Transient (BE Newton): RC discharge per-draw IC ==");
        let r_o   = 1_000.0_f32;
        let c_f   = 1e-9_f32;
        let dt    = r_o * c_f / 50.0;
        let n_steps: usize = 250;
        let t_end = n_steps as f32 * dt;

        for &n in &[16usize, 64, 256] {
            let mut c = Circuit::new();
            let vmid = c.alloc_unknown_net();
            let r   = Resistor { length: 10_000, id: "R1".into() };
            let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
            c.add_device(r.clone(),    &[vmid, NetId::GND]);
            c.add_storage(cap.clone(), [vmid, NetId::GND]);

            let mut params = HashMap::new();
            params.insert(Block::name(&r), r_o);
            params.insert(format!("{}_C", Block::name(&cap)), c_f);
            let v_init: Vec<f32> = (0..n).map(|i| {
                0.05 + 0.95 * (1.0 - i as f32 / (n.max(2) - 1) as f32)
            }).collect();
            let mut ic: HashMap<NetId, Vec<f32>> = HashMap::new();
            ic.insert(vmid, v_init.clone());
            let boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
            let mc_params: HashMap<String, Vec<f32>> = HashMap::new();
            let opt = NewtonOptions::default();

            let t0 = Instant::now();
            let _ = batched_transient_from(
                &c, n, &params, &mc_params, &boundary, &ic, dt, n_steps, opt,
            );
            let t_eda = t0.elapsed().as_secs_f64() * 1e3;

            let analysis = TransientAnalysis::new(dt as f64, t_end as f64);
            let t0 = Instant::now();
            for d in 0..n {
                let deck = format!(
                    "* RC discharge\n\
                     .options noecho method=gear maxord=1\n\
                     .ic v(vmid)={v}\n\
                     R1 vmid 0 {r}\n\
                     C1 vmid 0 {c}\n\
                     .end\n",
                    v = v_init[d], r = r_o as f64, c = c_f as f64,
                );
                let deck = deck.replace(".end\n",
                    &format!(".tran {} {} uic\n.end\n", dt as f64, t_end as f64));
                let _ = ng.run_transient_trace(
                    &deck, &analysis,
                    &[OutputRequest::NodeVoltage("vmid".into())],
                ).expect("ngspice tran");
            }
            let t_ng = t0.elapsed().as_secs_f64() * 1e3;
            println!("  N={:>4}  eda-mna {:>8.2} ms   ngspice {:>9.2} ms   speedup {:>6.1}×",
                n, t_eda, t_ng, t_ng / t_eda.max(1e-6));
        }
    }

    // ── Scan-folded transient: RC discharge auto-derived body ──
    fn bench_transient_scan(ng: &LocalBinary) {
        println!();
        println!("== Scan-folded transient (linear): same RC discharge ==");
        let r_o   = 1_000.0_f32;
        let c_f   = 1e-9_f32;
        let dt    = r_o * c_f / 50.0;
        let n_steps: u32 = 250;
        let t_end = n_steps as f32 * dt;

        let mut c = Circuit::new();
        let vmid = c.alloc_unknown_net();
        let r   = Resistor { length: 10_000, id: "R1".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
        c.add_device(r.clone(),    &[vmid, NetId::GND]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);
        let mut params = HashMap::new();
        params.insert(Block::name(&r), r_o);
        params.insert(format!("{}_C", Block::name(&cap)), c_f);

        let auto = build_linear_be_step_body(&c, &params, dt).expect("body");
        // N values capped — N=4096 forks of ngspice would take ~85s.
        for &n in &[64usize, 256, 1024] {
            let v_init: Vec<f32> = (0..n).map(|i| {
                0.05 + 0.95 * (1.0 - i as f32 / (n.max(2) - 1) as f32)
            }).collect();

            let mut g = Graph::new("rc_scan_outer");
            let init = g.input("v0", Shape::new(&[1], DType::F32));
            let traj = g.scan_trajectory(init, auto.body.clone(), n_steps);
            g.set_outputs(vec![traj]);
            let g_b = rlx_opt::vmap::vmap(&g, &["v0"], n);

            let mut exe = MlxExecutable::compile_with_mode(g_b, MlxMode::Lazy);
            let _ = exe.run(&[("v0", v_init.as_slice())]);    // warmup
            let t0 = Instant::now();
            let _ = exe.run(&[("v0", v_init.as_slice())]);
            let t_eda = t0.elapsed().as_secs_f64() * 1e3;

            // Real ngspice fork per draw.
            let analysis = TransientAnalysis::new(dt as f64, t_end as f64);
            let t0 = Instant::now();
            for d in 0..n {
                let deck = format!(
                    "* RC discharge\n\
                     .options noecho method=gear maxord=1\n\
                     .ic v(vmid)={v}\n\
                     R1 vmid 0 {r}\n\
                     C1 vmid 0 {c}\n\
                     .end\n",
                    v = v_init[d], r = r_o as f64, c = c_f as f64,
                );
                let deck = deck.replace(".end\n",
                    &format!(".tran {} {} uic\n.end\n", dt as f64, t_end as f64));
                let _ = ng.run_transient_trace(
                    &deck, &analysis,
                    &[OutputRequest::NodeVoltage("vmid".into())],
                ).expect("ngspice");
            }
            let t_ng = t0.elapsed().as_secs_f64() * 1e3;
            println!("  N={:>5}  eda-mna {:>8.2} ms   ngspice {:>10.2} ms   speedup {:>7.1}×",
                n, t_eda, t_ng, t_ng / t_eda.max(1e-6));
        }
    }

    // ── AC sweep: RC low-pass ──
    // ngspice does the entire AC sweep in ONE process via `.ac dec`,
    // so the natural comparison is one-process eda-mna vmap vs
    // one-process ngspice .ac. Per-fork-per-frequency would unfairly
    // penalise ngspice — its native sweep is what users actually run.
    fn bench_ac(ng: &LocalBinary) {
        println!();
        println!("== AC sweep: RC low-pass, vmap over frequency vs ngspice .ac dec ==");
        let r_o = 1_000.0_f32;
        let c_f = 1e-9_f32;
        let mut c = Circuit::new();
        let v_in = c.alloc_boundary_net();
        let vmid = c.alloc_unknown_net();
        let r = Resistor { length: 10_000, id: "R".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C".into() };
        c.add_device(r.clone(), &[v_in, vmid]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);
        let mut params = HashMap::new();
        params.insert(Block::name(&r), r_o);
        params.insert(format!("{}_C", Block::name(&cap)), c_f);
        let ac = build_ac_response_graph(&c, &params, v_in).expect("ac");

        let f0 = 1.0_f32 / (2.0 * std::f32::consts::PI * r_o * c_f);
        // ngspice `.ac dec n_per_decade fstart fstop` covers
        // ceil(log10(fstop/fstart)) · n_per_decade points. Our sweep is
        // 4 decades; pick n_per_decade to land near each target N.
        for &n in &[64usize, 256, 1024, 4096] {
            // eda-mna: vmap'd over N omegas.
            let omegas: Vec<f32> = (0..n).map(|i| {
                let log = -2.0 + 4.0 * i as f32 / (n.max(2) - 1) as f32;
                2.0 * std::f32::consts::PI * f0 * 10f32.powf(log)
            }).collect();
            let g_b = rlx_opt::vmap::vmap(&ac.graph, &["omega"], n);
            let mut exe = MlxExecutable::compile_with_mode(g_b, MlxMode::Lazy);
            let _ = exe.run(&[("omega", omegas.as_slice())]);
            let t0 = Instant::now();
            let _ = exe.run(&[("omega", omegas.as_slice())]);
            let t_eda = t0.elapsed().as_secs_f64() * 1e3;

            // ngspice: ONE `.ac dec` deck covering ~N points.
            let n_per_decade = ((n as f64) / 4.0).round().max(1.0) as usize;
            let f_start = (f0 as f64) * 1e-2;
            let f_stop  = (f0 as f64) * 1e2;
            let analysis = AcAnalysis::dec(n_per_decade, f_start, f_stop);
            let deck = format!(
                "* RC low-pass AC sweep\n\
                 .options noecho\n\
                 V1 vin 0 DC 0 AC 1\n\
                 R1 vin vmid {r}\n\
                 C1 vmid 0 {c}\n\
                 .end\n",
                r = r_o as f64, c = c_f as f64,
            );
            let t0 = Instant::now();
            let trace = ng.run_ac(&deck, &analysis,
                &[OutputRequest::NodeVoltage("vmid".into())]
            ).expect("ngspice ac");
            let t_ng = t0.elapsed().as_secs_f64() * 1e3;
            let n_freqs_actual = trace.frequency.len();
            println!("  N={:>5}  eda-mna {:>8.2} ms   ngspice {:>8.2} ms ({} pts)   speedup {:>7.1}×",
                n, t_eda, t_ng, n_freqs_actual,
                t_ng / t_eda.max(1e-6));
        }
    }

    pub fn run() {
        let ng = LocalBinary::from_env().expect("ngspice on PATH");
        bench_dc(&ng);
        bench_transient_newton(&ng);
        bench_transient_scan(&ng);
        bench_ac(&ng);
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    bench::run();
    #[cfg(not(target_os = "macos"))]
    eprintln!("requires macOS");
}
