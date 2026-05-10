//! End-to-end demo: Op::Scan + per-draw R MC on Apple GPU.
//!
//! Build the auto-derived RC body with R promoted as a per-draw
//! input, wrap in `scan_trajectory(length=250)`, vmap with batched
//! inputs `["v0", "R1_…"]` so each draw gets its own initial v_C
//! AND its own R value, dispatch in one MLX call.
//!
//! Compares per-draw final v(5τ) to the closed-form
//!   `v(5τ) = v_init · (1 + h/(R·C))^(-n_steps)`
//! since BE on a linear RC is exact at every step.
//!
//! Run with: `cargo run --example scan_rc_per_draw_r_mc -p eda-mna --release`

#[cfg(target_os = "macos")]
mod bench {
    use std::collections::HashMap;
    use std::time::Instant;

    use eda_hir::Block;
    use eda_mna::{
        build_linear_be_step_body_with_mc_params, Circuit, NetId,
    };
    use rlx_ir::{DType, Graph, Shape};
    use rlx_mlx::{MlxExecutable, MlxMode};
    use spike_divider_block::{Capacitor, Resistor};

    const C_FARADS: f32 = 1e-9;

    fn build_outer(body: &Graph, n_steps: u32, n_draws: usize, batched_names: &[&str]) -> Graph {
        let mut g = Graph::new("rc_per_draw_r_outer");
        let init = g.input("v0", Shape::new(&[1], DType::F32));
        let _ = g.scan_trajectory(init, body.clone(), n_steps);
        // Note: scan_trajectory binds the body's ONE Op::Input as carry.
        // For our with-mc-params body, additional Op::Inputs (R1) are
        // body bcasts — they must be supplied as scan bcasts via
        // scan_with_bcasts_and_xs. scan_trajectory is the carry-only
        // variant; we need the bcast variant. Construct manually below.
        unimplemented!("see build_outer_with_bcasts");
    }

    fn build_outer_with_bcasts(
        body: &Graph,
        n_steps: u32,
        bcast_names: &[&str],
    ) -> Graph {
        use rlx_ir::Op;
        // Outer graph: takes v0 [1] + each bcast [1], wraps body in
        // Op::Scan with save_trajectory=true + num_bcast = bcast_names.len().
        let mut g = Graph::new("rc_per_draw_r_outer");
        let init = g.input("v0", Shape::new(&[1], DType::F32));
        let mut bcast_ids = Vec::with_capacity(bcast_names.len());
        for nm in bcast_names {
            bcast_ids.push(g.input((*nm).to_string(), Shape::new(&[1], DType::F32)));
        }
        // Trajectory output shape: [length, *carry_shape] = [length, 1].
        let traj_shape = Shape::new(&[n_steps as usize, 1], DType::F32);
        let mut inputs = vec![init];
        inputs.extend_from_slice(&bcast_ids);
        let traj = g.add_node(
            Op::Scan {
                body: Box::new(body.clone()),
                length: n_steps,
                save_trajectory: true,
                num_bcast: bcast_names.len() as u32,
                num_xs: 0,
                num_checkpoints: 0,
            },
            inputs,
            traj_shape,
        );
        g.set_outputs(vec![traj]);
        g
    }

    fn closed_form_v_at_t(v0: f32, r: f32, c: f32, dt: f32, n_steps: u32) -> f32 {
        let coeff = 1.0 / (1.0 + dt / (r * c));
        v0 * coeff.powi(n_steps as i32)
    }

    pub fn run() {
        // Topology: vmid -- R1 -- gnd, vmid -- C1 -- gnd. Single
        // unknown vmid; per-draw R value, per-draw initial v_C.
        let mut c = Circuit::new();
        let vmid = c.alloc_unknown_net();
        let r   = Resistor { length: 10_000, id: "R1".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
        c.add_device(r.clone(),    &[vmid, NetId::GND]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);

        let r_key = Block::name(&r);
        let c_key = format!("{}_C", Block::name(&cap));

        let mut params: HashMap<String, f32> = HashMap::new();
        params.insert(c_key.clone(), C_FARADS);
        // R1 will be MC'd, intentionally not in `params` (it's
        // skipped from set_param via the mc_param promotion).

        let r_nominal = 1_000.0_f32;
        let tau_nom   = r_nominal * C_FARADS;
        let dt        = tau_nom / 50.0;
        let n_steps: u32 = 250;
        let _ = build_outer; // keep the helper for documentation

        // Build the body with R as mc_param.
        let auto = build_linear_be_step_body_with_mc_params(
            &c, &params, &[r_key.as_str()], dt,
        ).expect("auto-build with R as mc_param");
        assert_eq!(auto.n, 1);

        // Wrap in Scan with R as a bcast.
        let scan_g = build_outer_with_bcasts(&auto.body, n_steps, &[r_key.as_str()]);
        // vmap so v0 + R both become per-draw [B, 1].
        let batched = rlx_opt::vmap::vmap(&scan_g, &["v0", r_key.as_str()], 1);
        let _ = batched;    // shape sanity only — we'll re-vmap per-N below

        println!("== Op::Scan + per-draw R MC on Apple GPU ==\n");
        println!(
            "{:>6}  {:>10}  {:>14}  {:>14}",
            "N", "scan (ms)", "max |Δ| analytic", "ms / (draw × step)");

        for &n_draws in &[1usize, 4, 16, 64, 256, 1024] {
            // Per-draw v0 and R values.
            let v_init: Vec<f32> = (0..n_draws).map(|i| {
                0.05 + 0.95 * (1.0 - i as f32 / (n_draws.max(2) - 1) as f32)
            }).collect();
            let r_draws: Vec<f32> = (0..n_draws).map(|i| {
                // R spans 0.5×–2× nominal.
                r_nominal * (0.5 + 1.5 * i as f32 / (n_draws.max(2) - 1) as f32)
            }).collect();

            let scan_g = build_outer_with_bcasts(&auto.body, n_steps, &[r_key.as_str()]);
            let g_b = rlx_opt::vmap::vmap(&scan_g, &["v0", r_key.as_str()], n_draws);

            let mut exe = MlxExecutable::compile_with_mode(g_b, MlxMode::Lazy);
            exe.set_param(&c_key, &[C_FARADS]);
            // Warmup (compile + first dispatch).
            let _ = exe.run(&[
                ("v0",          v_init.as_slice()),
                (r_key.as_str(), r_draws.as_slice()),
            ]);
            let t0 = Instant::now();
            let outs = exe.run(&[
                ("v0",          v_init.as_slice()),
                (r_key.as_str(), r_draws.as_slice()),
            ]);
            let t_ms = t0.elapsed().as_secs_f64() * 1e3;
            let traj = &outs[0];   // [N, length, 1] flattened

            // Per-draw final-step value vs analytic.
            let mut max_drift = 0.0_f32;
            for d in 0..n_draws {
                let v_end_scan = traj[d * n_steps as usize + (n_steps as usize - 1)];
                let v_end_analytic = closed_form_v_at_t(
                    v_init[d], r_draws[d], C_FARADS, dt, n_steps,
                );
                max_drift = max_drift.max((v_end_scan - v_end_analytic).abs());
            }
            let per_unit = t_ms / (n_draws as f64 * n_steps as f64);
            println!(
                "{:>6}  {:>10.3}  {:>14.3e}  {:>14.3e}",
                n_draws, t_ms, max_drift, per_unit);
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    bench::run();
    #[cfg(not(target_os = "macos"))]
    eprintln!("requires macOS (MLX backend)");
}
