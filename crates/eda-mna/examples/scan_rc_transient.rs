//! Phase-5D MVP: whole RC transient folded into one `Op::Scan` graph,
//! vmap'd over draws, dispatched in a single MLX call.
//!
//! Run with: `cargo run --example scan_rc_transient -p eda-mna --release`
//!
//! ## What this demonstrates
//!
//! Instead of 250 BE steps × N draws of Rust-side orchestration
//! (each doing residual eval + Jacobian eval + inner solve + line
//! search), the entire transient becomes:
//!
//!   1. ONE rlx graph built from a per-step body (carry → next carry)
//!      wrapped in `Op::Scan` with `length = n_steps`.
//!   2. `vmap` lifts that to `[N, length, *carry]`.
//!   3. ONE `MlxExecutable` dispatch computes the full per-draw
//!      waveform on the Apple GPU.
//!
//! Compared to the `batched_transient_from` path (which still does
//! the per-step Newton orchestration in Rust), this should drop
//! transient runtime from O(n_steps × Rust-overhead) to O(one MLX
//! dispatch).
//!
//! ## Scope (linear-only MVP)
//!
//! The body graph is **hand-coded** for a 1-net RC discharge using
//! its closed-form BE update `v_new = v_prev / (1 + h/(RC))`. That's
//! exact for this circuit — no Newton iteration needed inside the
//! body. The same architecture extends to arbitrary linear MNA via
//! a pre-computed `K_inv·b` step body, and to nonlinear circuits via
//! either fixed-iter Newton inside the body or `Op::While` (each
//! their own multi-day project).

#[cfg(target_os = "macos")]
mod bench {
    use std::collections::HashMap;
    use std::time::Instant;

    use eda_hir::Block;
    use eda_mna::{
        batched_transient_from, build_linear_be_step_body, Circuit, NetId,
        NewtonOptions,
    };
    use rlx_ir::{op::BinaryOp, DType, Graph, NodeId, Op, Shape};
    use rlx_mlx::{MlxExecutable, MlxMode};
    use spike_divider_block::{Capacitor, Resistor};

    const R_OHMS:   f32 = 1_000.0;
    const C_FARADS: f32 = 1e-9;

    fn s_scalar() -> Shape { Shape::new(&[1], DType::F32) }

    fn const_scalar(g: &mut Graph, x: f32) -> NodeId {
        g.add_node(Op::Constant { data: x.to_le_bytes().to_vec() }, vec![], s_scalar())
    }

    /// Per-step BE body for the linear RC. Closed-form:
    ///   `v_new = v_prev / (1 + h/(RC))`.
    /// Body must have exactly one `Op::Input` (the carry) and one
    /// output (next carry) per `Graph::scan_trajectory` contract.
    fn build_rc_body(h_over_rc: f32) -> Graph {
        let mut g = Graph::new("rc_be_body");
        let v_prev = g.input("carry", s_scalar());
        let coeff = const_scalar(&mut g, 1.0 / (1.0 + h_over_rc));
        let v_new = g.binary(BinaryOp::Mul, v_prev, coeff, s_scalar());
        g.set_outputs(vec![v_new]);
        g
    }

    /// Outer transient graph: `v0` input → scan over `n_steps` →
    /// trajectory `[n_steps, 1]`.
    fn build_rc_transient_graph(h_over_rc: f32, n_steps: u32) -> Graph {
        let mut g = Graph::new("rc_transient_scan");
        let init = g.input("v0", s_scalar());
        let body = build_rc_body(h_over_rc);
        let traj = g.scan_trajectory(init, body, n_steps);
        g.set_outputs(vec![traj]);
        g
    }

    /// vmap the scan graph over a batch axis of size `n_draws`.
    /// `v0` input becomes `[n_draws, 1]`; output becomes
    /// `[n_draws, n_steps, 1]`.
    fn build_batched(h_over_rc: f32, n_steps: u32, n_draws: usize) -> Graph {
        let scalar_g = build_rc_transient_graph(h_over_rc, n_steps);
        rlx_opt::vmap::vmap(&scalar_g, &["v0"], n_draws)
    }

    pub fn run() {
        let tau     = R_OHMS * C_FARADS;
        let dt      = tau / 50.0;
        let n_steps: u32 = 250;
        let h_over_rc = dt / (R_OHMS * C_FARADS);     // = 1/50

        // ── Reference: batched_transient_from on the same RC. ──
        let mut c = Circuit::new();
        let vmid = c.alloc_unknown_net();
        let r   = Resistor { length: 10_000, id: "R1".into() };
        let cap = Capacitor { plate_size: 2_000, id: "C1".into() };
        c.add_device(r.clone(),    &[vmid, NetId::GND]);
        c.add_storage(cap.clone(), [vmid, NetId::GND]);
        let mut params = HashMap::new();
        params.insert(Block::name(&r), R_OHMS);
        params.insert(format!("{}_C", Block::name(&cap)), C_FARADS);

        println!("== Op::Scan-folded transient: hand-coded body vs auto-derived body vs `batched_transient_from` ==\n");
        println!(
            "{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}",
            "N", "hand (ms)", "auto (ms)", "newton (ms)",
            "h-spdup", "a-spdup", "max |Δ| auto");

        // Auto-derived body — built ONCE from the Circuit + params.
        // Same body works for every N because vmap lifts on dispatch.
        let auto = build_linear_be_step_body(&c, &params, dt)
            .expect("auto-derive linear body from circuit");
        // Single-unknown-only check for this demo; assert matches.
        assert_eq!(auto.n, 1, "auto demo expects 1 unknown");

        // Wrap auto body in scan_trajectory + vmap. Compile once,
        // re-bind input across draws.
        fn build_auto_batched(
            auto_body: &rlx_ir::Graph, n_steps: u32, n_draws: usize,
        ) -> rlx_ir::Graph {
            use rlx_ir::{DType, Graph, Shape};
            let mut g = Graph::new("auto_rc_transient_scan");
            let init = g.input("v0", Shape::new(&[1], DType::F32));
            let traj = g.scan_trajectory(init, auto_body.clone(), n_steps);
            g.set_outputs(vec![traj]);
            rlx_opt::vmap::vmap(&g, &["v0"], n_draws)
        }

        for &n_draws in &[1usize, 4, 16, 64, 256, 1024] {
            // Per-draw initial v_C: exponential spread, distinct per draw.
            let v_init: Vec<f32> = (0..n_draws).map(|i| {
                let t = i as f32 / (n_draws.max(2) - 1) as f32;
                0.05 + 0.95 * (1.0 - t)
            }).collect();

            // ── Path A: HAND-CODED body on Op::Scan ──
            let g_hand = build_batched(h_over_rc, n_steps, n_draws);
            let mut exe_hand = MlxExecutable::compile_with_mode(g_hand, MlxMode::Lazy);
            let _ = exe_hand.run(&[("v0", v_init.as_slice())]);    // warmup
            let t0 = Instant::now();
            let outs_hand = exe_hand.run(&[("v0", v_init.as_slice())]);
            let t_hand = t0.elapsed().as_secs_f64() * 1e3;
            let traj_hand = &outs_hand[0];

            // ── Path B: AUTO-DERIVED body from Circuit on Op::Scan ──
            let g_auto = build_auto_batched(&auto.body, n_steps, n_draws);
            let mut exe_auto = MlxExecutable::compile_with_mode(g_auto, MlxMode::Lazy);
            let _ = exe_auto.run(&[("v0", v_init.as_slice())]);    // warmup
            let t0 = Instant::now();
            let outs_auto = exe_auto.run(&[("v0", v_init.as_slice())]);
            let t_auto = t0.elapsed().as_secs_f64() * 1e3;
            let traj_auto = &outs_auto[0];

            // ── Path C: batched_transient_from (BE Newton baseline) ──
            let mut ic: HashMap<NetId, Vec<f32>> = HashMap::new();
            ic.insert(vmid, v_init.clone());
            let boundary: HashMap<NetId, Vec<f32>> = HashMap::new();
            let mc_params: HashMap<String, Vec<f32>> = HashMap::new();
            let opt = NewtonOptions::default();
            let t0 = Instant::now();
            let waveform = batched_transient_from(
                &c, n_draws, &params, &mc_params, &boundary, &ic,
                dt, n_steps as usize, opt,
            );
            let t_newton = t0.elapsed().as_secs_f64() * 1e3;

            // Drift: auto body's final v vs Newton baseline per draw.
            let mut max_auto_drift = 0.0_f32;
            for d in 0..n_draws {
                let v_auto    = traj_auto[d * n_steps as usize + (n_steps as usize - 1)];
                let v_newton  = waveform[n_steps as usize].voltages[&vmid][d];
                max_auto_drift = max_auto_drift.max((v_auto - v_newton).abs());
            }
            // Sanity: hand-coded matches auto exactly (same math).
            for d in 0..n_draws {
                let v_hand = traj_hand[d * n_steps as usize + (n_steps as usize - 1)];
                let v_auto = traj_auto[d * n_steps as usize + (n_steps as usize - 1)];
                debug_assert!(
                    (v_hand - v_auto).abs() < 1e-5,
                    "hand vs auto draw {d}: {v_hand} vs {v_auto}");
            }

            let h_spdup = t_newton / t_hand.max(1e-6);
            let a_spdup = t_newton / t_auto.max(1e-6);
            println!(
                "{:>6}  {:>10.3}  {:>10.3}  {:>10.3}  {:>9.1}×  {:>9.1}×  {:>14.3e}",
                n_draws, t_hand, t_auto, t_newton, h_spdup, a_spdup, max_auto_drift);
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    bench::run();
    #[cfg(not(target_os = "macos"))]
    eprintln!("Op::Scan-on-GPU bench requires macOS (MLX backend).");
}
