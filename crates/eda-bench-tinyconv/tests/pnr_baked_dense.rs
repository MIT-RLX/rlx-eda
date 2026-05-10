//! Third row of the silicon comparison matrix: **BakedConst + PnR**.
//!
//! Same architecture as the BakedConst row (`unrolled_rtl_sim.rs`),
//! but the placement of the multiplier+accumulator cells is
//! optimized via rlx-eda's differentiable HPWL placer instead of
//! left to a default grid. Reports:
//!
//! - HPWL_initial vs HPWL_final (the direct optimizer signal),
//! - projected period reduction (∝ √HPWL — Elmore RC scaling),
//! - projected switching-energy reduction (∝ HPWL — wire cap is
//!   linear in length under sky130 Mx Ohm/fF/µm).
//!
//! The architecture-only metrics (cycles, gate area) are identical
//! to the BakedConst row — placement doesn't change them. PnR only
//! moves the period / energy / noise needles.
//!
//! Runs separately from `unrolled_rtl_sim` so it can execute in
//! parallel: docker isn't involved, so the only competing resource
//! is CPU.
//!
//! ```sh
//! cargo test -p eda-bench-tinyconv --features bench-rtl-sim \
//!   --test pnr_baked_dense -- --ignored --nocapture
//! ```

#![cfg(feature = "bench-rtl-sim")]

use eda_pnr::{
    ad::{hpwl_loss_graph, position_param_ids, DifferentiablePlacement},
    Netlist,
};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape,
};
use klayout_pdk::pdk;
use rlx_fpga::model::{tinyconv_mnist_from_cortexm, Layer};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use spike_divider_block::{Adam, Optimizer};

pdk! {
    pub PnrBenchPdk {
        dbu: 1000,
        layers: { METAL1 = (10, 0) },
        ports: { Electrical },
    }
}

fn unit_cell(lib: &Library, pdk: &PnrBenchPdk, name: &str) -> CellId {
    let mut cb = CellBuilder::new(name);
    cb.add_shape(
        pdk.METAL1,
        Shape::Box(Rect::new(Bbox::new(
            Point::new(0, 0),
            Point::new(2_000, 1_000),
        ))),
    );
    cb.add_port(
        Port::new("p", pdk.METAL1, Point::new(0, 0), Angle90::E, 1_000)
            .with_kind(PnrBenchPdk::Electrical),
    );
    lib.insert(cb)
}

/// Build a netlist representing the BakedConst Dense architecture:
/// `out_features` accumulator hubs, each fed by `mults_per_oc`
/// multiplier instances. Each multiplier connects to its hub via a
/// 2-pin net carrying the partial product. Mirrors the actual
/// `acc[oc] += W[oc][ic] * (in_buf[ic] - X_ZP)` fan-in shape.
fn build_dense_netlist(
    lib: &Library,
    pdk: &PnrBenchPdk,
    out_features: usize,
    mults_per_oc: usize,
) -> Netlist {
    let mult = unit_cell(lib, pdk, "mult");
    let accum = unit_cell(lib, pdk, "accum");
    let mut nl = Netlist::new("baked_dense").with_default_signal_layer(pdk.METAL1);

    for oc in 0..out_features {
        let acc_idx = nl.add_instance(format!("accum_{oc}"), accum);
        for ic in 0..mults_per_oc {
            let m_idx = nl.add_instance(format!("mult_{oc}_{ic}"), mult);
            let net = format!("pp_{oc}_{ic}");
            nl.connect(&net, m_idx, "p");
            nl.connect(&net, acc_idx, "p");
        }
    }
    nl
}

#[test]
#[ignore = "Adam over ~200-cell netlist; ~30-90s; opt-in via --ignored"]
fn pnr_optimizes_baked_dense_layout() {
    // ── Read the real Dense layer dimensions so the netlist scale
    // matches what the silicon row will see (10 outputs × 400 inputs
    // for TinyConv-MNIST). For the AD smoke-test, sub-sample to
    // a tractable 20 mults per output (200 cells total) — the HPWL
    // ratio is what we're after, and that's scale-invariant.
    let dense = tinyconv_mnist_from_cortexm()
        .layers
        .iter()
        .find(|l| matches!(l, Layer::Dense { .. }))
        .expect("Dense layer present")
        .clone();
    let (in_features, out_features) = match &dense {
        Layer::Dense {
            in_features,
            out_features,
            ..
        } => (*in_features, *out_features),
        _ => unreachable!(),
    };
    // Sub-sample aggressively — Adam-on-HPWL evaluation cost scales
    // ~linearly with (cells × nets), and rlx-runtime is single-thread
    // CPU. 5 outs × 5 mults = 30 cells / 25 nets keeps per-step time
    // small enough that 200 steps run in seconds.
    let mults_per_oc = 5.min(in_features);
    let n_cells = out_features.min(5) * (mults_per_oc + 1);
    let out_features = out_features.min(5);
    eprintln!(
        "[pnr] modelling BakedConst Dense as {out_features} accumulators × \
         {mults_per_oc} multipliers each = {n_cells} cells, \
         {} nets (sub-sampled — HPWL ratio is scale-invariant)",
        out_features * mults_per_oc
    );

    // ── Build netlist + AD graph.
    let lib = PnrBenchPdk::new_library("pnr_dense");
    let pdk = PnrBenchPdk::register(&lib);
    let nl = build_dense_netlist(&lib, &pdk, out_features, mults_per_oc);

    // Naïve grid seed: lay cells out in a wide rectangle so the
    // initial HPWL is dominated by long inter-row connections.
    let cols = 16;
    let pitch_x = 4_000_f32; // 4 µm in DBU (dbu=1000)
    let pitch_y = 3_000_f32;
    let mut seed_xy: Vec<(f32, f32)> = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let r = (i / cols) as f32;
        let c = (i % cols) as f32;
        seed_xy.push((c * pitch_x, r * pitch_y));
    }

    let placement = DifferentiablePlacement {
        instance_xy: seed_xy.clone(),
        beta: 1e-4,
    };

    let fwd = hpwl_loss_graph(&nl, &lib, placement.beta);
    let pos_ids = position_param_ids(&fwd, &nl);
    let mut sess = Session::new(Device::Cpu).compile(grad_with_loss(&fwd, &pos_ids));

    for (i, (x, y)) in placement.instance_xy.iter().enumerate() {
        sess.set_param(&placement.x_param_name(&nl, i), &[*x]);
        sess.set_param(&placement.y_param_name(&nl, i), &[*y]);
    }

    let initial_hpwl = sess.run(&[("d_output", &[1.0_f32][..])])[0][0];
    eprintln!("[pnr] initial HPWL (naïve grid) = {initial_hpwl:.0} DBU");

    // ── Adam loop.
    let lr: f32 = 5_000.0;
    let max_steps: usize = 200;
    let mut params: Vec<f32> = seed_xy
        .iter()
        .flat_map(|(x, y)| [*x, *y])
        .collect();
    let mut adam = Adam::new(lr, params.len());

    let mut last_loss = initial_hpwl;
    for step in 0..max_steps {
        for (i, chunk) in params.chunks(2).enumerate() {
            sess.set_param(&placement.x_param_name(&nl, i), &chunk[0..1]);
            sess.set_param(&placement.y_param_name(&nl, i), &chunk[1..2]);
        }
        let outs = sess.run(&[("d_output", &[1.0_f32][..])]);
        last_loss = outs[0][0];
        let grads: Vec<f32> = (1..outs.len()).map(|i| outs[i][0]).collect();
        assert!(
            last_loss.is_finite() && grads.iter().all(|g| g.is_finite()),
            "Adam-on-HPWL diverged at step {step}: loss={last_loss}"
        );
        adam.step(&mut params, &grads);
        if step == 0 || step % 25 == 24 {
            eprintln!("[pnr] step {step:>3} HPWL = {last_loss:.0} DBU");
        }
    }

    let final_hpwl = last_loss;
    let hpwl_ratio = final_hpwl / initial_hpwl;
    eprintln!();
    eprintln!(
        "[pnr] final HPWL = {final_hpwl:.0} DBU  ({:.1}× shorter)",
        1.0 / hpwl_ratio
    );

    // ── Project onto silicon metrics. Wire RC delay scales as
    //    L² in long wires, but with sky130 buffers the effective
    //    scaling is closer to L (linear) for short hops and √L for
    //    medium ones. We use √-scaling as a defensible mid-range
    //    estimate; energy scales linearly with switched cap (≈ L).
    let period_factor = hpwl_ratio.sqrt();
    let energy_factor = hpwl_ratio;
    // Noise (capacitive coupling between adjacent long wires) tracks
    // total wire length × density. With shorter average hops the
    // density goes down ~linearly too, so noise scales with HPWL².
    let noise_factor = hpwl_ratio * hpwl_ratio;

    eprintln!();
    eprintln!("══════════════════════ PnR projection (BakedConst → BakedConst + PnR) ══════════════════════");
    eprintln!("┌────────────┬──────────────────────────┬──────────────────────────┐");
    eprintln!("│  Metric    │  before PnR              │  after PnR               │");
    eprintln!("├────────────┼──────────────────────────┼──────────────────────────┤");
    eprintln!(
        "│  HPWL      │  {:>10.0} DBU         │  {:>10.0} DBU         │  ({:.2}× shorter)",
        initial_hpwl, final_hpwl, 1.0 / hpwl_ratio,
    );
    eprintln!(
        "│  period    │  ×1.0 (baseline)         │  ×{:.3} (∝ √HPWL)        │",
        period_factor
    );
    eprintln!(
        "│  energy    │  ×1.0 (baseline)         │  ×{:.3} (∝ HPWL)         │",
        energy_factor
    );
    eprintln!(
        "│  noise     │  ×1.0 (baseline)         │  ×{:.3} (∝ HPWL²)        │",
        noise_factor
    );
    eprintln!("└────────────┴──────────────────────────┴──────────────────────────┘");
    eprintln!(
        "Cycles & gate area: identical to BakedConst (placement doesn't \
         change architecture)."
    );

    // Sanity bound: Adam should reduce HPWL relative to the naïve
    // grid seed. The smooth-max β floor prevents full collapse, so a
    // 1.15× shortening is the realistic expectation at β=1e-4.
    assert!(
        final_hpwl < initial_hpwl * 0.9,
        "PnR loop should cut HPWL by at least 1.1×; got {:.3}× ({:.0} → {:.0})",
        1.0 / hpwl_ratio,
        initial_hpwl,
        final_hpwl
    );
}
