//! Parallel-batch placement equivalence smoke. Builds the same
//! 6-instance / 4-net netlist as `ad_batched.rs`, runs the
//! `[B, N]`-shaped loss graph with `B=4` independent placements,
//! and asserts each batch element's loss matches the single-batch
//! `combined_loss_graph_batched` evaluated on its own positions.

use eda_pnr::ad::{
    combined_loss_graph_batched, combined_loss_graph_parallel_per_batch,
    position_param_ids_batched, DifferentiablePlacement, POSITIONS_X_PARAM, POSITIONS_Y_PARAM,
};
use eda_pnr::Netlist;
use klayout_core::{Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape};
use klayout_pdk::pdk;
use rlx_runtime::{Device, Session};

pdk! {
    pub TestPdk {
        dbu: 1000,
        layers: { METAL1 = (10, 0) },
        ports: { Electrical },
    }
}

fn build_unit(lib: &Library, pdk: &TestPdk, name: &str) -> CellId {
    let mut cb = CellBuilder::new(name);
    let half = 2_000_i64;
    cb.add_shape(
        pdk.METAL1,
        Shape::Box(Rect::new(Bbox::new(
            Point::new(-half, -half), Point::new(half, half),
        ))),
    );
    cb.add_port(
        Port::new("p", pdk.METAL1, Point::new(0, 0), Angle90::E, 4_000)
            .with_kind(TestPdk::Electrical),
    );
    lib.insert(cb)
}

fn build_netlist(lib: &Library, pdk: &TestPdk) -> Netlist {
    let cells: Vec<CellId> =
        (0..6).map(|i| build_unit(lib, pdk, &format!("U{i}"))).collect();
    let mut nl = Netlist::new("eq").with_default_signal_layer(pdk.METAL1);
    let inst: Vec<usize> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| nl.add_instance(format!("U{i}"), *c))
        .collect();
    nl.connect("netA", inst[0], "p");
    nl.connect("netA", inst[1], "p");
    nl.connect("netA", inst[2], "p");
    nl.connect("bridge", inst[2], "p");
    nl.connect("bridge", inst[3], "p");
    nl.connect("chain1", inst[3], "p");
    nl.connect("chain1", inst[4], "p");
    nl.connect("chain2", inst[4], "p");
    nl.connect("chain2", inst[5], "p");
    nl
}

#[test]
fn parallel_batch_matches_single_batch_per_element() {
    let lib = TestPdk::new_library("ad_parallel");
    let pdk = TestPdk::register(&lib);
    let nl = build_netlist(&lib, &pdk);

    // Four distinct placements — each row of the [B, N] tensor.
    let placements: Vec<Vec<(f32, f32)>> = vec![
        vec![
            (-20_000.0, -10_000.0),
            (-5_000.0, 5_000.0),
            (12_000.0, 8_000.0),
            (20_000.0, -15_000.0),
            (-10_000.0, 20_000.0),
            (15_000.0, 10_000.0),
        ],
        vec![
            (3_000.0, 7_000.0),
            (-8_000.0, 4_000.0),
            (10_000.0, -2_000.0),
            (5_000.0, -12_000.0),
            (-15_000.0, 6_000.0),
            (18_000.0, 1_000.0),
        ],
        vec![
            (0.0, 0.0),
            (10_000.0, 10_000.0),
            (-10_000.0, -10_000.0),
            (10_000.0, -10_000.0),
            (-10_000.0, 10_000.0),
            (5_000.0, 5_000.0),
        ],
        vec![
            (-30_000.0, 0.0),
            (30_000.0, 0.0),
            (0.0, 30_000.0),
            (0.0, -30_000.0),
            (15_000.0, 15_000.0),
            (-15_000.0, -15_000.0),
        ],
    ];
    let b = placements.len();
    let n = placements[0].len();
    let alpha = 1e-2_f32;
    let beta_h = 5e-4_f32;
    let beta_d = 1e-3_f32;

    // ── Per-batch single-placement losses (the reference) ─────────
    let mut single_losses = Vec::with_capacity(b);
    for xy in &placements {
        let placement = DifferentiablePlacement { instance_xy: xy.clone(), beta: 1e-4 };
        let g_b = combined_loss_graph_batched(&nl, &lib, &placement, alpha);
        let _ids = position_param_ids_batched(&g_b);
        let mut sess = Session::new(Device::Cpu).compile(g_b);
        let xs: Vec<f32> = xy.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f32> = xy.iter().map(|(_, y)| *y).collect();
        sess.set_param(POSITIONS_X_PARAM, &xs);
        sess.set_param(POSITIONS_Y_PARAM, &ys);
        let outs = sess.run(&[
            ("hpwl_beta",    &[beta_h]),
            ("density_beta", &[beta_d]),
        ]);
        single_losses.push(outs[0][0]);
    }

    // ── Parallel-batch loss: one graph, [B, N] positions ──────────
    // Use the per-batch variant so we can read per-element losses
    // directly. The AD-augmented `combined_loss_graph_parallel`
    // only exposes the scalar sum (since `grad_with_loss` requires
    // a single scalar output).
    let g_p = combined_loss_graph_parallel_per_batch(&nl, &lib, b, alpha);
    let mut sess = Session::new(Device::Cpu).compile(g_p);
    // Flatten [B, N] row-major: batch 0 first, then batch 1, ...
    let mut xs_flat = Vec::with_capacity(b * n);
    let mut ys_flat = Vec::with_capacity(b * n);
    for xy in &placements {
        for (x, _) in xy { xs_flat.push(*x); }
    }
    for xy in &placements {
        for (_, y) in xy { ys_flat.push(*y); }
    }
    sess.set_param(POSITIONS_X_PARAM, &xs_flat);
    sess.set_param(POSITIONS_Y_PARAM, &ys_flat);
    let outs = sess.run(&[
        ("hpwl_beta",    &[beta_h]),
        ("density_beta", &[beta_d]),
    ]);
    // outs[0] = [B] per-batch loss.
    let per_batch = &outs[0];
    assert_eq!(per_batch.len(), b, "per-batch loss should have B entries");

    println!("single losses:    {:?}", single_losses);
    println!("parallel  losses: {:?}", per_batch);

    for k in 0..b {
        let single = single_losses[k];
        let parallel = per_batch[k];
        let rel = (single - parallel).abs() / single.abs().max(1.0);
        assert!(
            rel < 1e-3,
            "batch {k}: single = {single}, parallel = {parallel}, rel = {rel:.3e}",
        );
    }
}
