//! Equivalence smoke: `combined_loss_graph_batched` should produce
//! the same scalar loss as `combined_loss_graph` for any given
//! placement, since both implement HPWL + density on the same
//! algebraic form. The batched version just packs positions into
//! `[N]` tensors so the density operator vectorizes for GPU
//! backends — numerics on CPU should be byte-identical (or within
//! `1e-3` relative, accounting for different op-ordering through
//! the rlx graph).

use eda_pnr::ad::{
    combined_loss_graph, combined_loss_graph_batched, position_param_ids,
    position_param_ids_batched, DifferentiablePlacement, POSITIONS_X_PARAM, POSITIONS_Y_PARAM,
};
use eda_pnr::Netlist;
use klayout_core::{Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape};
use klayout_pdk::pdk;
use rlx_opt::autodiff::grad_with_loss;
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

#[test]
fn batched_loss_matches_unbatched() {
    let lib = TestPdk::new_library("ad_batched");
    let pdk = TestPdk::register(&lib);
    let cells: Vec<CellId> =
        (0..6).map(|i| build_unit(&lib, &pdk, &format!("U{i}"))).collect();

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

    let xy = vec![
        (-20_000.0, -10_000.0),
        (-5_000.0,   5_000.0),
        (12_000.0,   8_000.0),
        (20_000.0, -15_000.0),
        (-10_000.0,  20_000.0),
        (15_000.0,   10_000.0),
    ];
    let placement = DifferentiablePlacement { instance_xy: xy.clone(), beta: 1e-4 };
    let beta_h = 5e-4_f32;
    let beta_d = 1e-3_f32;
    let alpha = 1e-2_f32;

    // ── Unbatched (scalar Param) loss ─────────────────────────────
    let g_a = combined_loss_graph(&nl, &lib, &placement, alpha);
    let mut sess_a = Session::new(Device::Cpu).compile(g_a.clone());
    for (i, (x, y)) in xy.iter().enumerate() {
        sess_a.set_param(&placement.x_param_name(&nl, i), &[*x]);
        sess_a.set_param(&placement.y_param_name(&nl, i), &[*y]);
    }
    let outs_a = sess_a.run(&[
        ("hpwl_beta",    &[beta_h]),
        ("density_beta", &[beta_d]),
    ]);
    let loss_unbatched = outs_a[0][0];

    // ── Batched ([N] Param) loss ──────────────────────────────────
    let g_b = combined_loss_graph_batched(&nl, &lib, &placement, alpha);
    let _pos_ids = position_param_ids_batched(&g_b);
    let mut sess_b = Session::new(Device::Cpu).compile(g_b);
    let xs: Vec<f32> = xy.iter().map(|(x, _)| *x).collect();
    let ys: Vec<f32> = xy.iter().map(|(_, y)| *y).collect();
    sess_b.set_param(POSITIONS_X_PARAM, &xs);
    sess_b.set_param(POSITIONS_Y_PARAM, &ys);
    let outs_b = sess_b.run(&[
        ("hpwl_beta",    &[beta_h]),
        ("density_beta", &[beta_d]),
    ]);
    let loss_batched = outs_b[0][0];

    println!("loss unbatched = {loss_unbatched:.6e}  batched = {loss_batched:.6e}");
    let rel = (loss_unbatched - loss_batched).abs() / loss_unbatched.abs().max(1.0);
    assert!(
        rel < 1e-3,
        "batched/unbatched mismatch: unbatched={loss_unbatched}, batched={loss_batched}, rel={rel:.3e}",
    );

    // ── AD also agrees within tolerance ───────────────────────────
    // (gradients via grad_with_loss; smoke check that the batched
    // graph produces a non-trivial gradient signal.)
    let pos_ids_b = position_param_ids_batched(
        &combined_loss_graph_batched(&nl, &lib, &placement, alpha),
    );
    assert_eq!(pos_ids_b.len(), 2, "batched should expose 2 position-Param tensors");
    let g_b2 = combined_loss_graph_batched(&nl, &lib, &placement, alpha);
    let pos_ids2 = position_param_ids_batched(&g_b2);
    let bwd = grad_with_loss(&g_b2, &pos_ids2);
    let mut sess_g = Session::new(Device::Cpu).compile(bwd);
    sess_g.set_param(POSITIONS_X_PARAM, &xs);
    sess_g.set_param(POSITIONS_Y_PARAM, &ys);
    let outs_g = sess_g.run(&[
        ("d_output",     &[1.0_f32]),
        ("hpwl_beta",    &[beta_h]),
        ("density_beta", &[beta_d]),
    ]);
    // outs[0] = scalar loss; outs[1] = dL/d(positions_x) [N]; outs[2] = dL/d(positions_y) [N]
    assert_eq!(outs_g[1].len(), xy.len(), "dL/dx tensor should have length N");
    assert_eq!(outs_g[2].len(), xy.len(), "dL/dy tensor should have length N");
    let any_nonzero = outs_g[1].iter().any(|&v| v.abs() > 1e-9)
        || outs_g[2].iter().any(|&v| v.abs() > 1e-9);
    assert!(any_nonzero, "batched gradients should be non-zero at this off-optimum placement");
}
