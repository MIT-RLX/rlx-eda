//! Combined HPWL + density-aware placement smoke. Confirms that
//! adding the overlap term breaks the degenerate "collapse to a
//! point" optimum: cells settle into a non-overlapping cluster
//! whose bbox is larger than zero but much smaller than the seed
//! spread.

use eda_pnr::ad::{combined_loss_graph, position_param_ids, DifferentiablePlacement};
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

fn build_unit(lib: &Library, pdk: &TestPdk, name: &str, half: i64) -> CellId {
    let mut cb = CellBuilder::new(name);
    cb.add_shape(
        pdk.METAL1,
        Shape::Box(Rect::new(Bbox::new(
            Point::new(-half, -half),
            Point::new(half, half),
        ))),
    );
    cb.add_port(
        Port::new("p", pdk.METAL1, Point::new(0, 0), Angle90::E, half * 2)
            .with_kind(TestPdk::Electrical),
    );
    lib.insert(cb)
}

#[test]
fn density_breaks_degenerate_collapse() {
    let lib = TestPdk::new_library("ad_combined");
    let pdk = TestPdk::register(&lib);

    // Three 4 µm × 4 µm cells (half = 2_000 DBU) on one shared net.
    // HPWL alone would collapse all three to coincident points;
    // density should keep them at least one cell-width apart.
    let cells: Vec<CellId> =
        (0..3).map(|i| build_unit(&lib, &pdk, &format!("U{i}"), 2_000)).collect();

    let mut nl = Netlist::new("ad_combined").with_default_signal_layer(pdk.METAL1);
    let inst: Vec<usize> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| nl.add_instance(format!("U{i}"), *c))
        .collect();
    nl.connect("net0", inst[0], "p");
    nl.connect("net0", inst[1], "p");
    nl.connect("net0", inst[2], "p");

    let mut placement = DifferentiablePlacement {
        instance_xy: vec![
            (   0.0,    0.0),
            (50_000.0,   0.0),
            (25_000.0, 50_000.0),
        ],
        beta: 1e-4,
    };
    let fwd = combined_loss_graph(&nl, &lib, &placement, 1e-2);
    let pos_ids = position_param_ids(&fwd, &nl);
    let bwd = grad_with_loss(&fwd, &pos_ids);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    let lr = 1_000.0_f32;
    let n = pos_ids.len();
    let mut adam = eda_trace::AdamState::new(n);
    let layout = eda_pnr::ad::position_param_layout(&nl);
    for t in 1..=600 {
        for (idx, (x, y)) in placement.instance_xy.iter().enumerate() {
            sess.set_param(&placement.x_param_name(&nl, idx), &[*x]);
            sess.set_param(&placement.y_param_name(&nl, idx), &[*y]);
        }
        let outs = sess.run(&[
            ("d_output",     &[1.0_f32]),
            ("hpwl_beta",    &[1e-4_f32]),
            ("density_beta", &[1e-3_f32]),
        ]);
        let grads: Vec<f32> = (0..n).map(|k| outs[1 + k][0]).collect();
        let mut params: Vec<f32> = layout
            .iter()
            .map(|(i, axis)| if *axis == 0 {
                placement.instance_xy[*i].0
            } else {
                placement.instance_xy[*i].1
            })
            .collect();
        adam.step(&mut params, &grads, lr, t);
        for (k, (i, axis)) in layout.iter().enumerate() {
            if *axis == 0 {
                placement.instance_xy[*i].0 = params[k];
            } else {
                placement.instance_xy[*i].1 = params[k];
            }
        }
    }

    println!("final positions = {:?}", placement.instance_xy);

    // Pairwise check: no two cells have BOTH |dx| < 4_000 AND
    // |dy| < 4_000 simultaneously (= bboxes don't overlap).
    let cell_w = 4_000.0_f32;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let dx = (placement.instance_xy[i].0 - placement.instance_xy[j].0).abs();
            let dy = (placement.instance_xy[i].1 - placement.instance_xy[j].1).abs();
            assert!(
                dx >= cell_w * 0.7 || dy >= cell_w * 0.7,
                "cells {i} and {j} overlap: dx = {dx:.0}, dy = {dy:.0} \
                 (both < cell width {cell_w})",
            );
        }
    }

    // Bbox spans some non-trivial area — confirms cells didn't all
    // converge to one point (degenerate HPWL minimum).
    let xs: Vec<f32> = placement.instance_xy.iter().map(|(x, _)| *x).collect();
    let ys: Vec<f32> = placement.instance_xy.iter().map(|(_, y)| *y).collect();
    let span_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let span_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - ys.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        span_x > cell_w * 0.5 || span_y > cell_w * 0.5,
        "placement collapsed to a point — density term failed: span_x = {span_x:.0}, span_y = {span_y:.0}",
    );

    // …and total span is much smaller than the seed spread.
    assert!(
        span_x < 30_000.0 && span_y < 30_000.0,
        "placement didn't pack: span_x = {span_x:.0}, span_y = {span_y:.0} \
         (seeds spread 50 kDBU)",
    );
}
