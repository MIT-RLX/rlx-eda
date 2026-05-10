//! Differential-pair / matching constraint smoke. Two cells share
//! one external pin so HPWL alone wants them coincident; a
//! `MatchKind::Mirror` declaration forces them onto opposite sides
//! of an axis at equal y. Adam-driven combined loss must satisfy
//! the constraint while still pulling them in close.

use eda_pnr::ad::{
    combined_loss_graph_with_symmetry, position_param_ids, position_param_layout,
    symmetry_loss_graph, DifferentiablePlacement,
};
use eda_pnr::{Netlist, SymmetryAxis};
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
fn mirror_pair_lands_symmetric_about_axis() {
    let lib = TestPdk::new_library("ad_diff_pair");
    let pdk = TestPdk::register(&lib);

    // Two 4 µm × 4 µm devices (NMOS pair stand-ins) tied to one
    // shared net. HPWL pulls them coincident; Mirror about x = 0
    // forces (x_a + x_b) → 0 and y_a → y_b.
    let cell = build_unit(&lib, &pdk, "M", 2_000);
    let mut nl = Netlist::new("ad_diff_pair").with_default_signal_layer(pdk.METAL1);
    let m1 = nl.add_instance("M1", cell);
    let m2 = nl.add_instance("M2", cell);
    nl.connect("tail", m1, "p");
    nl.connect("tail", m2, "p");
    nl.match_mirror("dp", m1, m2, SymmetryAxis::Vertical, 0.0);

    let mut placement = DifferentiablePlacement {
        // Asymmetric seed: not centered on the axis, different y.
        instance_xy: vec![
            (-30_000.0,  10_000.0),
            ( 50_000.0, -20_000.0),
        ],
        beta: 1e-4,
    };

    let fwd = combined_loss_graph_with_symmetry(&nl, &lib, &placement, 1e-2, 1.0);
    let pos_ids = position_param_ids(&fwd, &nl);
    let bwd = grad_with_loss(&fwd, &pos_ids);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    let lr = 500.0_f32;
    let n = pos_ids.len();
    let mut adam = eda_trace::AdamState::new(n);
    let layout = position_param_layout(&nl);
    for t in 1..=800 {
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

    let (xa, ya) = placement.instance_xy[0];
    let (xb, yb) = placement.instance_xy[1];
    println!("M1 = ({xa:.0}, {ya:.0})  M2 = ({xb:.0}, {yb:.0})");

    // Mirror residuals: x_a + x_b ≈ 0, y_a ≈ y_b.
    let mirror_dx = (xa + xb).abs();
    let mirror_dy = (ya - yb).abs();
    assert!(
        mirror_dx < 200.0,
        "mirror axis violated: x_a + x_b = {mirror_dx:.0} (want ≈ 0)",
    );
    assert!(
        mirror_dy < 200.0,
        "mirror y-equality violated: |y_a - y_b| = {mirror_dy:.0} (want ≈ 0)",
    );
    // …and they didn't all collapse onto the axis (HPWL would do
    // that — the density term should keep them apart).
    let separation = (xa - xb).abs();
    assert!(
        separation > 3_000.0,
        "diff-pair collapsed onto axis: |x_a - x_b| = {separation:.0} \
         (cells are 4_000 wide, density should keep them apart)",
    );
}

#[test]
fn symmetry_loss_zero_on_satisfied_mirror() {
    // Sanity: a placement that already satisfies the constraint
    // produces ~zero symmetry loss (no graph machinery surprise).
    let lib = TestPdk::new_library("ad_diff_pair_zero");
    let pdk = TestPdk::register(&lib);
    let cell = build_unit(&lib, &pdk, "M", 2_000);

    let mut nl = Netlist::new("nl").with_default_signal_layer(pdk.METAL1);
    let m1 = nl.add_instance("M1", cell);
    let m2 = nl.add_instance("M2", cell);
    nl.match_mirror("dp", m1, m2, SymmetryAxis::Vertical, 0.0);

    let placement = DifferentiablePlacement {
        instance_xy: vec![(-10_000.0, 5_000.0), (10_000.0, 5_000.0)],
        beta: 1e-4,
    };
    let g = symmetry_loss_graph(&nl, &placement);
    let mut sess = Session::new(Device::Cpu).compile(g);
    sess.set_param(&placement.x_param_name(&nl, 0), &[-10_000.0]);
    sess.set_param(&placement.y_param_name(&nl, 0), &[ 5_000.0]);
    sess.set_param(&placement.x_param_name(&nl, 1), &[ 10_000.0]);
    sess.set_param(&placement.y_param_name(&nl, 1), &[ 5_000.0]);
    let outs = sess.run(&[]);
    let loss = outs[0][0];
    assert!(loss.abs() < 1e-3, "symmetry loss not zero on satisfied mirror: {loss}");
}
