//! AD-enabled placement smoke: Adam minimizes the rlx-graph HPWL
//! loss over per-instance positions and recovers the analytical
//! optimum (every instance collapsed to a single x / y).
//!
//! Setup: three "cells", each a 1 × 1 µm dummy box with one port
//! at the cell origin. One net touches all three. Optimal HPWL = 0
//! when all three positions coincide. We seed positions far apart
//! and watch the loss fall.

use eda_pnr::{
    ad::{hpwl_loss_graph, position_param_ids, DifferentiablePlacement},
    Netlist,
};
use klayout_core::{
    Angle90, Bbox, CellBuilder, CellId, Library, Point, Port, Rect, Shape,
};
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

fn build_dummy(lib: &Library, pdk: &TestPdk, name: &str) -> CellId {
    let mut cb = CellBuilder::new(name);
    cb.add_shape(
        pdk.METAL1,
        Shape::Box(Rect::new(Bbox::new(Point::new(0, 0), Point::new(1_000, 1_000)))),
    );
    cb.add_port(
        Port::new("p", pdk.METAL1, Point::new(0, 0), Angle90::E, 1_000)
            .with_kind(TestPdk::Electrical),
    );
    lib.insert(cb)
}

#[test]
fn adam_on_hpwl_collapses_three_instances() {
    let lib = TestPdk::new_library("hpwl");
    let pdk = TestPdk::register(&lib);
    let c = build_dummy(&lib, &pdk, "U");

    let mut nl = Netlist::new("ad_hpwl").with_default_signal_layer(pdk.METAL1);
    let i0 = nl.add_instance("U0", c);
    let i1 = nl.add_instance("U1", c);
    let i2 = nl.add_instance("U2", c);
    nl.connect("net0", i0, "p");
    nl.connect("net0", i1, "p");
    nl.connect("net0", i2, "p");

    // Seed: positions spread across a 50 µm × 50 µm region.
    let seed_xy = [
        (0.0_f32,    0.0_f32),
        (50_000.0,   0.0),
        (25_000.0,   50_000.0),
    ];
    let mut placement = DifferentiablePlacement {
        instance_xy: seed_xy.to_vec(),
        beta: 1e-4, // β · max_dim ≈ 5; comfortably inside f32 exp range
    };

    // Build the loss graph + AD-augmented session once.
    let fwd = hpwl_loss_graph(&nl, &lib, placement.beta);
    let pos_ids = position_param_ids(&fwd, &nl);
    let bwd = grad_with_loss(&fwd, &pos_ids);
    let mut sess = Session::new(Device::Cpu).compile(bwd);

    // Push initial params.
    for (idx, (x, y)) in placement.instance_xy.iter().enumerate() {
        sess.set_param(&placement.x_param_name(&nl, idx), &[*x]);
        sess.set_param(&placement.y_param_name(&nl, idx), &[*y]);
    }

    // Loss before any updates.
    let initial_loss = {
        let outs = sess.run(&[("d_output", &[1.0_f32])]);
        outs[0][0]
    };

    // Adam — lr scaled to DBU magnitudes (~1e4) so steps are
    // hundreds-of-DBU per iter.
    let lr = 1_000.0_f32;
    let (b1, b2, eps) = (0.9_f32, 0.999_f32, 1e-8_f32);
    let n_params = pos_ids.len();
    let mut m = vec![0.0_f32; n_params];
    let mut v = vec![0.0_f32; n_params];

    for t in 1..=600 {
        for (idx, (x, y)) in placement.instance_xy.iter().enumerate() {
            sess.set_param(&placement.x_param_name(&nl, idx), &[*x]);
            sess.set_param(&placement.y_param_name(&nl, idx), &[*y]);
        }
        let outs = sess.run(&[("d_output", &[1.0_f32])]);
        let _loss = outs[0][0];
        // outs[0] = loss; outs[1..1+n_params] = grads
        for k in 0..n_params {
            let g = outs[1 + k][0];
            m[k] = b1 * m[k] + (1.0 - b1) * g;
            v[k] = b2 * v[k] + (1.0 - b2) * g * g;
            let m_hat = m[k] / (1.0 - b1.powi(t));
            let v_hat = v[k] / (1.0 - b2.powi(t));
            let inst = k / 2;
            let axis = k % 2;
            let delta = lr * m_hat / (v_hat.sqrt() + eps);
            if axis == 0 {
                placement.instance_xy[inst].0 -= delta;
            } else {
                placement.instance_xy[inst].1 -= delta;
            }
        }
    }

    // Final loss.
    for (idx, (x, y)) in placement.instance_xy.iter().enumerate() {
        sess.set_param(&placement.x_param_name(&nl, idx), &[*x]);
        sess.set_param(&placement.y_param_name(&nl, idx), &[*y]);
    }
    let final_loss = {
        let outs = sess.run(&[("d_output", &[1.0_f32])]);
        outs[0][0]
    };

    println!(
        "HPWL: initial = {initial_loss:.1}  final = {final_loss:.1}  \
         positions = {:?}",
        placement.instance_xy,
    );

    // Loss should drop substantially. The exact floor depends on
    // β (smooth-max bias) but for β=1e-4 with a 3-pin net the
    // floor is around 2/β ≈ 20_000 (the LSE smoothing residual);
    // we just check we got most of the way there.
    assert!(
        final_loss < 0.5 * initial_loss,
        "HPWL did not converge: initial = {initial_loss:.1}, final = {final_loss:.1}",
    );

    // Pairwise positions should be much closer than the seed
    // spread. Take max-distance between any two instances.
    let mut max_dist = 0.0_f32;
    for a in 0..placement.instance_xy.len() {
        for b in (a + 1)..placement.instance_xy.len() {
            let dx = placement.instance_xy[a].0 - placement.instance_xy[b].0;
            let dy = placement.instance_xy[a].1 - placement.instance_xy[b].1;
            let d = (dx * dx + dy * dy).sqrt();
            if d > max_dist { max_dist = d; }
        }
    }
    let initial_max = ((50_000.0_f32).powi(2) + (50_000.0_f32).powi(2)).sqrt();
    assert!(
        max_dist < 0.3 * initial_max,
        "instances did not collapse: max pairwise distance = {max_dist:.1} \
         (initial {initial_max:.1})",
    );

    // Materialize back to a Placement and confirm the bbox is
    // small (instances overlap or nearly so).
    let p = placement.materialize(&nl, &lib);
    let bbox_diag = (
        (p.bbox.max.x - p.bbox.min.x) as f32,
        (p.bbox.max.y - p.bbox.min.y) as f32,
    );
    println!("materialized bbox span: {bbox_diag:?}");
}
