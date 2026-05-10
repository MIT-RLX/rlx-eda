//! Inverse-design tier — SGD on the rlx-AD loss graph drives R1, R2 to
//! achieve a target Vout. This is the headline demonstration: define a
//! goal, watch the resistor parameters converge.

use spike_divider_block::*;

fn divider() -> RcDivider {
    RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    )
}

#[test]
fn converges_to_a_lower_target() {
    let div = divider();
    let opt = DcOptimizer::default();
    let res = div.optimize_to_target(
        /* v_in */ 1.0,
        /* target */ 0.4,
        /* r1_init */ 1_000.0,
        /* r2_init */ 3_000.0,
        &opt,
    );
    assert!(res.converged, "did not converge: {:?}", res);
    assert!((res.final_vout - 0.4).abs() < 1e-3,
        "final Vout {} not within 1e-3 of target 0.4", res.final_vout);
    // Manually verify the achieved R2 / (R1 + R2) ratio.
    let ratio = res.r2 / (res.r1 + res.r2);
    assert!((ratio - 0.4).abs() < 1e-3, "ratio = {}", ratio);
}

#[test]
fn converges_to_a_higher_target() {
    // Starts at vout = 0.75 (R1=1k, R2=3k), needs to push to 0.9.
    let div = divider();
    let opt = DcOptimizer { lr: 1e6, max_iters: 2_000, tol: 1e-4, r_min: 1.0 };
    let res = div.optimize_to_target(1.0, 0.9, 1_000.0, 3_000.0, &opt);
    assert!(res.converged, "did not converge: {:?}", res);
    assert!((res.final_vout - 0.9).abs() < 1e-3,
        "final Vout {} not within tol of 0.9", res.final_vout);
}

#[test]
fn adam_converges_to_target() {
    let div = divider();
    // Adam's adaptive scaling lets us pick a smaller, more universal lr.
    let mut adam = Adam::new(/* lr */ 50.0, /* n_params */ 2);
    let res = div.optimize_to_target_with(
        1.0, 0.4, 1_000.0, 3_000.0,
        &mut adam,
        /* max_iters */ 5_000, /* tol */ 1e-4, /* r_min */ 1.0,
    );
    assert!(res.converged, "Adam did not converge: {:?}", res);
    assert!((res.final_vout - 0.4).abs() < 1e-3,
        "Adam: final Vout {} not within tol of 0.4", res.final_vout);
}

#[test]
fn adamw_converges_with_weight_decay() {
    // AdamW with non-trivial weight decay still converges to the target,
    // it just biases solutions toward smaller R values. Check the target
    // is met and resistances stay positive.
    let div = divider();
    let mut adamw = AdamW::new(/* lr */ 50.0, /* weight_decay */ 1e-5, 2);
    let res = div.optimize_to_target_with(
        1.0, 0.4, 1_000.0, 3_000.0,
        &mut adamw,
        5_000, 1e-3, 1.0,
    );
    assert!(res.converged, "AdamW did not converge: {:?}", res);
    assert!(res.r1 > 0.0 && res.r2 > 0.0);
    assert!((res.final_vout - 0.4).abs() < 5e-3,
        "AdamW: final Vout {} not within tol of 0.4", res.final_vout);
}

#[test]
fn adam_handles_orders_of_magnitude_initial_param_disparity() {
    // R1 starts at 100 Ω, R2 at 100 kΩ — a factor of 1000 apart. SGD
    // with one lr struggles here (it's either too big for R1 or too small
    // for R2). Adam's per-param adaptive scaling handles it.
    let div = divider();
    let mut adam = Adam::new(20.0, 2);
    let res = div.optimize_to_target_with(
        1.0, 0.5,
        /* r1_init */ 100.0,
        /* r2_init */ 100_000.0,
        &mut adam,
        10_000, 1e-3, 1.0,
    );
    assert!(res.converged, "Adam failed on lopsided init: {:?}", res);
    assert!((res.final_vout - 0.5).abs() < 5e-3);
}

#[test]
fn loss_decreases_monotonically_within_lr_envelope() {
    // SGD on a smooth convex objective (in our reparametrization) should
    // monotonically decrease loss with a small enough lr. This catches
    // regressions where the gradient sign flips or the loss graph drifts.
    let div = divider();
    let opt = DcOptimizer { lr: 5e5, max_iters: 50, tol: 0.0, r_min: 1.0 };
    // tol=0 forces it to run all 50 iters so we get a real trajectory.

    // Reuse the optimizer's compile path manually to capture the per-step
    // loss. Easier than instrumenting the lib API.
    use rlx_opt::autodiff::grad_with_loss;
    use rlx_runtime::{Device, Session};
    let (fwd, r1_id, r2_id) = div.build_loss_graph();
    let bwd = grad_with_loss(&fwd, &[r1_id, r2_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let [n1, n2] = div.dc_param_names();

    let mut r1 = 1_000.0_f32;
    let mut r2 = 3_000.0_f32;
    let mut last_loss = f32::INFINITY;
    for _ in 0..opt.max_iters {
        compiled.set_param(&n1, &[r1]);
        compiled.set_param(&n2, &[r2]);
        let outs = compiled.run(&[
            ("V",        &[1.0_f32][..]),
            ("target",   &[0.4_f32][..]),
            ("d_output", &[1.0_f32][..]),
        ]);
        let loss = outs[0][0];
        // Allow tiny f32 jitter — require loss to not increase by more than
        // ~1 ulp of the magnitude.
        let envelope = (last_loss * 1e-6).max(1e-9);
        assert!(loss <= last_loss + envelope,
            "loss increased: {last_loss} → {loss}");
        last_loss = loss;
        r1 = (r1 - opt.lr * outs[1][0]).max(1.0);
        r2 = (r2 - opt.lr * outs[2][0]).max(1.0);
    }
}
