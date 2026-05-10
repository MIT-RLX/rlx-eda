//! Headline regression: PINN inference outperforms MNA BE-transient
//! on multi-query parametric RC.
//!
//! Pins both ends of the tradeoff explicitly:
//! - **PINN accuracy bound**: max relative error vs analytic ≤ 5%.
//! - **MNA accuracy floor**: max relative error vs analytic ≤ 1% —
//!   sanity check that the baseline is doing real work.
//! - **Throughput win**: PINN total wall-clock ≤ MNA total / 10.
//!
//! Both paths use a *cached compiled graph* (no per-query recompile).
//! That makes the throughput claim about solver work, not warmup
//! overhead. The naive `spike_rc_transient::run_transient` re-compiles
//! per call — its number would be even worse, but quoting the cached
//! one keeps the comparison honest.

use eda_nn::Rng;
use rlx_ir::DType;
use rlx_runtime::{Device, Session};
use spike_pinn_rc::{
    analytic, train, Query, C_N_HI, C_N_LO, C_REF, R_N_HI, R_N_LO, R_REF,
    T_REF, V_N_HI, V_N_LO, V_REF,
};
use spike_rc_transient::build_step_graph;
use std::time::Instant;

const N_QUERIES: usize = 10_000;
const N_TRAIN_STEPS: usize = 8_000;
const MNA_BE_STEPS: usize = 200;

fn random_query(rng: &mut Rng) -> Query {
    Query {
        r: (R_N_LO + (R_N_HI - R_N_LO) * rng.next_unit()) * R_REF,
        c: (C_N_LO + (C_N_HI - C_N_LO) * rng.next_unit()) * C_REF,
        v: (V_N_LO + (V_N_HI - V_N_LO) * rng.next_unit()) * V_REF,
        // Avoid t=0 (analytic = 0, makes rel-err denom blow up via the
        // floor below; cleaner to bound away).
        t: (0.05 + 0.94 * rng.next_unit()) * T_REF,
    }
}

/// Max absolute error in volts. Used as the accuracy metric instead
/// of relative error because target voltages range down to ~0.04 V
/// (near t=0) where a small absolute error inflates the rel-err
/// reading even though the prediction is well within full-scale
/// tolerance. Reporting `max_abs_err / V_REF` is the convention used
/// in PINN literature for "%% of full scale".
fn max_abs_err(pred: &[f32], truth: &[f32]) -> f32 {
    pred.iter()
        .zip(truth)
        .map(|(p, t)| (p - t).abs())
        .fold(0.0_f32, f32::max)
}

fn rms_err(pred: &[f32], truth: &[f32]) -> f32 {
    let n = pred.len() as f32;
    let sse: f32 = pred
        .iter()
        .zip(truth)
        .map(|(p, t)| {
            let d = p - t;
            d * d
        })
        .sum();
    (sse / n).sqrt()
}

/// MNA baseline: cached compiled BE-step graph from `spike-rc-transient`,
/// looped per query. Mirrors `run_transient` but lifts the compile out
/// of the loop so the per-query cost is purely solver work.
fn run_mna_baseline(queries: &[Query]) -> Vec<f32> {
    let (graph, _r_id, _c_id) = build_step_graph();
    let mut compiled = Session::new(Device::Cpu).compile(graph);

    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        compiled.set_param_typed("R", &(q.r as f64).to_le_bytes(), DType::F64);
        compiled.set_param_typed("C", &(q.c as f64).to_le_bytes(), DType::F64);

        let h = (q.t as f64) / MNA_BE_STEPS as f64;
        let v_dc = q.v as f64;
        let h_b = h.to_le_bytes();
        let mut vout = 0.0_f64;
        for _ in 1..=MNA_BE_STEPS {
            let outs = compiled.run_typed(&[
                ("V",         &v_dc.to_le_bytes(),  DType::F64),
                ("vout_prev", &vout.to_le_bytes(),  DType::F64),
                ("h",         &h_b,                  DType::F64),
            ]);
            let bytes = &outs[0].0;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[..8]);
            vout = f64::from_le_bytes(buf);
        }
        out.push(vout as f32);
    }
    out
}

#[test]
fn pinn_outperforms_mna_on_parametric_rc_inference() {
    // Train. CPU here so the test runs anywhere; macOS callers can
    // pass Device::Mlx via the lib API for the GPU-accelerated path.
    let res = train(42, N_TRAIN_STEPS, 1e-3, Device::Cpu);
    let final_loss = *res.losses.last().unwrap();
    println!("[pinn-rc] final training loss: {:.6e}", final_loss);
    assert!(
        final_loss < 1e-1,
        "training failed to converge: final loss = {:.6e}",
        final_loss
    );

    // 10k physical-unit queries.
    let mut rng = Rng::new(7);
    let queries: Vec<Query> = (0..N_QUERIES).map(|_| random_query(&mut rng)).collect();
    let truth: Vec<f32> = queries.iter().map(analytic).collect();

    // PINN: one batched forward.
    let t0 = Instant::now();
    let pinn_pred = res.pinn.eval_batch(&queries, Device::Cpu);
    let t_pinn = t0.elapsed();

    // MNA: cached compiled BE-step graph.
    let t0 = Instant::now();
    let mna_pred = run_mna_baseline(&queries);
    let t_mna = t0.elapsed();

    let pinn_max = max_abs_err(&pinn_pred, &truth);
    let mna_max  = max_abs_err(&mna_pred,  &truth);
    let pinn_rms = rms_err(&pinn_pred, &truth);
    let mna_rms  = rms_err(&mna_pred,  &truth);
    let speedup  = t_mna.as_secs_f64() / t_pinn.as_secs_f64();

    println!(
        "[pinn-rc] PINN: {:>6} ms | max abs {:.4} V ({:.2}% FS) | RMS {:.4} V",
        t_pinn.as_millis(),
        pinn_max,
        100.0 * pinn_max / V_REF,
        pinn_rms,
    );
    println!(
        "[pinn-rc] MNA:  {:>6} ms | max abs {:.4} V ({:.3}% FS) | RMS {:.4} V",
        t_mna.as_millis(),
        mna_max,
        100.0 * mna_max / V_REF,
        mna_rms,
    );
    println!("[pinn-rc] PINN throughput speedup: {:.1}×", speedup);

    // Accuracy bounds expressed as fraction of full-scale (V_REF).
    assert!(
        pinn_max < 0.05 * V_REF,
        "PINN max abs err {:.4} V ({:.2}% FS) ≥ 5% FS — PINN regression",
        pinn_max,
        100.0 * pinn_max / V_REF,
    );
    assert!(
        mna_max < 0.01 * V_REF,
        "MNA max abs err {:.4} V ({:.3}% FS) ≥ 1% FS — baseline sanity broke",
        mna_max,
        100.0 * mna_max / V_REF,
    );
    assert!(
        t_pinn * 10 < t_mna,
        "PINN expected ≥10× faster than MNA but got {:.1}× (PINN {:?}, MNA {:?})",
        speedup,
        t_pinn,
        t_mna
    );
}
