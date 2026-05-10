//! Tier 1: closed-form bounce-diagram check at known characteristic times.
//!
//! For a high-Z load (Γ_L = +1) and a Thevenin source impedance
//! `R_s = R_drv + R_term`, the receiver voltage is a piecewise-constant
//! staircase with steps at `t = (2k+1)·TD`. We verify three things:
//!
//! 1. Unterminated case: first arrival at `t = TD` overshoots to
//!    `2·V_+`; long-time limit settles at `V_s` (geometric series sums to
//!    `V_s` whenever Γ_S·Γ_L is a contraction).
//! 2. Matched case (`R_s = Z_0`): single arrival at `t = TD`, settles at
//!    `V_s` immediately, no further transitions.
//! 3. Numerical example matching the LinkedIn quiz screenshot:
//!    R_drv = 10 Ω → 5.5 V on a 3.3 V rail, undershoot to ~0.55 V·V_s_step.

use eda_hir::SourceWaveform;
use spike_tline_termination::{analytic_pulse_at, analytic_step_at, Topology};

fn assert_close(a: f64, b: f64, tol: f64, label: &str) {
    if (a - b).abs() > tol {
        panic!("[{label}] {a:.6} vs {b:.6}, diff {:.3e} > {tol:.3e}",
               (a - b).abs());
    }
}

/// The bundled `unterminated`/`series_matched` constructors set `r_load=1e9`
/// so the SPICE deck has a DC path; that gives Γ_L = 1 − 1e-7, ie. the
/// receiver-voltage closed forms drift by ~1e-7 from the ideal-high-Z
/// values these tests assert. Override `r_load` to `INFINITY` so the
/// analytic checks lock down exact integer/rational values.
fn ideal(mut topo: Topology) -> Topology {
    topo.r_load = f64::INFINITY;
    topo
}

#[test]
fn matched_case_settles_in_one_transit() {
    let td = 1e-9;
    let topo = ideal(Topology::series_matched(td));
    let vs = 3.3;

    // Just before TD: still zero.
    assert_close(analytic_step_at(topo, 0.5 * td, vs), 0.0, 1e-12,
                 "matched: pre-arrival");
    // At TD: V_+ = V_s/2 (since R_s = Z_0), receiver doubles to V_s.
    assert_close(analytic_step_at(topo, td, vs), vs, 1e-12,
                 "matched: arrival at TD");
    // Long after: still V_s, no second front (Γ_S = 0 absorbs reflection).
    assert_close(analytic_step_at(topo, 100.0 * td, vs), vs, 1e-12,
                 "matched: long after");
}

#[test]
fn unterminated_overshoots_then_rings() {
    let td = 1e-9;
    let topo = ideal(Topology::unterminated(td)); // R_drv=10, R_term=0, Z0=50
    let vs = 3.3;

    let vp = topo.v_plus(vs);            // 3.3 · 50/60 = 2.75
    let gs = topo.gamma_s();             // (10-50)/60 = -2/3
    assert_close(vp, 2.75, 1e-12, "v_plus");
    assert_close(gs, -2.0 / 3.0, 1e-12, "gamma_s");

    // First arrival at TD: 2·V_+ = 5.5 V — exactly the LinkedIn-quiz peak.
    assert_close(analytic_step_at(topo, td, vs), 2.0 * vp, 1e-12,
                 "first-arrival overshoot");
    // Sample inside [TD, 3·TD): plateau, no change since the next bounce
    // hasn't returned from the source yet.
    assert_close(analytic_step_at(topo, 2.5 * td, vs),
                 2.0 * vp, 1e-12, "first plateau");

    // Second arrival at 3·TD: undershoot. Adds 2·V_+·Γ_S·Γ_L = 2·V_+·(-2/3).
    let expected_after_2nd = 2.0 * vp + 2.0 * vp * gs;
    assert_close(analytic_step_at(topo, 3.0 * td, vs),
                 expected_after_2nd, 1e-12, "second-arrival undershoot");
    // Sanity: undershoots below V_s (the steady-state value).
    assert!(expected_after_2nd < vs,
            "expected undershoot below V_s; got {expected_after_2nd}");

    // Long-time limit: geometric series Σ (Γ_S·Γ_L)^k = 1/(1−Γ_S·Γ_L).
    // Receiver voltage = (1+Γ_L)·V_+ / (1−Γ_S·Γ_L) = V_s exactly for Γ_L=1.
    // Pick a t that has K=50 round trips; the residual is (Γ_S)^51 ≈ 1e-9.
    let t_far = (2 * 50 + 1) as f64 * td;
    assert_close(analytic_step_at(topo, t_far, vs), vs, 1e-8,
                 "long-time limit (50 round trips)");
}

#[test]
fn pulse_response_is_two_superposed_steps() {
    // A rising edge at td=1ns and falling edge at td+pw=4ns. Between the
    // two, the receiver should ring around v2; after the falling edge,
    // ring around v1 = 0.
    let td = 1e-9;
    let topo = ideal(Topology::unterminated(td));
    let v1 = 0.0;
    let v2 = 3.3;
    let edge_td = 1e-9;
    let pw = 3e-9;
    let w = SourceWaveform::pulse(v1, v2, edge_td, 0.0, 0.0, pw, 0.0);

    // Just after the rising edge but before its first arrival: still v1.
    let pre = analytic_pulse_at(topo, edge_td + 0.5 * td, &w);
    assert_close(pre, v1, 1e-12, "before first arrival");

    // First arrival of rising edge at t = edge_td + TD: 2·V_+(v2−v1).
    let t_arrive = edge_td + td;
    let v_arrive = analytic_pulse_at(topo, t_arrive, &w);
    let expected = v1 + 2.0 * topo.v_plus(v2 - v1);
    assert_close(v_arrive, expected, 1e-12, "rising-edge first arrival");

    // After the falling edge has had time to make its own first transit
    // (t = edge_td + pw + TD), the receiver-voltage formula is the sum of
    // the two step responses, both multi-bounced to that time. We just
    // assert it landed somewhere near v1 ± a bounce — a coarse smoke check.
    let t_late = edge_td + pw + 5.0 * td;
    let v_late = analytic_pulse_at(topo, t_late, &w);
    assert!(v_late.is_finite(), "non-finite late-time voltage");
    assert!((v_late - v1).abs() < (v2 - v1),
            "late-time should be ringing around v1, got {v_late}");
}
