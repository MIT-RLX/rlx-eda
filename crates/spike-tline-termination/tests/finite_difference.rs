//! Tier 2: discrete delay-line FDTD vs the closed-form bounce diagram.
//!
//! On a uniform grid where `TD/h` is integer, the FDTD is exact for a
//! lossless line: each timestep shifts both wave queues by exactly one
//! cell, and the boundary conditions reflect with the same gammas the
//! analytic uses. The two should agree to machine precision *except* at
//! the ideal step discontinuities, where any sample taken exactly on the
//! transition is ambiguous (left- vs right-continuous). We therefore
//! probe strictly inside the constant plateaus.

use eda_hir::SourceWaveform;
use spike_tline_termination::{analytic_pulse_at, fdtd_trace, Topology};

fn assert_close(a: f64, b: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * b.abs();
    if (a - b).abs() > env {
        panic!("[{label}] FDTD={a:.6e} analytic={b:.6e} diff={:.3e} > env={env:.3e}",
               (a - b).abs());
    }
}

#[test]
fn fdtd_unterminated_matches_analytic_step_response() {
    let td = 1e-9;
    let topo = Topology::unterminated(td);

    // 50 cells per direction → h = 20 ps.
    let h = td / 50.0;
    let t_stop = 12.0 * td;          // covers ~6 half-periods of ringing
    let n_steps = (t_stop / h).round() as usize;

    // Rising step at td=0.5·TD so the first front arrives mid-grid.
    let edge = 0.5 * td;
    let w = SourceWaveform::pulse(0.0, 3.3, edge, 0.0, 0.0, t_stop * 2.0, 0.0);

    let (t, v_fd) = fdtd_trace(topo, h, n_steps, &w);

    // Probe strictly inside each plateau region. k-th front arrives at
    // edge + (2k+1)·TD, so plateaus are between consecutive arrivals.
    // Use the midpoint of each plateau for a clean comparison.
    let plateau_mid = |k: usize| edge + (2.0 * k as f64 + 2.0) * td;
    for k in 0..5 {
        let tp = plateau_mid(k);
        let i = (tp / h).round() as usize;
        if i >= t.len() { break; }
        let ana = analytic_pulse_at(topo, t[i], &w);
        assert_close(v_fd[i], ana, 1e-12, 1e-12,
                     &format!("plateau {}", k));
    }
}

#[test]
fn fdtd_matched_settles_to_vs_after_one_transit() {
    let td = 1e-9;
    // Ideal high-Z so the receiver settles at exactly vs (the bundled
    // `series_matched` uses r_load=1e9 for ngspice's DC path; that gives
    // Γ_L = 1 − 5e-8 and the staircase undershoots by ~1.6e-7).
    let topo = Topology { r_load: f64::INFINITY, ..Topology::series_matched(td) };

    let h = td / 25.0; // any integer divisor works
    let t_stop = 8.0 * td;
    let n_steps = (t_stop / h).round() as usize;

    let edge = 0.0;
    let vs = 3.3;
    let w = SourceWaveform::pulse(0.0, vs, edge, 0.0, 0.0, t_stop * 2.0, 0.0);

    let (t, v_fd) = fdtd_trace(topo, h, n_steps, &w);

    // After TD the receiver should sit at exactly vs and never deviate.
    let i_arrive = (td / h).round() as usize;
    for i in i_arrive..t.len() {
        assert_close(v_fd[i], vs, 1e-12, 1e-12,
                     &format!("matched: i={i}, t={:.3e}", t[i]));
    }
}

#[test]
fn fdtd_pulse_returns_to_zero_long_after_falling_edge() {
    // Single pulse: receiver should ring to v2, then ring back to v1
    // after the falling edge. Geometric ringing means it takes a few
    // round trips to settle — this test checks "settled within tolerance"
    // many round trips later.
    let td = 1e-9;
    let topo = Topology::unterminated(td);
    let h = td / 20.0;
    let pw = 4.0 * td;
    let edge = 0.5 * td;
    let v1 = 0.0;
    let v2 = 3.3;

    // Run long enough that |Γ_S|^K is below tolerance after both edges.
    // |Γ_S| = 2/3, so 30 round trips → (2/3)^30 ≈ 5e-6.
    let t_stop = edge + pw + 60.0 * td;
    let n_steps = (t_stop / h).round() as usize;

    let w = SourceWaveform::pulse(v1, v2, edge, 0.0, 0.0, pw, 0.0);
    let (t, v_fd) = fdtd_trace(topo, h, n_steps, &w);

    // Compare last ~10 samples to v1; both edges should have rung out.
    let n_tail = 10;
    let tail_start = t.len() - n_tail;
    for i in tail_start..t.len() {
        let ana = analytic_pulse_at(topo, t[i], &w);
        assert_close(v_fd[i], ana, 1e-9, 1e-9,
                     &format!("tail i={i}"));
        assert!((v_fd[i] - v1).abs() < 1e-3,
                "tail not settled to v1: v_fd[{i}]={:.6}", v_fd[i]);
    }
}
