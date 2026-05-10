//! Print BE single-step + transient + ngspice references for an RC LP.

use spike_rc_transient::*;

fn main() {
    let v_dc = 1.0_f64;
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let rc = r * c;
    let t_stop = rc;            // 1 time constant
    let n_steps = 100;
    let h = t_stop / n_steps as f64;

    // ── Single BE step ──────────────────────────────────────────────
    let vout_prev = 0.3_f64;     // arbitrary state to make ∂/∂C non-zero
    let (vn_rlx, dr_rlx, dc_rlx) = run_step_and_grad(v_dc, vout_prev, r, c, h);
    let vn_an = analytic_step(v_dc, vout_prev, r, c, h);
    let dr_an = analytic_dstep_dr(v_dc, vout_prev, r, c, h);
    let dc_an = analytic_dstep_dc(v_dc, vout_prev, r, c, h);

    println!("BE single step:  V={v_dc}, vout_prev={vout_prev}, R={r}, C={c}, h={h:.3e}");
    println!("  vout_n");
    println!("    analytic:  {vn_an:+.12e}");
    println!("    rlx fwd:   {vn_rlx:+.12e}");
    println!("  ∂vout_n/∂R");
    println!("    analytic:  {dr_an:+.12e}");
    println!("    rlx AD:    {dr_rlx:+.12e}");
    println!("  ∂vout_n/∂C");
    println!("    analytic:  {dc_an:+.12e}");
    println!("    rlx AD:    {dc_rlx:+.12e}");

    // ── Multi-step transient (constant DC, zero IC) ─────────────────
    let vout_t_rlx       = run_transient(n_steps, h, r, c, 0.0, |_| v_dc);
    let vout_t_ref       = ref_transient(n_steps, h, r, c, 0.0, |_| v_dc);
    let vout_t_an        = analytic_transient_dc(v_dc, n_steps, h, r, c);
    let vout_t_continuum = continuum_transient_dc(v_dc, t_stop, r, c);

    println!();
    println!("BE transient:    N={n_steps} steps, T={t_stop:.3e} (= 1·RC), h={h:.3e}");
    println!("  vout(T)");
    println!("    rlx loop:        {vout_t_rlx:+.12e}");
    println!("    pure-Rust BE:    {vout_t_ref:+.12e}");
    println!("    analytic BE:     {vout_t_an:+.12e}");
    println!("    continuum:       {vout_t_continuum:+.12e}   ← BE → continuum as h → 0");

    #[cfg(feature = "ngspice")]
    {
        use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest, TransientAnalysis};
        match LocalBinary::from_env() {
            Ok(ng) => {
                let deck = spice_deck(v_dc, r, c);
                let analysis = TransientAnalysis::new(h, t_stop);
                match ng.run_transient_final(&deck, &analysis,
                    &[OutputRequest::NodeVoltage("vout".into())])
                {
                    Ok(r) => {
                        let v_ng = r.node_voltages["vout"];
                        println!("    ngspice .tran:   {v_ng:+.12e}");
                    }
                    Err(e) => eprintln!("ngspice run failed: {e}"),
                }
            }
            Err(e) => eprintln!("ngspice unavailable: {e}"),
        }
    }

    // ── End-to-end AD on the unrolled N-step graph ──────────────────
    let (v_n_unr, d_r_unr, d_c_unr) = run_unrolled_and_grad(v_dc, 0.0, h, r, c, n_steps);
    let d_r_an = analytic_dtransient_dr(v_dc, 0.0, n_steps, h, r, c);
    let d_c_an = analytic_dtransient_dc(v_dc, 0.0, n_steps, h, r, c);
    println!();
    println!("End-to-end AD across N={n_steps} unrolled BE steps");
    println!("  vout(T)        rlx unrolled  {v_n_unr:+.12e}");
    println!("  ∂vout(T)/∂R");
    println!("    analytic:    {d_r_an:+.12e}");
    println!("    rlx AD:      {d_r_unr:+.12e}");
    println!("  ∂vout(T)/∂C");
    println!("    analytic:    {d_c_an:+.12e}");
    println!("    rlx AD:      {d_c_unr:+.12e}");
}
