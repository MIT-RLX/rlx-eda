//! Quick demo: print the receiver-voltage table for a 100 MHz, 3.3 V clock
//! into a 50 Ω, 1 ns transmission line with a high-Z receiver. Two cases
//! side-by-side: unterminated (rings) and series-matched (clean).

use eda_hir::SourceWaveform;
use spike_tline_termination::{analytic_pulse_at, fdtd_trace, Topology};

fn main() {
    let td = 1e-9;             // 1 ns line, ≈ 6" of FR-4 microstrip
    let bad = Topology::unterminated(td);
    let good = Topology::series_matched(td);

    let v_high = 3.3_f64;
    // Single rising edge at 1 ns, hold high through the window. (We stay
    // in single-edge land so the analytic stays simple.)
    let w = SourceWaveform::pulse(0.0, v_high, 1e-9, 0.0, 0.0, 100e-9, 0.0);

    // h chosen so TD/h is integer (50 cells per direction).
    let h = td / 50.0;
    let t_stop = 10e-9;
    let n_steps = (t_stop / h).round() as usize;

    let (t_bad, vrx_bad)   = fdtd_trace(bad,  h, n_steps, &w);
    let (_t_good, vrx_good) = fdtd_trace(good, h, n_steps, &w);

    println!("T-line termination spike: 100 MHz, V_high = {v_high} V");
    println!("  Unterminated: R_drv={} Ω, R_term=0, Γ_S={:+.3}",
             bad.r_drv, bad.gamma_s());
    println!("  Matched:      R_drv={} Ω, R_term={} Ω, Γ_S={:+.3}",
             good.r_drv, good.r_term, good.gamma_s());
    println!();
    println!("{:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
             "t [ns]", "bad FDTD", "bad anal.", "good FDTD", "good anal.");

    let probe_indices = [0_usize, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500];
    for &i in &probe_indices {
        if i >= t_bad.len() { break; }
        let t = t_bad[i];
        let bad_a  = analytic_pulse_at(bad,  t, &w);
        let good_a = analytic_pulse_at(good, t, &w);
        println!("{:>10.3}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
                 t * 1e9, vrx_bad[i], bad_a, vrx_good[i], good_a);
    }

    let bad_max = vrx_bad.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bad_min = vrx_bad.iter().cloned().fold(f64::INFINITY, f64::min);
    let good_max = vrx_good.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!();
    println!("Unterminated: peak overshoot = {:.3} V ({:.0}% of V_high), undershoot = {:.3} V",
             bad_max, 100.0 * bad_max / v_high, bad_min);
    println!("Matched:      peak           = {:.3} V (settled at V_high)", good_max);
}
