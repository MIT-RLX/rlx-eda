//! Tiny driver: print a snapshot of the rlx vs analytic trace at a few
//! representative times for a 0→1V step into a 1kΩ × 1nF RC LP.

use eda_hir::SourceWaveform;
use spike_pulse_rc::{analytic_pulse_at, run_transient_trace};

fn main() {
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let rc = r * c;

    // Single 1V pulse, 50 ns delay, 200 ns wide, no rise/fall, no repeat.
    let w = SourceWaveform::pulse(0.0, 1.0, 50e-9, 0.0, 0.0, 200e-9, 0.0);

    let n_steps = 1000;
    let t_stop = 400e-9;
    let h = t_stop / n_steps as f64;
    let (t, v) = run_transient_trace(n_steps, h, r, c, 0.0, &w);

    println!("RC LP pulse spike: R={r}Ω C={c:.0e}F τ={rc:.2e}s");
    println!("{:>10}  {:>14}  {:>14}  {:>10}", "t [ns]", "rlx vout", "analytic", "Δ");
    for &k in &[0, 250, 499, 500, 700, 999] {
        let tk = t[k];
        let vk = v[k];
        let ana = analytic_pulse_at(tk, 0.0, 1.0, 50e-9, 200e-9, r, c);
        println!("{:>10.2}  {:>+14.6e}  {:>+14.6e}  {:>+10.2e}",
                 tk * 1e9, vk, ana, vk - ana);
    }
}
