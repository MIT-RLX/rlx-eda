//! Print a Bode-table snapshot of the rlx vs analytic RC LP response.

use spike_ac::{analytic_mag, analytic_phase, run_ac_sweep};

fn main() {
    let r = 1_000.0_f64;
    let c = 1e-9_f64;
    let f_start = 1e3;
    let f_stop = 1e9;
    let (freq, re, im) = run_ac_sweep(f_start, f_stop, 4, r, c);

    println!("RC LP AC sweep: R={r}Ω C={c:.0e}F → fc = {:.2}MHz", 1.0 / (2.0 * std::f64::consts::PI * r * c) / 1e6);
    println!("{:>10}  {:>14}  {:>14}  {:>14}  {:>14}", "f [Hz]", "|H| rlx", "|H| analytic", "∠H rlx [°]", "∠H ana [°]");
    for (i, &f) in freq.iter().enumerate() {
        let omega = 2.0 * std::f64::consts::PI * f;
        let mag = (re[i] * re[i] + im[i] * im[i]).sqrt();
        let phase_deg = im[i].atan2(re[i]).to_degrees();
        let mag_a = analytic_mag(omega, r, c);
        let phase_a = analytic_phase(omega, r, c).to_degrees();
        println!("{:>10.2e}  {:>14.6}  {:>14.6}  {:>+14.4}  {:>+14.4}",
                 f, mag, mag_a, phase_deg, phase_a);
    }
}
