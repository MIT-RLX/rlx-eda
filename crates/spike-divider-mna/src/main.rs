//! Print all four references side-by-side for the MNA divider.

use spike_divider_mna::*;

fn main() {
    let v = 1.0_f64;
    let r1 = 1_000.0_f64;
    let r2 = 1_000.0_f64;

    let (vout_ad, d_r1_ad, d_r2_ad) = run_forward_and_grad_mna(v, r1, r2);
    let vout_an = analytic_vout(v, r1, r2);
    let d_r1_an = analytic_dvout_dr1(v, r1, r2);
    let d_r2_an = analytic_dvout_dr2(v, r1, r2);

    // f64 FD with relative perturbation; sweet spot is eps^(1/3) of magnitude.
    let h1 = (1e-5 * r1).max(1e-9);
    let h2 = (1e-5 * r2).max(1e-9);
    let d_r1_fd = (analytic_vout(v, r1 + h1, r2) - analytic_vout(v, r1 - h1, r2)) / (2.0 * h1);
    let d_r2_fd = (analytic_vout(v, r1, r2 + h2) - analytic_vout(v, r1, r2 - h2)) / (2.0 * h2);

    println!("voltage divider via MNA + DenseSolve");
    println!("  V = {v}, R1 = {r1}, R2 = {r2}");
    println!();
    println!("  Vout");
    println!("    analytic:  {vout_an:+.12e}");
    println!("    rlx solve: {vout_ad:+.12e}");
    println!();
    println!("  dVout/dR1");
    println!("    analytic:  {d_r1_an:+.12e}");
    println!("    rlx AD:    {d_r1_ad:+.12e}");
    println!("    FD (f64):  {d_r1_fd:+.12e}");
    println!();
    println!("  dVout/dR2");
    println!("    analytic:  {d_r2_an:+.12e}");
    println!("    rlx AD:    {d_r2_ad:+.12e}");
    println!("    FD (f64):  {d_r2_fd:+.12e}");

    #[cfg(feature = "ngspice")]
    {
        use eda_extern_ngspice::{Invoker, LocalBinary, OutputRequest};
        match LocalBinary::from_env() {
            Ok(ng) => {
                let deck = spice_deck(v, r1, r2);
                let req = vec![OutputRequest::NodeVoltage("vout".into())];
                match ng.run_dc(&deck, &req) {
                    Ok(r) => {
                        let v_ng = r.node_voltages["vout"];
                        println!();
                        println!("  ngspice");
                        println!("    Vout:      {v_ng:+.12e}");
                    }
                    Err(e) => eprintln!("ngspice run failed: {e}"),
                }
            }
            Err(e) => eprintln!("ngspice unavailable: {e}"),
        }
    }
}
