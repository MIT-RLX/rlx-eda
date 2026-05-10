//! Run the voltage divider, print all four references side-by-side.
//!
//! Use `cargo run -p spike-divider` for a quick eyeball check. The
//! integration tests do the actual assertions.

use spike_divider::*;

fn main() {
    let v = 1.0_f32;
    let r1 = 1_000.0_f32;
    let r2 = 1_000.0_f32;

    let (vout_ad, d_r1_ad, d_r2_ad) = run_forward_and_grad(v, r1, r2);
    let vout_an = analytic_vout(v, r1, r2);
    let d_r1_an = analytic_dvout_dr1(v, r1, r2);
    let d_r2_an = analytic_dvout_dr2(v, r1, r2);

    // Relative perturbation: f32 absolute 1e-3 cancels for kΩ-scale values.
    let h1 = (1e-3 * r1).max(1e-6);
    let h2 = (1e-3 * r2).max(1e-6);
    let d_r1_fd = (analytic_vout(v, r1 + h1, r2) - analytic_vout(v, r1 - h1, r2)) / (2.0 * h1);
    let d_r2_fd = (analytic_vout(v, r1, r2 + h2) - analytic_vout(v, r1, r2 - h2)) / (2.0 * h2);

    println!("voltage divider:  V = {v}, R1 = {r1}, R2 = {r2}");
    println!();
    println!("  Vout");
    println!("    analytic:  {vout_an:+.6e}");
    println!("    rlx fwd:   {vout_ad:+.6e}");
    println!();
    println!("  dVout/dR1");
    println!("    analytic:  {d_r1_an:+.6e}");
    println!("    rlx AD:    {d_r1_ad:+.6e}");
    println!("    FD:        {d_r1_fd:+.6e}");
    println!();
    println!("  dVout/dR2");
    println!("    analytic:  {d_r2_an:+.6e}");
    println!("    rlx AD:    {d_r2_ad:+.6e}");
    println!("    FD:        {d_r2_fd:+.6e}");

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
                        println!("    Vout:      {v_ng:+.6e}");
                    }
                    Err(e) => eprintln!("ngspice run failed: {e}"),
                }
            }
            Err(e) => eprintln!("ngspice unavailable: {e}"),
        }
    }
}
