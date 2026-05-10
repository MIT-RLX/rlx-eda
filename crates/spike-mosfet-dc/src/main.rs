//! Print Id at a few representative operating points: well inside
//! cutoff / triode / saturation, and right at the saturation boundary.

use spike_mosfet_dc::{
    analytic_did_dkp_saturation, analytic_did_dlam_saturation, analytic_did_dvth_saturation,
    id_strict, run_id, run_id_grad,
};

fn main() {
    let vth = 0.5_f64;
    let kp = 100e-6_f64;     // 100 µA/V², treated as already W/L'd-in.
    let lam = 0.02_f64;

    println!("L1 NMOS Id (Vth={vth}, kp_eff={kp:.0e}, λ={lam}):");
    println!("{:>8} {:>8} {:>14} {:>14} {:>10} {:>14}",
        "Vgs", "Vds", "rlx Id [µA]", "strict [µA]", "Δrel", "region");

    let cases = &[
        // (Vgs, Vds, label)
        (0.30, 2.0, "cutoff"),
        (2.00, 0.3, "triode"),
        (2.00, 1.5, "saturation (Vds=Vov)"),
        (2.00, 2.0, "saturation"),
        (2.00, 5.0, "deep saturation"),
        (3.30, 1.8, "high-bias saturation"),
    ];

    for &(vgs, vds, label) in cases {
        let id_rlx = run_id(vgs, vds, vth, kp, lam);
        let id_a = id_strict(vgs, vds, vth, kp, lam);
        let rel = if id_a.abs() > 1e-12 { (id_rlx - id_a).abs() / id_a.abs() } else { 0.0 };
        println!("{:>8.2} {:>8.2} {:>14.3} {:>14.3} {:>10.2e}  {label}",
            vgs, vds, id_rlx * 1e6, id_a * 1e6, rel);
    }

    println!("\n[debug] run_id_grad at saturation Vgs=2.0 Vds=2.0:");
    let (id, dvth, dkp, dlam) = run_id_grad(2.0, 2.0, vth, kp, lam);
    let an_dvth = analytic_did_dvth_saturation(2.0, 2.0, vth, kp, lam);
    let an_dkp  = analytic_did_dkp_saturation(2.0, 2.0, vth, kp, lam);
    let an_dlam = analytic_did_dlam_saturation(2.0, 2.0, vth, kp, lam);
    println!("  Id   = {id:.6e}");
    println!("  dVth: AD = {dvth:.6e}  analytic = {an_dvth:.6e}");
    println!("  dKp:  AD = {dkp:.6e}  analytic = {an_dkp:.6e}");
    println!("  dLam: AD = {dlam:.6e}  analytic = {an_dlam:.6e}");
}
