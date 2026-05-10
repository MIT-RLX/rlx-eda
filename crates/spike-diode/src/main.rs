//! Print Newton convergence + AD vs analytic vs FD for the diode spike.

use spike_diode::*;

fn main() {
    let v   = 1.0_f32;
    let r   = 1_000.0_f32;       // 1 kΩ
    let is_ = 1e-15_f32;          // typical silicon diode saturation current
    let vt  = VT;
    let n   = 20;

    let vmid_rlx = run_forward(v, r, is_, vt, n);
    let vmid_ref = ref_newton(v, r, is_, vt, n);

    let (vmid_ad, d_r_ad, d_is_ad) = run_forward_and_grad(v, r, is_, vt, n);
    let d_r_an  = analytic_dvmid_dr(v, r, is_, vt, vmid_rlx);
    let d_is_an = analytic_dvmid_dis(v, r, is_, vt, vmid_rlx);

    // FD as a third witness (relative perturbation).
    let h_r  = 1e-3 * r;
    let h_is = 1e-3 * is_;
    let d_r_fd  = (run_forward(v, r + h_r, is_, vt, n) - run_forward(v, r - h_r, is_, vt, n)) / (2.0 * h_r);
    let d_is_fd = (run_forward(v, r, is_ + h_is, vt, n) - run_forward(v, r, is_ - h_is, vt, n)) / (2.0 * h_is);

    println!("Diode-Resistor DC operating point (Newton, {n} unrolled iters)");
    println!("  V={v} V, R={r} Ω, Is={is_:.0e} A, Vt={vt:.5} V");
    println!();
    println!("  Vmid");
    println!("    rlx forward:  {vmid_rlx:+.6e} V");
    println!("    rlx + AD:     {vmid_ad:+.6e} V");
    println!("    Rust newton:  {vmid_ref:+.6e} V");
    println!();
    println!("  ∂Vmid/∂R");
    println!("    analytic IFT: {d_r_an:+.6e}");
    println!("    rlx AD:       {d_r_ad:+.6e}");
    println!("    FD (rlx):     {d_r_fd:+.6e}");
    println!();
    println!("  ∂Vmid/∂Is");
    println!("    analytic IFT: {d_is_an:+.6e}");
    println!("    rlx AD:       {d_is_ad:+.6e}");
    println!("    FD (rlx):     {d_is_fd:+.6e}");
}
