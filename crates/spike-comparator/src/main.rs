//! Print a transfer-function snapshot — sweep `v+` − `v−` from −10 mV
//! to +10 mV and show the smooth approach to the ideal step.

use spike_comparator::{run_vout, vout_ideal, Comparator};

fn main() {
    let comp = Comparator::default();
    println!(
        "Comparator behavioral: voh={}, vol={}, k={}/V",
        comp.voh, comp.vol, comp.k
    );
    println!("{:>12}  {:>14}  {:>14}", "Δv [mV]", "rlx vout [V]", "ideal [V]");
    let v_minus = comp.voh * 0.5; // common-mode at mid-rail
    for &dv_mv in &[-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let v_plus = v_minus + dv_mv * 1e-3;
        let v_rlx = run_vout(v_plus, v_minus, comp.k, comp.voh, comp.vol);
        let v_ideal = vout_ideal(v_plus, v_minus, comp.voh, comp.vol);
        println!("{:>12.3}  {:>14.6}  {:>14.3}", dv_mv, v_rlx, v_ideal);
    }
}
