//! Train the MOSFET thermal surrogate and verify it actually learned
//! the underlying analytic surface.
//!
//! Two assertions:
//! 1. **Loss collapse.** Final mean loss is at least 100× below the
//!    initial loss — confirms Adam + the MLP architecture made
//!    progress at all.
//! 2. **Pointwise agreement.** At a small held-out grid of (Vgs, Vds, T)
//!    points covering all three operating regions × all three thermal
//!    corners, the surrogate's `Id` prediction is within `MAX_REL`
//!    relative of the analytic ground truth (or `MAX_ABS` absolute,
//!    whichever is looser — important near cutoff where Id ≈ 0 and
//!    relative error explodes).

use spike_mosfet_dc::run_id_at_temp;
use spike_mosfet_dc::surrogate::{predict, train};

const VTH0: f64 = 0.5;
const KP0:  f64 = 100e-6;
const LAM:  f64 = 0.02;

const MAX_REL: f32 = 0.10;   // 10 % relative tolerance
const MAX_ABS: f32 = 5e-6;   // 5 µA absolute tolerance (noise floor for cutoff)

#[test]
fn surrogate_learns_thermal_id_surface() {
    let result = train(/*n_steps=*/ 2000, /*lr=*/ 1e-2, /*seed=*/ 0xC0FFEE);

    // 1. Loss collapse.
    let initial = result.losses[..10].iter().copied().sum::<f32>() / 10.0;
    let final_  = result.losses[result.losses.len()-10..].iter().copied().sum::<f32>() / 10.0;
    let drop = initial / final_.max(1e-12);
    eprintln!("initial loss = {initial:.4e}, final loss = {final_:.4e}, drop = {drop:.1}×");
    assert!(drop > 100.0, "loss only dropped {drop:.1}× — surrogate didn't learn");

    // 2. Pointwise agreement on a held-out grid.
    let bias_points = &[
        (0.4, 0.5, "subthreshold"),
        (0.9, 0.3, "triode"),
        (0.9, 1.5, "saturation knee"),
        (1.5, 1.5, "deep saturation"),
        (1.8, 1.0, "max overdrive"),
    ];
    let temps = &[-40.0_f64, 27.0, 125.0];

    let mut max_rel_seen = 0.0_f32;
    let mut n_checked = 0;
    for &(vgs, vds, label) in bias_points {
        for &t in temps {
            let truth = run_id_at_temp(vgs, vds, VTH0, KP0, LAM, t) as f32;
            let pred  = predict(&result.final_weights, vgs as f32, vds as f32, t as f32);

            let abs = (pred - truth).abs();
            let rel = abs / truth.abs().max(1e-12);
            // Only enforce relative tolerance when the absolute error
            // is above the noise floor — at deep cutoff Id ≈ 0 and the
            // surrogate's tiny absolute slop makes rel meaningless.
            let pass = abs < MAX_ABS || rel < MAX_REL;
            assert!(pass,
                "[{label}, T={t}°C] truth={truth:.4e} A, pred={pred:.4e} A, abs={abs:.2e}, rel={rel:.3}");

            if abs > MAX_ABS { max_rel_seen = max_rel_seen.max(rel); }
            n_checked += 1;
        }
    }
    eprintln!("checked {n_checked} (bias, T) points; worst above-noise rel error = {max_rel_seen:.3}");
}
