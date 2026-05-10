//! Tier 1: rlx-side smooth `Id` vs strict-piecewise analytic `Id`,
//! across cutoff, triode, and saturation.
//!
//! The graph implements the textbook smooth Level-1 formula:
//!
//!   `Vov_s   = (1/β)·log(1 + exp(β·(Vgs - Vth)))`     (softplus)
//!   `Vds_eff = ½·(Vds + Vov_s − √((Vds − Vov_s)² + δ))` (smooth-min)
//!   `Id      = kp · (Vov_s · Vds_eff − Vds_eff²/2) · (1 + λ·Vds)`
//!
//! With β=200 (~5 mV cutoff width) and δ=1e-4 V² (~10 mV knee width),
//! agreement with strict piecewise is sub-ppm well inside each region
//! and ~few-percent in the smoothing windows.

use spike_mosfet_dc::{id_strict, run_id};

const VTH: f64 = 0.5;
const KP: f64 = 100e-6;
const LAM: f64 = 0.02;

#[track_caller]
fn assert_close_or_zero(rlx: f64, strict: f64, rtol: f64, atol: f64, label: &str) {
    let env = atol + rtol * strict.abs();
    if (rlx - strict).abs() > env {
        panic!("[{label}] not close: rlx={rlx:+.6e} strict={strict:+.6e} |Δ|={:.3e} env={env:.3e}",
            (rlx - strict).abs());
    }
}

#[test]
fn deep_cutoff_id_essentially_zero() {
    // 200 mV below threshold (Vov = -0.2): softplus → essentially zero.
    // exp(-40) ≈ 4e-18, log(1+4e-18) ≈ 4e-18, Vov_s ≈ 2e-20. Then Id
    // is dominated by Vov_s² · kp ~ 1e-44 ≈ 0 to f64 precision.
    let id_rlx = run_id(0.30, 2.0, VTH, KP, LAM);
    // f64 round-off in the smooth-min cancellation can leak ~1e-14;
    // anything below that is "zero" for any physical interpretation.
    assert!(id_rlx.abs() < 1e-13,
        "deep cutoff Id should be ~0, got {id_rlx:.3e}");
    let id_s = id_strict(0.30, 2.0, VTH, KP, LAM);
    assert_eq!(id_s, 0.0, "strict cutoff Id must be exactly 0, got {id_s:.3e}");
}

#[test]
fn near_cutoff_smoothing_window_bounds() {
    // 5 mV below threshold (Vov = -0.005): right inside the softplus
    // smoothing window (β=200 ⇒ width ~5 mV). Smooth Id is non-zero
    // but small. Just check it's positive (rather than spuriously
    // negative due to numerics) and below 1% of the on-current.
    let id_rlx = run_id(VTH - 5e-3, 2.0, VTH, KP, LAM);
    let id_on = run_id(2.0, 2.0, VTH, KP, LAM);
    assert!(id_rlx >= 0.0, "near-cutoff Id should be non-negative, got {id_rlx:.3e}");
    assert!(id_rlx < 0.01 * id_on,
        "near-cutoff Id = {id_rlx:.3e} > 1% of on-current {id_on:.3e}");
}

#[test]
fn deep_triode_matches_strict() {
    // Vov = 1.5, Vds = 0.3 ≪ Vov: deep triode. Smooth-min rounding at
    // δ=1e-4 V² contributes O(δ/(Vov-Vds)²) ≈ 1e-4/1.44 ≈ 7e-5 relative.
    let rlx = run_id(2.0, 0.3, VTH, KP, LAM);
    let strict = id_strict(2.0, 0.3, VTH, KP, LAM);
    assert_close_or_zero(rlx, strict, 1e-3, 1e-12, "triode interior");
}

#[test]
fn shallow_triode_matches_strict() {
    // Vov = 1.5, Vds = 1.0: half-overdrive triode. Still well-resolved.
    let rlx = run_id(2.0, 1.0, VTH, KP, LAM);
    let strict = id_strict(2.0, 1.0, VTH, KP, LAM);
    assert_close_or_zero(rlx, strict, 1e-3, 1e-12, "shallow triode");
}

#[test]
fn saturation_at_vov_boundary_matches_strict() {
    // Vov = 1.5, Vds = Vov: right at the saturation/triode knee. The
    // smooth-min rounding (O(√δ) ≈ 10 mV here) shows up most strongly
    // at this point; even so, the relative error stays < 0.1%.
    let rlx = run_id(2.0, 1.5, VTH, KP, LAM);
    let strict = id_strict(2.0, 1.5, VTH, KP, LAM);
    assert_close_or_zero(rlx, strict, 1e-3, 1e-12, "saturation @ Vds=Vov");
}

#[test]
fn deep_saturation_matches_strict() {
    // Vov = 1.5, Vds = 2.0: comfortably in saturation. smooth-min's
    // residual is O(δ / (Vds - Vov)²) ≈ 4e-4 relative.
    let rlx = run_id(2.0, 2.0, VTH, KP, LAM);
    let strict = id_strict(2.0, 2.0, VTH, KP, LAM);
    assert_close_or_zero(rlx, strict, 1e-3, 1e-12, "saturation interior");
}

#[test]
fn far_saturation_matches_strict() {
    // Vds = 5.0, well above Vov: O(δ / (Vds - Vov)²) → ~1e-5.
    let rlx = run_id(2.0, 5.0, VTH, KP, LAM);
    let strict = id_strict(2.0, 5.0, VTH, KP, LAM);
    assert_close_or_zero(rlx, strict, 1e-4, 1e-12, "deep saturation");
}

#[test]
fn higher_overdrive_scales_quadratically_in_saturation() {
    // Id ∝ (Vgs - Vth)² in saturation. Doubling Vov should ~4× Id.
    let id_low  = run_id(VTH + 0.5, 2.0, VTH, KP, LAM);
    let id_high = run_id(VTH + 1.0, 2.0, VTH, KP, LAM);
    let ratio = id_high / id_low;
    assert!((ratio - 4.0).abs() < 0.02,
        "Vov-doubling Id ratio = {ratio:.4}, expected ≈ 4");
}

#[test]
fn lambda_zero_yields_nearly_lambda_independent_id_in_saturation() {
    // λ = 0 in saturation: strict Id is exactly Vds-independent. With
    // smooth_min in the graph, Vds_eff approaches Vov_s asymptotically
    // as Vds grows, leaving a residual O(δ/(Vds-Vov)²) dependence on
    // Vds — ~5e-4 at Vds=2, ~5e-6 at Vds=5.
    let id_a = run_id(2.0, 2.0, VTH, KP, 0.0);
    let id_b = run_id(2.0, 5.0, VTH, KP, 0.0);
    let rel = (id_a - id_b).abs() / id_a.abs();
    assert!(rel < 1e-3,
        "λ=0 Id varied with Vds: {id_a:.6e} vs {id_b:.6e} (rel = {rel:.3e})");
}
