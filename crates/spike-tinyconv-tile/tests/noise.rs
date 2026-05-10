//! `NoiseModel` closed-form smoke tests. Pure-function — no SPICE,
//! no foundry library. Verifies the qualitative shape of the
//! placeholder model:
//!   - σ grows when Vdd drops below nominal,
//!   - σ shrinks as transistors get larger (Pelgrom 1/√(WL)),
//!   - σ is positive everywhere defined,
//!   - calibrate() reports honest "untrusted" status until the real
//!     fitter lands.

use spike_tinyconv_tile::{NoiseModel, NoiseStats, TileParams};

fn nominal_params() -> TileParams {
    TileParams {
        w_l_n: 1.0,
        w_l_p: 1.0,
        vdd: 1.8,
        bias_v: 0.0,
        weight_bits: 8,
    }
}

#[test]
fn evaluate_returns_finite_positive_sigma() {
    let m = NoiseModel::default();
    let stats = m.evaluate(nominal_params());
    assert_eq!(stats.mean_lsb, 0.0, "balanced design → mean offset 0");
    assert!(stats.sigma_lsb > 0.0, "thermal floor → σ > 0");
    assert!(stats.sigma_lsb.is_finite());
}

#[test]
fn lower_vdd_increases_sigma() {
    let m = NoiseModel::default();
    let nom = m.evaluate(nominal_params());
    let dropped = m.evaluate(TileParams {
        vdd: 0.9, // half of nominal
        ..nominal_params()
    });
    assert!(
        dropped.sigma_lsb > nom.sigma_lsb,
        "halving Vdd should raise σ: nom={} dropped={}",
        nom.sigma_lsb,
        dropped.sigma_lsb,
    );
}

#[test]
fn larger_transistors_reduce_sigma() {
    let m = NoiseModel::default();
    let small = m.evaluate(TileParams { w_l_n: 0.5, w_l_p: 0.5, ..nominal_params() });
    let large = m.evaluate(TileParams { w_l_n: 4.0, w_l_p: 4.0, ..nominal_params() });
    assert!(
        large.sigma_lsb < small.sigma_lsb,
        "larger W/L should lower σ (Pelgrom): small={} large={}",
        small.sigma_lsb,
        large.sigma_lsb,
    );
}

#[test]
fn above_nominal_vdd_does_not_increase_supply_term() {
    // Vdd > nominal: supply contribution clamps to 0; σ only from
    // Pelgrom + thermal. Should equal the nominal-Vdd case where
    // supply also contributes 0.
    let m = NoiseModel::default();
    let high = m.evaluate(TileParams { vdd: 2.5, ..nominal_params() });
    let nom = m.evaluate(nominal_params());
    assert!((high.sigma_lsb - nom.sigma_lsb).abs() < 1e-9);
}

#[test]
fn evaluate_handles_pathological_sizing_without_nan() {
    // Zero W/L → guarded clamp; result must stay finite.
    let m = NoiseModel::default();
    let stats = m.evaluate(TileParams { w_l_n: 0.0, w_l_p: 0.0, ..nominal_params() });
    assert!(stats.sigma_lsb.is_finite(), "zero W/L should not NaN");
    assert!(stats.sigma_lsb > 0.0);
}

#[test]
fn calibrate_reports_untrusted_until_real_fitter_lands() {
    // v1 stub semantics: residual = ∞ regardless of sample count.
    // When the real least-squares fitter lands, this test changes
    // shape (assert residual decreases with more samples).
    let mut m = NoiseModel::default();
    let r1 = m.calibrate(&[]);
    assert_eq!(r1, f64::INFINITY);

    let s = NoiseStats { mean_lsb: 0.0, sigma_lsb: 1.0, calibration_residual: 0.0 };
    let r2 = m.calibrate(&[(nominal_params(), s)]);
    assert_eq!(r2, f64::INFINITY);
}
