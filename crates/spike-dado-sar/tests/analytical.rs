//! Integration tests for the analytical noise budget.
//!
//! Random-design sanity sweep + a "best-on-paper" design check.

use spike_dado_sar::{
    catalog::{self, *},
    score_analytical::{enob_from_score, score_analytical},
    Rng,
};

fn random_design(rng: &mut Rng) -> Design {
    let mut x = [0u8; L];
    for i in 0..L { x[i] = (rng.next_u32() as usize % D) as u8; }
    x
}

#[test]
fn random_designs_score_finite() {
    let mut rng = Rng::new(7);
    for _ in 0..200 {
        let x = random_design(&mut rng);
        let (s, comps) = score_analytical(&x);
        assert!(s.is_finite() && s <= 0.0, "score not in valid range: {s}");
        let sum: f64 = comps.iter().sum();
        assert!((s - sum).abs() < 1e-12, "components don't sum to total: s={s}, Σcomps={sum}");
    }
}

#[test]
fn handcrafted_strong_design_beats_handcrafted_weak_design_at_fixed_vref() {
    // Hold vref at the same bin in both designs — vref scales every
    // noise term quadratically, so comparing across vref values isn't
    // monotonic in raw noise². ENOB *is* monotonic, since it adds the
    // signal/noise ratio back in. We test both.
    let mut strong: Design = NOMINAL;
    strong[I_C_HOLD]      = 4;  // 500 fF — tiny thermal
    strong[I_SH_NMOS_W]   = 4;
    strong[I_SH_PMOS_W]   = 4;
    strong[I_COMP_K]      = 4;  // 10k — tiny offset
    strong[I_DAC_MATCH]   = 0;  // 0.1%
    strong[I_SAR_NAND_W]  = 4;
    strong[I_SAR_INV_W]   = 4;
    // I_VREF stays at NOMINAL (idx 2 → 1.0 V).

    let mut weak: Design = NOMINAL;
    weak[I_C_HOLD]      = 0;    // 30 fF — large thermal
    weak[I_SH_NMOS_W]   = 0;
    weak[I_SH_PMOS_W]   = 0;
    weak[I_COMP_K]      = 0;    // 100 — large offset
    weak[I_DAC_MATCH]   = 4;    // 5% mismatch
    weak[I_SAR_NAND_W]  = 0;
    weak[I_SAR_INV_W]   = 0;

    let (s_strong, _) = score_analytical(&strong);
    let (s_weak,   _) = score_analytical(&weak);
    assert!(s_strong > s_weak,
            "strong ({s_strong:.3e}) should beat weak ({s_weak:.3e}) at fixed vref");

    let enob_strong = enob_from_score(&strong);
    let enob_weak   = enob_from_score(&weak);
    assert!(enob_strong > enob_weak,
            "ENOB ordering wrong: strong={enob_strong:.2}, weak={enob_weak:.2}");
}

#[test]
fn catalog_lookups_match_indices() {
    // Spot-check a few catalog entries to guard against silent reordering.
    assert_eq!(catalog::c_hold(0), 30e-15);
    assert_eq!(catalog::c_hold(4), 500e-15);
    assert_eq!(catalog::comp_voh(2), 1.8);
    assert_eq!(catalog::dac_r_ohms(2), 10e3);
}
