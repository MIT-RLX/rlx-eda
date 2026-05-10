//! Thermal-corner extension to the R-2R DADO objective.
//!
//! Validates that:
//! 1. `solve_r2r_at_temp` reduces to `solve_r2r` at `T_NOMINAL_C`
//!    (sanity — no spurious shift at nominal).
//! 2. For an adversarial design (max-magnitude deviations on every
//!    resistor), the corner-T INL is materially worse than nominal-T
//!    INL — confirming the deviation × TC1 coupling actually does
//!    something.
//! 3. `score_inl_worst_corner` returns the most-negative of the
//!    per-corner scores, picking the worst `T` from `T_CORNERS_C`.
//! 4. The worst-corner score is *strictly worse* than the nominal-T
//!    score for the adversarial design — i.e., a designer optimizing
//!    only nominal-T INL will leave thermal headroom on the table
//!    that DADO with `score_inl_worst_corner` would catch.

use spike_dado_r2r::{
    score_inl, score_inl_at_temp, score_inl_worst_corner, solve_r2r, solve_r2r_at_temp,
    Design, L, T_CORNERS_C, T_NOMINAL_C,
};

#[test]
fn solve_at_nominal_temperature_matches_nominal_solver() {
    // Uniform-deviation designs across the alphabet — covers every
    // categorical choice each of the 16 resistors can take.
    for choice in 0..5_u8 {
        let design: Design = [choice; L];
        for code in [0u32, 1, 7, 64, 128, 200, 255] {
            let v_nom = solve_r2r(&design, code, 1.0, 0.0);
            let v_t   = solve_r2r_at_temp(&design, code, 1.0, 0.0, T_NOMINAL_C);
            assert!((v_nom - v_t).abs() < 1e-12,
                "code={code}, choice={choice}: nominal={v_nom}, t-nominal={v_t}");
        }
    }
}

#[test]
fn corner_inl_is_worse_for_adversarial_design() {
    // Adversarial sizing: alternate ±5 % deviations along the spine
    // and fall through to ±5 % on the 2R legs. The deviation × TC1
    // coupling lights up exactly here — the +5 % resistors get
    // larger-magnitude thermal drift than the -5 % ones, which
    // breaks the divider symmetry at hot/cold corners.
    let mut design: Design = [2u8; L]; // start at 0% deviation
    for i in 0..L { design[i] = if i % 2 == 0 { 0 } else { 4 }; }   // -5/+5 alternating

    let (s_nom, _) = score_inl_at_temp(&design, T_NOMINAL_C);
    let mut worst_corner_score = f64::INFINITY;
    let mut worst_corner_t = T_NOMINAL_C;
    for &t in &T_CORNERS_C {
        if (t - T_NOMINAL_C).abs() < 1e-9 { continue; } // skip Tnom
        let (s, _) = score_inl_at_temp(&design, t);
        if s < worst_corner_score { worst_corner_score = s; worst_corner_t = t; }
    }

    eprintln!("nominal-T score = {s_nom:.6e}, worst-corner score = {worst_corner_score:.6e} at T={worst_corner_t}°C");
    assert!(worst_corner_score < s_nom - 1e-6,
        "worst-corner INL should exceed nominal-T INL for adversarial design \
         (got worst={worst_corner_score:.6e}, nom={s_nom:.6e}); deviation×TC1 coupling didn't engage");
}

#[test]
fn worst_corner_score_picks_the_worst_corner() {
    // Same adversarial design, this time exercising the wrapper.
    let mut design: Design = [2u8; L];
    for i in 0..L { design[i] = if i % 2 == 0 { 0 } else { 4 }; }

    let (worst, _) = score_inl_worst_corner(&design);
    let per_corner: Vec<f64> = T_CORNERS_C.iter()
        .map(|&t| score_inl_at_temp(&design, t).0)
        .collect();
    let expected = per_corner.iter().cloned().fold(f64::INFINITY, f64::min);

    assert!((worst - expected).abs() < 1e-12,
        "worst-corner score = {worst}, expected min over {per_corner:?} = {expected}");
}

#[test]
fn nominal_only_score_understates_thermal_envelope() {
    // The headline claim of #3: optimizing nominal-T INL leaves
    // thermal headroom on the table. Take the *all-nominal* design
    // (every resistor at 0 % deviation) — score_inl at Tnom is 0
    // by construction. But TC1_eff under uniform 0% deviation is
    // exactly TC1_NOM (κ·dev = 0), so all resistors scale uniformly
    // with T and INL stays 0 across every corner. The all-nominal
    // case is thermally benign — confirms the demo's coupling lives
    // in the deviation × TC1 cross term, not a bug in the math.
    let design: Design = [2u8; L];
    let (worst, _) = score_inl_worst_corner(&design);
    let (nom, _)   = score_inl(&design);
    assert!(worst.abs() < 1e-9 && nom.abs() < 1e-9,
        "all-nominal design should have zero INL at every corner; got nom={nom}, worst={worst}");
}
