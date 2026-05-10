//! Inverse layout — optimize the simulation, get a converged GDS for
//! free. The test runs:
//!
//!   1. Initial divider with R1=10 µm, R2=30 µm body lengths.
//!   2. Adam optimizer drives `R → target Vout`.
//!   3. Converged R values are mapped back to `length` fields.
//!   4. A new layout is generated from the new lengths.
//!
//! Assertion: the new layout's R values (back-derived from its body
//! lengths via `length_to_resistance`) reproduce the optimized R values
//! within the discretization round-off introduced at step 3.

use spike_divider_block::*;

#[test]
fn optimize_then_relayout_uses_converged_lengths() {
    let lib = RcDemo::new_library("inv_layout");
    let pdk = RcDemo::register(&lib);
    let div = RcDivider::new(
        Resistor { length: 10_000, id: "R1".into() },
        Resistor { length: 30_000, id: "R2".into() },
    );

    let mut adam = Adam::new(50.0, 2);
    let (res, new_div, _top) = div.optimize_and_relayout(
        /* v_in */ 1.0,
        /* target */ 0.4,
        &mut adam,
        /* max_iters */ 5_000,
        /* tol */ 1e-4,
        /* r_min */ 1.0,
        &lib, &pdk,
    );
    assert!(res.converged);

    // Round-trip: new lengths → R values should match the optimizer's
    // converged R within the DBU rounding floor (0.1 Ω = 1 DBU).
    let r1_from_layout = length_to_resistance(new_div.r1.length);
    let r2_from_layout = length_to_resistance(new_div.r2.length);
    assert!((r1_from_layout - res.r1).abs() < 0.5,
        "round-trip R1: optimizer={}, layout={}", res.r1, r1_from_layout);
    assert!((r2_from_layout - res.r2).abs() < 0.5,
        "round-trip R2: optimizer={}, layout={}", res.r2, r2_from_layout);

    // The new layout's resistors must be physically realizable.
    assert!(new_div.r1.length >= 100, "R1 length = {}", new_div.r1.length);
    assert!(new_div.r2.length >= 100, "R2 length = {}", new_div.r2.length);

    // And the layout-derived Vout matches the target.
    let vout_from_layout = 1.0 * r2_from_layout / (r1_from_layout + r2_from_layout);
    assert!((vout_from_layout - 0.4).abs() < 1e-2,
        "back-derived Vout = {} ≠ 0.4", vout_from_layout);
}

#[test]
fn length_resistance_round_trip_is_exact() {
    // Forward + reverse should round-trip exactly (modulo the >=100 DBU
    // clamp). Validates the bridge function is consistent.
    for r in [100.0_f32, 1_000.0, 1_234.5, 10_000.0, 50_000.0] {
        let l = resistance_to_length(r);
        let r_back = length_to_resistance(l);
        assert!((r_back - r).abs() < 0.1,
            "round-trip drift: r={r}, l={l}, r_back={r_back}");
    }
}
