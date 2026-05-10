//! `Mosfet` primitive — 4-terminal square-law NMOS / PMOS through both
//! `Layout<P: MosfetPdk>` and `NonlinearDcBehavioral`.
//!
//! Two surfaces, both validated:
//!
//! 1. **Layout** — each polarity lays out under all three lite PDKs,
//!    drawing diff + poly + metal1 contacts on the right per-PDK GDS
//!    layers, plus an n-well only for PMOS. Four ports always: D/G/S/B.
//!
//! 2. **DC behavior** — closed-form square-law (cutoff / triode /
//!    saturation) matched to ~ulp from `rlx`-evaluated `currents()`. The
//!    PMOS check is a polarity mirror: same |I_D| as the NMOS reference
//!    with sign flipped at D and S terminals.

use eda_hir::{Block, Layout, NonlinearDcBehavioral, Schematic, SymbolKind};
use klayout_core::LayerInfo;
use rlx_ir::{DType, Graph, Shape as TensorShape};
use rlx_runtime::{Device, Session};
use spike_divider_block::pdks::{Gf180Lite, Sky130Lite};
use spike_divider_block::{MosModel, MosPolarity, Mosfet, RcDemo};

// ── Layout ─────────────────────────────────────────────────────────────

#[test]
fn nmos_lays_out_under_rcdemo_with_4_ports_and_no_nwell() {
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 2_000, l: 1_000, id: "M1".into() };
    let lib = RcDemo::new_library("nmos_rcdemo");
    let pdk = RcDemo::register(&lib);
    let id = m.layout(&lib, &pdk);
    let cell = lib.get(id);

    // 4 ports: D, G, S, B.
    assert_eq!(cell.ports().len(), 4, "expected 4 ports, got {}", cell.ports().len());
    let names: Vec<&str> = cell.ports().iter().map(|p| p.name.as_str()).collect();
    for need in ["d", "g", "s", "b"] {
        assert!(names.contains(&need), "missing port {need:?}: {names:?}");
    }

    // Diff (51,0) and poly (52,0) populated; nwell (53,0) empty for NMOS;
    // n+ implant (54,0) drawn over diff; p+ implant (55,0) absent.
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(51, 0))).count() > 0, "diff empty");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(52, 0))).count() > 0, "poly empty");
    assert_eq!(cell.shapes_on(lib.layer(LayerInfo::gds(53, 0))).count(), 0,
        "nwell drawn for NMOS — should be skipped");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(54, 0))).count() > 0,
        "NMOS must have n+ implant");
    assert_eq!(cell.shapes_on(lib.layer(LayerInfo::gds(55, 0))).count(), 0,
        "p+ implant drawn for NMOS — should be skipped");
}

#[test]
fn pmos_draws_nwell_and_pplus_no_nplus_under_rcdemo() {
    let m = Mosfet { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw, w: 2_000, l: 1_000, id: "M2".into() };
    let lib = RcDemo::new_library("pmos_rcdemo");
    let pdk = RcDemo::register(&lib);
    let id = m.layout(&lib, &pdk);
    let cell = lib.get(id);

    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(53, 0))).count() > 0,
        "PMOS must have nwell");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(55, 0))).count() > 0,
        "PMOS must have p+ implant");
    assert_eq!(cell.shapes_on(lib.layer(LayerInfo::gds(54, 0))).count(), 0,
        "n+ implant drawn for PMOS — should be skipped");
}

#[test]
fn mosfet_lays_out_under_sky130lite_with_correct_layers() {
    // Sky130: diff=(65,20), poly=(66,20), nwell=(64,20), met1=(68,20).
    let m = Mosfet { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw, w: 2_000, l: 1_000, id: "M3".into() };
    let lib = Sky130Lite::new_library("sky130_pmos");
    let pdk = Sky130Lite::register(&lib);
    let id = m.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(65, 20))).count() > 0, "sky130 diff");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(66, 20))).count() > 0, "sky130 poly");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(64, 20))).count() > 0, "sky130 nwell");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(68, 20))).count() > 0, "sky130 met1");
    // PMOS → psdm (94,20).
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(94, 20))).count() > 0, "sky130 psdm");
}

#[test]
fn mosfet_lays_out_under_generated_sky130_with_correct_layers() {
    // Same layer assertions as the Sky130Lite test, but against the
    // build-time `klayout_pdk::pdk!`-generated `Sky130` from `eda-pdks`.
    // Exercises the `MosfetPdk for eda_pdks::Sky130` impl in lib.rs.
    if !spike_divider_block::pdks_foundry::HAS_SKY130 {
        eprintln!("skipping: sky130 lyp absent at build time");
        return;
    }
    use spike_divider_block::pdks_foundry::Sky130;
    let m = Mosfet { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw, w: 2_000, l: 1_000, id: "M3g".into() };
    let lib = Sky130::new_library("sky130_pmos_gen");
    let pdk = Sky130::register(&lib);
    let id = m.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(65, 20))).count() > 0, "sky130 gen diff");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(66, 20))).count() > 0, "sky130 gen poly");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(64, 20))).count() > 0, "sky130 gen nwell");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(68, 20))).count() > 0, "sky130 gen met1");
    // PMOS → psdm (94,20).
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(94, 20))).count() > 0, "sky130 gen psdm");
}

#[test]
fn mosfet_lays_out_under_generated_gf180mcu_with_correct_layers() {
    if !spike_divider_block::pdks_foundry::HAS_GF180MCU {
        eprintln!("skipping: gf180mcu lyp absent at build time");
        return;
    }
    use spike_divider_block::pdks_foundry::Gf180mcu;
    let m = Mosfet { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw, w: 2_000, l: 1_000, id: "M4g".into() };
    let lib = Gf180mcu::new_library("gf180_pmos_gen");
    let pdk = Gf180mcu::register(&lib);
    let id = m.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(22, 0))).count() > 0, "gf180 gen comp");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(30, 0))).count() > 0, "gf180 gen poly2");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(21, 0))).count() > 0, "gf180 gen nwell");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(34, 0))).count() > 0, "gf180 gen metal1");
    // PMOS → pplus (31,0).
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(31, 0))).count() > 0, "gf180 gen pplus");
}

#[test]
fn mosfet_lays_out_under_gf180lite_with_correct_layers() {
    // gf180mcu: comp=(22,0), poly2=(30,0), nwell=(21,0), metal1=(34,0).
    let m = Mosfet { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw, w: 2_000, l: 1_000, id: "M4".into() };
    let lib = Gf180Lite::new_library("gf180_pmos");
    let pdk = Gf180Lite::register(&lib);
    let id = m.layout(&lib, &pdk);
    let cell = lib.get(id);
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(22, 0))).count() > 0, "gf180 comp");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(30, 0))).count() > 0, "gf180 poly2");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(21, 0))).count() > 0, "gf180 nwell");
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(34, 0))).count() > 0, "gf180 metal1");
    // PMOS → pplus (31,0).
    assert!(cell.shapes_on(lib.layer(LayerInfo::gds(31, 0))).count() > 0, "gf180 pplus");
}

// ── DC behavior — closed-form square-law match ─────────────────────────

const KP:  f32 = 100e-6;       // 100 µA/V² — typical 180 nm scale.
const VTH: f32 = 0.5;
// Reference square-law I_D, NMOS sign convention (positive when V_DS > 0).
fn closed_form_id(v_gs: f32, v_ds: f32, kp: f32, vth: f32, w_over_l: f32) -> f32 {
    let v_ov = (v_gs - vth).max(0.0);
    let v_ds_clip = v_ds.clamp(0.0, v_ov);
    kp * w_over_l * (v_ov * v_ds_clip - 0.5 * v_ds_clip * v_ds_clip)
}

/// Compile a graph that exposes the four terminal voltages as Inputs
/// and outputs the four `currents()` slots in [D, G, S, B] order.
fn build_iv_probe(m: &Mosfet) -> Graph {
    let mut g = Graph::new(format!("{}_iv_probe", <Mosfet as Block>::name(m)));
    let s = TensorShape::new(&[1], DType::F32);
    let v_d = g.input("v_d", s.clone());
    let v_g = g.input("v_g", s.clone());
    let v_s = g.input("v_s", s.clone());
    let v_b = g.input("v_b", s);
    let outs = m.currents(&[v_d, v_g, v_s, v_b], &mut g);
    g.set_outputs(outs);
    g
}

/// Run the I-V probe with explicit `kp`/`vth` and λ=γ=0 — i.e. bare
/// square-law, as if the body-effect / CLM extensions weren't there.
fn run_iv(m: &Mosfet, v_d: f32, v_g: f32, v_s: f32, v_b: f32, kp: f32, vth: f32) -> [f32; 4] {
    run_iv_full(m, v_d, v_g, v_s, v_b, kp, vth, /*lambda*/ 0.0, /*gamma*/ 0.0)
}

/// Probe with all five Mosfet params under explicit control. `2φF` is
/// pinned at 0.7 V (a common silicon value); pass γ=0 to disable body
/// effect entirely regardless of `2φF`.
fn run_iv_full(
    m: &Mosfet,
    v_d: f32, v_g: f32, v_s: f32, v_b: f32,
    kp: f32, vth: f32, lambda: f32, gamma: f32,
) -> [f32; 4] {
    let g = build_iv_probe(m);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let n = <Mosfet as Block>::name(m);
    compiled.set_param(&format!("{n}_Kp"),      &[kp]);
    compiled.set_param(&format!("{n}_Vth"),     &[vth]);
    compiled.set_param(&format!("{n}_Lambda"),  &[lambda]);
    compiled.set_param(&format!("{n}_Gamma"),   &[gamma]);
    compiled.set_param(&format!("{n}_TwoPhiF"), &[0.7]);
    compiled.set_param(&format!("{n}_N"),       &[1.0]);
    let outs = compiled.run(&[
        ("v_d", &[v_d][..]), ("v_g", &[v_g][..]),
        ("v_s", &[v_s][..]), ("v_b", &[v_b][..]),
    ]);
    [outs[0][0], outs[1][0], outs[2][0], outs[3][0]]
}

#[test]
fn nmos_cutoff_zero_current() {
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "M".into() };
    // V_GS = 0.3 < V_th = 0.5 → cutoff.
    let [i_d, i_g, i_s, i_b] = run_iv(&m, 1.0, 0.3, 0.0, 0.0, KP, VTH);
    assert!(i_d.abs() < 1e-12, "I_D in cutoff: {}", i_d);
    assert_eq!(i_g, 0.0);
    assert!(i_s.abs() < 1e-12);
    assert_eq!(i_b, 0.0);
}

#[test]
fn nmos_triode_matches_closed_form() {
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "M".into() };
    // V_GS=1.5, V_ov=1.0, V_DS=0.5 < V_ov → triode.
    let [i_d_term, _, i_s_term, _] = run_iv(&m, 0.5, 1.5, 0.0, 0.0, KP, VTH);
    let id_ref = closed_form_id(1.5, 0.5, KP, VTH, 1.0);
    assert!(id_ref > 0.0);
    // i_d_term = -I_D (device pulls from D), i_s_term = +I_D.
    assert!((-i_d_term - id_ref).abs() < 1e-9,
        "triode I_D@D: got {}, expected {}", -i_d_term, id_ref);
    assert!((i_s_term - id_ref).abs() < 1e-9,
        "triode I_D@S: got {}, expected {}", i_s_term, id_ref);
}

#[test]
fn nmos_saturation_matches_closed_form() {
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "M".into() };
    // V_GS=1.5, V_ov=1.0, V_DS=2.0 ≥ V_ov → saturation, V_DS_clip=V_ov.
    let [i_d_term, _, i_s_term, _] = run_iv(&m, 2.0, 1.5, 0.0, 0.0, KP, VTH);
    let id_ref = closed_form_id(1.5, 2.0, KP, VTH, 1.0);
    let id_sat = 0.5 * KP * 1.0 * 1.0_f32.powi(2);
    assert!((id_ref - id_sat).abs() < 1e-12, "closed-form sanity");
    assert!((-i_d_term - id_ref).abs() < 1e-9,
        "sat I_D@D: got {}, expected {}", -i_d_term, id_ref);
    assert!((i_s_term - id_ref).abs() < 1e-9);
}

#[test]
fn pmos_polarity_mirrors_nmos() {
    let nmos = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "Mn".into() };
    let pmos = Mosfet { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "Mp".into() };

    // NMOS in saturation at V_S=0, V_G=1.5, V_D=2.0.
    let [n_d, _, n_s, _] = run_iv(&nmos, 2.0, 1.5, 0.0, 0.0, KP, VTH);
    // PMOS at the mirrored bias: V_S=0, V_G=-1.5, V_D=-2.0 — same V_GS_eff,
    // V_DS_eff folded through `sign`, so identical |I_D|, opposite signs at
    // D and S relative to NMOS.
    let [p_d, _, p_s, _] = run_iv(&pmos, -2.0, -1.5, 0.0, 0.0, KP, VTH);

    let id_ref = closed_form_id(1.5, 2.0, KP, VTH, 1.0);
    assert!((-n_d - id_ref).abs() < 1e-9);
    // PMOS pushes current to D (sign-flipped) and pulls from S.
    assert!((p_d - id_ref).abs() < 1e-9,
        "PMOS I@D: got {}, expected {}", p_d, id_ref);
    assert!((-p_s - id_ref).abs() < 1e-9,
        "PMOS I@S: got {}, expected {}", -p_s, id_ref);
    // Magnitude match between polarities.
    assert!((n_s - (-p_s)).abs() < 1e-9);
}

#[test]
fn mosfet_w_over_l_scales_current_linearly() {
    let m_1x = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "wide1".into() };
    let m_4x = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 4_000, l: 1_000, id: "wide4".into() };

    let [d1, _, _, _] = run_iv(&m_1x, 2.0, 1.5, 0.0, 0.0, KP, VTH);
    let [d4, _, _, _] = run_iv(&m_4x, 2.0, 1.5, 0.0, 0.0, KP, VTH);
    let ratio = (-d4) / (-d1);
    assert!((ratio - 4.0).abs() < 1e-5, "W/L scaling: got {}× (expected 4×)", ratio);
}

// ── CLM (channel-length modulation) ────────────────────────────────────

#[test]
fn nmos_clm_lifts_saturation_current_linearly_in_vds() {
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "Mclm".into() };
    let lambda = 0.05_f32;     // 1/V — moderate λ for a textbook test.

    // V_GS=1.5, V_ov=1.0 → I_D_sat0 = ½·K_p·V_ov² = 50 µA at λ=0.
    // Two V_DS points in saturation: 1.5 V and 3.0 V.
    let [d_lo, _, _, _] = run_iv_full(&m, 1.5, 1.5, 0.0, 0.0, KP, VTH, lambda, 0.0);
    let [d_hi, _, _, _] = run_iv_full(&m, 3.0, 1.5, 0.0, 0.0, KP, VTH, lambda, 0.0);
    let id_lo = -d_lo;
    let id_hi = -d_hi;

    let id0 = 0.5 * KP * 1.0_f32.powi(2);
    let exp_lo = id0 * (1.0 + lambda * 1.5);
    let exp_hi = id0 * (1.0 + lambda * 3.0);
    assert!((id_lo - exp_lo).abs() < 1e-9, "I_D@V_DS=1.5: got {}, expected {}", id_lo, exp_lo);
    assert!((id_hi - exp_hi).abs() < 1e-9, "I_D@V_DS=3.0: got {}, expected {}", id_hi, exp_hi);

    // Slope ∂I_D/∂V_DS = λ·I_D0 — pin via the 2-point estimate.
    let slope = (id_hi - id_lo) / (3.0 - 1.5);
    assert!((slope - lambda * id0).abs() < 1e-9,
        "saturation slope: got {}, expected {}", slope, lambda * id0);
}

#[test]
fn nmos_clm_lambda_zero_is_bare_square_law() {
    // Sanity: with λ=0 the augmented model collapses to the original
    // square-law value used in the saturation test above.
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "Mlam0".into() };
    let [d, _, _, _] = run_iv_full(&m, 2.0, 1.5, 0.0, 0.0, KP, VTH, 0.0, 0.0);
    let exp = closed_form_id(1.5, 2.0, KP, VTH, 1.0);
    assert!((-d - exp).abs() < 1e-9, "λ=0 baseline: got {}, expected {}", -d, exp);
}

// ── Body effect (γ, 2φF) ───────────────────────────────────────────────

#[test]
fn nmos_body_effect_raises_vth_per_root_formula() {
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "Mbody".into() };
    let gamma   = 0.4_f32;
    let two_phi = 0.7_f32;
    // V_S=0, V_B=-1 → V_SB=+1 V (NMOS body reverse-biased).
    // V_th_eff = 0.5 + 0.4·(√(0.7+1) − √0.7) = 0.5 + 0.4·(1.30384 − 0.83666)
    //          ≈ 0.68687 V.
    let v_sb = 1.0_f32;
    let vth_eff = VTH + gamma * ((two_phi + v_sb).sqrt() - two_phi.sqrt());

    // Probe in saturation: V_GS = 1.5 → V_ov_eff = 1.5 − vth_eff.
    let v_gs = 1.5_f32;
    let v_ds = 2.0_f32;
    let id_expected = closed_form_id(v_gs, v_ds, KP, vth_eff, 1.0);

    let [d, _, _, _] = run_iv_full(&m, v_ds, v_gs, 0.0, -v_sb, KP, VTH, 0.0, gamma);
    assert!((-d - id_expected).abs() < 1e-9,
        "body-effect I_D: got {}, expected {}", -d, id_expected);
}

#[test]
fn nmos_body_effect_zero_gamma_is_no_op() {
    // γ=0 should give the same I_D as if V_B were tied to V_S.
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "Mg0".into() };
    let [d_tied,  _, _, _] = run_iv_full(&m, 2.0, 1.5,  0.0, 0.0, KP, VTH, 0.0, 0.0);
    let [d_split, _, _, _] = run_iv_full(&m, 2.0, 1.5,  0.0, -2.0, KP, VTH, 0.0, 0.0);
    assert!((d_tied - d_split).abs() < 1e-12,
        "γ=0 must be insensitive to V_SB: tied={}, split={}", d_tied, d_split);
}

#[test]
fn pmos_body_effect_uses_same_polarity_folded_v_sb() {
    // Mirror of the NMOS body-effect test: PMOS at V_S=0, V_B=+1 V →
    // sign·(V_S − V_B) = -1·(0 − 1) = +1 V → same V_SB_eff → same V_th
    // shift. |I_D| should match the NMOS reference.
    let pmos = Mosfet { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "Mpb".into() };
    let gamma   = 0.4_f32;
    let two_phi = 0.7_f32;
    let v_sb = 1.0_f32;
    let vth_eff = VTH + gamma * ((two_phi + v_sb).sqrt() - two_phi.sqrt());
    let id_expected = closed_form_id(1.5, 2.0, KP, vth_eff, 1.0);

    let [d, _, _, _] = run_iv_full(&pmos, -2.0, -1.5, 0.0, 1.0, KP, VTH, 0.0, gamma);
    // PMOS pushes current to D → current at D is +|I_D|.
    assert!((d - id_expected).abs() < 1e-9,
        "PMOS body-effect I@D: got {}, expected {}", d, id_expected);
}

// ── Schematic ──────────────────────────────────────────────────────────

#[test]
fn nmos_schematic_picks_nmos_symbol_with_w_over_l_value() {
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 2_000, l: 500, id: "M1".into() };
    let lib = RcDemo::new_library("nmos_schem");
    let pdk = RcDemo::register(&lib);
    let ir = <Mosfet as Schematic<RcDemo>>::schematic(&m, &pdk);

    assert_eq!(ir.symbols.len(), 1);
    let sym = &ir.symbols[0];
    assert_eq!(sym.kind, SymbolKind::Nmos);
    assert_eq!(sym.label, "M1");
    assert_eq!(sym.value.as_deref(), Some("W/L = 2000/500"));
}

#[test]
fn pmos_schematic_picks_pmos_symbol() {
    let m = Mosfet { polarity: MosPolarity::Pmos, model: MosModel::SquareLaw, w: 1_500, l: 1_000, id: "M2".into() };
    let lib = RcDemo::new_library("pmos_schem");
    let pdk = RcDemo::register(&lib);
    let ir = <Mosfet as Schematic<RcDemo>>::schematic(&m, &pdk);
    assert_eq!(ir.symbols[0].kind, SymbolKind::Pmos);
}

#[test]
fn mos_symbolkind_reports_4_terminals() {
    assert_eq!(SymbolKind::Nmos.n_terminals(), 4);
    assert_eq!(SymbolKind::Pmos.n_terminals(), 4);
}

#[test]
fn default_params_helper_seeds_all_five_keys() {
    let m = Mosfet { polarity: MosPolarity::Nmos, model: MosModel::SquareLaw, w: 1_000, l: 1_000, id: "Mdef".into() };
    let p = m.default_params();
    let n = <Mosfet as Block>::name(&m);
    for suffix in ["Kp", "Vth", "Lambda", "Gamma", "TwoPhiF", "N"] {
        assert!(p.contains_key(&format!("{n}_{suffix}")),
            "missing default param {suffix}");
    }
}

// ── EKV-lite model ─────────────────────────────────────────────────────

/// Probe an EKV-lite Mosfet at a given operating point. Body effect
/// disabled (γ=0) so the test isolates the EKV interpolation; n_slope
/// is configurable so we can sweep the strong-inv match knob.
fn run_iv_ekv(
    m: &Mosfet,
    v_d: f32, v_g: f32, v_s: f32, v_b: f32,
    kp: f32, vth: f32, n_slope: f32,
) -> [f32; 4] {
    let g = build_iv_probe(m);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let n = <Mosfet as Block>::name(m);
    compiled.set_param(&format!("{n}_Kp"),      &[kp]);
    compiled.set_param(&format!("{n}_Vth"),     &[vth]);
    compiled.set_param(&format!("{n}_Lambda"),  &[0.0]);
    compiled.set_param(&format!("{n}_Gamma"),   &[0.0]);
    compiled.set_param(&format!("{n}_TwoPhiF"), &[0.7]);
    compiled.set_param(&format!("{n}_N"),       &[n_slope]);
    let outs = compiled.run(&[
        ("v_d", &[v_d][..]), ("v_g", &[v_g][..]),
        ("v_s", &[v_s][..]), ("v_b", &[v_b][..]),
    ]);
    [outs[0][0], outs[1][0], outs[2][0], outs[3][0]]
}

#[test]
fn ekv_strong_inversion_saturation_matches_square_law_at_n_one() {
    // n=1 makes EKV-lite saturation collapse to ½·K_p·(W/L)·V_ov² —
    // the same as square-law saturation. Within the EKV softplus
    // smoothing tolerance (≪1 µA at strong overdrive).
    let mq = Mosfet::nmos(1_000, 1_000, "ekv1").with_model(MosModel::EkvLite);
    // V_GS=1.5, V_th=0.5 → V_ov=1.0, deep strong-inv at U_T=0.026.
    // V_DS=2.0 → saturation.
    let [d_ekv, _, _, _] = run_iv_ekv(&mq, 2.0, 1.5, 0.0, 0.0, KP, VTH, 1.0);
    let id_ekv = -d_ekv;
    let id_sq  = closed_form_id(1.5, 2.0, KP, VTH, 1.0);
    let rel = (id_ekv - id_sq).abs() / id_sq;
    assert!(rel < 5e-3,
        "EKV vs square-law sat: ekv={:.3e}, sq={:.3e}, rel={:.2e}",
        id_ekv, id_sq, rel);
}

#[test]
fn ekv_subthreshold_has_60mv_per_decade_swing() {
    // Subthreshold swing S = n · U_T · ln(10) ≈ 60 mV/decade at n=1.
    // Sweep V_GS through cutoff and check the I_D ratio for two points
    // 60 mV apart matches 10× to within EKV's smooth approx.
    let mq = Mosfet::nmos(1_000, 1_000, "ekvSub").with_model(MosModel::EkvLite);
    let n_slope = 1.0_f32;
    let [d_lo, _, _, _] = run_iv_ekv(&mq, 1.0, 0.30, 0.0, 0.0, KP, VTH, n_slope);
    let s_v = n_slope * 0.025852 * 10.0_f32.ln();    // ≈ 0.0595 V
    let [d_hi, _, _, _] = run_iv_ekv(&mq, 1.0, 0.30 + s_v, 0.0, 0.0, KP, VTH, n_slope);
    let id_lo = -d_lo;
    let id_hi = -d_hi;
    let ratio = id_hi / id_lo;
    // Allow 5% slack: in deep subthreshold the softplus is exactly
    // exp(x), but at V_GS just below V_th the (1+exp) bias drags the
    // ratio down a bit. The decade-slope holds asymptotically.
    assert!(id_lo > 0.0 && id_hi > 0.0, "subthreshold currents not both positive");
    assert!((ratio - 10.0).abs() / 10.0 < 0.05,
        "subthreshold decade ratio: got {}× (expected 10×)", ratio);
}

#[test]
fn ekv_id_is_monotone_increasing_in_vgs_across_threshold() {
    // EKV's whole point: smooth, monotone, no piecewise discontinuity.
    // Sweep V_GS through the V_th transition and check strict monotone
    // increase in I_D.
    let mq = Mosfet::nmos(1_000, 1_000, "ekvMono").with_model(MosModel::EkvLite);
    let mut prev = -1e30_f32;
    for &v_gs in &[0.30_f32, 0.40, 0.45, 0.49, 0.50, 0.51, 0.55, 0.7, 1.0, 1.5] {
        let [d, _, _, _] = run_iv_ekv(&mq, 1.0, v_gs, 0.0, 0.0, KP, VTH, 1.0);
        let i_d = -d;
        assert!(i_d > prev,
            "I_D not monotone at V_GS={}: prev={:.3e}, now={:.3e}",
            v_gs, prev, i_d);
        prev = i_d;
    }
}

#[test]
fn ekv_zero_overdrive_gives_finite_nonzero_current() {
    // At V_GS = V_th exactly, square-law would give I_D = 0 (cutoff
    // boundary). EKV-lite gives a small finite leakage current — that's
    // the smoothness payoff. Just check it's positive and small.
    let mq = Mosfet::nmos(1_000, 1_000, "ekv0").with_model(MosModel::EkvLite);
    let [d, _, _, _] = run_iv_ekv(&mq, 1.0, 0.5, 0.0, 0.0, KP, VTH, 1.0);
    let i_d = -d;
    assert!(i_d > 0.0, "expected positive I_D at V_GS=V_th, got {}", i_d);
    // Strong-inv saturation at V_ov=0 is 0; EKV smoothing puts us in
    // the transition region, so I_D should be tiny — much less than
    // the saturation current at V_ov=1V (50 µA).
    assert!(i_d < 5e-6, "I_D at V_GS=V_th should be ≪ 50 µA, got {}", i_d);
}

#[test]
fn ekv_pmos_polarity_mirrors_nmos() {
    // PMOS folded-polarity sanity: at mirrored bias, EKV-lite gives
    // the same |I_D| as NMOS, with sign flipped at D and S.
    let nmos = Mosfet::nmos(1_000, 1_000, "ekvN").with_model(MosModel::EkvLite);
    let pmos = Mosfet::pmos(1_000, 1_000, "ekvP").with_model(MosModel::EkvLite);

    let [n_d, _, _, _] = run_iv_ekv(&nmos,  2.0,  1.5, 0.0, 0.0, KP, VTH, 1.0);
    let [p_d, _, _, _] = run_iv_ekv(&pmos, -2.0, -1.5, 0.0, 0.0, KP, VTH, 1.0);
    let id_n = -n_d;
    let id_p =  p_d;     // PMOS pushes to D → currents[D] = +|I_D|
    assert!((id_n - id_p).abs() / id_n.abs() < 1e-4,
        "EKV PMOS mirror: nmos={}, pmos={}", id_n, id_p);
}
