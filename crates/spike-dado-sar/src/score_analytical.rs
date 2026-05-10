//! Closed-form ADC noise budget.
//!
//! Standard system-level model: total error² is the sum of per-block
//! noise terms, each a function of variables in only that block. This is
//! exactly the `Σ_i C_i(x̂_i)` shape DADO is designed for. ENOB =
//! (SNDR_dB − 1.76) / 6.02; we score by the negative noise² total
//! directly (monotonic in ENOB, simpler to interpret).
//!
//! The formulas are intentionally textbook-grade — not foundry-exact.
//! Each is easy to pin to a citation:
//!   thermal noise on a sampling cap     = kT / C
//!   droop over a hold window            = ΔV = vref · t / (R_off · C)
//!   comparator finite-gain offset       = vref / k
//!   DAC quantisation noise              = vref² / (12 · 2^(2N))
//!   DAC mismatch noise                  = (σ_R · vref)²
//!   SAR digital metastability proxy     ∝ 1 / (W_nand · W_inv)
//!
//! With realistic catalog values these terms span ~6 orders of
//! magnitude, so the optimum is genuinely "balance them": pump up the
//! sample-hold cap and the comparator gain together.

use crate::catalog::{
    self, Design, CL_COMP, CL_DAC, CL_SAR, CL_SH, N_CLIQUES,
};

const K_BOLTZ: f64 = 1.380_649e-23;        // J/K
const T_KELVIN: f64 = 300.0;               // 27 °C
const T_HOLD: f64   = 1e-6;                // 1 µs sample-hold window
const R_OFF_REF: f64 = 1e9;                // 1 GΩ off-state for nominal W_nm·W_pm = 200 µm²
const C_GATE_REF: f64 = 1e-15;             // 1 fF input cap reference for the comparator
const SAR_NOISE_REF: f64 = 1e-9;           // 1 nV² metastability budget at nominal sizing

/// Score the design under the analytical noise budget.
/// Returns `(score = -total_noise², per-clique-components)` (V²; higher
/// is better). Components are negative noise² and (approximately) sum
/// to total — DADO's per-clique signal is exactly informative.
pub fn score_analytical(design: &Design) -> (f64, [f64; N_CLIQUES]) {
    let vref = catalog::vref(design[catalog::I_VREF]);

    // Sample-Hold clique: thermal + droop.
    let c_hold = catalog::c_hold(design[catalog::I_C_HOLD]);
    let w_n    = catalog::sh_nmos_w(design[catalog::I_SH_NMOS_W]);
    let w_p    = catalog::sh_pmos_w(design[catalog::I_SH_PMOS_W]);
    let l_sh   = catalog::sh_l(design[catalog::I_SH_L]);
    // Off-state resistance scales like 1 / (W_n · W_p) and like L (longer
    // gate = better off-state). Reference R_OFF_REF at nominal W·W = 200
    // µm² and L = 2 µm.
    let r_off = R_OFF_REF * (200e-12 / (w_n * w_p)) * (l_sh / 2e-6);
    let n_thermal_sq = K_BOLTZ * T_KELVIN / c_hold;
    let droop_v = vref * T_HOLD / (r_off * c_hold);
    let n_droop_sq = droop_v * droop_v;
    let sh_noise_sq = n_thermal_sq + n_droop_sq;

    // Comparator clique: finite-gain offset + clipping.
    let comp_k_val = catalog::comp_k(design[catalog::I_COMP_K]);
    let comp_voh   = catalog::comp_voh(design[catalog::I_COMP_VOH]);
    let comp_vol   = catalog::comp_vol(design[catalog::I_COMP_VOL]);
    let n_offset_sq = (vref / comp_k_val).powi(2);
    // Input-referred shot-style term that improves with bigger comparator
    // (which we don't actually vary — proxy with k since higher k usually
    // means more transconductance). Keep small.
    let n_input_sq = K_BOLTZ * T_KELVIN / (C_GATE_REF * comp_k_val.max(1.0));
    // Clipping penalty when the comparator output rails can't span vref.
    let clip_v = (vref - (comp_voh - comp_vol)).max(0.0);
    let n_clip_sq = clip_v * clip_v;
    let comp_noise_sq = n_offset_sq + n_input_sq + n_clip_sq;

    // DAC clique: quantisation + mismatch + a tiny thermal term that
    // prefers smaller R (reduces kT/C-equivalent at the DAC output node).
    let n_bits   = 4_i32; // SarAdc<4>
    let dac_r    = catalog::dac_r_ohms(design[catalog::I_DAC_R_OHMS]);
    let match_sigma = catalog::dac_match(design[catalog::I_DAC_MATCH]);
    let n_quant_sq = vref * vref / (12.0 * (1u64 << (2 * n_bits)) as f64);
    let n_match_sq = (match_sigma * vref).powi(2);
    let n_dac_thermal_sq = K_BOLTZ * T_KELVIN * dac_r * 1e-9; // weak; just for a knob
    let dac_noise_sq = n_quant_sq + n_match_sq + n_dac_thermal_sq;

    // SAR Logic clique: digital-metastability proxy. Smaller gates →
    // weaker drive → higher logic-noise envelope. Doesn't depend on vref.
    let w_nand = catalog::sar_nand_w(design[catalog::I_SAR_NAND_W]);
    let w_inv  = catalog::sar_inv_w(design[catalog::I_SAR_INV_W]);
    let sar_noise_sq = SAR_NOISE_REF * (10e-6 * 10e-6) / (w_nand * w_inv);

    let mut comps = [0.0_f64; N_CLIQUES];
    comps[CL_SH]   = -sh_noise_sq;
    comps[CL_COMP] = -comp_noise_sq;
    comps[CL_DAC]  = -dac_noise_sq;
    comps[CL_SAR]  = -sar_noise_sq;
    let total: f64 = comps.iter().sum();
    (total, comps)
}

/// Convenience: (analytical) ENOB derived from the scored noise.
pub fn enob_from_score(design: &Design) -> f64 {
    let (score, _) = score_analytical(design);
    let total_noise_sq = -score;
    let vref = catalog::vref(design[catalog::I_VREF]);
    let signal_power = (vref / 2.0).powi(2) / 2.0;
    let sndr_db = 10.0 * (signal_power / total_noise_sq.max(1e-30)).log10();
    (sndr_db - 1.76) / 6.02
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::*;

    #[test]
    fn nominal_design_scores_finite_and_negative() {
        let (s, comps) = score_analytical(&NOMINAL);
        assert!(s.is_finite());
        assert!(s <= 0.0);
        // Components should approximately sum to total.
        let sum: f64 = comps.iter().sum();
        assert!((s - sum).abs() < 1e-12);
    }

    #[test]
    fn larger_c_hold_reduces_thermal_noise() {
        // Hold everything fixed, increase only c_hold from idx 0 to idx 4.
        let mut a = NOMINAL; a[I_C_HOLD] = 0;
        let mut b = NOMINAL; b[I_C_HOLD] = 4;
        let (_, ca) = score_analytical(&a);
        let (_, cb) = score_analytical(&b);
        assert!(cb[CL_SH] > ca[CL_SH], "bigger c_hold should improve SH noise");
    }

    #[test]
    fn higher_comp_k_improves_comparator_noise() {
        let mut a = NOMINAL; a[I_COMP_K] = 0;
        let mut b = NOMINAL; b[I_COMP_K] = 4;
        let (_, ca) = score_analytical(&a);
        let (_, cb) = score_analytical(&b);
        assert!(cb[CL_COMP] > ca[CL_COMP], "bigger comp_k should improve comparator noise");
    }

    #[test]
    fn lower_dac_match_pct_improves_dac_noise() {
        let mut a = NOMINAL; a[I_DAC_MATCH] = 4; // 5% mismatch
        let mut b = NOMINAL; b[I_DAC_MATCH] = 0; // 0.1% mismatch
        let (_, ca) = score_analytical(&a);
        let (_, cb) = score_analytical(&b);
        assert!(cb[CL_DAC] > ca[CL_DAC], "tighter matching should improve DAC noise");
    }
}
