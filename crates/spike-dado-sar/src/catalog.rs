//! 12-variable discrete catalog over the SAR ADC sub-block parameters.
//!
//! Each variable picks one of `D = 5` values. Total design space
//! `5¹² ≈ 2.4 × 10⁸`. Variables are grouped into four cliques (one per
//! sub-block), with empty separators (no shared variables between
//! blocks). A `Design` is a fixed-length array of categorical indices.

#![allow(clippy::approx_constant)]

/// Alphabet size — number of bins per variable.
pub const D: usize = 5;
/// Number of design variables.
pub const L: usize = 12;
/// Number of cliques (one per sub-block).
pub const N_CLIQUES: usize = 4;

/// Categorical-index vector of length `L`. Each entry is in `0..D`.
pub type Design = [u8; L];

// ── Variable layout ──────────────────────────────────────────────────────
//
//   idx | clique     | variable          | catalog
//   ----+------------+-------------------+----------------------------------------
//    0  | SH         | c_hold            | {30, 50, 100, 200, 500} fF
//    1  | SH         | sh_nmos_w         | {2, 5, 10, 20, 40} µm
//    2  | SH         | sh_pmos_w         | {5, 10, 20, 40, 80} µm
//    3  | Comparator | comp_k            | {100, 500, 1k, 5k, 10k}
//    4  | Comparator | comp_voh          | {1.2, 1.5, 1.8, 2.5, 3.3} V
//    5  | DAC        | dac_r_ohms        | {1k, 5k, 10k, 20k, 50k} Ω
//    6  | DAC        | dac_match_pct     | {0.1, 0.5, 1, 2, 5} %  (analytical only)
//    7  | SAR Logic  | sar_nand_w        | {2, 5, 10, 20, 40} µm
//    8  | SAR Logic  | sar_inv_w         | {2, 5, 10, 20, 40} µm
//    9  | DAC        | vref              | {0.6, 0.9, 1.0, 1.5, 1.8} V
//   10  | SH         | sh_l              | {0.5, 1, 2, 4, 8} µm
//   11  | Comparator | comp_vol          | {0.0, 0.1, 0.2, 0.3, 0.5} V

// Indices.
pub const I_C_HOLD: usize         = 0;
pub const I_SH_NMOS_W: usize      = 1;
pub const I_SH_PMOS_W: usize      = 2;
pub const I_COMP_K: usize         = 3;
pub const I_COMP_VOH: usize       = 4;
pub const I_DAC_R_OHMS: usize     = 5;
pub const I_DAC_MATCH: usize      = 6;
pub const I_SAR_NAND_W: usize     = 7;
pub const I_SAR_INV_W: usize      = 8;
pub const I_VREF: usize           = 9;
pub const I_SH_L: usize           = 10;
pub const I_COMP_VOL: usize       = 11;

// Clique IDs.
pub const CL_SH: usize    = 0;
pub const CL_COMP: usize  = 1;
pub const CL_DAC: usize   = 2;
pub const CL_SAR: usize   = 3;

// ── Catalogs ─────────────────────────────────────────────────────────────

const C_HOLD: [f64; D]    = [30e-15, 50e-15, 100e-15, 200e-15, 500e-15];
const SH_NMOS_W: [f64; D] = [2e-6, 5e-6, 10e-6, 20e-6, 40e-6];
const SH_PMOS_W: [f64; D] = [5e-6, 10e-6, 20e-6, 40e-6, 80e-6];
const COMP_K:    [f64; D] = [100.0, 500.0, 1000.0, 5000.0, 10_000.0];
const COMP_VOH:  [f64; D] = [1.2, 1.5, 1.8, 2.5, 3.3];
const DAC_R:     [f64; D] = [1e3, 5e3, 10e3, 20e3, 50e3];
const DAC_MATCH: [f64; D] = [0.001, 0.005, 0.01, 0.02, 0.05]; // fractional σ
const SAR_NAND_W:[f64; D] = [2e-6, 5e-6, 10e-6, 20e-6, 40e-6];
const SAR_INV_W: [f64; D] = [2e-6, 5e-6, 10e-6, 20e-6, 40e-6];
const VREF:      [f64; D] = [0.6, 0.9, 1.0, 1.5, 1.8];
const SH_L:      [f64; D] = [0.5e-6, 1e-6, 2e-6, 4e-6, 8e-6];
const COMP_VOL:  [f64; D] = [0.0, 0.1, 0.2, 0.3, 0.5];

#[inline] pub fn c_hold(idx: u8)        -> f64 { C_HOLD[idx as usize] }
#[inline] pub fn sh_nmos_w(idx: u8)     -> f64 { SH_NMOS_W[idx as usize] }
#[inline] pub fn sh_pmos_w(idx: u8)     -> f64 { SH_PMOS_W[idx as usize] }
#[inline] pub fn comp_k(idx: u8)        -> f64 { COMP_K[idx as usize] }
#[inline] pub fn comp_voh(idx: u8)      -> f64 { COMP_VOH[idx as usize] }
#[inline] pub fn dac_r_ohms(idx: u8)    -> f64 { DAC_R[idx as usize] }
#[inline] pub fn dac_match(idx: u8)     -> f64 { DAC_MATCH[idx as usize] }
#[inline] pub fn sar_nand_w(idx: u8)    -> f64 { SAR_NAND_W[idx as usize] }
#[inline] pub fn sar_inv_w(idx: u8)     -> f64 { SAR_INV_W[idx as usize] }
#[inline] pub fn vref(idx: u8)          -> f64 { VREF[idx as usize] }
#[inline] pub fn sh_l(idx: u8)          -> f64 { SH_L[idx as usize] }
#[inline] pub fn comp_vol(idx: u8)      -> f64 { COMP_VOL[idx as usize] }

/// Variables in clique `i`. Order matters — it's the canonical layout
/// used to encode tabular table positions.
pub fn clique_vars(i: usize) -> Vec<usize> {
    match i {
        CL_SH   => vec![I_C_HOLD, I_SH_NMOS_W, I_SH_PMOS_W, I_SH_L],
        CL_COMP => vec![I_COMP_K, I_COMP_VOH, I_COMP_VOL],
        CL_DAC  => vec![I_DAC_R_OHMS, I_DAC_MATCH, I_VREF],
        CL_SAR  => vec![I_SAR_NAND_W, I_SAR_INV_W],
        _ => panic!("clique index out of range: {i}"),
    }
}

/// Friendly label for plotting / debug.
pub fn var_label(v: usize) -> &'static str {
    match v {
        I_C_HOLD => "c_hold",
        I_SH_NMOS_W => "sh_nmos_w",
        I_SH_PMOS_W => "sh_pmos_w",
        I_COMP_K => "comp_k",
        I_COMP_VOH => "comp_voh",
        I_DAC_R_OHMS => "dac_r_ohms",
        I_DAC_MATCH => "dac_match",
        I_SAR_NAND_W => "sar_nand_w",
        I_SAR_INV_W => "sar_inv_w",
        I_VREF => "vref",
        I_SH_L => "sh_l",
        I_COMP_VOL => "comp_vol",
        _ => "?",
    }
}

pub fn clique_label(c: usize) -> &'static str {
    match c {
        CL_SH => "Sample-Hold",
        CL_COMP => "Comparator",
        CL_DAC => "DAC",
        CL_SAR => "SAR Logic",
        _ => "?",
    }
}

/// "Nominal" design: every variable's middle bin (idx 2). Used in tests
/// and as a visualization reference.
pub const NOMINAL: Design = [2u8; L];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_resolves_to_workspace_defaults() {
        // Spot-check that the nominal-bin catalog values match the
        // crate-default values from spike-sar-adc + sub-blocks.
        assert_eq!(c_hold(2),     100e-15);   // SampleHold default 100 fF
        assert_eq!(comp_k(2),     1000.0);    // Comparator default k = 1000
        assert_eq!(comp_voh(2),   1.8);
        assert_eq!(dac_r_ohms(2), 10e3);      // R2RDac default 10 kΩ
        assert_eq!(vref(2),       1.0);
    }

    #[test]
    fn cliques_cover_every_variable_disjointly() {
        use std::collections::BTreeSet;
        let mut covered: BTreeSet<usize> = BTreeSet::new();
        for c in 0..N_CLIQUES {
            for v in clique_vars(c) {
                assert!(covered.insert(v), "var {v} appears in two cliques");
            }
        }
        assert_eq!(covered.len(), L, "not every variable in a clique");
    }
}
