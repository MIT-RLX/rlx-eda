//! Frozen experiment constants — mirror of `preregistration.md`.
//!
//! Every value here is asserted by `tests/pre_registration_check.rs`
//! to match the pre-registration document. Edits to either side
//! without updating the other fail CI. **Edits after training begins
//! invalidate the experimental result.**
//!
//! When in doubt, treat this file as read-only. The methodology
//! point of the crate is that this file does not change once locked.

// ── §3. Parameter ranges (in-distribution) ───────────────────────────

pub const R_LO:    f32 = 1.0e3;    // Ω
pub const R_HI:    f32 = 1.0e5;    // Ω
pub const IS_LO:   f32 = 1.0e-14;  // A
pub const IS_HI:   f32 = 1.0e-12;  // A
pub const C_LO:    f32 = 1.0e-10;  // F
pub const C_HI:    f32 = 1.0e-8;   // F
pub const VDC_LO:  f32 = 0.5;      // V
pub const VDC_HI:  f32 = 1.5;      // V
pub const T_OVER_TAU_LO: f32 = 0.01;
pub const T_OVER_TAU_HI: f32 = 5.0;

// ── §3. OOD slice ────────────────────────────────────────────────────

pub const R_OOD_LO:    f32 = 1.0e2;
pub const R_OOD_HI:    f32 = 1.0e3;
pub const IS_OOD_LO:   f32 = 1.0e-12;
pub const IS_OOD_HI:   f32 = 1.0e-11;
pub const C_OOD_LO:    f32 = 1.0e-8;
pub const C_OOD_HI:    f32 = 1.0e-7;
pub const VDC_OOD_LO:  f32 = 1.5;
pub const VDC_OOD_HI:  f32 = 2.0;

// ── §3. Reference scales for input encoding ──────────────────────────

pub const R_REF:   f32 = 1.0e4;     // Ω
pub const IS_REF:  f32 = 1.0e-13;   // A
pub const C_REF:   f32 = 1.0e-9;    // F
pub const V_REF:   f32 = 1.0;       // V

/// Thermal voltage at T ≈ 300 K. Matches `spike_diode::VT`.
pub const VT:      f32 = 0.025_852;

// ── §4. Splits ───────────────────────────────────────────────────────

pub const N_TRAIN: usize = 12_000;
pub const N_VAL:   usize = 4_000;
pub const N_TEST:  usize = 4_000;
pub const N_OOD:   usize = 4_000;

pub const SPLIT_SEED_LHS:  u64 = 0xD10DE_5EED;
pub const SPLIT_SEED_TEST: u64 = 0xCAFE_BABE;
pub const SPLIT_SEED_OOD:  u64 = 0xDEAD_BEEF;

// ── §5. Architecture ─────────────────────────────────────────────────

pub const ARCH_DIMS: &[usize] = &[5, 64, 64, 64, 1];

/// Sum-product of weights + biases. Asserted in the pre-registration
/// test against §5 of the document.
pub const TOTAL_PARAMS: usize = 5 * 64 + 64
                              + 64 * 64 + 64
                              + 64 * 64 + 64
                              + 64 * 1  + 1;

// ── §6. Hyperparameters ──────────────────────────────────────────────

pub const LR:           f32   = 3.0e-4;
pub const BETA1:        f32   = 0.9;
pub const BETA2:        f32   = 0.999;
pub const BATCH:        usize = 256;
pub const N_STEPS:      usize = 20_000;
pub const EPS_T_NORM:   f32   = 1.0e-3;
pub const LAMBDA_IC:    f32   = 10.0;
pub const N_SEEDS:      usize = 10;

// ── §7. Loss normalisation ───────────────────────────────────────────

/// Reference saturation current used to non-dimensionalise the KCL
/// residual. The middle of the in-distribution `Is` range on a log
/// scale.
pub const IS_TYPICAL:   f32 = 1.0e-13;

// ── §9. Ablation rows ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ablation {
    pub row: char,
    pub lambda_phys: f32,
    pub lambda_data: f32,
    pub lambda_ic:   f32,
}

pub const ABL_PURE_PINN: Ablation = Ablation {
    row: 'A', lambda_phys: 1.0, lambda_data: 0.0, lambda_ic: LAMBDA_IC,
};
pub const ABL_PURE_SURROGATE: Ablation = Ablation {
    row: 'B', lambda_phys: 0.0, lambda_data: 1.0, lambda_ic: 0.0,
};
pub const ABL_HYBRID: Ablation = Ablation {
    row: 'H', lambda_phys: 1.0, lambda_data: 1.0, lambda_ic: LAMBDA_IC,
};

pub const ABLATIONS: &[Ablation] = &[
    ABL_PURE_PINN,
    ABL_PURE_SURROGATE,
    ABL_HYBRID,
];

// ── §10. Baseline configurations ─────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct MnaConfig {
    pub id: &'static str,
    pub n_newton_step: usize,
    pub steps_per_tau: usize,
}

pub const MNA_COARSE:  MnaConfig = MnaConfig { id: "M-coarse",  n_newton_step: 2, steps_per_tau:   40 };
pub const MNA_DEFAULT: MnaConfig = MnaConfig { id: "M-default", n_newton_step: 4, steps_per_tau:  200 };
pub const MNA_FINE:    MnaConfig = MnaConfig { id: "M-fine",    n_newton_step: 8, steps_per_tau: 1000 };

pub const MNA_BASELINES: &[MnaConfig] = &[MNA_COARSE, MNA_DEFAULT, MNA_FINE];

/// Lookup-table memory baseline: 16⁵ grid of f32 ngspice values.
pub const LOOKUP_GRID_PER_AXIS: usize = 16;
pub const LOOKUP_BYTES_PER_VALUE: usize = 4;
pub const LOOKUP_TABLE_BYTES: usize =
    LOOKUP_GRID_PER_AXIS.pow(5) * LOOKUP_BYTES_PER_VALUE;

/// Polynomial-regression baseline degree (per-axis).
pub const POLY_DEGREE: usize = 4;

// ── §11. Statistics ──────────────────────────────────────────────────

pub const ALPHA: f64 = 0.05;
/// Holm-Bonferroni family size per §11. {A-vs-H, B-vs-H, H-vs-each-of-3-MNA-baselines}.
pub const HOLM_FAMILY_SIZE: usize = 5;

// ── §12. Acceptance criteria thresholds ──────────────────────────────

pub const C2_OOD_RATIO_MAX:        f32 = 2.0;
pub const C3_HYBRID_BEATS_DATA_BY: f32 = 1.0; // multiples of std-dev
pub const C4_LOOKUP_MEMORY_RATIO:  f32 = 100.0;
pub const C5_OOD_MAX_ABS_ERR_FS:   f32 = 0.10; // 10% of full-scale (V_REF)
