//! Frozen experiment constants — mirror of `preregistration.md`.

// ── §2 SAR ADC + mismatch ────────────────────────────────────────
pub const N_BITS: usize = 8;
pub const VREF:   f32   = 1.0;
pub const LEVELS: u32   = 1u32 << N_BITS;

pub const SIGMA_R:      f64 = 5.0e-2;
pub const SIGMA_OFFSET: f64 = 5.0e-3;

// ── §3 Inputs ────────────────────────────────────────────────────
/// 1 (vin) + N_BITS (per-bit weight errors) + 1 (comparator offset).
pub const INPUT_DIM: usize = 1 + N_BITS + 1;

// ── §4 Splits ────────────────────────────────────────────────────
pub const N_TRAIN: usize = 12_000;
pub const N_VAL:   usize = 4_000;
pub const N_TEST:  usize = 4_000;

pub const SPLIT_SEED_TRAIN: u64 = 0xCAB5_5AAD_5AAD_BEEF;
pub const SPLIT_SEED_TEST:  u64 = 0xC0DE_BABE_BABE_C0DE;

// ── §5 Architecture ──────────────────────────────────────────────
pub const ARCH_DIMS: &[usize] = &[10, 64, 64, 1];

/// 10·64+64 + 64·64+64 + 64·1+1 = 640+64 + 4096+64 + 64+1 = 4929.
pub const TOTAL_PARAMS: usize = 10 * 64 + 64
                              + 64 * 64 + 64
                              + 64 * 1  + 1;

// ── §6 Hyperparameters ───────────────────────────────────────────
pub const LR:      f32   = 3.0e-4;
pub const BATCH:   usize = 256;
pub const N_STEPS: usize = 20_000;
pub const N_SEEDS: usize = 10;

// ── §10 Baselines ────────────────────────────────────────────────
pub const POLY_DEGREES: &[usize] = &[1, 2, 4];

// ── §11 Statistics ───────────────────────────────────────────────
pub const ALPHA: f64 = 0.05;
pub const HOLM_FAMILY_SIZE: usize = 3;

// ── §12 Acceptance thresholds ────────────────────────────────────
pub const C5_ONE_LSB: f32 = 1.0 / LEVELS as f32;
