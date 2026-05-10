//! Frozen experiment constants — mirror of `preregistration.md`.

// ── §3 Parameter range (1-D, unit interval) ──────────────────────
pub const X_LO: f32 = 0.0;
pub const X_HI: f32 = 1.0;

// ── §2 SAR ADC config ────────────────────────────────────────────
pub const N_BITS: usize = 8;
pub const VREF:   f32   = 1.0;
pub const LEVELS: u32   = 1u32 << N_BITS; // 256

// ── §4 Splits ────────────────────────────────────────────────────
pub const N_TRAIN: usize = 12_000;
pub const N_VAL:   usize = 4_000;
pub const N_TEST:  usize = 4_000;

pub const SPLIT_SEED_TRAIN: u64 = 0xCAB5_5AAD;
pub const SPLIT_SEED_TEST:  u64 = 0xC0DE_BABE;

// ── §5 Architecture ──────────────────────────────────────────────
pub const ARCH_DIMS: &[usize] = &[1, 32, 32, 1];

/// `Σ (in_i · out_i + out_i)` — asserted in the parity test.
/// 1·32+32 + 32·32+32 + 32·1+1 = 32 + 32 + 1024 + 32 + 32 + 1 = 1153.
pub const TOTAL_PARAMS: usize = 1 * 32 + 32
                              + 32 * 32 + 32
                              + 32 * 1  + 1;

// ── §6 Hyperparameters ───────────────────────────────────────────
pub const LR:      f32   = 3.0e-4;
pub const BATCH:   usize = 128;
pub const N_STEPS: usize = 20_000;
pub const N_SEEDS: usize = 10;

// ── §10 Baseline parameters ──────────────────────────────────────
pub const POLY_DEGREES:    &[usize] = &[4, 8, 16];
pub const LOOKUP_SIZES:    &[usize] = &[16, 64, 256];

// ── §11 Statistics ───────────────────────────────────────────────
pub const ALPHA: f64 = 0.05;
/// Family size: 1 PINN vs each of 6 baselines = 6 pairwise tests.
pub const HOLM_FAMILY_SIZE: usize = 6;

// ── §12 Acceptance thresholds ────────────────────────────────────
/// ½ LSB on `code/256` scale = `1 / (2 · 256) = 1/512`.
pub const C5_HALF_LSB: f32 = 1.0 / (2.0 * LEVELS as f32);
