# spike-pinn-sar-mc pre-registration

**Status:** locked. Frozen 2026-05-10.

Mirrors `spike-pinn-diode` and `spike-pinn-sar` — pre-registration
locked **before** any training, mirrored as `pub const` items in
`src/config.rs`, parity enforced by `pre_registration_check.rs`.

The two prior experiments (smooth diode-RC and ideal 1-D 8-bit SAR)
were both won by polynomial regression. Both were low-dimensional —
diode 5-D, SAR-ideal 1-D. Polynomial monomial count grows as
`C(d+k, k)`, so at low d every reasonable degree fits in memory and
training data is plentiful. The high-D regime is where the
combinatorial blow-up actually kicks in. This crate tests the PINN
claim there.

---

## 1. Hypothesis

> A 10→64→64→1 MLP trained on 12,000 (vin + 8 bit-weight mismatches
> + comparator offset) samples beats the best polynomial regressor
> (degrees 1, 2, 4) on max-abs error across K=10 seeds with
> statistical significance, and stays under 1 LSB.

Falsifiable by: max-abs ≥ 1 LSB on the test split (mean across
seeds), OR p ≥ α/family for any polynomial comparison, OR PINN
dominated by Poly-d4 (the highest-capacity polynomial baseline) on
the (max-abs, train-time) Pareto.

## 2. Problem

Circuit: `BehavioralSar` from `spike-sar-adc/src/behavioral.rs`,
`n_bits=8`, `vref=1.0`, `comp_noise_sigma=0`. **Mismatch active**:
each sample carries its own `Realization` with `bit_weight_err`
drawn from `N(1, σ_R·√2)` and `comp_offset` from `N(0, σ_offset)`,
with σ values chosen to make the mismatch correction observable
above the quantisation floor:

- `σ_R = 5e-2` (5% per-resistor σ — substantially above
  Sky130-class trimming; intentional to exercise the model).
- `σ_offset = 5e-3` (5 mV comparator offset σ).

The PINN predicts `code/256 ∈ [0, 1]` from a **10-dimensional**
input.

## 3. Inputs

10-D, all normalised to approximately `[−1, 1]`:

| Index | Quantity | Encoding |
|---|---|---|
| 0 | `vin/vref` | `2·(vin/vref) − 1` (range `[−1, 1]`) |
| 1..8 | `bit_weight_err[0..7]` | `(err − 1) / (3·σ_R·√2)` |
| 9 | `comp_offset` | `offset / (3·σ_offset)` |

Bit-weight and offset normalisations divide by `3σ` so the
distribution sits near `[−1, 1]` with sub-percent mass outside.

## 4. Splits

Latin-hypercube sampling on the 10-D parameter cube, frozen seeds.

| Split | N | Purpose | Touched during dev |
|---|---|---|---|
| Train | 12,000 | Adam | yes |
| Val | 4,000 | hyperparameter tuning | yes |
| Test | 4,000 | final reporting | **once at the end** |

`SPLIT_SEED_TRAIN = 0xCAB5_5AAD_5AAD_BEEF`,
`SPLIT_SEED_TEST = 0xC0DE_BABE_BABE_C0DE`.

No OOD slice — same reasoning as `spike-pinn-sar`: bounded domain on
each axis, no extrapolation regime.

## 5. Architecture (frozen)

MLP `10 → 64 → 64 → 1`, Tanh hidden, sigmoid output. Param count:

`10·64 + 64 + 64·64 + 64 + 64·1 + 1 = 640 + 64 + 4096 + 64 + 64 + 1
= 4,929`.

(Asserted in parity test.)

## 6. Hyperparameters (frozen)

| Knob | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `3e-4` |
| β₁, β₂ | 0.9, 0.999 |
| Batch | 256 |
| Train steps | 20,000 |
| Seeds | `[1..=10]` |
| Glorot init | yes |
| Sigmoid output | yes |
| Per-element grad clip `[−1, 1]` | yes |

## 7. Loss

Pure data MSE — no physics term, no IC. Same Row B' style as
`spike-pinn-sar`.

## 8. Metrics

Same eight as `spike-pinn-sar`: max-abs (units), max-abs (LSB),
RMS, p99 / mean inference latency, training wall-clock, parameters,
memory bytes.

## 9. Ablation grid

Single row (Row B'). No physics ablation — same reasoning as the
ideal-SAR protocol.

## 10. Baseline grid

Three polynomial baselines on the 10-D normalised inputs.
**No lookup baseline**: 10-D uniform grid at 16 nodes per axis is
`16^10 ≈ 10^12` entries — infeasible by 5 orders of magnitude. This
is the central methodological point of the high-D regime: lookup
fails by construction, polynomial blows up combinatorially as `k`
rises, leaving learned surrogates as the natural option.

| ID | Method | Monomials |
|---|---|---|
| Poly-d1 | linear regression | `C(11, 1) = 11` |
| Poly-d2 | linear + pure quadratics + cross terms | `C(12, 2) = 66` |
| Poly-d4 | up to total degree 4 | `C(14, 4) = 1,001` |

Poly-d4 is the headline competitor: capacity-comparable to a small
MLP and well-conditioned at `N_TRAIN = 12k`.

## 11. Statistics

Same as prior crates: K=10 seeds, paired Wilcoxon signed-rank
(exact 2^K), Cliff's δ, Holm-Bonferroni at α=0.05 across the family
of 3 pairwise tests.

## 12. Acceptance criteria

| ID | Criterion |
|---|---|
| C1'' (capacity) | PINN max-abs ≤ Poly-d4 max-abs, p < α/3 |
| C2'' (capacity-progression) | PINN max-abs ≤ Poly-d2, AND Poly-d4 ≤ Poly-d2, AND PINN < Poly-d4 — i.e. capacity ordering matches |
| C5'' (functional) | PINN max-abs μ < 1 LSB = 1/256 |

## 13. Reported regardless

All 8 metrics × all 4 methods × per-seed PINN trace × wall-clock
× hardware metadata in `docs/results.md`.

## 14. Reproducibility

All seeds frozen. Behavioral oracle pure-Rust deterministic. No
external simulator dependency.

## 15. What this experiment does not claim

- Realistic mismatch parameters. σ_R = 5% is much larger than
  Sky130 trimming; chosen to exercise the model. A separate
  pre-registration with realistic σ would test whether PINN
  advantage (if found here) survives at production scales.
- High-bit-count SAR. `n_bits = 8` locked.
- Per-instance noise (comp_noise_sigma > 0). Stochastic SAR is its
  own experiment.
- Industrial relevance. Methodological pre-registration vehicle, not
  a circuit-design tool.

## 16. Amendments

(none yet)
