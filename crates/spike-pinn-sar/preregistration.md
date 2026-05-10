# spike-pinn-sar pre-registration

**Status:** locked. Frozen 2026-05-10.

This document defines the SAR-ADC PINN/surrogate experiment **before**
any training is run. Mirrors the methodological shape of
`spike-pinn-diode` (pre-registration → parity test → ablation →
baselines → stats → results). Constants here are mirrored as `pub
const` items in `src/config.rs`; the test
`pre_registration_check.rs` fails the build if either side drifts.

The diode-RC experiment falsified its hypothesis (polynomial
regression beat the PINN at machine precision because the
diode-RC's parametric output is smooth in log coordinates). This
experiment moves to a structurally different problem — the SAR
ADC's input→output map is a discrete staircase — where polynomial
fits are *expected* to fail. The empirical question this
pre-registration commits to: does that expectation hold?

---

## 1. Hypothesis

> A small MLP trained on `vin/vref → code/256` for an ideal 8-bit
> SAR ADC achieves max-abs error < ½ LSB across K=10 seeds and
> beats every polynomial baseline up to degree 16, with statistical
> significance.

Falsifiable by: max-abs ≥ ½ LSB, OR p ≥ α/family for any
polynomial comparison, OR Lookup-64 dominating PINN on the
(accuracy, memory) Pareto.

If falsified, the report is *that* — not a retuned re-run.

## 2. Problem

Circuit: ideal 8-bit `BehavioralSar` from
`spike-sar-adc/src/behavioral.rs` with `Realization::ideal` (all
mismatch errors zero), `vref = 1.0`, all noise σ = 0. The oracle
is deterministic: `convert(vin) → code` where
`code ∈ {0, 1, ..., 255}`.

The PINN predicts `code/256 ∈ [0, 1]` from a single normalised
input `x = vin/vref ∈ [0, 1]`.

There is no closed form for the SAR transfer function in the form
that polynomial regression handles well. The output is piecewise-
constant with 256 steps; a polynomial of degree d cannot represent
more than ≈d/2 inflections, so polynomial fits should fail
catastrophically for `d < 256`. This problem is the polar opposite
of diode-RC's smooth-everywhere structure — exactly where neural
nets with sigmoid-style activations have a structural advantage.

## 3. Parameter range

| Parameter | Distribution | Range |
|---|---|---|
| `x = vin/vref` | uniform | `[0, 1)` |

Single 1-D input. **No OOD slice**: with bounded domain on a 1-D
input there is no extrapolation regime that is not pathological.
Acceptance criteria that reference OOD (C2, C3 from the diode
protocol) are not evaluated for SAR.

## 4. Splits

Uniform random sampling on `[0, 1)` with frozen RNG seeds.

| Split | N | Purpose | Touched during dev |
|---|---|---|---|
| Train | 12,000 | Adam steps over batches | yes |
| Val | 4,000 | hyperparameter tuning, early stopping | yes |
| Test | 4,000 | final reporting | **once at the end** |

Seeds: `SPLIT_SEED_TRAIN = 0xCAB5_5AAD`, `SPLIT_SEED_TEST =
0xC0DE_BABE`. Listed in `config.rs` and asserted by the parity
test.

## 5. Architecture (frozen)

MLP `1 → 32 → 32 → 1` with `Activation::Tanh` on hidden layers,
**sigmoid** on the output. Total params:
`1·32 + 32 + 32·32 + 32 + 32·1 + 1 = 1,153`. Asserted in the
parity test.

The smaller architecture (vs diode's `5→64→64→64→1` at 8,769
params) reflects the smaller input dimension and the fact that a
1-D regression problem doesn't need the same capacity. Smaller
PINN also makes the *capacity-efficiency* baseline comparison
(C1') meaningful — if PINN at 1,121 params can't beat a 17-param
polynomial, that's the answer.

## 6. Hyperparameters (frozen)

| Knob | Value | Rationale |
|---|---|---|
| Optimizer | Adam | matches eda-nn defaults |
| Learning rate | `3e-4` | same as diode protocol |
| β₁, β₂ | 0.9, 0.999 | defaults |
| Batch size | 128 | smaller graph than diode |
| Train steps | 20,000 | locked, no early stop on test |
| Glorot init | yes | matches eda-nn |
| Sigmoid output | yes | bounds prediction in [0, 1] |
| Per-element grad clip | `[-1, 1]` | training-stability hygiene |
| Seeds | `[1..=10]` | K=10 trials |

LR schedule, dropout, weight decay, BatchNorm: all explicitly
disabled.

## 7. Loss

Pure data MSE — no physics residual, no IC term. SAR is a
discrete iterative algorithm, not an ODE; there is no smooth
residual to enforce.

```
v_pred = sigmoid(MLP(x))
L      = mean((v_pred − y_truth)²)
```

The diode protocol's three ablation rows (A: pure-PINN,
B: surrogate, H: hybrid) collapse to a single row here. This is
labelled as **Row B'** to distinguish it from diode's Row B.

## 8. Metrics (locked)

All metrics computed in normalised output units (`code/256`); ½
LSB = 1/512 ≈ 0.00195.

| # | Metric | Type |
|---|---|---|
| 1 | max-abs (units of 1) | primary |
| 2 | RMSE | primary |
| 3 | max-abs in LSBs (× 256) | primary, normalised |
| 4 | p99 query latency (µs) | primary, throughput |
| 5 | mean query latency (µs) | secondary |
| 6 | training wall-clock (s) | reported |
| 7 | analytic params count | primary, capacity |
| 8 | analytic memory bytes | primary, memory |

## 9. Ablation grid

Single row: PINN with pure-data MSE loss. Diode protocol's Row A
(pure-PINN) and Row H (hybrid) are N/A. Documented explicitly in
`§16c` of `spike-pinn-diode/preregistration.md` (which is where
the original methodology was defined; this crate inherits and
specialises).

## 10. Baseline grid

Six baselines, all evaluated on the same test split.

| ID | Method | Params |
|---|---|---|
| Poly-d4 | 1-D polynomial regression, degree 4 | 5 |
| Poly-d8 | 1-D polynomial regression, degree 8 | 9 |
| Poly-d16 | 1-D polynomial regression, degree 16 | 17 |
| Lookup-16 | uniform grid + linear interp, 16 nodes | 16 |
| Lookup-64 | uniform grid + linear interp, 64 nodes | 64 |
| Lookup-256 | uniform grid + linear interp, 256 nodes | 256 |

`Lookup-256` is exact-at-grid-points (matches the 256-step
output's boundaries); it is effectively the *upper bound* of any
interpolation-based 1-D method. PINN beating Lookup-256 would be
unexpected; the meaningful question is PINN vs `Lookup-64` and
`Poly-d16`.

No MNA baseline: the behavioral oracle in `spike-sar-adc` is the
ground truth, and there is no independent MNA solve for SAR-as-
system in tree.

## 11. Statistics

K = 10 seeds. Same protocol as diode:
- Mean ± std + 95% bootstrap CI per metric.
- Paired Wilcoxon signed-rank (exact enumeration over `2^K`
  permutations) for PINN vs each baseline.
- Cliff's δ + magnitude bins.
- Holm-Bonferroni correction across the family of 6 pairwise
  tests, α = 0.05.

## 12. Acceptance criteria

| ID | Criterion |
|---|---|
| C1' (capacity) | PINN max-abs ≤ Poly-d16 max-abs, p < α/6 |
| C2' (memory) | PINN max-abs ≤ Lookup-64 max-abs, p < α/6 |
| C5' (sub-LSB) | PINN max-abs < ½ LSB = 1/512, mean across seeds |

All three must hold for hypothesis acceptance. Each failure is a
reportable result and is **not** retried.

If C1' fails: result is "polynomial of modest degree is the right
tool for SAR-staircase regression; PINN's smoothing advantage is
illusory at this fidelity".

If C2' fails: result is "PINN is a parameter-inefficient surrogate
compared to a 64-point lookup table".

If C5' fails: result is "PINN at this architecture and training
budget cannot represent an 8-bit SAR transfer function below 1
LSB; either capacity, training time, or representation choice is
inadequate".

## 13. Reported regardless of outcome

- All 8 metrics × all 7 methods (PINN + 6 baselines) committed to
  `docs/results.md`.
- Per-seed PINN trace.
- rlx commit, rlx-eda commit, host architecture, device used.
- Training wall-clock per seed.

## 14. Reproducibility

- All seeds frozen in `config.rs`.
- Behavioral oracle is pure-Rust, deterministic given seeds.
- No PDK or external simulator dependency — fully self-contained.

## 15. What this experiment does not claim

- Real SAR ADC silicon correlation. The oracle is the *behavioral*
  simulator (pure algorithm, optionally with mismatch/noise). The
  silicon-correlation tier from `spike-pinn-diode` §15 applies
  here too: this experiment validates AD/methodology, not silicon
  performance.
- Multi-bit (`n_bits ≠ 8`) results. A single n_bits is locked;
  scaling to other bit counts would require its own
  pre-registration.
- Non-ideal SAR (with mismatch/noise). The protocol locks
  `Realization::ideal` and all σ = 0 — a deterministic oracle.
  Stochastic SAR is a separate experiment.
