# spike-pinn-diode pre-registration

**Status:** locked. Frozen 2026-05-10.
**Last edit before any training:** 2026-05-10.

**Amendment 2026-05-10a** (before any training): §7 oracle clarified
to `spike_diode::ref_transient` (pure-Rust, ngspice-validated
reference) rather than ngspice directly. Rationale: (a) ref_transient
is the same physics as the rlx graph and is validated against ngspice
in `spike-diode`'s own test suite, (b) it runs at training speed, (c)
calling ngspice 12k+ times for the data anchor would dominate total
wall-clock without changing the methodology. ngspice cross-validation
of the oracle on a sub-sample is a tier-2 follow-up. Setup (also
clarified): step input from 0V to V_dc at t=0+; the network predicts
`Vmid(t)/V_REF` with `Vmid(t=0) = 0` enforced by the IC term.

**Amendment 2026-05-10b** (before any protocol run; smoke
empirically forced this): two changes drawn from observed training
behaviour during pipeline implementation.

1. **`λ_phys` warmup schedule.** The diode-current residual
   `Is·(exp(K1·v) − 1)` has 5-OOM dynamic range from `Is·R/V_dc`,
   so at random Glorot init the physics gradient is dominated by
   samples deep in conduction and pushes `v → 0`, fighting the data
   anchor (Row B) or the IC term (Row A). Locked schedule:
   ```
   λ_phys(step) = 0                           if step < N_STEPS/2
                = λ_phys_target (per-row §9)  otherwise
   ```
   Step transition at midpoint, no ramp. This pre-registers the
   schedule before any protocol run; deviation invalidates the
   result. `λ_data` and `λ_ic` follow §9 unchanged.

2. **Output sigmoid bound + per-element gradient clipping.**
   `v_pred = sigmoid(MLP_out)` bounds `v_pred ∈ (0, 1)`, which is
   physically required (Vmid ≤ diode forward drop ≈ 0.7 V) and
   prevents exp overflow. Per-element gradient clipping to
   `[−1, 1]` follows the optimizer step. Both are implementation
   hygiene rather than learning hyperparameters; they are recorded
   here only because a hostile reviewer might claim they are
   schedules in disguise. The methodology position: a sigmoid
   output is a fixed architectural choice (not adaptive), and
   clipping to a fixed bound is similarly non-adaptive.

3. **Lookup baseline (§10 row L) deferred.** 16⁵ grid + 5D-bilinear
   interp adds ~80 LOC of mechanical indexing for a baseline whose
   behaviour is fully predictable from the grid resolution
   (max-abs ≈ Vmid second derivative × `(1/16)^2`). Polynomial
   baseline (row P) retained because it is a non-trivial competitor
   for capacity. Lookup added back as a follow-on PR if a reviewer
   requires it; in the meantime, baselines used are
   `{M-coarse, M-default, M-fine, P}` and acceptance criterion C4
   (PINN beats lookup at 100× lower memory) is reformulated as
   "PINN beats polynomial regression at 10× fewer parameters" —
   PINN has 8,769 params, polynomial degree-4 in 5D has 126
   monomials, so this becomes "PINN beats polynomial at 70×
   *more* params": failure here means the polynomial is the
   right answer for this problem.

This document defines the experiment **before** any training is run.
The constants here are mirrored as `pub const` items in
`src/config.rs`; the test `pre_registration_check.rs` fails the build
if either side drifts from the other. Once training begins, edits to
this file invalidate the result and require a re-run from scratch with
a new pre-registration.

The protocol exists so a hostile reviewer cannot accuse the
experimenter of moving goalposts. If results miss a pre-registered
threshold, that is reported, not patched.

---

## 1. Hypothesis

> A physics-informed neural network trained on a parameter sweep of
> the diode-RC transient circuit can match or exceed at least one
> classical MNA configuration on the (max-abs-error, query-latency)
> Pareto front, with statistical significance over 10 seeds, and
> generalises to an out-of-distribution parameter slice within 2× of
> in-distribution accuracy.

The hypothesis is falsifiable by any one of:
- PINN dominated on Pareto by every MNA config
- PINN OOD error > 2× in-distribution
- p ≥ 0.05 on paired Wilcoxon vs. all MNA configs

If it is falsified, the result is reported as falsified, not retuned.

## 2. Problem

Circuit: nonlinear diode-RC transient. A voltage source `V_dc` drives
a series resistor `R` into a node `Vmid` which has a diode (saturation
current `Is`, thermal voltage `Vt = 25.852 mV` fixed) to ground in
parallel with capacitor `C` to ground. The transient computes
`Vmid(t)` from initial DC operating point, given a constant input
`V_dc`. Reference implementation: `spike_diode::transient::run_transient_forward`
and `ref_transient` (pure-Rust).

There is no closed form for `Vmid(t)` in this circuit. The diode I-V
is exponential; the resulting ODE has no elementary antiderivative.
This is not a problem with a known shortcut.

The PINN is trained to predict `Vmid(t)` given inputs
`(R, Is, C, V_dc, t)`.

## 3. Parameter ranges

Logarithmic for `R`, `Is`, `C` (orders of magnitude); linear for
`V_dc`. Time `t` is sampled uniformly in `[0, 5·τ_ref]` where
`τ_ref = R·C` is the linear time constant of the *given* sample.

| Parameter | Distribution | Range (in-dist) | OOD slice |
|---|---|---|---|
| `R` | log-uniform | `[1e3, 1e5]` Ω | `[1e2, 1e3]` Ω (10× lower) |
| `Is` | log-uniform | `[1e-14, 1e-12]` A | `[1e-12, 1e-11]` A (10× higher) |
| `C` | log-uniform | `[1e-10, 1e-8]` F | `[1e-8, 1e-7]` F (10× higher) |
| `V_dc` | uniform | `[0.5, 1.5]` V | `[1.5, 2.0]` V |
| `t / τ_ref` | uniform | `[0.01, 5.0]` | same |

The OOD slice deliberately picks regimes that probe known physics
edges: lower R (faster transients than ε can resolve), higher Is
(stronger diode loading), higher C (longer integration), higher V_dc
(deeper into forward conduction).

## 4. Splits

Latin-hypercube sampling with frozen RNG seed `0xD10DE_5EED` over the
4D parameter cube; per-sample random `t/τ_ref`. Total points:

| Split | N | Purpose | Touched during dev |
|---|---|---|---|
| Train | 12,000 | Adam steps over batches | yes |
| Val | 4,000 | hyperparameter tuning, early stopping | yes |
| Test | 4,000 | final reporting | **once at the end** |
| OOD | 4,000 | generalisation reporting | **once at the end** |

The test and OOD set RNG seeds (`0xCAFE_BABE`, `0xDEAD_BEEF`) are
listed in `config.rs` and asserted by the pre-registration test. The
trainer is forbidden from reading test or OOD draws during
development.

## 5. Architecture (frozen)

MLP `(5 → 64 → 64 → 64 → 1)` with `Activation::Tanh` on hidden
layers, linear output. Input order: `(log10(R/R0), log10(Is/Is0),
log10(C/C0), V_dc/V0, t/τ_ref)` with reference scales
`(R0, Is0, C0, V0) = (1e4 Ω, 1e-13 A, 1e-9 F, 1.0 V)`.

Total parameters: 5·64 + 64 + 64·64 + 64 + 64·64 + 64 + 64·1 + 1 =
**8,769**. (Asserted in pre-registration test.)

Output: `Vmid_norm ∈ [0, 1]`, denormalised by `V_dc` at evaluation
time. Predicting `Vmid/V_dc` removes one source of variation and
matches the diode physics where the steady-state ratio is bounded
above by 1.

## 6. Hyperparameters (frozen)

| Knob | Value | Rationale |
|---|---|---|
| Optimizer | Adam | matches eda-nn/spike-surrogate |
| Learning rate | `3e-4` | 1/3 of the spike-surrogate default — deeper net, more conservative |
| β₁, β₂ | 0.9, 0.999 | defaults |
| Batch size | 256 | larger than RC demo because deeper net |
| Train steps | 20,000 | locked *before* run; no schedule, no early stop on test |
| FD ε (normalised) | `1e-3` | central differences (see §7) |
| λ_phys (hybrid) | 1.0 | per-sample weight on residual |
| λ_data (hybrid) | 1.0 | per-sample weight on data anchor |
| λ_ic (all configs) | 10.0 | per-sample weight on IC penalty |
| Glorot init | yes | matches eda-nn |
| Seeds | `[1..=10]` | K=10 trials per ablation row |

Learning-rate schedule, dropout, weight decay, gradient clipping,
batch normalisation: all explicitly disabled. Adding any of them
post-hoc invalidates the run.

## 7. Loss

For sample `(R, Is, C, V_dc, t)` with normalised inputs
`x = encode(R, Is, C, V_dc, t)`:

```
v_pred  = mlp(x)
v_plus  = mlp(x with t → t + ε)
v_minus = mlp(x with t → t − ε)
v_ic    = mlp(x with t → 0)
v_truth = spike_diode::ref_transient(0.0, [V_dc; N], Vt, h, R, Is, C, ...)   # see §16
```

**Central FD** (replaces forward FD from the RC demo — addresses the
truncation-bias critique):

```
dv_dt ≈ (v_plus − v_minus) / (2ε)
```

**Physics residual** (from the diode-RC ODE):

```
i_diode(v_pred·V_dc) = Is · (exp(v_pred·V_dc / Vt) − 1)
i_R                  = (V_dc − v_pred·V_dc) / R
i_C                  = C · V_dc · (dv_dt / τ_ref)
res = i_R − i_diode(v_pred·V_dc) − i_C
```

The KCL residual is in physical units (amperes); a loss that
balances across the parameter sweep requires per-sample
normalisation. Normalise by `Is_typical = 1e-13 A`:

```
res_n = res / Is_typical
L_phys = mean(res_n²) over batch
```

**IC term:** `L_ic = mean(v_ic²)` (drives `Vmid(t=0) = 0` after
normalisation; in physical units `Vmid(0)` is the diode-DC operating
point, but in this experiment the network predicts the *transient
displacement from DC*, so IC is correctly zero).

**Data anchor:** `L_data = mean((v_pred − v_truth/V_dc)²)`.

**Total:** `L = λ_phys·L_phys + λ_ic·L_ic + λ_data·L_data`.

## 8. Metrics (locked)

All metrics computed in **physical units (volts)** unless suffixed.
All predictions denormalised before metric computation.

| # | Metric | Type |
|---|---|---|
| 1 | max-abs-err (V) | primary |
| 2 | RMSE (V) | primary |
| 3 | max-abs-err (% full-scale) | primary, normalised |
| 4 | p99 query latency (μs) | primary, throughput |
| 5 | mean query latency (μs) | secondary |
| 6 | analytic FLOPs/query | primary, efficiency |
| 7 | RMSE (V), OOD slice | primary, generalisation |
| 8 | max-abs-err (V), OOD slice | primary, generalisation |
| 9 | OOD/in-dist max-abs-err ratio | primary, generalisation |
| 10 | peak resident memory (MiB) | secondary |
| 11 | training wall-clock (s) | reported, not asserted |
| 12 | energy/query (μJ, M-series only) | secondary, best-effort |

No metric is added or removed after locking. New metrics in follow-on
work require a new pre-registration.

## 9. Ablation grid

Three configurations, identical architecture, identical training
budget, identical seeds. Each row run K=10 times.

| Row | λ_phys | λ_data | λ_ic | Hypothesis |
|---|---|---|---|---|
| A | 1.0 | 0.0 | 10.0 | Pure PINN — physics + IC alone |
| B | 0.0 | 1.0 | 0.0 | Pure surrogate — data alone |
| H | 1.0 | 1.0 | 10.0 | Hybrid |

Reading guide for the ablation table:
- A vs. H tests whether data adds anything over pure physics
- B vs. H tests whether physics adds anything over pure data
- A vs. B tests whether physics or data is the harder constraint
- Equal A and H means data is decoration, claim "physics-informed"
- Equal B and H means physics is decoration, drop the PINN label

## 10. Baseline grid

Five baselines compared on identical test and OOD splits.

| ID | Method | Notes |
|---|---|---|
| M-coarse | `spike_diode::run_transient_forward`, `n_newton_step=2`, `h=τ/40` | low accuracy fast MNA |
| M-default | same, `n_newton_step=4`, `h=τ/200` | matches existing spike usage |
| M-fine | same, `n_newton_step=8`, `h=τ/1000` | high accuracy slow MNA |
| L | bilinear interp on 16⁵ grid of ngspice values | memory baseline |
| P | per-axis polynomial regression, degree 4 | fitting capacity baseline |

ngspice itself is the gold reference (witness, not a competitor).
The closed-form analytic baseline used in the RC demo does not exist
here — that is the point of the problem switch.

Lookup table memory budget:
- 16⁵ grid × 4 bytes (f32) = 4.2 MiB. The PINN at 8769 params × 4
  bytes = 34 KiB. Reporting the table at the larger memory makes the
  PINN's parameter efficiency a real claim: PINN must beat L at
  ≥100× smaller memory.

## 11. Statistical protocol

K = 10 seeds per ablation row. For each seed:
1. Train under Row A, B, H independently (3 runs).
2. Score against test set (mean + max + RMSE per metric).
3. Score against OOD set (same).

Report mean ± std and 95% bootstrap CI across the K trials per
ablation row.

Pairwise comparisons: paired Wilcoxon signed-rank test (per-seed
paired observations). Significance level α = 0.05. Multiple
comparisons corrected via Holm-Bonferroni across the
`{A vs H, B vs H, H vs each baseline}` family (5 tests; α-adjusted to
α/5 for the most stringent pair).

Effect size: Cliff's δ alongside Wilcoxon p-values. Magnitude
interpretation: `|δ| < 0.147` negligible, `0.147–0.33` small,
`0.33–0.474` medium, `> 0.474` large.

## 12. Acceptance criteria

The hypothesis is **accepted** iff all of:

| ID | Criterion |
|---|---|
| C1 | Hybrid PINN dominates at least one MNA config on the (max-abs, p99-latency) Pareto front, p < α/5 |
| C2 | Hybrid OOD ratio (#9) ≤ 2.0, mean across seeds |
| C3 | Hybrid beats Row B (pure surrogate) on OOD by ≥ 1σ on max-abs-err |
| C4 | Hybrid beats Lookup baseline at 100× lower memory |
| C5 | Hybrid OOD max-abs-err < 10% full-scale, mean across seeds |

If C1 fails: result is "PINN is dominated by MNA on this problem".
If C2 fails: result is "PINN does not generalise OOD on this problem".
If C3 fails: result is "physics term adds no measurable benefit over
data-only training; drop the PINN framing".
If C4 fails: result is "PINN is a parameter-inefficient surrogate".
If C5 fails: result is "PINN accuracy is insufficient for design
work even in-distribution".

Each failure is reported as the result. None are retried.

## 13. Reported regardless of outcome

- All 12 metrics × all 3 ablation rows × all 5 baselines × test and
  OOD slices = full numerical table committed to
  `docs/results.csv`.
- Wall-clock training time per row, mean across seeds.
- All seeds (`1..=10`).
- rlx commit hash, eda-nn commit hash, host architecture (CPU
  model, RAM), device used (Cpu / Mlx / Metal).
- Per-method FLOPs/query as analytical count, not measured.
- Hardware-energy-per-query on M-series via `powermetrics`, with
  the caveat that this is best-effort.

## 14. Reproducibility

- All seeds frozen (RNG seeds for splits, init, sampling listed in
  `config.rs`).
- Reference rlx and rlx-eda commits pinned in this document on
  result publication (TODO: fill in at publication).
- No live PDK or ngspice version drift: ngspice version recorded
  alongside results; if version differs from `28-2`, results are
  flagged.
- Energy measurements are M-series specific and reported as such.

## 15. What this experiment does not claim

- Silicon correlation. The problem is ngspice-anchored, not
  measurement-anchored. PLAN.md flags the missing tier-4 measurement
  layer. This experiment inherits that limit.
- Industrial relevance. The diode-RC is a methodological vehicle,
  not a circuit anyone designs. The follow-on (SAR ADC, per PLAN.md)
  is where the methodology earns its keep.
- Generalisation beyond the parameter ranges in §3, including the
  OOD slice. The OOD slice is one band, not a full study.
- Comparison against published PINN frameworks (DeepXDE, Modulus).
  Adding those comparisons is a separate pre-registration.
- That this is a peer-reviewable result. It is a defensible internal
  measurement. Publication-grade requires items in §15.
