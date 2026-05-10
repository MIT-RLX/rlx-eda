# Worked example — `spike-dado-sar` end-to-end

The SAR ADC experiment is the most informative one in the workspace
because it exercises DADO's intended use case (per-sub-block
decomposition) *and* the practical surrogate-then-verify pipeline
that actually matters in real EDA flows. This document walks
through what happens, in enough detail to reproduce the result.

## Discrete-optimization context

The `spike-dado-*` crates apply [DADO][dado] (Decomposition-Aware
Distributional Optimization, Bowden/Levine/Listgarten, ICLR 2026) to
EDA design problems. Both compare DADO against a decomposition-unaware
EDA baseline, write artifacts (charts, schematics, GDS, ngspice
cross-validation) under `crates/spike-dado-*/docs/`, and emit a
`STORY.md` narrative built from real run numbers.

[dado]: https://arxiv.org/abs/2511.03032

| Crate | Granularity | Status |
| --- | --- | --- |
| [`spike-dado-r2r`](../crates/spike-dado-r2r/README.md) | Per-resistor sizing of one 8-bit R-2R DAC | Negative result on real R-2R objectives (max-INL, Σ-INL², max-DNL); decisive DADO win on a synthetic Σ-decomposable benchmark. |
| [`spike-dado-sar`](../crates/spike-dado-sar/README.md) | Per-sub-block discrete catalog choice for an SAR ADC | Three evaluators: analytical noise budget, ngspice on `SarAdc<4>`, and a **hybrid surrogate-then-verify pipeline** that's **36× faster than direct SPICE** at ~5% relative quality loss. |

## 1. Decomposition: which 12 parameters DADO breaks down

`spike-sar-adc::SarAdc<4>` composes four sub-blocks. For each
sub-block, the catalog (`crates/spike-dado-sar/src/catalog.rs`)
exposes a small set of discrete choices, picked from realistic
ranges. DADO's junction tree has one **clique per sub-block**, with
**empty separators** (no shared variables between blocks) — each
block's choices are statistically independent in the search
distribution.

| clique | variables (5 bins each) | catalog values |
| --- | --- | --- |
| **Sample-Hold** | `c_hold` | `{30, 50, 100, 200, 500}` fF |
|  | `sh_nmos_w`, `sh_pmos_w` | `{2, 5, 10, 20, 40}` µm / `{5, 10, 20, 40, 80}` µm |
|  | `sh_l` | `{0.5, 1, 2, 4, 8}` µm |
| **Comparator** | `comp_k` (gain) | `{100, 500, 1k, 5k, 10k}` |
|  | `comp_voh`, `comp_vol` | `{1.2…3.3}` V / `{0…0.5}` V |
| **DAC** | `dac_r_ohms` | `{1k, 5k, 10k, 20k, 50k}` Ω |
|  | `dac_match_pct` *(analytical only)* | `{0.1, 0.5, 1, 2, 5}` % σ |
|  | `vref` *(analytical only)* | `{0.6, 0.9, 1.0, 1.5, 1.8}` V |
| **SAR Logic** | `sar_nand_w`, `sar_inv_w` | `{2, 5, 10, 20, 40}` µm each |

12 variables × 5 bins = `5¹² ≈ 2.4 × 10⁸` distinct ADC designs.
`dac_match_pct` and `vref` only affect the analytical model — the
SPICE deck pins `vdd = 1.8 V` and ignores per-resistor mismatch (CMOS
logic-margin reasons; see the crate README).

## 2. Optimization: how DADO and naive EDA differ

Both fit the **same** factorised tabular categorical:

```
p_θ(x) = p_SH(x_SH) · p_Comp(x_Comp) · p_DAC(x_DAC) · p_SAR(x_SAR)
```

Each iteration: draw `K = 100` designs, score each, refit the four
per-clique tables by **weighted MLE**. Weights are `exp(score / τ)`
with `τ = 1.0` plus a small smoothing prior `α = 0.1`.

The **only** difference between DADO and naive EDA is what gets
weighted:

| algorithm | weight on clique-`c`'s table update |
| --- | --- |
| Naive EDA | scalar `f(x)` — same for all 4 cliques |
| **DADO** | `Q_c(x_c) = Σ_{j ≥ c} C_j(x_j)` — suffix sum of per-block components along the chain JT |

`C_c` is the per-block contribution to the analytical noise budget
(thermal + droop for SH, finite-gain offset for Comparator, quant +
match² for DAC, etc.). DADO's `Q_c` only credits clique `c` with the
part of `f` it can actually influence (its own component plus
descendants); EDA gives every clique the full scalar score.

## 3. Why optimizing analytically is fast

Each per-block noise term is a closed-form formula (`kT/C`,
`vref / k`, etc.), so one design evaluation is **a handful of
multiplies and additions** — about a microsecond. K × n_iters × seeds
× 2 algorithms ≈ 192 000 evaluations finish in **0.6 seconds**, vs
~20 minutes for the same workload through ngspice (~0.7 s/eval).

## 4. Surrogate-then-verify: how the analytical phase enables SPICE to run faster

Wrap the analytical scoring in a closure that **records every unique
(design, score) pair seen** during the run (`HashMap<Design, f64>`).
After phase A finishes, sort that pool by analytical score and
SPICE-verify only the top `N`. For `N = 50` and ~190 000 unique
designs in the pool, that's a **3800× reduction** in the SPICE-eval
budget.

| pipeline | optimization | SPICE evals | total time | best SPICE found |
| --- | ---: | ---: | ---: | ---: |
| A only (analytical, no SPICE) | 0.6 s | 0 | 0.6 s | — |
| **B only (direct SPICE optimization)** | 0 | 1 800 | **20.2 min** | **`0.00`** (perfect) |
| **Hybrid (A → top-50 SPICE-rerank)** | 0.6 s | 50 | **34 s** | `−0.75` (~95% accurate) |

**36× speedup** for ~5% relative quality loss on this circuit at
`N = 50`. Wider pools (`N = 200`, `N = 500`) cost proportionally
more SPICE evals but stay well below B-direct.

## Pros, cons, and when to use each

**Pros of the hybrid pipeline:**

- **Wall-clock speedup is large and predictable.** You pay (analytical
  optimization, ~ms) + (N × SPICE-eval-time). Tune `N` directly to
  trade quality for time.
- **Drop-in.** No algorithm change. Reuses the same DADO/EDA
  optimization machinery; only the score wrapper that collects
  candidates is new (~10 LOC).
- **Composes with any surrogate**, not just analytical noise budgets.
  A small SPICE-trained MLP, a corner-extracted ROM, an LTSpice
  pre-pass — anything cheap enough to evaluate in bulk.
- **Catches surrogate failures quickly.** If the top-`N` analytical
  designs all SPICE-fail, you learn that immediately at ~0.7 s/eval
  rather than discovering it after 20 minutes of direct SPICE.

**Cons / where it can mislead you:**

- **Quality loss is real.** On this circuit at `N = 50`, hybrid
  picked SPICE = `−0.75` while B-direct found SPICE = `0.00`. The
  analytical model's top tier didn't include the SPICE-perfect basin.
  You're trading verification for speed.
- **Hybrid is bounded by the surrogate's recall, not its precision.**
  If the analytical top-`N` doesn't *contain* a SPICE-good design,
  no amount of SPICE-reranking inside the pool can recover. This is
  why we recommend SPICE as the final-sign-off step regardless.
- **Per-clique decomposition didn't help.** Both DADO and naive EDA
  found the same analytical optimum — at `K = 100` samples × 80
  iterations, the per-clique signal isn't decisive at this
  problem scale. If you want DADO's algorithmic advantage to *also*
  show up, you need a wider problem (more cliques) or a sample-
  starved budget.

## Precisions and checks built into the experiment

- **`tests/analytical.rs`** — analytical-model monotonicity (more
  `c_hold` → less thermal noise; lower `dac_match_pct` → less DAC
  noise; sane behaviour on 200 random designs); strong-vs-weak
  handcrafted ordering at fixed vref.
- **`tests/evaluator.rs` (in `spike-dado-r2r`)** — non-ideal MNA
  solver agrees with the closed-form `ideal_vout` to `1e-12` V at
  every code; uniform R-scaling preserves output (topology
  invariant).
- **ngspice cross-validation in `spike-dado-r2r`** — every final
  design is swept against ngspice across all 256 codes; agreement
  is at machine precision (`max |ngspice − analytical| ≈ 5 × 10⁻⁸ V`).
  Confirms the analytical evaluator the optimizer is calling is
  computing the same answer SPICE would.
- **Cross-evaluation table** — every winner from one phase is
  re-scored under the other phase's metric, so you can directly
  read off whether the analytical model is a faithful proxy.
- **DADO/EDA convergence check** — the synthetic decomposable
  benchmark in `spike-dado-r2r` reproduces the paper's Fig. 1c
  (DADO = 0, EDA ≈ −3.75, paired *t* ≈ 17, *p* ≈ 0). Confirms our
  DADO implementation is correct before applying it to real
  circuits.

See [`../crates/spike-dado-sar/docs/STORY.md`](../crates/spike-dado-sar/docs/STORY.md)
for the auto-generated narrative with this run's exact numbers, and
[`../crates/spike-dado-r2r/docs/STORY.md`](../crates/spike-dado-r2r/docs/STORY.md)
for the companion R-2R-level result.

## Design choices: what was picked, what wasn't, what it would buy you to revisit

This wasn't a clean ablation study (we didn't sweep every knob with
the others held fixed), but every choice below was made deliberately,
and most have a documented test or result that supports it. If a
later experimenter wants to dig further, this is the map.

### a. Optimization algorithm — **DADO + naive EDA**

| dimension | chosen | alternative | rationale / observed impact |
| --- | --- | --- | --- |
| inner loop | DADO (per-clique value-function `Q_c = Σ_{j ≥ c} C_j`) | naive EDA (scalar `f(x)` weight everywhere) | Both implemented; same harness. **Result: tied** at this problem scale (`K = 100`, 4 cliques, max table 5⁴ = 625 logits). Per-clique attribution doesn't help when EDA already has enough samples per table. |
| not implemented | — | simulated annealing, CMA-ES, Bayesian opt | DADO's claim is sample-efficiency on decomposable objectives; SA / CMA-ES would be the natural baselines to add if we ever see DADO win and want to claim that win is *because of* decomposition. |
| sanity check | synthetic `Σ Cᵢ(x̂ᵢ)` benchmark in `spike-dado-r2r` | — | DADO converges to optimum, EDA plateaus at `−3.75` (paired *t* ≈ 17, *p* ≈ 0). Confirms our DADO implementation matches Fig. 1c of the paper *before* applying it to circuits. |

### b. Junction-tree topology — **disjoint cliques (one per sub-block)**

| dimension | chosen | alternative | rationale / observed impact |
| --- | --- | --- | --- |
| separators | empty (each variable in exactly one clique) | overlapping cliques (e.g. `dac_r_ohms` shared between DAC + Comparator) | Disjoint mirrors the SAR ADC's actual block-level interface (each sub-block exposes one terminal voltage, not internal state). Empty separators turn the chain JT into independent per-block categoricals — simpler to fit but **may not exercise DADO's strength**: the suffix-sum `Q_c` carries useful info only if cliques have meaningful descendants. With 4 disjoint cliques, the JT has 4 cliques worth of `Q` granularity and 0 of separator-conditional structure. |
| chain JT used in `spike-dado-r2r` | overlapping (size-1 separators on shared spine resistors) | — | Even with proper overlapping JT structure, R-2R sizing didn't show DADO winning either — the issue there was max-INL not decomposing, not the JT shape. |

### c. Search distribution — **tabular categorical**

| dimension | chosen | alternative | rationale / observed impact |
| --- | --- | --- | --- |
| parameterization | one categorical table per clique (`D^\|clique\|` logits, max 625) | small MLP / VAE per clique (the paper's actual choice) | Tabular is simple, has no hyperparameters, and doesn't generalize across separator slices (irrelevant here since separators are empty). NN would give better extrapolation on **larger** problems where data is sparse vs table size. At our scale (max 625-logit table, ~2 400 effective samples per table) tabular is fine. |
| update rule | weighted MLE on softmax weights with smoothing `α = 0.1` | gradient-based fitting, KL-projected updates | Closed-form for tabular case; matches the EDA-family literature directly. |

### d. Objective design and per-clique decomposition — **the binding constraint**

| dimension | chosen | alternative | rationale / observed impact |
| --- | --- | --- | --- |
| reduction over codes (R-2R) | tested **all three**: max-INL, Σ-INL², max-DNL | (one of the above) | Σ-decomposable objectives compose with DADO's suffix-sum value function; `max` reductions don't. Result: max-INL gap = `−6 %` (worse), Σ-INL² gap = `−25 %`, DNL gap = `+2 %`. None statistically significant. **The reduction matters more than the algorithm** in our setting. |
| SAR ADC analytical decomposition | per-block noise² (`kT/C` for SH, finite-gain for Comparator, quant + match² for DAC, digital-noise proxy for SAR) | every-block-couples-everywhere model with shared parameters | Disjoint decomposition is exactly the case DADO wants. Both DADO and EDA still tied — see (a). |
| SAR ADC SPICE decomposition | 50 / 50 between Comparator and DAC cliques (SH + SAR cliques get 0) | per-bit attribution of digital-code error | Static-input MSE doesn't naturally decompose by sub-block. The 50/50 split was a documented heuristic. A *better* decomposition (per-bit DNL-style, dynamic-input ENOB) would expose more structure for DADO to exploit. |

### e. Hyperparameters — **DADO-friendly defaults from a sweep**

| dimension | chosen | alternative | rationale / observed impact |
| --- | --- | --- | --- |
| `K` (samples / iter) | 100 (analytical), 20 (SPICE) | 32, 64, 200 | Picked from `crates/spike-dado-r2r/examples/sweep.rs` which grid-searched `K ∈ {64, 100, 200}` × `α ∈ {0.5, 0.1, 0.01}` × `τ ∈ {1.0, 0.5, 2.0}` over the synthetic objective. K = 100 was the smallest that gave EDA a fair shot. |
| `α` (Dirichlet smoothing) | 0.1 | 0.01, 0.5 | `α = 0.5` smoothed the data away; `α = 0.01` collapsed early. 0.1 was the sweet spot. |
| `τ` (Boltzmann temperature) | 1.0 | 0.5, 2.0 | 0.5 was slightly sharper; 2.0 too uniform. 1.0 robust. |
| `n_iters` | 80 (analytical), 15 (SPICE) | — | 80 covers the convergence plateau in spike-dado-r2r's trajectory plot; SPICE's smaller budget chosen for time. |
| seeds | 12 (analytical), 3 (SPICE) | more | 12 is enough to hit *p* < 0.05 on a clear effect; 3 is the floor for any t-test. SPICE seeds are wall-clock-bound. |

### f. Evaluator and the hybrid pool

| dimension | chosen | alternative | rationale / observed impact |
| --- | --- | --- | --- |
| analytical model fidelity | textbook closed-form noise budget (kT/C, droop, finite-gain offset, quant + match²) | SPICE-trained MLP surrogate, ROM from corner sweeps | Closed-form is interpretable and free; calibration would tighten the analytical-vs-SPICE correlation but adds complexity. |
| ADC bit width | 4 | 6, 8, 10 (`SarAdc<N>` is const-generic) | 4 keeps each transient ≤ 6 µs simulated. 8 would 4× the SPICE time per design — pushes B-direct from 20 min to over an hour without changing the scientific conclusion. |
| `vref` in SPICE | pinned to **`vdd = 1.8 V`** | per-design from catalog | Catalog includes `0.6 V`, `0.9 V`, `1.0 V` which break SAR logic (CMOS Vt = 0.5 V → no margin). Pinning avoids systematic SPICE failures; `vref` becomes analytical-only (along with `dac_match_pct`). |
| `n_vins` per SPICE design | 4 | 1, 16 | At 1, the score has no signal (one bit of resolution per eval). At 16, you 4× the SPICE time. 4 was the smallest that gave coherent INL coverage. |
| ngspice backend | host (default) + Docker (pinned image) | only host, only Docker | Both supported via `eda-extern-ngspice`'s `Invoker` trait, switched by `NGSPICE_BACKEND=docker`. Docker adds ~250 ms / call but pins the version. |
| **hybrid pool size `N`** | 50 | 100, 200, 500, 1000 | Each extra finalist costs ~0.7 s SPICE, so even N = 1000 is ~12 min vs B-direct's 20 min. **At N = 50 we hit SPICE = `−0.75` (95 % accurate) — the SPICE-perfect basin is reachable from analytical-top-200+ likely, but we haven't run it.** This is the open follow-up. |

### g. What an honest ablation would change

For each row above, the corresponding "alternative" column points at a
single-knob experiment that's worth running if you want a rigorous
ablation. The infrastructure to do it is mostly in place — `K`,
`alpha`, `tau`, `n_iters`, `n_seeds`, hybrid pool size, and the
evaluator are all driver-level constants in
`crates/spike-dado-sar/src/main.rs`. The non-trivial ones to add are:

- **Replace tabular distribution with a small NN.** Touches
  `crates/spike-dado-sar/src/lib.rs:CliqueDist`. ~150 LOC.
- **Overlapping junction tree.** Mostly affects the catalog (re-grouping
  variables) and the per-clique component computation. ~80 LOC.
- **Better SPICE decomposition.** Per-bit DNL attribution, like
  `score_dnl` in `spike-dado-r2r`. ~30 LOC adapt.
- **Wider hybrid pool sweep.** Loop the `N_HYBRID` constant. ~20 LOC.

That's a half-day of work to turn this from a worked example into a
real DADO-vs-circuit-optimization ablation paper. Open to a contributor
who wants to do it.
