# spike-dado-sar

[DADO][dado] (Decomposition-Aware Distributional Optimization, ICLR
2026) applied at the **SAR ADC system level** — discrete catalog
choice for each sub-block (sample-hold, comparator, DAC, SAR logic).
Three evaluators competing on the same 12-variable design space:

* **A — Analytical noise budget.** Closed-form `Σ_block noise²`
  (kT/C, droop, finite-gain offset, quant + match²). Σ-decomposable
  by construction — designed to be the case where DADO's per-clique
  value-function update *should* shine.
* **B — ngspice transient.** Drives the actual `SarAdc<4>` from
  `spike-sar-adc` at 4 representative `vin` levels per design,
  scores by mean squared digital-code error.
* **H — Hybrid surrogate-then-verify.** Run A to optimize over the
  cheap analytical model, take the top-50 candidates by analytical
  score, SPICE-rerank them, return the SPICE-best. Tests whether
  the analytical model is a good *filter* even if it's a bad
  *ranker*.

The four DADO/EDA winners across A and B are cross-evaluated under
both metrics, plus the hybrid finalist is added — answering "does
DADO transfer to ADC system-level?", "is the analytical model a
faithful enough proxy?", and "would surrogate-then-verify save us
time?"

[dado]: https://arxiv.org/abs/2511.03032

## Run

The `dado` wrapper script handles ngspice / Docker selection and shows
live progress bars (via [`indicatif`](https://docs.rs/indicatif)):

```sh
./scripts/dado sar              # host ngspice (~25 min)
./scripts/dado sar --docker     # Docker ngspice, auto-builds image first run
just dado sar                   # same, via Justfile
```

Direct invocations also work:

```sh
cargo run --release -p spike-dado-sar
NGSPICE_BACKEND=docker cargo run --release -p spike-dado-sar
just run-dado-sar
just run-dado-sar-docker
```

A finishes in seconds; B is ngspice-bound (~0.7 s / design eval; total
~20 minutes at the default budget of K=20, n_iters=15, 3 seeds);
H adds ~35 s on top of A. Artifacts and the data-driven
[`docs/STORY.md`](docs/STORY.md) narrative write to `docs/`.

## Result summary

| design | analytical (V²) | SPICE (mean code² err) | wall clock |
| --- | ---: | ---: | ---: |
| A-DADO  | `−1.18 × 10⁻⁴` | `−0.75`     | 0.6 s   |
| A-EDA   | `−1.18 × 10⁻⁴` | `−4.00`     | 0.6 s   |
| B-DADO  | `−9.15 × 10⁻²` | **`0.00`**  | 20.2 min |
| B-EDA   | `−2.76 × 10⁻³` | **`0.00`**  | 20.2 min |
| **Hybrid** (A → top-50 SPICE rerank) | `−1.18 × 10⁻⁴` | `−0.75` | **34 s** |

**Three findings from one run:**

1. **DADO didn't beat EDA at either evaluation level** (paired *p* = 0.18 on A, *p* = 1.0 on B). With K=100 samples and 4 disjoint cliques of size 2-4, naive EDA's per-clique tabular MLE already has plenty of data — DADO's per-clique attribution doesn't add discriminative signal at this problem scale.

2. **The analytical model is a coarse SPICE proxy.** A-DADO and A-EDA tie analytically (`-1.18e-4`) but score 5× differently in SPICE (`-0.75` vs `-4.00`). Same in reverse: B-DADO and B-EDA both hit SPICE = 0 but differ analytically by a factor of 30. Designs that look identical analytically can have very different SPICE behaviour and vice-versa.

3. **Hybrid is 36× faster but lossy.** Top-50 analytical candidates → SPICE-rerank lands at SPICE = `-0.75` in 34 s; direct SPICE finds `0.00` in 20 min. Same answer as A-DADO — the top-50 analytical pool didn't contain a SPICE-perfect design. For early DSE this is a clear win (~5% relative quality loss for 36× speedup); for final sign-off, run direct SPICE.

See [`docs/STORY.md`](docs/STORY.md) for the data-driven narrative
(setup table, trajectory plots, head-to-head + wall-clock tables, and
recovery options like wider hybrid pools or a SPICE-calibrated
surrogate).

## Design choices and what they'd cost to revisit

Quick ablation map — every choice below was deliberate, but most
weren't grid-searched. The workspace
[`README.md`](../../README.md#design-choices-what-was-picked-what-wasnt-what-it-would-buy-you-to-revisit)
has the full table; this is the abbreviated version for reference
inside the crate.

| dimension | chosen here | alternative | what changing it would test |
| --- | --- | --- | --- |
| **algorithm** | DADO + naive EDA (both run, compared) | SA, CMA-ES, BO | whether *any* discrete optimizer beats DADO on this granularity |
| **junction tree** | disjoint cliques (one per sub-block) | overlapping cliques sharing terminals | whether DADO's per-clique signal needs separator structure to bite |
| **search distribution** | tabular categorical, `D^\|clique\| ≤ 625` logits | small per-clique NN | matters when problem grows past tabular budget |
| **A objective** | Σ-block noise² (textbook ADC budget) | SPICE-calibrated MLP | tightens analytical↔SPICE correlation, helps hybrid recall |
| **B SPICE decomposition** | 50 / 50 split between Comparator + DAC; SH and SAR get 0 | per-bit DNL-style attribution | gives DADO a real per-clique signal under SPICE |
| **K, n_iters, τ, α** | 100 / 80 / 1.0 / 0.1 (A); 20 / 15 / 1.0 / 0.1 (B) | grid search around these | swept once in `crates/spike-dado-r2r/examples/sweep.rs` |
| **seeds** | 12 (A), 3 (B) | more | reduces variance; 3 is the t-test floor for B |
| **ADC bit width** | `SarAdc<4>` | 6, 8, 10 (const-generic) | 8 = 4× B time, doesn't change conclusions |
| **`n_vins` per SPICE** | 4 | 1, 8, 16 | trades coverage for time per design |
| **`vref` in SPICE** | pinned to `vdd = 1.8 V` | per-design from catalog | low-vref designs break SAR-logic margin, so `vref` is analytical-only |
| **hybrid pool `N`** | 50 | 100, 200, 500, 1000 | **the open follow-up** — wider pool likely catches the SPICE-perfect basin we miss at N = 50 |

The non-trivial knobs to turn (the ones that aren't just driver
constants) are in **bold** in the workspace README. Each is a ~30–150
LOC change.

## Design space

12 categorical variables (alphabet `D = 5`) over 4 disjoint cliques:

| clique | variables |
| --- | --- |
| Sample-Hold | `c_hold`, `sh_nmos_w`, `sh_pmos_w`, `sh_l` |
| Comparator | `comp_k`, `comp_voh`, `comp_vol` |
| DAC | `dac_r_ohms`, `dac_match_pct` (analytical-only), `vref` (analytical-only) |
| SAR Logic | `sar_nand_w`, `sar_inv_w` |

Total design space `5¹² ≈ 2.4 × 10⁸`. `dac_match_pct` and `vref` only
affect the analytical model — the SPICE deck pins `vdd = 1.8 V` (CMOS
logic margin) and ignores per-resistor mismatch.

## Architecture

Disjoint cliques → empty separators → per-clique conditionals collapse
to independent unconditional categoricals. DADO's value-function `Q_c`
becomes the suffix sum of per-block components from clique `c` onward;
naive EDA uses the scalar `f(x)` everywhere.

## ngspice backend

Selected via the `NGSPICE_BACKEND` env var:

* unset (default) — host ngspice (uses `LocalBinary::from_env()`).
* `docker` — pinned image `rlx-ngspice:local` built from
  `docker/ngspice/Dockerfile` (centralized under the workspace
  `docker/` tree). Build it once with `just deps-docker-ngspice`
  (or `just deps-docker` for every image); subsequent runs auto-detect
  and reuse it.

Both backends share parsing logic via the private `NgspiceRunner` trait
in `eda-extern-ngspice` — only the subprocess invocation differs.

## Companion

[`spike-dado-r2r`](../spike-dado-r2r/README.md) tests DADO at the
single-block resistor-sizing level on the same family of objectives.
That experiment showed DADO doesn't transfer to fully-coupled
single-block problems; this crate is the natural follow-up at the
sub-block-composition granularity.
