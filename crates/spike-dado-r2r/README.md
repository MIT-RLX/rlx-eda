# spike-dado-r2r

[DADO][dado] (Decomposition-Aware Distributional Optimization, ICLR
2026) applied to discrete sizing of the 8-bit R-2R DAC from
`spike-dac-r2r`. Each of the 16 resistors picks one of 5 deviations
(`{−5%, −2.5%, 0%, +2.5%, +5%}`) from nominal; DADO and a naive EDA
baseline compete on three real-circuit objectives plus a synthetic
sanity benchmark.

[dado]: https://arxiv.org/abs/2511.03032

## Run

```sh
./scripts/dado r2r        # progress bars via indicatif
just dado r2r             # same, via Justfile
cargo run --release -p spike-dado-r2r   # raw cargo
just run-dado-r2r         # raw cargo via Justfile
```

End-to-end run takes ~30 seconds. Artifacts land in `docs/` (see
[`docs/STORY.md`](docs/STORY.md) for the narrative built from this
exact run's numbers).

## What the run produces

* **Trajectories.** `00_trajectory_synth.png`, `00_trajectory_inl.png`
  — DADO vs EDA best/mean over iterations, mean across 12 seeds.
* **Per-snapshot packages** for both DADO and EDA at iters
  `{0, 9, 24, 49, 79}`: marginals (per-resistor probability tracks
  split into spine / feeder), schematic SVG with annotated values,
  real GDS layout (open in KLayout), analytical staircase + INL.
* **Final designs.** Annotated schematics + GDS layouts for each
  algorithm's best-of-run design.
* **ngspice cross-validation.** 256-code sweep of each final design
  against the in-house analytical 8×8 MNA solver. Agreement is at
  machine precision (`max |ngspice − analytical| ≈ 5e-8 V`).

## Result summary

| objective | DADO mean | EDA mean | gap | *p* |
| --- | ---: | ---: | ---: | ---: |
| synthetic decomposable (sanity) | `0.000` | `−3.75` | huge | `≈ 0` |
| max-INL (V) | `−9.2e-4` | `−8.7e-4` | `−6%` | `0.66` |
| Σ-INL² (V²) | `−4.4e-5` | `−3.5e-5` | `−25%` | `0.17` |
| max-DNL (V) | `−1.03e-3` | `−1.05e-3` | `+2%` | `0.90` |

Algorithm verifies on the synthetic case but **doesn't transfer to any
of the natural single-block R-2R objectives**. The R-2R network is
genuinely fully coupled (every output code involves all 16 resistors),
so no per-bit decomposition gives DADO a real lever. See
[`docs/STORY.md`](docs/STORY.md) for the frame-by-frame walk-through
and discussion.

## Companion

[`spike-dado-sar`](../spike-dado-sar/README.md) tests DADO at the SAR
ADC *system* level — discrete catalog choice for each sub-block —
which is the granularity where the algorithm should plausibly help.
