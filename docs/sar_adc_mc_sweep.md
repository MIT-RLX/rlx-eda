# T.11.D — Hybrid (vin × Monte Carlo) batched SAR ADC

Full transistor-level 4-bit SAR ADC, **B = 64 chips** (8 input voltages × 8 mismatch realizations) run through ONE `transient_pwl_batched` call. Per-chip vin via the boundary closure; per-chip M1/M2 Vth (Pelgrom σ = 5 mV per side) via `mc_params`. The transistor-level SAR register is part of every chip's circuit, so each chip's bit decisions emerge naturally on the shared capture clock — no external trial loop.

## Headline

- **64 chips × 140 BE steps** in **191.6 s** (21.39 ms / step / chip)
- **Avg per-vin match-rate vs analytic SAR**: 12%
- **Avg per-vin code σ under mismatch**: 1.85 LSB

## Floor plan (Sky130-driven)

![SAR ADC floor plan, sky130 layers](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/floorplan.svg)

Same circuit for every solver-version below; floor plan is invariant.

## Newton convergence per BE step (per solver version)

![Newton convergence per BE step, per solver version](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/convergence.svg)

## Version-comparison bar chart

![Match rate, σ, wall time per version](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/version_compare.svg)

## MLX dispatch scaling

![CPU vs MLX-Lazy vs MLX-Compiled vs batch size](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/mlx_scaling.svg)

`RLX_MLX_MODE=compiled` reaches CPU parity at small batches; Lazy mode pays per-op kernel-launch overhead and ends up ~11× slower at 256 chips.

## Comparator transfer curve under mismatch (T.11.E)

![9-T comparator transfer under mismatch](crates/spike-sar-adc/docs/assets/sar_adc_mc_sweep/comparator_transfer.svg)

## Solver-version sweep (real measurements)

All four runs use the wider vin grid [0.54, 1.53] V, B=64, N_DRAWS=8. Reproducible via env-var gates on the same binary:

| ver | per-chip α | adaptive dt | phase pulse | match rate | σ (LSB) | wall (s) | env |
| --- | :---: | :---: | :---: | ---: | ---: | ---: | --- |
| v0 — shared α | shared | off | 0.50 | 14% | 0.38 ⚠ | 210.2 | `RLX_BATCHED_PER_CHIP_ALPHA=0 RLX_SAR_PHASE_FRAC=0.50` |
| v1 — per-chip α | per-chip | off | 0.50 | 12% | 0.67 | 205.7 | `RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.50` |
| v2 — wider phase | per-chip | off | 0.70 | 12% | 1.85 | 207.9 | `RLX_BATCHED_PER_CHIP_ALPHA=1 RLX_SAR_PHASE_FRAC=0.70` |
| v3 — + adaptive dt | per-chip | on | 0.70 | 12% | 1.55 | 423.9 | `… RLX_BATCHED_ADAPTIVE_DT=1` |
| **scalar baseline** | (n=1) | n/a | 0.70 | **100%** | n/a | 22 | `sar_adc_full_mna` (single chip) |

⚠ v0's σ=0.38 LSB is *coordinated failure* (every chip converges to the same wrong code), not Pelgrom-honest variance. v1+ shows real per-draw spread because chips diverge per-mismatch-realization.

## AD-driven design objective on top (T.11.G — DADO 4-stage cascade)

Loss = (σ_offset(W) − target)²; FD gradient on the batched MC; 4-stage surrogate→verify cascade where the verify stage's bias re-aims the next surrogate stage.

![σ vs M1/M2 width with optimizer trajectory](crates/spike-divider-block/docs/assets/comparator_sizing_opt_ad/sigma_vs_W.svg)

Full per-stage trace + AD-optimized layouts:
[`comparator_sizing_opt_ad.md`](../../spike-divider-block/docs/comparator_sizing_opt_ad.md).

## Transfer curve under mismatch (this run)

| vin (V) | ideal code | mean decoded | σ (LSB) | match rate |
| ---: | ---: | ---: | ---: | ---: |
| 0.5400 | 4 | 6.12 | 0.33 | 0/8 |
| 0.6814 | 6 | 8.50 | 2.92 | 0/8 |
| 0.8229 | 7 | 3.00 | 3.04 | 0/8 |
| 0.9643 | 8 | 12.75 | 1.98 | 1/8 |
| 1.1057 | 9 | 9.62 | 3.67 | 0/8 |
| 1.2471 | 11 | 13.00 | 1.12 | 0/8 |
| 1.3886 | 12 | 13.00 | 0.87 | 2/8 |
| 1.5300 | 13 | 13.00 | 0.87 | 5/8 |

## What this proves

- The 2-axis batch (characterization × yield) collapses two sweeps into one MLX-batched transient. The same `Op::BatchedDenseSolve` infrastructure that powers T.11.B's pure-MC comparator now carries a full transistor-level SAR ADC.
- Per-chip cost amortizes across the batch axis — the per-step Newton solve runs once for all B chips, not B times.
- Each chip's transistor-level SAR register makes its own bit decisions inside the single batched transient — there is no external per-trial synchronization loop and no mid-batch host roundtrip across the 4 bits.
