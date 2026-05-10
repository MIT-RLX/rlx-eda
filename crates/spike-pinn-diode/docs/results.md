# spike-pinn-diode protocol results

device: MLX  | n_train=12000 | n_test=4000 | n_ood=4000 | n_steps=20000 | K_SEEDS=10

## Baselines

| name | test max-abs (V) | test % FS | OOD max-abs (V) | OOD % FS | test time (ms) | params |
|---|---|---|---|---|---|---|
| M-coarse | 0.0037 | 0.371 | 0.0029 | 0.289 | 4 | 0 |
| M-default | 0.0005 | 0.049 | 0.0004 | 0.039 | 49 | 0 |
| M-fine | 0.0002 | 0.016 | 0.0001 | 0.013 | 497 | 0 |
| Poly-d4 | 0.0000 | 0.000 | 0.0000 | 0.000 | 71 | 126 |

## PINN ablations (K seeds)

| row | in-dist max-abs (V, μ±σ, 95% CI) | OOD max-abs (V, μ±σ, 95% CI) | mean train (ms) | mean infer (µs) |
|---|---|---|---|---|
| A | 0.0874 ± 0.0306 V (95% CI [0.0700, 0.1060]) | 0.1022 ± 0.0319 V (95% CI [0.0833, 0.1199]) | 25288 | 678 |
| B | 0.0158 ± 0.0023 V (95% CI [0.0144, 0.0171]) | 0.0486 ± 0.0136 V (95% CI [0.0406, 0.0576]) | 25612 | 563 |
| H | 0.0649 ± 0.0223 V (95% CI [0.0523, 0.0796]) | 0.1534 ± 0.0787 V (95% CI [0.1139, 0.2107]) | 22581 | 530 |

## Pairwise (Wilcoxon + Cliff's δ + Holm-Bonferroni @ α=0.05)

| comparison | p-value | Holm threshold | reject? | δ | δ magnitude |
|---|---|---|---|---|---|
| Hybrid vs M-coarse | 1.953e-3 | 8.333e-3 | ✓ | +1.000 | large |
| Hybrid vs M-default | 1.953e-3 | 1.000e-2 | ✓ | +1.000 | large |
| Hybrid vs M-fine | 1.953e-3 | 1.250e-2 | ✓ | +1.000 | large |
| Hybrid vs Poly-d4 | 1.953e-3 | 1.667e-2 | ✓ | +1.000 | large |
| Hybrid vs Pure-Surrogate (OOD) | 1.953e-3 | 2.500e-2 | ✓ | +1.000 | large |
| Hybrid vs Pure-PINN (OOD) | 8.398e-2 | 5.000e-2 | · | +0.480 | large |

## Acceptance criteria (§12)

- **C1 [FAIL]** dominated by all baselines
- **C2 [FAIL]** OOD ratio 2.36× (≤ 2)
- **C3 [FAIL]** hybrid OOD μ=0.1534 V, surrogate OOD μ=0.0486 V, Δ=-0.1048 V, σ-threshold=0.0787 V
- **C4 [FAIL]** hybrid 0.0649 V vs polynomial 0.0000 V (PINN 8769 params, poly 126 params, ratio 69.6×)
- **C5 [FAIL]** OOD max-abs/V_REF = 0.153 (< 0.1)

**Verdict:** HYPOTHESIS NOT ACCEPTED
