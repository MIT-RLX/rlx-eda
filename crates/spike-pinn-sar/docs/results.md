# spike-pinn-sar protocol results

device: MLX  | n_train=12000 | n_test=4000 | n_steps=20000 | K_SEEDS=10

## Baselines

| name | max-abs | LSB | RMS | time (ms) | params | bytes |
|---|---|---|---|---|---|---|
| Poly-d4 | 0.00200 | 0.512 | 0.00112 | 0 | 5 | 20 |
| Poly-d8 | 0.00207 | 0.529 | 0.00112 | 0 | 9 | 36 |
| Poly-d16 | 0.00213 | 0.545 | 0.00112 | 1 | 17 | 68 |
| Lookup-16 | 0.00387 | 0.991 | 0.00160 | 0 | 16 | 64 |
| Lookup-64 | 0.00383 | 0.980 | 0.00154 | 0 | 64 | 256 |
| Lookup-256 | 0.00387 | 0.991 | 0.00160 | 0 | 256 | 1024 |

## PINN (K seeds)

max-abs (units): 0.07063 ± 0.01373 (95% CI [0.06212, 0.07795])

max-abs (LSB): mean 18.080


## Pairwise (Wilcoxon + Cliff's δ + Holm-Bonferroni @ α=0.05)

| comparison | p-value | Holm threshold | reject? | δ | δ magnitude |
|---|---|---|---|---|---|
| PINN vs Poly-d4 | 1.953e-3 | 8.333e-3 | ✓ | +1.000 | large |
| PINN vs Poly-d8 | 1.953e-3 | 1.000e-2 | ✓ | +1.000 | large |
| PINN vs Poly-d16 | 1.953e-3 | 1.250e-2 | ✓ | +1.000 | large |
| PINN vs Lookup-16 | 1.953e-3 | 1.667e-2 | ✓ | +1.000 | large |
| PINN vs Lookup-64 | 1.953e-3 | 2.500e-2 | ✓ | +1.000 | large |
| PINN vs Lookup-256 | 1.953e-3 | 5.000e-2 | ✓ | +1.000 | large |

## Acceptance criteria

- **C1' [FAIL]** PINN max-abs μ=0.07063 vs Poly-d16 max-abs=0.00213 (p=1.953e-3 thr 1.250e-2)
- **C2' [FAIL]** PINN max-abs μ=0.07063 vs Lookup-64 max-abs=0.00383 (p=1.953e-3 thr 2.500e-2); PINN 4612 bytes vs Lookup-64 256 bytes
- **C5' [FAIL]** PINN max-abs μ = 0.07063 (< ½ LSB = 0.00195)

**Verdict:** HYPOTHESIS NOT ACCEPTED
