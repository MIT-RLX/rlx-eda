# spike-pinn-sar-mc protocol results

device: MLX  | n_train=12000 | n_test=4000 | n_steps=20000 | K=10

## Polynomial baselines (10-D)

| name | max-abs | LSB | RMS | fit (ms) | predict (ms) | params | bytes |
|---|---|---|---|---|---|---|---|
| Poly-d1 | 0.08420 | 21.554 | 0.01987 | 2 | 0 | 11 | 44 |
| Poly-d2 | 0.07618 | 19.503 | 0.01278 | 19 | 1 | 66 | 264 |
| Poly-d4 | 0.06622 | 16.953 | 0.00953 | 5234 | 36 | 1001 | 4004 |

## PINN (K seeds)

max-abs (units): 0.04203 ± 0.00434 (95% CI [0.03967, 0.04473])

max-abs (LSB): mean 10.759


## Pairwise (Wilcoxon + Cliff's δ + Holm-Bonferroni @ α=0.05)

| comparison | p-value | Holm threshold | reject? | δ | δ mag |
|---|---|---|---|---|---|
| PINN vs Poly-d1 | 1.953e-3 | 1.667e-2 | ✓ | -1.000 | large |
| PINN vs Poly-d2 | 1.953e-3 | 2.500e-2 | ✓ | -1.000 | large |
| PINN vs Poly-d4 | 1.953e-3 | 5.000e-2 | ✓ | -1.000 | large |

## Acceptance criteria

- **C1'' [PASS]** PINN max-abs μ=0.04203 vs Poly-d4 max-abs=0.06622 (p=1.953e-3 thr 5.000e-2)
- **C2'' [PASS]** Poly-d1=0.08420 ≥ Poly-d2=0.07618 ≥ Poly-d4=0.06622 ≥ PINN=0.04203 ?  ordering ok | PINN<d4 yes
- **C5'' [FAIL]** PINN max-abs μ = 0.04203 (< 1 LSB = 0.00391)

**Verdict:** HYPOTHESIS NOT ACCEPTED
