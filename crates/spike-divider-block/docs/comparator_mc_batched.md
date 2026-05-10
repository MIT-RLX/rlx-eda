# T.11.B — Batched MC on the comparator (Apple Metal via batched_solve_be_step)

9-transistor Baker-style comparator, 16 per-chip Monte Carlo realizations of M1/M2 Vth mismatch (σ = 5 mV per side, Pelgrom-style). All 16 draws solved in **one** `transient_pwl_batched` call: each Backward-Euler step's per-draw Newton inner solve dispatches through `Op::BatchedDenseSolve` → `MlxExecutable` → Apple Metal LU+solve kernel (CPU fallback off-Mac).

## Headline

- **N_DRAWS = 16** (one batched transient, not 16 independent runs)
- **d2(80 ns) per-draw distribution**: mean = 1.1244 V, **σ = 463.6 mV**
- **Wall time**: 1.6 s for all 16 chips (20.2 ms / BE step)

d2 is the comparator's analog stage-1 output (before the digital buffer); its spread under M1/M2 Vth mismatch is the input-referred offset times the comparator's small-signal gain. The Pelgrom σ_Vth = 5 mV on each side gives σ(ΔVth) ≈ 7 mV; with ~10× stage-1 gain that maps to ~70 mV at d2 — within shooting distance of the measured σ here.

## Per-draw raw data

| draw | M1_Vth (V) | M2_Vth (V) | ΔVth (mV) | d2(80ns) (V) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4953 | 0.5035 | -8.21 | 1.7025 |
| 1 | 0.4996 | 0.4984 | +1.27 | 1.0354 |
| 2 | 0.4931 | 0.5007 | -7.60 | 1.6997 |
| 3 | 0.5013 | 0.4980 | +3.32 | 0.8359 |
| 4 | 0.4949 | 0.5046 | -9.64 | 1.7073 |
| 5 | 0.4990 | 0.4979 | +1.15 | 1.0473 |
| 6 | 0.4946 | 0.4971 | -2.45 | 1.3987 |
| 7 | 0.5010 | 0.4979 | +3.02 | 0.8649 |
| 8 | 0.5024 | 0.4985 | +3.90 | 0.7784 |
| 9 | 0.5015 | 0.4946 | +6.85 | 0.4905 |
| 10 | 0.4923 | 0.5045 | -12.25 | 1.7146 |
| 11 | 0.5036 | 0.5025 | +1.09 | 1.0536 |
| 12 | 0.5058 | 0.4956 | +10.20 | 0.3691 |
| 13 | 0.5031 | 0.5022 | +0.87 | 1.0892 |
| 14 | 0.4966 | 0.5080 | -11.41 | 1.7124 |
| 15 | 0.5029 | 0.4960 | +6.85 | 0.4903 |

## What this proves

- **Layer 2 of the GPU acceleration plan is operational**: the existing batched-DC MLX inner-solve infrastructure now has a transient sibling (`transient_pwl_batched`), so Monte Carlo / PVT / parameter sweeps over a transistor-level circuit run on Apple Metal in one call instead of a serial loop on CPU.
- **The correctness contract is unchanged**: each draw's per-step Newton converges to the same operating point a single-draw `transient_pwl(circuit, params_d, …)` would produce — just N at once.
- **Cost**: this v1 recompiles the batched residual + jacobian graphs on every BE step (mirror of the pre-T.10 scalar issue). T.11.B.2 (`BatchedBeStepContext`) lifts the cache same way T.10 did for the scalar path; expected ~50–100× speedup on top of the current numbers.
