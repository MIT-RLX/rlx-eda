# Main contributions

The seven contributions below are the substantive results in this
workspace. The README summarises each in two lines; this document
expands each one with the underlying numbers, tables, and reproducer
pointers.

## 1. A working Rust EDA stack with cross-validated solvers

In-house MNA + autodiff matched against ngspice at machine precision
(`max | analytical − ngspice | ≈ 5 × 10⁻⁸ V` over 256 R-2R DAC
codes; see `../crates/spike-dado-r2r/docs/00_final_ngspice_*.txt`). Same
typed-block IR drives both the differentiable inner loop and the
SPICE deck — no double-bookkeeping.

## 2. A pluggable ngspice backend (`eda-extern-ngspice`)

Both a host `LocalBinary` invoker and a `DockerInvoker` against a pinned
image. Both share a private `NgspiceRunner` trait so all parsing
logic is shared; only the subprocess invocation differs. Picks
automatically via `NGSPICE_BACKEND=docker`.

## 3. A surrogate-then-verify hybrid pipeline (`spike-dado-sar`)

For discrete circuit-design optimization. Optimize on a cheap
closed-form analytical noise budget (~µs/eval), record every unique
design seen, SPICE-verify only the top *N* by analytical score.
**Empirical result on a 4-bit SAR ADC across a `5¹² ≈ 2.4 × 10⁸`
design space: 36× faster than direct SPICE optimization (34 s
vs 20.2 min) at ~5% relative SPICE-quality loss at *N* = 50.** The
pipeline is drop-in (~10 LOC of score-wrapper) and composes with
any cheap surrogate, not just analytical noise budgets. See
[`dado-sar-worked-example.md`](dado-sar-worked-example.md).

## 4. An honest negative result on per-block decomposition (DADO)

Across two crates, four objectives, and ~25 distinct experiments,
no statistically significant DADO-vs-naive-EDA win on any real
circuit metric — only on a synthetic Σ-decomposable benchmark
(where the algorithm reproduces the paper's Fig. 1c at *p* ≈ 0).
Documented with cross-evaluation tables and reproducible
`STORY.md` narratives so future work can pick up exactly where we
stopped.

## 5. GPU-accelerated Monte Carlo via Apple Metal

A custom Metal LU+solve kernel in `rlx-mlx` lowers `Op::DenseSolve` /
`Op::BatchedDenseSolve` to the Apple GPU; `Op::Scan` lowering +
`vmap` lift any analysis (DC, BE-Newton transient, scan-folded
transient, AC) to one `MlxExecutable` dispatch with the per-draw
axis (or per-frequency axis) batched on the GPU. Measured against
ngspice on the same circuits + same M-series hardware:

| Analysis | Workload | N | eda-mna | ngspice | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| **DC MC (per-draw R)** | linear divider | 256 | 0.7 ms | 2902 ms | **4034×** |
| **BE-Newton transient** | RC discharge per-draw IC | 256 | 92 ms | 3029 ms | 33× |
| **Scan-folded transient** | same RC discharge | 1024 | 3.6 ms | 11953 ms | **3315×** |
| **AC sweep** (one-process ngspice `.ac dec`) | RC low-pass | 4096 | 0.15 ms | 14.3 ms | 95× |

Per-draw drift vs ngspice is sub-µV / sub-µA across all four
analyses (f32 noise floor). Architecture, builder API surface,
and the `examples/ngspice_cross_bench.rs` reproducer documented
in [`gpu-monte-carlo.md`](gpu-monte-carlo.md).

## 6. Differentiable place-and-route on the GPU (`eda-pnr`)

Place-and-route as a first-class layer with positions exposed as
`rlx_ir::Param[B, N]` tensors so half-perimeter wirelength + a
smooth bbox-overlap density penalty form a single differentiable
loss across `B` parallel placements. The same Adam/cosine-LR/
β-anneal recipe used for the LNA's `Lg` and the MZI's `n_eff_A`
drives the placer; the parallel-batch dimension turns the
`[B, N, N]` density operator into a real GPU workload that
amortizes Apple-GPU launch overhead. Measured on an M-series
host at `N = 64` instances, 32 nets, 300 Adam steps:

| B | Cpu wall | Mlx wall | Speedup | Best-of-B loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.06 s | 2.43 s | 0.025× | — |
| 16 | 0.97 s | 2.14 s | 0.45× | 6.99 e6 |
| 64 | 3.99 s | 2.73 s | **1.46×** | 7.21 e6 |
| 128 | 7.74 s | 3.60 s | **2.15×** | 6.63 e6 |
| 256 | 15.17 s | 5.60 s | **2.71×** | 6.55 e6 |

Same numerical answer on both backends (`4.9596e8 = 4.9596e8`
at `B = 64`), only wall time changes. Crossover at `B ≈ 30`;
per-placement MLX cost drops 110× from `B = 1` to `B = 256`
(2.43 s → 22 ms). Full charts + CSV +
methodology in [`../crates/eda-pnr/docs/gpu_scaling.md`](../crates/eda-pnr/docs/gpu_scaling.md);
overview in [`eda-pnr.md`](eda-pnr.md).

## 7. Pre-registered PINN/surrogate experiment series

Four runs, three falsifications, one partial pass. Across four
pre-registered K=10 protocols (linear RC, nonlinear diode-RC,
ideal 1-D 8-bit SAR, 10-D SAR with mismatch), the central PINN
claim — that a physics-informed neural network beats classical
regression baselines as a circuit-block surrogate — was
falsified on three problems and partially supported on the
fourth. Polynomial regression at modest degree dominates PINN
at input dimensionality `d ≤ 5`, including on the discrete
256-step SAR staircase where the prior intuition predicted
PINN would naturally win. **At `d = 10` PINN beats Poly-d4 by
36% on max-abs error (Wilcoxon p=2e-3, Cliff's δ = −1.0
large)** — the first positive PINN finding — but neither
method clears the absolute sub-LSB threshold needed to ship as
an actual SAR surrogate. The methodology infrastructure
(pre-registration parity test enforcing markdown ↔ `pub const`
consistency, ablation grid, paired Wilcoxon + Cliff's δ +
Holm-Bonferroni statistical pipeline) is reusable for further
protocols. Full write-up + charts + CSVs at
[`pinn-experiments.md`](pinn-experiments.md);
per-crate pre-registrations and result tables under
[`../crates/spike-pinn-{diode,sar,sar-mc}/`](../crates/).
<a id="8-hybrid-2-axis-batch-per-chip-newton-and-ad-driven-sizing-t11d-t11e-t11g"></a>
## 8. Hybrid 2-axis batch + per-chip α Newton + AD-driven sizing (T.11.D / T.11.E / T.11.G)

One `transient_pwl_batched` call sweeps `B = N_VIN × N_DRAWS`
chips through both axes — characterization (per-chip vin via the
boundary closure) **and** Monte Carlo (per-chip device parameters
via `mc_params`) — in a single MLX dispatch. Bit decisions stay
*inside* the circuit: each chip's transistor-level SAR register
makes its own latching decisions on the shared capture clock, so
the batched transient covers all `N_BITS` decisions for every
chip with no external trial loop.

Four load-bearing fixes landed under this contribution; each is
reproducible from the current code via env-var gates so the
solver-version sweep below can be replayed verbatim:

- **Per-chip α backtracking** in
  `eda_mna::batched_solve_be_step_with_ctx` replaces the prior
  shared-α version. Under shared α a single stiff chip in the
  batch could halve α for everyone and stall the entire Newton
  step; under per-chip α each chip's line search proceeds
  independently. Gated by `RLX_BATCHED_PER_CHIP_ALPHA={0,1}`
  (default `1`).
- **`RLX_MLX_MODE=compiled`** wires `mlx::compile` for persistent
  trace fusion. Default `Lazy` mode pays per-op kernel-launch
  overhead and was 11× slower than CPU at 256 chips on the
  comparator demo; `Compiled` mode reaches CPU parity at small
  batches. (Honest measurement: `Compiled` mode also doesn't
  *beat* CPU at the batch sizes we measured — at B=4096 it runs
  18.7 s vs CPU 7.5 s. The fix moves MLX from "broken default"
  to "competitive option," not to "always-best.")
- **Duplicate-output bug fix** in `rlx_mlx::lower_with_env` —
  output collection used `env.remove`, which broke any vmap'd
  graph reusing the same NodeId across multiple output slots
  (common when a tangent feeds two outputs). Switched to
  `env.get + Array::clone_handle()` (Arc-like clone, no data
  copy) so duplicate output slots all hash to the same env entry.
- **Adaptive sub-step driver** in
  `eda_mna::transient_pwl_batched`, gated by
  `RLX_BATCHED_ADAPTIVE_DT=1`. When too few chips converge for the
  full `dt` step, the driver retries the same interval as 2 / 4 /
  8 sub-steps; helps the SAR ADC's bistable DffSR cross-couple
  flip during phase-transition steps.

### T.11.E — clean correctness + perf demo (`comparator_vin_sweep_mc`)

Standalone 9-T comparator, **B = 256 chips** in **0.5 s on CPU**;
same 0.5 s with `RLX_MLX_MODE=compiled` on Apple Metal. Per-draw
input-referred offset σ = **7.06 mV** under 5 mV-per-side Pelgrom
mismatch — within 0.05 mV of the analytic √2·σ_Vth = 7.07 mV.
This is the cleanest validation that the hybrid 2-axis batch
architecture is numerically sound on a non-bistable circuit.

### T.11.D — full transistor-level SAR ADC (`sar_adc_mc_sweep`)

Same hybrid 2-axis batch on the full SAR ADC (S/H + R-2R DAC +
9-T comparator + 4 × DffSR, ~241 transistors). Real measurements
from the env-gate-driven solver-version sweep over the wider vin
grid `[0.54, 1.53] V`, B = 64, N_DRAWS = 8:

| ver | per-chip α | adaptive dt | phase pulse | match rate | σ (LSB) | wall (s) |
| --- | :---: | :---: | :---: | ---: | ---: | ---: |
| v0 — shared α | shared | off | 0.50 | 14% | 0.38 ⚠ | 210.2 |
| v1 — per-chip α | per-chip | off | 0.50 | 12% | 0.67 | 205.7 |
| v2 — wider phase | per-chip | off | 0.70 | 12% | 1.85 | 207.9 |
| v3 — + adaptive dt | per-chip | on | 0.70 | 12% | 1.55 | 423.9 |
| **scalar baseline** | (n=1) | n/a | 0.70 | **100%** | n/a | 22 |

⚠ v0's σ = 0.38 LSB is *coordinated failure* — every chip
converges to the same wrong code, so cross-draw σ trivially
shrinks. Not honest Pelgrom variance. v1+ shows real
per-mismatch-realization spread.

The remaining gap to the scalar baseline's 100% match rate is a
**circuit-level** issue, not a solver issue: the DffSR's
cross-coupled SR latch is bistable, and vmap'd numerics can push
the latch over its tipping point during set_b release. Documented
honestly in [`../crates/spike-sar-adc/docs/sar_adc_mc_sweep.md`](../crates/spike-sar-adc/docs/sar_adc_mc_sweep.md).

### T.11.G — gradient-driven comparator sizing (`comparator_sizing_opt_ad`)

The same hybrid-batch infrastructure is the inner loop of a real
circuit-design optimization. **Loss = (σ_offset(W) − target)²**
where W is the matched-pair M1/M2 channel width; gradient via
central finite-difference on the batched MC; outer loop is gradient
descent. Implements a **DADO 4-stage cascade**:

1. **Stage 1 — cheap surrogate** (N_DRAWS = 8 from W = 2 µm). Fast
   inner MC, ~10 outer gradient iters at ~1 s/iter.
2. **Stage 2 — verify** (N_DRAWS = 64 at the surrogate's W).
   Reveals σ_v1 = 7.72 mV (the Stage 1 surrogate had over-fit
   N_DRAWS=8 noise to 4.0 mV).
3. **Stage 3 — re-targeted surrogate** (N_DRAWS = 32, internal
   target shifted by the verify-stage bias so gradient descent
   pushes W in the right direction even though the surrogate's
   absolute σ number is biased).
4. **Stage 4 — final verify** (N_DRAWS = 64 at the cascade's
   converged W). Reports σ_v2 = 6.10 mV, **closing 44 % of the
   residual gap to the 4 mV target** in 34.9 s end-to-end.

This is contribution #4's "honest negative result" applied to a
continuous design objective: a cheap surrogate gives a biased
gradient signal; the cascade quantifies the bias and trades a few
more verify calls to recover. The σ-vs-W chart's two distinct
trajectories (Stage 1 red-×, Stage 3 magenta-×) sit visibly off
the verify-stage Pelgrom 1/√W curve in different ways — Stage 1
diverges into a noise pocket, Stage 3 climbs back onto the curve.

Generated artifacts (all rendered against Sky130 layers in the same
PDK as the rest of the workspace): the SAR ADC top-level floor
plan, the four supporting charts (per-step Newton convergence,
version-comparison bars, MLX-dispatch scaling, comparator transfer
under mismatch), the cascade's loss curve and σ-vs-W trajectory,
and three AD-optimized M1 footprints at W = 2 µm / 9.32 µm / 25 µm
showing the diffusion-height-scales-linearly-with-W relationship.

Full write-ups, runnable bins, and per-stage traces:
[`../crates/spike-sar-adc/docs/sar_adc_mc_sweep.md`](../crates/spike-sar-adc/docs/sar_adc_mc_sweep.md),
[`../crates/spike-divider-block/docs/comparator_sizing_opt_ad.md`](../crates/spike-divider-block/docs/comparator_sizing_opt_ad.md),
and the bins
[`comparator_vin_sweep_mc.rs`](../crates/spike-divider-block/src/bin/comparator_vin_sweep_mc.rs)
·
[`sar_adc_mc_sweep.rs`](../crates/spike-sar-adc/src/bin/sar_adc_mc_sweep.rs)
·
[`comparator_sizing_opt_ad.rs`](../crates/spike-divider-block/src/bin/comparator_sizing_opt_ad.rs).

