# Differentiable place-and-route (`eda-pnr`)

Place-and-route lifted out of every spike's bespoke `Layout::layout`
and into a shared crate with three first-class concerns:

1. **`Netlist`** — declarative connectivity. Instances reference
   already-laid-out child cells; `Net`s name pin-tuples (with optional
   per-net `weight` for timing-driven HPWL); `ExternalPin`s expose
   selected nets at the top level; `MatchGroup`s declare analog
   matching constraints (differential pairs, common-centroid groups,
   interdigitated arrays). Instances can be marked `fixed = true` so
   I/O pads stay pinned while internal blocks move under AD.

2. **`PnrFlow`** — `Placer + Router → CellId`. Built-ins:
   `ManualPlacer`, `GridPlacer`, `ManhattanRouter` (wrapping
   `klayout-route::ManhattanPlanner`) with `WireStyle::{Path,
   Polygon}` and `MultiPinStrategy::{Star, Steiner}` (the Steiner
   path wraps `klayout-route::rsmt`; saves ~25 % wire area on dense
   ≥3-pin nets).

3. **AD-first placement** — the `ad` module exposes
   `combined_loss_graph_batched` (positions as `Param[N]` tensors,
   density as one `[N, N]` outer-difference + reduce) and
   `combined_loss_graph_parallel` (positions as `Param[B, N]` —
   `B` independent placements as one fused tensor). Loss = HPWL
   (smooth-max via log-sum-exp) + α · density (smooth swish-relu of
   pairwise bbox overlap). Hand to `rlx_opt::autodiff::grad_with_loss`
   and Adam — same path the LNA uses for `Lg`, the MZI uses for
   `n_eff_A`.

The harness side lives in
[`../crates/eda-trace/src/optim.rs`](../crates/eda-trace/src/optim.rs):
`AdamState`, `LrSchedule::{Constant, Cosine, LinearDecay, StepDecay}`,
`BetaSchedule::{Constant, LinearAnneal, GeometricAnneal,
CosineAnneal}`, plus `default_device()` that picks `Device::Mlx`
on macOS hosts where `rlx-mlx` is linked. Three trace bins are
ported onto it: `spike-lna::lna_match_trace` (RF, β-anneal +
cosine LR drove `rel err 1.02e-7 → 0.00e0`), `spike-waveguide-block::mzi_match_trace`
(photonic, `|T_through|² 1.87e-4 → 1.83e-9`),
`eda-pnr::hpwl_optim_trace` (layout, 66.5 % → 93.4 % HPWL reduction).

## GPU scaling — `[B, N]` parallel-batch placement

Positions live as `Param[B, N]` tensors so the rlx graph's
`[B, N, N]` density operator does B× the work per kernel launch
without changing the dispatch count. CPU sees this as B sequential
placements; MLX amortizes its per-launch overhead across all B
placements at once. Measured on an M-series host at `N = 64`
instances, 32 nets, 300 Adam steps, identical seeds and
hyperparameters across runs:

| B | Cpu wall | Mlx wall | Speedup | Best-of-B loss | Cpu/B | Mlx/B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 0.06 s | 2.43 s | 0.025× | — | 60 ms | 2430 ms |
| **16** | 0.97 s | 2.14 s | 0.45× | 6.99 e6 | 60 ms | 134 ms |
| **64** | 3.99 s | 2.73 s | **1.46×** | 7.21 e6 | 62 ms | 43 ms |
| **128** | 7.74 s | 3.60 s | **2.15×** | 6.63 e6 | 60 ms | 28 ms |
| **256** | 15.17 s | 5.60 s | **2.71×** | 6.55 e6 | 59 ms | 22 ms |

Same loss on both backends (`4.9596e8 = 4.9596e8` at `B=64`,
byte-identical). CPU/B stays flat at ~60 ms (sequential placement);
MLX/B drops 110× from 2.43 s to 22 ms — the per-placement MLX cost
keeps falling as `B` grows because the launch overhead is fixed.
Crossover (MLX > CPU) at `B ≈ 30` on this host.

| Wall vs B (log-y) | Speedup vs B |
| --- | --- |
| ![Wall vs B](../crates/eda-pnr/docs/assets/gpu_scaling/wall_vs_b.svg) | ![Speedup vs B](../crates/eda-pnr/docs/assets/gpu_scaling/speedup_vs_b.svg) |

| Per-placement wall (the amortization story) | Best-of-B converged loss |
| --- | --- |
| ![ms per placement](../crates/eda-pnr/docs/assets/gpu_scaling/per_placement_vs_b.svg) | ![best loss](../crates/eda-pnr/docs/assets/gpu_scaling/best_loss_vs_b.svg) |

Raw data: [`../crates/eda-pnr/docs/assets/gpu_scaling/gpu_scaling.csv`](../crates/eda-pnr/docs/assets/gpu_scaling/gpu_scaling.csv).
Methodology + interpretation: [`../crates/eda-pnr/docs/gpu_scaling.md`](../crates/eda-pnr/docs/gpu_scaling.md).

What this unlocks (the use cases that were previously CPU-bound):

- **Best-of-K sampling**: `B` random seeds in one Adam invocation,
  pick the lowest-loss converged placement.
- **Hyperparameter sweep**: `B = n_lr × n_β` configurations
  evaluated jointly, pick the Pareto front in one run.
- **Monte-Carlo placement under PVT / mismatch**: `B` process-corner
  draws → a placement robust across the distribution.
- **DADO-style architecture search**: `B` candidate netlists sharing
  the same position grid, each with its own gradient signal.

Reproduce:

```sh
# the optimization itself (set BATCH_SIZE in the bin source per row)
cargo run --release -p eda-pnr --bin hpwl_at_scale_trace

# refresh charts + CSV + the standalone gpu_scaling.md after a fresh sweep
cargo run -p eda-pnr --bin gpu_scaling_chart
```
