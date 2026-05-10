# Differentiability quick reference — AD, MNA, PINN, DADO

One-page map of the four optimization machineries in `rlx-eda`:
**what each is**, **what can be differentiated / optimized over**,
**what's a good vs bad target**, and **how it's validated**. For
acronyms see [`glossary.md`](glossary.md). For the layered IR
picture see [`architecture.md`](architecture.md).

| Machinery | Question it answers | Variables | Output |
| --- | --- | --- | --- |
| **MNA** | What does the circuit *do*? | node voltages, branch currents | residual + Jacobian as `rlx_ir::Graph` |
| **AD** | How does the answer change with circuit params? | continuous device params (W, L, Vth, R, C, …) | gradient of any scalar loss |
| **DADO** | Which discrete catalog choice wins? | per-sub-block categorical bins | distribution over catalog tuples |
| **PINN** | Can we replace the simulator with a fast surrogate? | NN weights | (param → output) regression |

## MNA — Modified Nodal Analysis

The substrate everything else stands on. `eda-mna::build_residual_graph`
emits the residual `F(x, p) = 0` and its Jacobian `∂F/∂x` as
**`rlx_ir::Graph` instances** — not opaque matrices. That's what
makes the rest of this page possible.

- **Differentiable wrt**: any continuous device parameter that
  appears in a stamp. AD reuses the same graphs.
- **Good target**: small/medium analog blocks (≤ a few hundred
  transistors), full-stack mixed-signal (T.9 SAR ADC: 241 FETs /
  120 nodes runs end-to-end). Digital primitives validated under
  MNA at truth-table granularity.
- **Bad target**: statistical work at scale (Monte Carlo, PVT
  sweeps) — use the behavioral path or ngspice. The transistor MNA
  loop recompiles graphs per BE step today; T.10 caching will close
  that.
- **Validation**: analytic + finite-difference + ngspice witnesses
  per the validation-pyramid convention.
- **Read**: [`architecture.md`](architecture.md),
  [`digital_primitives_mna.md`](digital_primitives_mna.md) (T.8.B,
  truth tables), [`sar_adc_mna.md`](sar_adc_mna.md) (T.8.C, analog
  front-end), [`sar_adc_full_mna.md`](sar_adc_full_mna.md) (T.9,
  end-to-end).

## AD — Automatic Differentiation

Gradient of any scalar loss wrt any param flows through the same
`rlx_ir::Graph` MNA emits, via `rlx-opt`. No SPICE in the loop, no
finite differences during training.

- **Differentiable wrt**: device geometry (W, L), threshold (Vth),
  passives (R, C, L), bias voltages, anything you can route into a
  stamp. ∂(top-level scalar) / ∂(param) is a single AD pass.
- **Good target**: smooth operating regions with bounded gain
  between the param and the loss probe — e.g. T.8.A probes the
  comparator's analog stage-1 output (`d2`) *before* the digital
  output buffer, because the buffer's hard-saturating gain would
  collapse gradients to zero at the rails.
- **Bad target**: high-gain switching points (loss surface is a
  step), discrete choices (use DADO), and anything stiffer than
  Newton+BE can land — gradients exist but optimization stalls
  because the forward problem stalls.
- **Validation**: AD vs FD at the smooth operating point. T.8.A
  comparator: AD = +1.6521e0, FD = +1.5460e0, **6.86% relative
  error** — see [`comparator_sizing_ad.md`](comparator_sizing_ad.md).
  The doc explicitly explains why FD diverges from AD at the
  *converged* point and why the honest comparison is in the smooth
  region.
- **Status**: Relu/Sigmoid/Sqrt + softplus chain all FD-pass via
  `cpu_sqrt_grad.rs` witness in `../rlx`.
- **Read**: [`comparator_sizing_ad.md`](comparator_sizing_ad.md)
  (T.8.A, the canonical AD-validation example),
  [`sar_adc_full_mna.md`](sar_adc_full_mna.md) (cost / scaling
  notes for AD on big nets).

## DADO — Decomposition-Aware Distributional Optimization

Discrete optimization over per-sub-block catalogs, with a junction
tree exploiting decomposability. Bowden/Levine/Listgarten, ICLR
2026 ([arXiv:2511.03032](https://arxiv.org/abs/2511.03032)).

- **Optimizes over**: categorical bins per clique. SAR example: 12
  parameters across 4 cliques (Sample-Hold, Comparator, DAC, SAR
  register) with empty separators — independent in the search
  distribution.
- **Good target**: problems where the objective genuinely
  decomposes per-block (the synthetic Σ-decomposable benchmark in
  `spike-dado-r2r` shows the win cleanly). Catalog-style discrete
  EDA choices: which sub-block variant to pick.
- **Bad target**: tightly coupled real objectives — `spike-dado-r2r`
  produces a **negative result** on max-INL, Σ-INL², max-DNL on a
  real R-2R DAC. The decomposition assumption is the load-bearing
  part; if it doesn't hold, DADO doesn't win.
- **Hybrid pipeline**: surrogate-then-verify is **36× faster than
  direct SPICE** at ~5% relative quality loss
  ([`dado-sar-worked-example.md`](dado-sar-worked-example.md)).
- **Validation**: against analytical noise budget + ngspice on
  `SarAdc<4>`, plus a decomposition-unaware EDA baseline.
- **Read**: [`dado-sar-worked-example.md`](dado-sar-worked-example.md),
  crate READMEs under `crates/spike-dado-*/`.

## PINN — Physics-Informed Neural Network

A surrogate that adds the MNA residual as a soft loss term.
Raissi/Perdikaris/Karniadakis, JCP 2019.

- **Trained on**: (param → output) pairs from MNA, with a physics
  loss penalizing residual violation.
- **Good target**: input dimensionality **d ≥ 10** with **loose
  absolute accuracy bounds**. Experiment 4 (SAR + mismatch, d=10)
  is the only PINN positive in the suite — beats Poly-d4 by 36% on
  max-abs (p=2e-3, δ=−1.0). Both still fail absolute sub-LSB.
- **Bad target**: low-d smooth problems. Experiment 2 (Diode-RC,
  d=5): Poly-d4 hits machine precision at 126 params; physics term
  *hurts* (Hybrid worse than Surrogate on OOD, p=2e-3). Experiment
  3 (SAR-1D): linear regression at 5 params hits ½ LSB; PINN at
  1153 params is 35× worse.
- **Validation**: pre-registered protocol frozen 2026-05-10. K=10
  seeds → paired Wilcoxon (exact 2^K enumeration) → Cliff's δ +
  magnitude bins → Holm-Bonferroni. Acceptance criteria mirrored
  in `pub const` items and enforced by a parity test — no metric
  switching after results land.
- **Read**: [`pinn-experiments.md`](pinn-experiments.md) for the
  full executive summary table, methodology, and per-experiment
  verdicts.

## Picking the right tool

| Situation | Use |
| --- | --- |
| "Does my circuit work?" | MNA (or ngspice for cross-check) |
| "Tune Vth / W / R to hit a target" | AD on MNA gradients |
| "Pick the best variant from a discrete catalog per block" | DADO |
| "I'll evaluate this 10⁶ times, can I replace MNA?" | PINN — but only if d ≥ 10 and absolute bound is loose; otherwise polynomial regression |
| "Statistical / PVT sweeps" | behavioral path + ngspice (not transistor MNA, not yet) |
| "High-gain switching point" | none of the above — pick a smoother probe |

## Cross-cutting validation pattern

Per [`validation_pyramid`](../README.md) convention, every component
lands with **analytic + FD + ngspice** witnesses:

- **MNA** ↔ analytic / truth-table / ngspice
- **AD** ↔ finite differences (at smooth operating points)
- **DADO** ↔ ngspice on the decoded catalog tuple
- **PINN** ↔ pre-registered K=10 statistical tests against polynomial baseline
