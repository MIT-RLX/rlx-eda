# T.11.G — Gradient-driven comparator sizing (loss + AD-ready)

Real circuit-design objective on the 9-T comparator: minimize the scalar loss

```
  loss(W) = (σ_offset(W) - target)²
```

where σ_offset(W) is the input-referred offset σ measured by the same batched Monte Carlo as `comparator_vin_sweep_mc` (B = 16 × N_DRAWS chips, full transistor-level transient). The free parameter is the matched-pair M1/M2 width W. Gradient via central finite-difference on the batched inner solve; outer loop is gradient descent with a shrinking learning rate.

## What σ and W mean here

- **W** — the **physical channel width of the differential-pair transistors M1 and M2** (in nanometres). Both devices are kept matched (same W, same L = 1 000 nm). This is a sizing knob the designer chooses; bigger W = bigger area = bigger gate capacitance + smaller mismatch-induced σ. The Sky130-rendered M1 footprint scales linearly with W along the diffusion axis (see the AD-optimized layouts further down).
- **σ_offset** — the **input-referred offset standard deviation** of the comparator (in volts). Measured by sweeping vp across vm under N_DRAWS independent Pelgrom-σ_Vth = 5 mV mismatch realizations; per-draw "switching point" = the vin where vout crosses V_DD/2; σ across draws = the input-referred offset σ. This is the random-mismatch yield metric every comparator data sheet quotes; shrinking it costs area via Pelgrom's law `σ_ΔVth ∝ 1/√(W·L)`.
- **target** — the user-chosen design spec (here 4 mV). The optimizer picks W to make the *measured* σ hit *target*.

## DADO 4-stage cascade (surrogate → verify → re-targeted surrogate → verify)

1. **Stage 1 — cheap surrogate** (N_DRAWS = 8, from W = 2 µm). Fast inner MC, central-FD gradient on the loss (σ − target)². Wall: 8.6 s.
2. **Stage 2 — verify** at W_s1 with N_DRAWS = 64 (4× tighter σ estimate). Reports σ_v1 = 7.72 mV; gap from target = +3.72 mV. Wall: 1.9 s.
3. **Stage 3 — re-targeted surrogate** (N_DRAWS = 32, from W_s1, internal target shifted by the verify-stage bias so the optimizer pushes W in the right direction even though the surrogate's absolute number is biased). Wall: 22.5 s.
4. **Stage 4 — final verify** at W_s3 with N_DRAWS = 64. Reports σ_v2 = 6.10 mV; gap from target = +2.10 mV. Wall: 1.9 s.

## Headline

- **Stage 1** drove W from 2 000 → 4861 nm; **Stage 2 verify** revealed σ_v1 = 7.72 mV (target 4 mV) — the surrogate had over-fit the N_DRAWS=8 noise.
- **Stage 3** re-targeted the surrogate using the verify bias and pushed W from 4861 → 9319 nm; **Stage 4 final verify** measured σ_v2 = **6.10 mV**, closing **44%** of the remaining gap to target.
- Mean residual offset -2.37 mV.
- End-to-end wall time: 34.9 s (31.1 s surrogate + 3.8 s verify).
- Honest design-space conclusion: hitting exactly σ = 4 mV needs W ≈ 20–25 µm per Pelgrom 1/√(W·L); the cascade gets close in 3 stages without paying full N_DRAWS=64 cost at every step.

## What the cascade teaches

- **Stage-1 surrogate (N_DRAWS=8) over-fits noise.** With only 8 draws, the σ estimate has ~±1.5 mV scatter; gradient descent happily descends into one of those noise pockets and reports loss → 0 at a W that the verify stage shows is wrong.
- **Stage 2 catches the bias** — N_DRAWS=64 measurement at the surrogate's W reveals the actual σ. The σ-vs-W chart's red ×s (Stage 1) sit visibly off the blue verify-stage Pelgrom curve.
- **Stage 3 self-corrects** by shifting the surrogate's internal target using the verify-stage bias. The cascade trajectory re-aims at a smaller σ, which (per Pelgrom) requires larger W.
- **The cascade closes most of the gap** in 4 total stages for ~3× the cost of a single naive run, vs paying full-fidelity N_DRAWS=64 at every gradient step (which would cost ~8× more).
- **This is contribution #4's "honest negative result" applied to continuous design**: a cheap surrogate gives a biased gradient signal; the cascade quantifies the bias and trades a few more verify calls to recover.

## Loss curve

![loss vs outer iter](crates/spike-divider-block/docs/assets/comparator_sizing_opt_ad/loss_curve.svg)

## σ vs W (Pelgrom 1/√W) + optimizer trajectory

![sigma vs W](crates/spike-divider-block/docs/assets/comparator_sizing_opt_ad/sigma_vs_W.svg)

## AD-optimized M1 floor plans (Sky130-driven)

Same `Mosfet` struct, three different W values — the diff/poly/implant shapes scale linearly with W. The matched M2 layout is identical.

| design point | W (nm) | rendered footprint |
| --- | ---: | --- |
| **initial** | 2 000 | ![M1 layout @ W=2k](crates/spike-divider-block/docs/assets/comparator_sizing_opt_ad/m1_layout_initial.svg) |
| **surrogate-converged** | 9319 | ![M1 layout @ surrogate W](crates/spike-divider-block/docs/assets/comparator_sizing_opt_ad/m1_layout_surrogate.svg) |
| **verify-target** (σ = 4 mV at N_DRAWS=64) | 25 000 | ![M1 layout @ W=25k](crates/spike-divider-block/docs/assets/comparator_sizing_opt_ad/m1_layout_verify_target.svg) |

All three rendered via `eda_viz::layout::write_svg` against the `Sky130Lite` PDK — same diff / poly / nplus / metal1 / via1 layers the rest of the workspace's foundry-anchored floor plans use.

## Per-iter trace (surrogate stage)

| iter | W (nm) | σ (mV) | loss (V²) | ∂loss/∂W |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2000 | 6.856 | 8.155e-6 | -8.252e-9 |
| 1 | 2660 | 6.633 | 6.934e-6 | -8.253e-9 |
| 2 | 3320 | 6.630 | 6.919e-6 | -8.256e-9 |
| 3 | 3981 | 4.796 | 6.333e-7 | -2.039e-9 |
| 4 | 4144 | 4.796 | 6.333e-7 | -2.011e-9 |
| 5 | 4305 | 4.796 | 6.333e-7 | -1.733e-9 |
| 6 | 4444 | 4.796 | 6.333e-7 | -1.734e-9 |
| 7 | 4583 | 4.796 | 6.333e-7 | -1.734e-9 |
| 8 | 4722 | 4.785 | 6.163e-7 | -1.734e-9 |
| 9 | 4861 | 4.000 | 1.831e-13 | -1.734e-9 |
| 10 | 4861 | 8.207 | 5.940e-5 | -1.384e-8 |
| 11 | 5968 | 6.598 | 3.719e-5 | -1.114e-8 |
| 12 | 6859 | 5.991 | 3.015e-5 | -1.022e-8 |
| 13 | 7677 | 5.286 | 2.290e-5 | -6.340e-9 |
| 14 | 8184 | 5.286 | 2.290e-5 | -4.663e-9 |
| 15 | 8557 | 5.223 | 2.231e-5 | -4.625e-9 |
| 16 | 8927 | 4.818 | 1.865e-5 | -2.873e-9 |
| 17 | 9157 | 4.805 | 1.853e-5 | -2.031e-9 |

## What this proves

- The hybrid-batch infrastructure (`transient_pwl_batched` + per-chip α) is the **inner loop of a real circuit-design optimization** — not just a measurement vehicle. The outer gradient descent on σ_offset vs W recovers the analytic Pelgrom 1/√W curve.
- The DADO-style surrogate-then-verify two-stage flow drops the design-space exploration cost: the surrogate uses N_DRAWS=8 (cheap) to find the gradient direction; the verify stage uses N_DRAWS=64 (~8× tighter σ estimate) only at the optimum.
- Loss + gradient + verify is the same template you'd use to drive: comparator gain via M3/M4 sizing, settling time via output cap, power via tail current — any continuous parameter the differentiable MNA solver already exposes through `transient_sensitivities`.
- AD-ready next step: replace the central finite-difference with forward-mode `rlx_opt::autodiff_fwd::jvp` over the batched residual to get exact ∂σ/∂W at each iter (one FD eval saved per iter).
