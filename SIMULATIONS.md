# Simulation paths

Five paths run circuits in this workspace. They live at different
fidelity-vs-cost points and cross-validate each other; nothing here
is "the simulator" and nothing is purely a witness — every path is
load-bearing for some workflow, and every other path is the witness
that catches its lies.

| Path | Where | Per-eval cost | What it's good at | What it can't do |
| --- | --- | --- | --- | --- |
| Analytic device + rlx graph | `spike-mosfet-dc/src/lib.rs:94` (`id_subgraph`) | µs (compiled), ms (compile+run) | Closed-form `Id(Vgs, Vds; Vth, kp, λ)` with reverse-mode AD on every parameter. Building block for everything else. | Single device. No KCL on a netlist. |
| Circuit-level MNA | `eda-mna/src/lib.rs:885` (`solve_dc`), `:1869` (`transient_pwl`) | 1–100 ms per Newton solve | Differentiable BE-step transient over arbitrary `Mosfet` / `Resistor` / `Diode` / `LinearCap` compositions; gradients of any output w.r.t. any param via `transient_sensitivities`. | No frequency-domain. Newton stalls on degenerate DC corners (use transient + caps). |
| SPICE bridge | `eda-extern-ngspice/src/lib.rs:333` (`LocalBinary`), `:438` (`DockerInvoker`) | 50–500 ms per `.op`, seconds per `.tran` | The cross-validator. Anything ngspice can do (BSIM4, sky130 .lib, `.temp`, AC, MC) lands here. Picks Docker-pinned image when `NGSPICE_BACKEND=docker`. | Slow for inner-loop optimization. Not differentiable. |
| ML surrogate | `spike-mosfet-dc/src/surrogate.rs`, `spike-surrogate/src/lib.rs` | µs per batched matmul once trained; minutes one-time train | 10⁴–10⁶ evals across (operating point × corner) without per-call graph compile. Gradients through every input — including `T_celsius`. | Only as accurate as the training distribution. Out-of-envelope queries silently extrapolate. |
| Distributional optimization (DADO) | `spike-dado-r2r/src/lib.rs:563` (`run`), `spike-dado-sar/` | Seconds–minutes for K=100 × 80 iter | Discrete-design optimization with worst-corner / multi-objective scoring functions. Per-clique decomposition for chain-junction-tree-shaped designs. | Decomposability bottlenecks the gain (see the negative result in the DADO R-2R STORY). |

The paths compose: the analytic device feeds the MNA stamps, MNA feeds
the surrogate's training labels (or ngspice does, when sky130 BSIM4 is
the truth), and DADO scores call into MNA or the surrogate as its
inner loop.

## Validation pyramid

Every component lands with three witnesses, run as separate test files:

1. **Analytic.** Closed-form formula in Rust. Cheap, exact, unit-testable.
   Example: `spike_mosfet_dc::id_strict` (`crates/spike-mosfet-dc/src/lib.rs:217`).
2. **Finite difference.** Central FD on the analytic forward, compared
   against rlx-graph reverse-mode AD. Catches AD bugs without needing
   ngspice. Example: `crates/spike-mosfet-dc/tests/finite_difference.rs`.
3. **ngspice cross-engine.** Same operating point through ngspice via
   the SPICE bridge. Catches model bugs FD can't see (sign conventions,
   unit errors, pre-DC convergence quirks). Example:
   `crates/spike-mosfet-dc/tests/ngspice.rs`.

A fourth tier — **LTspice** — kicks in when present (soft-skipped on
Linux/macOS-only setups). See `crates/spike-cmos-gates/tests/truth_tables.rs`
for the gated-by-feature pattern.

## Thermal corners

PVT corner sweeps run on every path without rebuilding circuits.
Single source of truth for the physics:

```text
Vth(T) = Vth0 + KT1 · (T − Tnom)            KT1 = -1 mV/°C
kp(T)  = kp0 · (T_K / Tnom_K)^UTE           UTE = -1.5
T_NOM_C = 27 °C
```

Constants live in `spike-mosfet-dc/src/lib.rs:97` and are duplicated
verbatim in `spike-divider-block/src/lib.rs::thermal` (no shared
upstream crate yet — physics, not implementation, so duplication is
tolerable; the validation pyramid catches divergence).

### Per-path entry points

| Path | Entry point | Pattern |
| --- | --- | --- |
| Analytic device | `spike_mosfet_dc::run_id_at_temp(vgs, vds, vth0, kp0, lam, t_celsius)` | Parameter remap before graph build. T is **not** an AD edge. |
| Circuit MNA | `spike_divider_block::thermal::remap_mosfet_params_for_temp(&mut params, t_celsius)` | Post-hoc walk over the params HashMap built by the gate constructors. Same circuit graph, remapped scalars. |
| SPICE bridge | `spike_mosfet_dc::spice_deck_at_temp` (single device) or any deck text containing `.temp <T>` | ngspice's built-in LEVEL 1 scaling (μ ∝ T^-1.5 + bandgap-driven VTO shift) does the rest. |
| ML surrogate | T enters as a graph input (3rd column of `x`); see `spike_mosfet_dc::surrogate` | T **is** an AD edge — useful for thermal sensitivity / worst-corner-via-ascent. |
| DADO | `spike_dado_r2r::score_inl_worst_corner(design)` returns `min_T |INL(design, T)|` (negated). | Scores against `T_CORNERS_C = {-40, 27, 125}` ⁰C. |

### Cross-engine numbers

| Test | Tnom agreement | Worst-corner agreement | File |
| --- | --- | --- | --- |
| Analytic vs ngspice (single NMOS, V=2.0, V=2.0) | exact at Tnom | 0.3 % rel at -40 °C, 0.08 % at 125 °C | `crates/spike-mosfet-dc/tests/thermal_sweep.rs` |
| MNA inverter Vout swing across corners (Vin=V_M=0.97) | nominal V_M = 0.97 V | 0.25 V at -40 °C, 1.52 V at 125 °C | `crates/spike-cmos-gates/tests/thermal_sweep.rs` |
| Surrogate MLP vs analytic on (bias × T) grid | — | < 5 µA absolute everywhere | `crates/spike-mosfet-dc/tests/thermal_surrogate.rs` |
| DADO worst-corner INL (adversarial design) | nominal-T INL = 26.2 mV | -40 °C INL = 28.8 mV (~10 % worse) | `crates/spike-dado-r2r/tests/thermal.rs` |

### Tolerance schedule for new corner-sweep tests

Per `crates/spike-mosfet-dc/tests/thermal_sweep.rs`:
- Tnom: 1 % rel envelope (matches existing nominal tests).
- ±100 °C corners: 7 % rel envelope (linear KT1 vs ngspice's bandgap-based VTO formulas — in practice agreement is ~0.3 %, but don't tighten further without a reason).
- Sanity guards: `Id(125 °C) < Id(−40 °C)` (mobility-dominated trend) and `|ΔId|/Id > 5 %` (rules out silent no-ops where T failed to thread through).

### Known gotchas

- **`solve_dc` on inverters at extreme rails.** Newton blows up to nonsense voltages because the off-MOSFET + off-rail-PMOS Jacobian goes near-singular. Use `transient_pwl` with cap-stabilized BE for inverter / gate corner sweeps — that's the path `digital_primitives_mna.rs` uses and what passes consistently across corners.
- **Bias near `V_M`** to amplify corner shifts. For a 4 µm:2 µm PMOS:NMOS sizing, `V_M ≈ 1.054 - 0.172·|Vth| ≈ 0.97 V`. At `V_M`, the high-gain region multiplies the few-mV `V_M` shift across corners into a hundreds-of-mV `Vout` shift — easy to assert. Far from `V_M`, the signal is dominated by whichever transistor is rail-pinned and corner sensitivity collapses.
- **All-nominal designs are thermally benign by construction in DADO.** The `TC1_DEV_KAPPA × dev` cross-term zeroes when every resistor sits at 0 % deviation. The interesting designs for worst-corner are the adversarial ones with mixed deviations.
- **ngspice batch + `/dev/stdin`.** Piping a deck through stdin sometimes mis-parses the title line. The MNA + spike-mosfet-dc tests work fine, but if you hit silent failures, write the deck to a temp file and pass the path. See `memory/ngspice_stdin_title.md` for context.

## Choosing a path

Decision tree for new work:

1. **Single-device characterization or AD demo.** → Analytic + rlx graph. Add a `run_*_at_temp` wrapper if the corner story matters.
2. **Multi-device circuit at DC or transient with one or a few solves.** → Circuit-level MNA via `solve_dc` / `transient_pwl`. Use the MNA stamps in `spike-divider-block` (`Mosfet`, `Resistor`, `Diode`).
3. **Final correctness check.** → Add an ngspice tier-3 test that runs the same operating point or transient. Tolerance budget: 1 % at Tnom, 7 % at corners. The bridge auto-skips when ngspice isn't on `PATH`.
4. **Inner loop of an outer optimization with > 10⁴ evals.** → Train a surrogate (`spike-surrogate` for the divider; `spike-mosfet-dc::surrogate` for thermal MOSFET). Validate against the analytic ground truth on a held-out grid before using it for science.
5. **Discrete-design optimization with corner robustness.** → DADO with a worst-corner / multi-objective score. The `spike-dado-r2r::score_inl_worst_corner` shape is the template; replace `score_inl_at_temp` with whatever per-corner score your problem cares about.

## File-path quick reference

```text
crates/eda-mna/src/lib.rs                        — MNA assembly + Newton + transient
crates/eda-extern-ngspice/src/lib.rs             — ngspice bridge (LocalBinary + DockerInvoker)
crates/eda-spice-emit/src/lib.rs                 — SPICE deck emitter (Netlist, primitives)
crates/spike-mosfet-dc/src/lib.rs                — analytic LEVEL=1 NMOS + thermal
crates/spike-mosfet-dc/src/surrogate.rs          — MLP surrogate (Vgs, Vds, T) → Id
crates/spike-divider-block/src/lib.rs            — Mosfet / Resistor / Diode MNA stamps + thermal::*
crates/spike-cmos-gates/src/mna.rs               — gate constructors (add_inverter, add_nand2, …)
crates/spike-dado-r2r/src/lib.rs                 — discrete R-2R + DADO + thermal::*
```

Tests for the thermal-corner work specifically:

```text
crates/spike-mosfet-dc/tests/thermal_sweep.rs       — single device, three tiers across corners
crates/spike-mosfet-dc/tests/thermal_surrogate.rs   — MLP surrogate vs analytic
crates/spike-cmos-gates/tests/thermal_sweep.rs      — inverter via MNA across corners
crates/spike-dado-r2r/tests/thermal.rs              — worst-corner DADO score
```
