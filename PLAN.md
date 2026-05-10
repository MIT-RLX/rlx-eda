# rlx-eda PLAN

Roadmap for the rlx-eda workspace — Rust EDA libraries + spike
crates that validate differentiable circuit-simulation primitives
on top of the rlx ML compiler at `../rlx`.

This file mirrors `../rlx/PLAN.md`'s structure: a "Landed" section
for completed work (with the validation surface that proves it),
and a "Future" section organised by leverage. The split between
the two repos is intentional — `rlx` stays JAX-shaped (generic
differentiable runtime), and circuit-specific code (component
models, MNA stamping, spike workflows) lives here.

## Landed

The toolchain hosts a substantially complete Circulax-style
differentiable circuit simulator. Each item below pairs an
analytic / closed-form Tier 1 oracle with an FD-on-AD Tier 2
witness and (where applicable) an ngspice Tier 3 cross-validation.

### EDA core libraries

- **`eda-hir`** (589 LOC) — typed circuit HIR. `Block` /
  `NonlinearDcBehavioral` / `MnaDevice` / `TransientStorage`
  traits define the contract every circuit primitive implements.
  `schematic` provides node/port/edge graph for visual rendering.

- **`eda-mna`** (1401 LOC) — Modified Nodal Analysis assembler.
  - `Circuit` builder: `alloc_unknown_net` / `alloc_boundary_net`
    / `add_device` / `add_storage`.
  - `build_residual_graph` (DC) and `build_be_step_residual_graph`
    (transient) emit rlx graphs from a topological netlist.
  - `solve_dc(circuit, params, boundary, NewtonOptions) →
    DcOperatingPoint` runs damped Newton.
  - `solve_be_step` / `transient` for backward-Euler integration.
  - `sensitivities(circuit, params, boundary, op, wrt)` does
    block-IFT parameter Jacobian assembly at the converged OP —
    the canonical Circulax "block-Jacobian" / "hierarchical AD"
    pattern. **21 tests across 9 files**: R+D circuit, MOSFET DC,
    MOSFET transient, voltage source, inverter, optimization,
    Newton solve, sensitivity, transient.

- **`eda-spice-emit`** — SPICE netlist emission for ngspice /
  LTspice cross-validation. Used by every spike crate's Tier 3
  test.

- **`eda-validate`** — assertion helpers (`assert_close`,
  `assert_traces_close`, FD utilities) shared across spike tests.

- **`eda-waveform`** — time-domain trace I/O + plotting helpers.

- **`eda-viz`** — schematic + layout rendering. Optional `png`
  feature uses resvg/usvg for byte-stable PNG output.

- **`eda-pdks` / `eda-pdk-ingest`** — process design kit ingestion.

- **`eda-extern-ngspice`** — `Invoker` trait + `LocalBinary`
  shells out to a native ngspice. `run_dc` / `run_transient_final`
  / `run_transient_trace` / `run_ac`.

- **`eda-extern-ltspice`** — sibling for LTspice (skeleton).

### Differentiable spike workflows

- **`spike-diode`** — diode-RC nonlinear circuit, end-to-end:
  - `op_ift` — DC operating point via `Op::CustomFn` with IFT
    vjp_body / jvp_body. O(1) backward via the implicit-function
    theorem (no differentiating-through-the-loop).
  - `transient` — diode-RC transient via `scan_with_bcasts_and_xs`
    + Griewank checkpointing. ngspice-validated against `.tran`.
  - `optimize` — Adam in log-space recovers a target waveform from
    a perturbed parameter init. **First end-to-end proof the
    AD pipeline is useful for design optimization, not just
    correct.** Loss drops 4.7 orders of magnitude in 100 Adam
    steps.
  - 21 passing tests across newton / ad / op_ift / transient /
    ngspice / optimize.

- **`spike-mosfet-dc`** — Shichman-Hodges (SPICE LEVEL=1) NMOS:
  - Smooth `Id(Vgs, Vds)` with softplus cutoff + smooth-min
    Vds_eff for saturation/triode boundary.
  - `inverter` — CMOS inverter DC operating point. Two MOSFETs
    sharing one node; nonlinear KCL at V_out solved via damped
    Newton (step-magnitude cap fixes the steep-transition
    divergence). 4 passing tests including AD-vs-FD on all
    6 device parameters.
  - 12+ tests across analytic, finite_difference, ngspice,
    inverter.

- **`spike-ac`** — AC small-signal analysis:
  - Linear RC low-pass via 2N×2N real-block complex MNA
    encoding (no native complex dtype needed).
  - Diode-RC linearised at DC OP (`diode_rc` module): computes
    Vmid* + g_d via Rust pre-pass, plugs into the AC graph as
    an extra Input. 6 tests across analytic, finite_difference,
    ngspice, diode_rc_analytic, diode_rc_fd, diode_rc_ngspice.

- **`spike-rc-transient`** — linear RC LP transient (predecessor
  of spike-diode/transient; uses unrolled `dense_solve` chain
  rather than scan).

- **`spike-cmos-gates`** — gate-level standard-cell library:
  Inverter, Nand2, Nand3, Nor2, And2, Or2, DLatch, DFF, DLatchSR,
  DffSR. Each as a `SpiceEmit` block composing eda-spice-emit
  Nmos/Pmos primitives. Truth-table tests for digital correctness.

- **`spike-divider*`** — voltage divider in three flavours:
  unrolled (`spike-divider`), block-shaped (`spike-divider-block`),
  layout-bound (`spike-divider-layout`), MNA-routed
  (`spike-divider-mna`).

### Tooling-level wins

- **End-to-end ngspice cross-validation** — every spike with a
  numerical claim has a `tests/ngspice.rs` confirming agreement
  with an external simulator.
- **f32 + f64 type discipline** — most rlx-side bugs found
  during this work were dtype-mismatch silent miscompute landmines
  (f32 ActivationBackward, f64 trajectory upstream-add). Each
  was caught by an FD-vs-AD parity test.
- **Damped Newton (step-magnitude cap)** — added to the inverter
  body; lets the unrolled-Newton-in-graph pattern survive
  steep-transition regions where `f' ≈ 0`.

### Cicsim-parity simulation harness

Functional parity with Carsten Wulff's
[`cicsim`](https://github.com/wulffern/cicsim) flow plus a few
features we go past it on. Validated against Carsten's
`rply_ex0_sky130nm` reference cell run inside his published
`wulffern/aicex:26.04_latest` Docker image.

- **`eda-sim-harness`** — testbench → corner-set → ngspice →
  measure → spec-check → report. Mirrors `cicsim`'s
  `make typical etc mc summary` pipeline. `Testbench` trait with
  `build_netlist` / `measurements` / `analysis` / `derive` /
  `plot_signals` hooks. `derive` is the cicsim `tran.py`
  analogue — emits derived measurements (e.g.
  `ibn_settl_err = ibns_20u - ibns_20u_9n`) folded into the
  `MeasureLog` before spec checks.
- **`rlx-eda-cli`** binary with subcommands:
  - `pdk install/list/show/register/forget/path` —
    wraps `ciel` (with `volare` fallback) for sky130A/B,
    gf180mcuA-D, ihp-sg13g2. Auto-discovers section names from
    each PDK's `.lib` file.
  - `doctor` — checks ngspice/ciel/volare on PATH, registry
    config readable, registered + ciel-discovered PDKs (lib_path
    exists, sections parse). Live: 5 ok / 7 warn / 0 fail with
    concrete remediation hints.
  - `dashboard --root <path>` — walks `crates/*/docs/*/`,
    parses CSV `OK` columns, generates `docs/index.html` with
    pass/fail badges, per-bucket counts, age timestamps.
- **`spike-lelo-ex`** — sky130A 1:4 NMOS current mirror with
  three integration tests producing artifacts in
  `crates/spike-lelo-ex/docs/`:
  - `sky130a_tt_ff_ss/` — 3-corner PVT sweep. Live:
    mirror ratio 4.55× / 4.74× / 5.00× (all pass).
  - `sky130a_mc/` — 8-draw MC against `mc_pr_switch=1`
    mismatch. Live: σ = 1.9 % of mean. `Spec::check_mc` gates
    against `µ ± 3σ`.
  - `sky130a_sch_vs_lay/` — Schematic vs. Layout-extracted
    view. Live: Δ = −0.50 % from a stub `Rdrain` parasitic
    (real LPE pending — see Future).
- **Reporter outputs** — per-corner HTML (SVG range bars,
  Δ-from-typ %, embedded waveform PNG, collapsible deck + log
  accordions), Markdown summary, multi-page PDF with embedded
  300dpi plots, inline MC histogram SVG with mean line + ±σ
  band, cicsim-shape CSV per `(view, kind)` bucket
  (`<tb>_Sch_typical.csv` style).
- **Differentiable Monte Carlo proof-of-concept** in
  `spike-mosfet-dc::mc` — 200 LEVEL=1 mirror draws in **127 ms**
  (≈ 800× faster than the equivalent ngspice MC sweep), with
  Pelgrom σ scaling observed exactly (ratio = √4 = 2.00) and AD
  gradient `∂E[Iout]/∂W_M2` matching finite-difference baseline
  to **0.000 % relative error**. Demonstrates what we can do
  past cicsim once a real PDK model is in rlx (see Future:
  BSIM4 in rlx).
- **Robustness wins** — content-hash cache (survives PDK path
  moves; in-process memo so multi-MB libs hash once); first-error
  extraction from ngspice stderr (`Error in netlist line N:`
  surfaced cleanly, full text behind `RLX_EDA_VERBOSE=1`);
  HSPICE-A compat injected via deck-side `.spiceinit` so
  sky130's `montecarlo.spice` parses.

**Validation surface:** 39 unit tests across 4 crates + 5
sky130A integration tests + 1 RC smoke. `rlx-eda doctor`
precheck. Two memories saved (`ngspice_stdin_title.md`,
`sky130_mc_composition.md`) for the non-obvious gotchas.

## Validation framing — what the pyramid does and doesn't prove

The Landed section pairs every numerical claim with an
analytic / FD / ngspice witness. That stack proves *AD and
numerical consistency*, not silicon performance. Three caveats
worth carrying forward when reporting results or scoping
follow-ons:

1. **Ngspice is the bottom of our pyramid, not ground truth.**
   It is another simulator, with simplified PDK models —
   `spike-mosfet-dc`'s LEVEL=1 + KT1/UTE thermal remap, ngspice's
   BSIM4 implementation, no package parasitics, no EM coupling,
   layout extraction only as far as `magic` hands us.
   "Ngspice-validated" means consistent with a particular
   simulator stack, not predictive of silicon.

2. **Tolerance ≠ accuracy.** Several spike-* tests chase very
   tight numerical agreement (≈0.3 % at thermal corners,
   AD-vs-FD to many digits, 0.000 % rel-err on the MC gradient).
   These are AD-correctness witnesses — we're checking math, not
   predicting silicon. The MC and PVT-corner work in
   `eda-sim-harness` (Pelgrom σ scaling, `µ ± 3σ` spec-checks)
   is the closer analog of a real silicon-yield claim, but is
   still bounded by caveat 1.

3. **No measurement tier yet.** There is no silicon-correlation
   layer below ngspice. The Future items `Real layout-parasitic
   extraction` and `BSIM4 in rlx` are the natural lead-ins to
   one — once LPE is real and BSIM4 lives in rlx, a
   `measurement` witness alongside analytic / FD / ngspice would
   close the loop. `spike-lelo-ex`'s Sch-vs-Lay output already
   carries this caveat (called out as a *demonstration* until
   real LPE lands); the same applies to the rest of the suite.

Actionable consequence: surface numbers in dashboards and write-
ups as "AD/numerical consistency" or "ngspice-correlated", not
"accuracy". Don't chase further tightening of FD tolerances past
what AD verification needs — additional digits buy nothing
silicon-side until tier 4 exists.

## Future

Items not yet in tree, ordered by leverage. None are blocking
current workloads — their priority should be set by what real
circuits drive through the toolchain next.

### Workload-driven scaling (wait for real bottleneck)

These help when circuits get big enough that the current
implementation hurts. Speculative implementation risks designing
for needs that don't materialise.

- **`Op::SparseSolve` in rlx**. Current `Op::DenseSolve` is
  `O(N³)`; circuits past ~30 nodes feel it. Substantial: CSR/COO
  storage + reordering (AMD or COLAMD for fill-in) + symbolic +
  numeric factorization + AD via IFT exploiting sparsity. Either
  own implementation (~1000 LOC) or shim to KLU/SuiteSparse via
  FFI (~300 LOC, maintenance trade). Triggers: `spike-sar-adc`
  or any multi-stage analog circuit.

- **`jacrev`-driven block-Jacobian assembly in
  `eda-mna::sensitivities`**. Currently `n` separate
  `grad_with_loss` compiles + runs (`O(n²)` total); a single
  multi-output reverse-mode AD pass collapses to `O(n)`. Pure
  perf win, transparent to callers. ~150 LOC. Triggers: any
  circuit with >50 unknown nets.

- **Multi-level hierarchical sensitivities**. Currently flat at
  the device level — sub-circuits-within-sub-circuits aren't
  modelled. Natural for any circuit organised as composable
  blocks (`spike-sar-adc`'s DAC + comparator + register +
  clock-decoder hierarchy is the canonical example). ~300 LOC
  extension to `eda-mna`. Pairs with the SAR ADC port.

- **Globalised Newton variants** beyond step-magnitude cap:
  Levenberg-Marquardt damping, line search with Armijo
  backtracking, homotopy continuation (parameter ramp). Current
  cap works for steep-transition CMOS gates; LM/homotopy would
  help convergence on harder problems (regenerative comparators,
  oscillators). ~200 LOC each. Triggers:
  `spike-comparator-cmos` (regenerative latch).

### Specialised analysis types

Each serves a narrow workflow. Add when a circuit demands it.

- **PSS / shooting method** for periodic steady-state. Finds
  limit cycles via Newton on the period map; combines FFT +
  DenseSolve + `custom_vjp`. The natural mode for switching
  converters and oscillators. ~500 LOC. Triggers: any switching
  circuit with persistent-cycle behavior (`spike-clocks`,
  ring-oscillators in `spike-cmos-gates`).

- **Harmonic balance** for steady-state in frequency domain.
  Solves `f(X) - jωCX = 0` directly. Useful for RF and analog
  filters. ~400 LOC. Pairs with PSS for shooting+HB hybrid.

- **Adaptive-timestep transient** with LTE-controlled refinement.
  Replaces uniform-h scan-based BE. Needs AD through the step
  controller (`custom_vjp` on the implicit step). Substantial:
  ~600 LOC. Useful for stiff systems (clock edges, comparator
  latching) where uniform timestep wastes work in slow regions.

- **Mixed-signal time-domain simulation**. Combine analog
  (continuous via BE/IRK) and digital (event-driven, threshold-
  triggered transitions) in one transient run. Required for
  any controller/datapath simulation including the SAR ADC.
  Substantial — likely needs its own scan body that handles
  both element types. ~800 LOC.

- **Implicit Runge-Kutta integrators** (Radau, Lobatto). Higher-
  order than BE for stiff transients. Each step is a small
  nonlinear solve — natural fit for `custom_vjp`. ~400 LOC.

- **Noise / variance propagation analysis**. Builds on AC: each
  device contributes a noise current; the small-signal
  admittance matrix maps it to output noise voltage. Standard
  analog characterisation; useful when noise figure matters.

- **S-parameters / two-port characterisation**. Frequency-domain
  driven, builds on AC analysis. Specialised but standard for
  RF design. ~250 LOC.

- **Stochastic / Monte Carlo via vmap'd parameter draws**.
  Already buildable with the current toolchain; just needs a
  demo and clean API. ~200 LOC. Useful for yield analysis.

### Diffrax-inspired solver architecture

Diffrax (JAX differentiable ODE/SDE/CDE library) overlaps heavily
with `eda-mna`'s domain — MNA produces index-1 DAEs, and the
transient spikes are ODE solvers in disguise. These items lift
patterns from diffrax that aren't already covered above; some
cross-reference existing entries.

- **Term / Solver / StepSizeController refactor in `eda-mna`**.
  Today `solve_be_step` / `transient` hardcode backward-Euler with
  a fixed step. Diffrax's three-way split (residual term, stepping
  algorithm, adaptive controller) lets each evolve independently
  and is the right shape *before* a second solver lands. Refactor
  is mostly a trait reshuffle on existing code. ~250 LOC. Triggers:
  the moment a second integrator (IRK, adaptive BE, trapezoidal)
  is wanted.

- **DAE-aware solver interface (mass-matrix form)**. MNA naturally
  produces `M(x)·dx/dt = f(x, p, t)` with `M` rank-deficient
  (resistors → algebraic rows). Currently the BE step embeds this
  implicitly; exposing `M` as a first-class object makes
  index-reduction, consistent-IC computation, and stiff-solver
  selection tractable. ~200 LOC, pairs with the IRK item above.

- **Adjoint sensitivity for transient inverse design**. Today
  `eda-mna::sensitivities` runs at the *converged DC OP* only.
  Extending inverse design to transient targets (settling time on
  `spike-sample-hold`, eye-opening on the SAR DAC, jitter on
  `spike-clocks`) currently means unrolled backprop through every
  BE step — memory grows linearly in step count. The continuous
  adjoint method gives `O(1)` memory in trajectory length by
  solving a backward ODE. Substantial: ~500 LOC plus careful
  interaction with checkpointing. Triggers: any transient
  inverse-design or training-through-transient workflow.

- **Dense (continuous) output in `eda-waveform`**. Diffrax solvers
  emit per-step interpolants rather than samples; downsampling and
  sub-step queries become free, and event detection (next item)
  gets exact root times. Hermite interpolation for BE, native
  dense output for IRK. ~150 LOC in `eda-waveform` plus solver-side
  hooks. Triggers: anything that wants accurate edge timing
  without grid refinement.

- **Event detection via root-finding on dense output**. For
  `spike-comparator`, `spike-clocks`, `spike-ripple-counter`,
  threshold crossings are currently approximated by the sample
  grid. Bracketed root-find on the per-step interpolant gives
  exact crossing times with no grid refinement. ~200 LOC, depends
  on dense output. Subsumes part of the "Mixed-signal time-domain
  simulation" item above — the event half of mixed-signal is
  event-detection-on-continuous-trace.

- **SDE solvers for time-domain noise**. The "Noise / variance
  propagation analysis" item above is small-signal (frequency
  domain). The diffrax shape — same `Term` abstraction, different
  driver — extends naturally to thermal-noise-driven transients
  (Euler-Maruyama, Milstein, SRK). Useful for jitter histograms
  on the SAR clock or kT/C noise on sample-and-hold caps.
  ~400 LOC, pairs with the term/solver split.

- **Batched solves via vmap-style compilation**. The "Stochastic /
  Monte Carlo via vmap'd parameter draws" item above already
  flags this; diffrax's contribution is that the *same* compiled
  solver handles batched IC and parameter sweeps, so PDK-corner
  Monte Carlo and surrogate-training data generation share
  infrastructure. Cross-reference, not new work.

### Polish on existing capabilities

- **C64 element-wise kernels** (Add/Sub via 2N-f32 dispatch;
  Mul/Div as dedicated complex kernels). Completes the
  `DType::C64` work landed in scaffolding form in rlx.
  ~250 LOC in rlx-cpu. The 2N-real-block convention validated
  in `spike-ac` continues to work; native C64 is the natural
  successor when convenience > backwards-compat.

- **Wirtinger reverse-mode AD on C64**. Conjugate-aware VJPs
  (`∂/∂z` paired with `∂/∂z̄`). ~300 LOC in rlx-opt. Pairs with
  the C64 kernels above.

- **Multi-output `Op::CustomFn`** in rlx. Today fwd_body has a
  single output; real Newton solvers often want primal +
  residual + diagnostics as a tuple. IR change with knock-on
  effects in AD and lowering. ~200 LOC.

### Cicsim-parity follow-ups

Open items from the harness work above. None block current
flows; they extend reach.

- **BSIM4 in rlx** — implement BSIM4's main equations
  (`Vth_sat` with NDEP/NSD/short-channel terms, mobility,
  `Vds_sat`, channel-length modulation) as rlx graph ops. Once
  this lands, sky130A native MC runs entirely on the rlx side
  without shelling out to ngspice — the proper differentiable-MC
  story on a real PDK. Combined with the existing
  `spike-mosfet-dc::mc` AD-gradient demo this is what closes the
  cicsim-parity gap on differentiation. Multi-week. Triggers: a
  real analog block needing parameter-gradient yield
  optimization where the LEVEL=1 surrogate isn't accurate enough.
- **Real layout-parasitic extraction** — wire `make lpe` (magic)
  into the harness so `corner.view == View::Layout` consumes a
  real `.lpe.spi` instead of the hand-decorated `Rdrain` stub in
  `spike-lelo-ex`. Needs magic in the container or host install
  + the `cicpy sch2mag` step from Carsten's flow. ~1 week. Until
  this lands, Sch-vs-Lay regression numbers are a *demonstration*,
  not a real verification.
- **Continuous dashboard** — `rlx-eda dashboard --watch`
  re-renders `docs/index.html` when `crates/*/docs/` changes.
  ~50 LOC on the `notify` crate.
- **Regression-baseline diffing** — dashboard reads a committed
  `baseline.csv` per testbench (e.g.
  `crates/<crate>/docs/<run>/baseline.csv`) and shows a Δ column
  on each measurement; flags a run as "regressed" if any value
  moved by >σ. ~100 LOC. Pairs with the cicsim-shape CSV emit
  already in tree.
- **Surrogate-fallback fast MC** — when the rlx native solver
  isn't available for a PDK (e.g. sky130A pre-BSIM4-in-rlx),
  train a `spike-surrogate`-style ML model on a smaller ngspice
  MC sample, then sample N×100 from the surrogate for
  distribution shape. Bridges between ngspice's slow accurate MC
  and pure-rlx's fast graph MC. ~300 LOC + a notebook.
- **Differentiable mismatch parameters** — expose Pelgrom AVT as
  a graph input so optimizers can compute `∂yield/∂AVT` and
  trade device area against mismatch headroom in a single solve.
  Builds on the spike-mosfet-dc MC demo's gradient story. ~150
  LOC once BSIM4-in-rlx exists.

### Industry-tool parity (close the `none` items)

The comparison against Cadence/Synopsys/Siemens (see README) leaves
seven cells marked `none` for rlx-eda. Each item below names one of
those gaps and the smallest credible step to flip it from `none` to
"first version present". Several reduce to wire-up work because the
sibling `klayout-rs` workspace already ships the heavy machinery
(`klayout-drc`, `klayout-connect`, `klayout-io`, `klayout-validate`,
`klayout-route`).

#### Layout editor / schematic capture — **KLayout-native export**

- **`eda-viz` GDSII export** via `klayout-io`. Today `eda-viz` renders
  layout to SVG/PNG only — useful for documentation, useless for any
  downstream foundry flow. `klayout-io` already provides GDS read /
  write; the work is a `viz::layout::to_gds(layout, &PdkLayerMap) →
  klayout_geom::Layout` adapter that maps `eda-pdks` typed layers to
  GDS layer/datatype pairs and emits one cell per `Macro`. ~250 LOC
  + a roundtrip test (`spike-divider-layout` → GDS → KLayout
  headless → cell-bbox + layer histogram match). Pairs with the DRC
  item below (DRC consumes the same GDS).
- **OASIS export** as a follow-on — same call site, different
  serialiser inside `klayout-io`. ~30 LOC once GDS is in.
- **Xschem netlist export** for schematic-capture interop. The
  schematic graph already exists in `eda-hir::schematic`; emit
  Xschem `.sch` + symbol `.sym` files so a human can open an
  rlx-eda block in Xschem, edit, and round-trip back through the
  existing SPICE netlist. ~400 LOC. Triggers: any human-in-the-loop
  schematic-review workflow.
- **KLayout `.lyp` consumption end-to-end**. `eda-pdk-ingest` already
  parses `.lyp`; surface the parsed palette in `eda-viz` SVG output
  so screenshots match KLayout colours exactly. ~80 LOC.

#### DRC / LVS / PEX — **wire `klayout-drc` + `klayout-connect`**

- **DRC harness** (`eda-drc` crate). Thin wrapper around
  `klayout-drc` that runs a foundry rule deck on a layout produced
  by `eda-viz::to_gds`. Sky130A's `sky130A.drc` and gf180mcuA's
  `gf180mcu.drc` are public; `klayout-drc` already executes them.
  Output: `DrcReport { violations: Vec<DrcViolation> }` consumable
  in tests. ~300 LOC + per-PDK fixture. Triggers: any spike crate
  emitting real geometry that we want to claim is foundry-clean.
- **LVS via `klayout-connect`**. LVS = compare extracted-from-layout
  netlist against schematic-derived netlist; `klayout-connect` does
  the connectivity extraction. Build an `eda-lvs` crate that emits
  the schematic side from `eda-spice-emit`, the layout side from
  `klayout-connect`, and reports node/device-level mismatches.
  ~500 LOC. Triggers: closing the `spike-lelo-ex` Sch-vs-Lay loop
  with real LVS instead of the current stub.
- **Parasitic extraction (PEX)**. Two tiers: (1) MVP — capacitive-
  only, parallel-plate `C = ε·A/d` per overlapping layer pair
  derived from the PDK stack; (2) Real — shell out to `magic
  ext2spice` for full RC. Tier 1 is ~200 LOC pure Rust; tier 2 is
  a `~/.magicrc` + container hop. Replaces the hand-decorated
  `Rdrain` stub in `spike-lelo-ex`. **Cross-reference: this is the
  "Real layout-parasitic extraction" item already in
  Cicsim-parity follow-ups; expand it there with the tier
  split.**

#### AMS / mixed-signal — **event-driven layer**

Already partially captured under "Specialised analysis types →
Mixed-signal time-domain simulation". Sharpening:

- **`eda-event` crate** — a discrete-event scheduler that consumes
  threshold crossings from the analog transient (via the "Event
  detection via root-finding" item in Diffrax-inspired) and emits
  digital-state transitions. Simulates a Verilog-shaped subset
  (always-blocks on edges, blocking/non-blocking assigns, no full
  HDL parse). ~600 LOC. Triggers: SAR ADC end-to-end, any
  sample-and-hold + comparator + register chain.
- **VerilogA → rlx behavioral compiler** — already present as
  "Verilog-A-style behavioral model compiler" under Architecturally
  bigger. Pairs with this section: VerilogA covers the *device-
  model* side of AMS, `eda-event` covers the *system-simulation*
  side; both are needed for true Spectre AMS / Xcelium AMS parity.

#### RF / EM — **frequency-domain breadth**

Several existing items already form the spine; this subsection
collects them and names what's missing.

- **S-parameters / two-port** — already in "Specialised analysis
  types"; promote to higher priority once an RF spike crate
  (`spike-lna` already exists but is a stub) has a forward model
  that needs it.
- **Harmonic balance** — already in "Specialised analysis types".
- **Noise / variance propagation** — already in "Specialised
  analysis types"; flagged as the noise-figure prerequisite.
- **EM coupling (3D field solver)**. Out of scope for in-house
  implementation — the right move is an adapter to an external
  open-source solver (`openEMS` FDTD, `palace` FEM) that consumes
  geometry from `eda-viz::to_gds` and returns S-parameter touchstone
  files re-imported into `spike-ac`. ~400 LOC of adapter + parser,
  no own solver. Triggers: any RF block where layout coupling
  matters (`spike-lna`, future PLL).

#### P&R, STA, power — **OpenROAD / OpenSTA adapters**

- **`eda-extern-openroad` adapter**. Parallels `eda-extern-ngspice`:
  drives OpenROAD's TCL flow (floorplan → place → CTS → route)
  against a LEF/DEF emitted from `eda-viz` + `eda-stdcells`. Returns
  a routed DEF that can be re-rendered or shipped to GDS. ~500 LOC.
  Triggers: any cell-based digital block big enough that hand-
  routing in `eda-viz` is silly (>~50 cells).
- **`eda-extern-opensta` adapter**. STA on the same routed DEF +
  Liberty + SPEF. Returns slack/setup/hold reports as Rust structs
  consumable in tests (`assert!(min_slack > 0.0)`). ~300 LOC.
  Triggers: anything with a clock domain.
- **Power analysis (dynamic + leakage)**. Two paths: (1) post-
  simulation — integrate `i(vdd)·v(vdd)` from a transient run
  produced by `eda-mna` or ngspice (~80 LOC, available now);
  (2) activity-driven — switching activity from `eda-event` ×
  Liberty cell power tables (~250 LOC, depends on Liberty
  ingestion). Tier 1 is a free win once the SAR ADC transient lands.

#### Reliability / aging / EM (electromigration)

- **Electromigration current-density check**. Per-net `J = I/W·t`
  derived from a transient + the `eda-pdks` layer stack; flags any
  segment exceeding the foundry `Jmax`. ~200 LOC, pure post-
  processing. Cross-checks against KLayout's existing
  current-density DRC where the foundry deck includes one.
- **BSIM4 aging (HCI / NBTI / BTI)**. Stress-time-accumulating
  `ΔVth(t)` extension to the rlx BSIM4 model (see "BSIM4 in rlx"
  in Cicsim-parity follow-ups). Once `ΔVth` is a graph parameter,
  AD gives `∂lifetime/∂device_size` for free — a story commercial
  flows handle in separate Spectre RelXpert / PrimeSim Reliability
  passes. ~400 LOC on top of BSIM4.
- **Self-heating / thermal coupling**. Per-device temperature as a
  solved unknown coupled to power dissipation; thermal RC network
  on the substrate. ~500 LOC. Triggers: any high-current analog
  (LDOs, bandgaps).

#### Foundry sign-off

Acknowledged out-of-scope: sign-off requires foundry-licensed decks,
calibration data, and tool qualification. The realistic terminal
state is "produces artifacts a sign-off team can ingest" — i.e.
GDSII (above), SPICE netlist with parasitics (PEX above), Liberty
+ SPEF for digital, IBIS for IO. Each falls out of items above
once they exist; no separate workstream warranted.

### PDK breadth — fill out `eda-pdks`

`eda-pdks` currently feature-gates `sky130` and `gf180mcu` only,
even though `rlx-eda-cli pdk install` can fetch more via ciel.
Expand the typed-layer-map crate to cover what the install path
already supports:

- **`ihp-sg13g2`** (IHP 130 nm BiCMOS, open-source). Layer property
  file at `IHP-Open-PDK/ihp-sg13g2/libs.tech/klayout/tech/sg13g2.lyp`
  once the repo is cloned to `/Users/Shared/mtl/ihp-sg13g2`. SPICE
  side already known to `rlx-eda-cli doctor`. Adds first BiCMOS
  device family — bipolar HBTs absent from sky130/gf180. ~150 LOC
  build.rs entry + tests. Triggers: `spike-lna` (low-noise amp,
  bipolar-friendly).
- **`gf180mcuA` / `gf180mcuB` / `gf180mcuC` / `gf180mcuD`**. Today's
  feature is unsplit `gf180mcu`; ciel exposes the four sub-variants.
  Differentiate with feature flags `gf180mcu_a..d` while keeping
  `gf180mcu` as default-A for back-compat. ~80 LOC build.rs.
- **`sky130B`**. The B variant differs from A in metal stack and
  device offering; ciel handles both. Add `sky130b` feature flag
  alongside today's `sky130` (= sky130A). ~80 LOC.
- **`asap7`** (academic 7 nm predictive). Useful for digital-flow
  benchmarking against published OpenROAD numbers. Layer map ships
  with the asap7 GitHub repo. ~150 LOC.
- **`gpdk45`** (Cadence academic 45 nm). Common in coursework /
  papers; license-free for academic use. ~150 LOC.
- **Photonic PDKs** for `spike-waveguide-block`: **`AIM-PDK`**
  (open photonic PDK from AIM Photonics) and **`IHP-SG13G2-photonic`**
  (the photonic add-on layers on IHP). Each ~200 LOC + a layer
  abstraction split, since photonic layers (waveguides, grating
  couplers) don't cleanly fit the current
  metal/poly/diff/well taxonomy. Triggers: photonic spike work
  past today's bare waveguide.
- **Out of scope (foundry-NDA)**: TSMC, GF, Samsung, Intel
  production nodes — would require licensed PDKs. No work item.

`eda-pdks` becomes the test of whether the typed-layer abstraction
actually generalises; if a fifth PDK can't fit without trait
surgery, the abstraction needs work — that's a useful forcing
function, not failure.

### Differentiable surrogates and PINNs

Two new scaffolded crates — `eda-nn` and `eda-pinn` — extend the
differentiable-physics stack with NN-shaped consumers. The premise is
that `eda-mna` already gives AD through Newton + BE transient, and
MOSFET physics is a graph subgraph (`spike-mosfet-dc::id_subgraph`),
so neural surrogates and physics-informed losses compose with existing
machinery instead of replacing it.

- **`eda-nn`** (scaffolded) — minimal `Linear` / `Mlp` /
  `ParamSpec` / `Adam` / xorshift32 `Rng` over rlx. Lifts the
  proven pattern from `spike-surrogate` (same Adam, same Glorot
  init, same flat-weight slicing convention) into a reusable
  layer so PINN trainers and future surrogate spikes share one
  well-understood path. Three unit tests cover param-count
  consistency, bias-zeroing in Glorot init, and Adam convergence
  on a quadratic. Deliberately narrow: no dataloader, no
  `Module` trait, no checkpointing — add only when a second
  consumer needs it.

- **`eda-pinn`** (scaffolded) — physics-informed training on top
  of `eda-nn` + `eda-mna`. Two modules:

  - `mosfet_surrogate` — drop-in replacement for
    `spike_mosfet_dc::id_subgraph` with the same call shape
    `(g, vgs, vds, vth, kp, lam) → NodeId`, routed through a
    learned MLP. Same signature means existing consumers
    (`spike_divider_block::Mosfet`, `spike-cmos-gates`)
    swap behind a feature flag with no further change. Loss
    convention: data MSE against ngspice/analytic samples plus
    a shape-anchor regulariser against the LEVEL=1 form
    (`λ · ‖id_nn(x) − id_smooth(x)‖²` at sampled points) to
    keep the surrogate honest in data-sparse regions. Status:
    `Mlp` allocation + `TrainConfig` defaults are real; the
    `id_subgraph` body needs the rlx scalar-stacking
    convention picked (concat vs stack) before it's callable.

  - `kcl_residual` — PINN-style loss that lets an NN predict
    node voltages and supervises with `Σ KCL_i(v_pred)²`.
    Reuses `eda_mna::build_residual_graph` so physics is not
    duplicated. Needs a small `eda-mna` extension —
    `build_residual_graph_at(circuit, v_pred)` — that
    substitutes NN outputs into the unknown-net input slots
    of the residual graph. ~20 LOC change in `eda-mna`,
    unlocks the entire PINN family.

  Status: scaffold; both module bodies are `unimplemented!()`
  with cross-references back to this section.

**Validation pyramid** (mirrors the convention every other crate
follows):
- *Analytic* — surrogate `id_nn` vs `id_smooth` (LEVEL=1) on
  a (Vgs, Vds) grid, bounded relative error.
- *FD* — finite-difference gradient of `id_nn` w.r.t. inputs,
  parity-checked against AD (matches the FD-vs-AD pattern in
  `spike-diode/finite_difference.rs`, `cpu_sqrt_grad.rs`, etc.).
- *Ngspice* — full circuit (e.g. `spike-divider-mna` or
  `spike-cmos-gates::Inverter`) run with surrogate vs. ngspice
  golden, with the surrogate swapped in via feature flag.
- *Behavioural* — re-run `comparator_sizing_ad` with the
  surrogate. If gradients still drive M1_Vth toward target,
  AD has flowed end-to-end through Newton + BE + surrogate.
  This is the headline regression test.

**Staging.** First PR: `eda-nn` filled in (already
compilable), plus the surrogate trainer wired up using a
sweep of `spike-mosfet-dc`'s analytic `id_subgraph` as ground
truth (no ngspice yet — the analytic form is the cheapest
oracle and lets us validate the trainer in isolation). Second
PR: ngspice ground-truth via `eda-extern-ngspice` device-sweep
fixtures, sky130 NMOS at TT corner. Third PR: KCL-residual
PINN — needs the `build_residual_graph_at` hook in `eda-mna`
first, then trains on `spike-rc-transient` or another small
linear circuit before scaling to MOSFET-bearing nets.

**What this is *not*.** Not RL (gradient-based sizing
already wins via existing `comparator_sizing_ad` template).
Not transformers/diffusion (data-hungry, opaque, throw away
encoded physics). Not GNNs (no netlist-graph layer in
`eda-hir`/`eda-tile` yet — premature). The fit here is
specifically the methods that *reuse* the differentiable
physics substrate.

### Architecturally bigger

- **Verilog-A-style behavioral model compiler**. Declarative
  component models (textual VA description → Rust trait impl
  via macro). Production-quality EDA needs this. Multi-week
  project; would unblock arbitrary device models without
  hand-coding each one.

- **Topology-aware AD beyond flat sensitivities**: detect
  symmetries (matched pairs, identical fingers) and exploit
  them in the Jacobian assembly. Real Circulax production
  teams care about this for scaling to industrial circuits.

- **Sparse + hierarchical AD combined**. The endgame: each
  hierarchical block exposes its sparse residual contribution;
  the top-level assembler stitches them via a single sparse
  factorization with full hierarchy preserved. Pairs everything
  above into one architectural piece. Real Circulax-grade
  workflow. ~2000 LOC.

## Recommended next step

**Stop adding rlx primitives and run a SAR ADC end-to-end** with
the existing toolchain. The first thing that breaks tells you
exactly what to build next, and you'll have an actual benchmark
for whichever item above to prioritise.

The spike-* crates already in tree (`spike-comparator`,
`spike-comparator-cmos`, `spike-sample-hold`, `spike-sar-register`,
`spike-sar-logic`, `spike-dac-r2r`, `spike-clocks`,
`spike-clock-decoder`, `spike-output-door`, `spike-ripple-counter`)
suggest this is already the trajectory — they're the SAR ADC's
sub-blocks. A working `spike-sar-adc` end-to-end:

1. **Pressure-tests sparse solver** (the full ADC has hundreds
   of nodes; DenseSolve will struggle).
2. **Pressure-tests hierarchical sensitivities** (DAC +
   comparator + register + clock + output-buffer is naturally
   multi-level).
3. **Pressure-tests switching-circuit Newton** (the comparator's
   regenerative latch is convergence-hard; current damped Newton
   may need globalisation).
4. **Pressure-tests long mixed-signal transients** (a full
   N-bit conversion = N bit-cycles × clock period, with both
   analog cap settling and digital edges).

Whichever of those breaks first names the next milestone. Until
one of them does, additional rlx primitives are speculative.
