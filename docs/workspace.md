# Workspace overview

This workspace is organized under `crates/*` and currently contains
33 crates. The composition hierarchy and dependency layering are
covered in [`architecture.md`](architecture.md); this document is
the per-crate inventory.

## Core libraries

| Crate | Purpose |
| --- | --- |
| `eda-hir` | User-facing high-level IR traits for blocks, schematics, and layouts. |
| `eda-mna` | Modified Nodal Analysis assembly from behavioral devices into solvable circuit residuals. |
| `eda-spice-emit` | SPICE netlist emission for cross-validation in external simulators. |
| `eda-waveform` | Common waveform IR plus Nutmeg/CSV/VCD read-write and plotting utilities. |
| `eda-viz` | SVG/PNG rendering for layout and symbolic schematic views. |
| `eda-validate` | Numerical validation helpers: tolerance checks, finite differences, grad checks, trace comparison. |

## Composition hierarchy

Circuits are organized along the standard semiconductor stack —
`Device → Cell → Macro → Tile → Core → Die → Reticle → Wafer → Lot`.
All rungs sit on top of `eda-hir`'s `Block` base trait (`Hash + Eq +
name()`); each rung adds only the obligation specific to that level.
The hierarchy traits live in
[`../crates/eda-hir/src/hierarchy.rs`](../crates/eda-hir/src/hierarchy.rs);
`Tile` keeps its richer abutment contract in
[`../crates/eda-tile/src/tile.rs`](../crates/eda-tile/src/tile.rs).

| rung | trait | obligations beyond `Block` | concrete consumers today |
| ---: | --- | --- | --- |
| 1 | `Device: Block` | `terminal_names()` | every `MnaDevice` impl (Mosfet, Diode, Resistor, Capacitor, …) |
| 2 | `Cell<P>: Block + Layout<P>` | `pins()` | `eda-stdcells::StdCell` (foundry sky130 cells) |
| 3 | `Macro<P>: Block + Layout<P>` | `pins()`, `boundary()` | `spike-divider-block`, `spike-sar-adc`, `spike-waveguide-block` (de facto macros) |
| 4 | `Tile<P>: Block + Layout<P>` *(in `eda-tile`)* | `pitch()`, `rails()`, `edge_ports()` | `spike-tinyconv-tile::Mac8x8Tile` |
| 5 | `Core<P>: Block + Layout<P>` | `io_pins()` | — *(speculative; no consumer yet)* |
| 6 | `Die<P>: Block + Layout<P>` | `outline()`, `io_pads()`, `scribe_clearance_dbu()` | — *(speculative)* |
| 7 | `Reticle: Block` | `field_size()`, `fields()` | — *(speculative)* |
| 8 | `Wafer: Block` | `diameter_mm()`, `edge_exclusion_mm()`, `step_pattern()` | — *(speculative)* |
| 9 | `Lot: Block` | `wafer_count()`, `process_run_id()` | — *(speculative)* |

`Macro` is the standard EDA term for what the literature also calls a
"block" or "hard macro"; we use `Macro` because `Block` is already the
crate-wide identity-and-equality contract every composable thing
satisfies. Per `eda-hir`'s philosophy ("traits earn their place"), the
speculative rungs carry only the minimal contract a future consumer
would need; richer obligations are added when a real use case lands.

## External simulator adapters

| Crate | Purpose |
| --- | --- |
| `eda-extern-ngspice` | ngspice invocation and result parsing for `.op`, transient, and AC-style validation flows. |
| `eda-extern-ltspice` | LTspice adapter with a shape aligned to the ngspice invoker and shared waveform parsing. |

## PDK pipeline

| Crate | Purpose |
| --- | --- |
| `eda-pdk-ingest` | Parses foundry layer property files (for example `.lyp`) into Rust structures. |
| `eda-pdks` | Build-time generated PDK definitions, feature-gated per foundry. |

## Design automation

| Crate | Purpose |
| --- | --- |
| `eda-pnr` | Differentiable place-and-route. Positions exposed as `rlx_ir::Param[B, N]`; HPWL + density are one differentiable loss. |
| `eda-tile` | Tile abutment contract: pitch, rails, edge ports. Consumed by `spike-tinyconv-tile`. |
| `eda-stdcells` | Liberty-backed standard-cell library views (sky130). |
| `eda-drc` / `eda-em` / `eda-pex` | Design-rule, electromigration, and parasitic-extraction surfaces (mostly stubs at this point). |

## Harnesses and utilities

| Crate | Purpose |
| --- | --- |
| `eda-trace` | Markdown / SVG / CSV report generation; `optim::AdamState`, `LrSchedule`, `BetaSchedule`, `default_device()`. |
| `eda-sim-harness` | Shared simulation-harness scaffolding for cross-simulator triangulation experiments. |
| `eda-extract` | Layout → SPICE extraction (RcDivider round-trip ground truth). |
| `eda-container` | `DockerRun` builder and image registry shared by `eda-extern-ngspice` and `eda-bench-tinyconv`. |
| `eda-bench-tinyconv` | TinyConv-MNIST silicon-flow benchmark umbrella (in-house sky130 + ORFS Docker + FPGA). |

## Spike crates

Spike crates are intentionally narrow experiments that prove one
capability, stress one API edge, or triangulate one model.

| Crate | Focus |
| --- | --- |
| `spike-divider` | Divider forward model and AD gradient sanity checks. |
| `spike-divider-mna` | Divider represented as MNA stamps solved through linear-system ops. |
| `spike-divider-layout` | Divider physical layout using typed PDK layers, ports, and routing. |
| `spike-divider-block` | Divider as composable typed blocks implementing HIR layout traits. |
| `spike-diode` | First nonlinear DC circuit (resistor + diode) with solver validation. |
| `spike-rc-transient` | Backward-Euler transient flow for an RC low-pass. |
| `spike-pulse-rc` | Time-varying pulse-driven RC transient validation. |
| `spike-ac` | AC small-signal/Bode flow via complex MNA in real arithmetic. |
| `spike-mosfet-dc` | Shichman-Hodges (SPICE level-1 style) NMOS DC model validation. |
| `spike-mosfet` | CMOS inverter circuit as first MOSFET composition target. |
| `spike-cmos-gates` | Gate-level CMOS cells (inverter/NAND/AND and sequential building blocks). |
| `spike-comparator` | Behavioral comparator model with smooth differentiable transfer. |
| `spike-dac-r2r` | 8-bit R-2R DAC topology and analysis checks. |
| `spike-sample-hold` | CMOS transmission-gate sample-and-hold behavior. |
| `spike-ripple-counter` | Asynchronous ripple counter for digital timing/control experiments. |
| `spike-clock-decoder` | Counter-state decoder that produces SAR timing strobes. |
| `spike-clocks` | Full SAR clock generator composed from counter + decoder. |
| `spike-sar-register` | N-bit successive-approximation register behavior and wiring. |
| `spike-sar-logic` | SAR logic state-machine placeholder/stub for planned expansion. |
| `spike-sar-adc` | Const-generic N-bit SAR ADC composed of SH + Comparator + DAC + SAR logic. |
| `spike-output-door` | Parallel latch stage for SAR output capture and hold. |
| `spike-surrogate` | Surrogate-model workflow for optimization over expensive simulations. |
| `spike-triangulate` | End-to-end cross-simulator triangulation pipeline demo. |
| `spike-waveguide-block` | Photonic block and optical-PDK trait experiments. |
| `spike-lna` | RF inductively-degenerated cascode LNA at ~2.4 GHz with closed-form 2-port S-params on rlx graph. |
| `spike-tinyconv-tile` | 8×8 MAC array implementing `Tile<P>`; the workspace's only active `Tile` consumer. |
| `spike-tinyconv-array` | Tiled CNN compute fabric built on `spike-tinyconv-tile`. |
| `spike-pinn-rc` / `spike-pinn-diode` / `spike-pinn-sar` / `spike-pinn-sar-mc` | Pre-registered PINN-vs-polynomial-regression surrogate experiments. |
| `spike-dado-r2r` | DADO (decomposition-aware EDA) on R-2R DAC discrete sizing. See [`../crates/spike-dado-r2r/docs/STORY.md`](../crates/spike-dado-r2r/docs/STORY.md). |
| `spike-dado-sar` | DADO at the SAR ADC system level — analytical noise budget vs ngspice transient, head-to-head. See [`../crates/spike-dado-sar/docs/STORY.md`](../crates/spike-dado-sar/docs/STORY.md). |

## Repository layout

```text
rlx-eda/
  Cargo.toml
  README.md
  PLAN.md
  docs/                 # this directory
  scripts/dado          # one-shot DADO experiment driver
  Justfile              # common dev commands
  docker/               # one Dockerfile per shipped image
  crates/
    eda-*/
    spike-*/
```

## Dependencies outside this repo

This workspace uses local path dependencies to sibling repositories:

- `../rlx` for `rlx-ir`, `rlx-opt`, `rlx-runtime`, `rlx-mlx`.
- `../mtl/klayout-rs` for `klayout-*` crates.

Without those paths available, full workspace builds will fail.
