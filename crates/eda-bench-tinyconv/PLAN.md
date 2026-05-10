# TinyConv-MNIST Silicon Plan

End-to-end code-defined silicon flow for the TinyConv-MNIST model that
already runs on `rlx-fpga`. Lives entirely inside this workspace. Uses
`rlx-eda`'s differentiable HIR + sky130 PDK for the design path, and
Yosys/OpenROAD-in-docker as a ground-truth oracle for benchmarking — not
as a production flow. Functional accuracy on the MNIST test set is the
load-bearing metric; physical metrics ride on top.

## Status (live)

**Pipeline runs end-to-end** in `cargo test` against the mock sky130
library — manifest → tile/array layout → inhouse measurement → markdown
report → optimization loop. **118 tests pass** across the five new
crates with no foundry GDS / docker / ngspice dependencies. A
runnable demo binary
(`cargo run -p eda-bench-tinyconv --features demo-bin --bin demo_report`)
emits `target/bench/demo/report.md` with real Liberty-derived
foundry numbers in ~2 seconds.

| Layer | What works | What's stubbed |
|---|---|---|
| `eda-stdcells` | sky130_fd_sc_hd ingest (`ScHdLibrary::load` with PDK-install probing), Liberty parser, mock cells, area aggregation (`sum_area_um2_x1000`) | foundry GDS not checked out |
| `eda-tile` | `Tile<P>` trait, `RailSpec`, `tile_grid` composer with optional PDN gate | thermal model |
| `spike-tinyconv-tile` | `Mac8x8Tile` (Block + Layout + Tile + DcBehavioral), 4-row Digital floorplan with 202 cell instances, closed-form energy/delay/area + noise model, **`area_baseline_um2` overrideable from Liberty sum**, `cell_inventory` | analog topologies (CR/CM) |
| `spike-tinyconv-array` | `ArrayBlock::layout` (tile_grid composition), `cell_inventory` cascading, **`lower(model, config)` from `rlx_fpga::model::Model` + `weight_count` + `min_required_tiles`** | controller FSM placement |
| `eda-bench-tinyconv` | reproducibility manifest, yield gate, bisection, ORFS JSON parser, inner Adam loop with full 4-term loss + accuracy gate, outer brute-force loop, `InhouseBackend` (tile + array scope), **`LossWeights::with_inhouse_baseline` builder**, markdown reporter with file output, demo binary | ORFS docker not pinned, no FPGA backend body, calibration constants are placeholders |

**Cross-cutting requirements** (from "Cross-cutting requirements" below):

| # | Status |
|---|---|
| 1 — Reproducibility manifest | ✓ `Manifest::capture` + JSON round-trip |
| 2 — OpenRCX parity | scaffolded; activates with `bench-orfs` feature |
| 3 — Yield gate | ✓ `YieldGate::evaluate` + `pass_rate` |
| 4 — PDN check | ✓ `current_density_check` + tile_grid integration |
| 5 — Failure bisection | ✓ `bisect::bisect` + Report rendering |
| 6 — Cross-backend energy definition | type-level (Option<f64>); inhouse populates area only |

**Co-design loop end-to-end**: 4-term loss
`α·P + β·delay + γ·area + λ·max(0, k_acc·σ − ε)` differentiable through
one `(w_l_n, w_l_p, vdd)` Param triple, gradient-stepped via Adam,
clamped to bounds, gated by `OptError::InnerDiverged` on NaN/Inf.

## Goal

Demonstrate that a single Rust source-of-truth can produce:

1. A trained MNIST classifier (already exists, `rlx-cortexm` weights).
2. An FPGA bitstream (already exists, `rlx-fpga`).
3. A sky130 GDS layout with parametric, differentiable analog MAC tiles
   tiled into a TinyConv array, validated bit-for-functional against the
   FPGA path under PVT.

…and that the inner design parameters (tile sizing, supply, parallelism,
weight bit-width) can be co-optimized with quantization-aware training so
the silicon ships with a *known* accuracy / energy / area Pareto trade.

## Repo organization

Stays in `rlx-eda`. Convention is explicit (`circuit logic stays in
rlx-eda`), every dependency already lives here, and the spike-crate
pattern (`spike-dado-r2r`, `spike-sar-adc`) is the established home for
research-scale efforts. Heavyweight bench dependencies (ORFS docker,
sky130 download, FPGA toolchain) sit behind Cargo features
(`bench-orfs`, `bench-fpga`) and a `just bench-tinyconv` target so
default CI is not affected.

## Crate plan

### New crates (in dependency order)

**`eda-stdcells`** — thin wrapper. Ingests `sky130_fd_sc_hd` via
`eda-pdk-ingest`, exposes a `StdCell` trait shim that implements
`Block + Layout<Sky130> + Schematic<Sky130>` over the foundry library.
*Not* a from-scratch parametric library. This is what the digital glue
(controller FSM, address decoders, ping-pong control) lowers to.
Justification: matches what ORFS will use anyway, so any in-house ↔ ORFS
divergence is in floorplan, not cell library.

**`eda-tile`** — pitch-matched abutment + power-rail helpers.

```rust
pub trait Tile: Block + Layout<P> {
    fn pitch(&self) -> Vec2;
    fn rails(&self) -> RailSpec;
    fn edge_ports(&self, side: Side) -> &[EdgePort];
}
pub fn tile_grid<T, P>(tile: &T, nx: usize, ny: usize, ...) -> CellId;
```

Plus a coarse current-density check on power straps (item #4 below).
Sits on `klayout_route::ManhattanPlanner` for inter-tile bus routing.

**`spike-tinyconv-tile`** — the custom analog MAC tile. Carries
`Block + Layout<Sky130> + Schematic<Sky130> + DcBehavioral` (the four
HIR traits) plus the full validation pyramid (analytic + FD + ngspice
tt/mc, using the `mc_*_switch` override pattern). Parametric over
(`w_l_n`, `w_l_p`, `vdd`, `pitch`, `weight_bits`).

Critically, it also exposes a **noise model**: mean and σ on MAC output
as a closed-form function of the optimized parameters, calibrated
against ngspice. This is what the inference-time validator consumes so
the inner ML loop never has to invoke SPICE per image.

**`spike-tinyconv-array`** — full TinyConv silicon. Lowers
`rlx_fpga::ir::Graph` (yes, reuse the FPGA IR) to a tiled instantiation
of `spike-tinyconv-tile` plus a controller FSM built from
`eda-stdcells`. Two lowerings now share the IR: `rlx-fpga` →
SystemVerilog, `spike-tinyconv-array` → `Block` composition. Driven by
`rlx_fpga::tune` for the discrete architectural sweep.

**`eda-bench-tinyconv`** (this crate) — three-backend bench harness
with two metric arms. See *Bench harness* below.

### Reused crates

`eda-hir`, `eda-pdks` (sky130 already), `eda-pdk-ingest`,
`eda-sim-harness`, `eda-spice-emit`, `eda-validate`, `eda-vfit`,
`eda-waveform`, `klayout-rs` (entire stack), `rlx-fpga::ir` +
`rlx-fpga::reference` + `rlx-fpga::tune`.

## Validation: five functional levels + the analog pyramid

The analog pyramid (analytic → FD → ngspice) is the per-tile electrical
contract — same as every other rlx-eda block. On top of that, the chip
has to *classify MNIST* at every backend stage:

| Level | Catches | Cost / img | When |
|---|---|---|---|
| L1 — Q31 reference (Rust) | algorithmic baseline; defines "correct" | µs | every commit |
| L2 — RTL functional sim | RTL ≠ reference (codegen bug) | ms | every commit, 200-img golden subset |
| L3 — Gate-level sim w/ SDF back-annotation | synthesis bugs, X-prop, latch inference | seconds | per ORFS run, golden subset |
| L4 — Mixed-signal post-layout sim w/ parasitics | analog MAC glitches, IR drop, near-threshold corruption, supply coupling | minutes | tagged candidates, ~20 representative images |
| L5 — PVT × MC sweep on full 10k test set | corner-induced accuracy collapse, yield estimate | hours; FPGA + ORFS only | release gates |

L1+L2 are essentially free today (FPGA path covers them). L3 comes from
ORFS. L4 is the new expensive one — gated to small image sets. L5 is
why FPGA stays in CI: it is the only backend fast enough to validate
accuracy at scale, and silicon trusts FPGA, FPGA trusts the Rust
reference, physical metrics ride on top.

## Co-design optimization

Two loops, two scopes, two optimizers:

| Loop | Scope | Variables | Optimizer | Validated by |
|---|---|---|---|---|
| Inner (continuous) | one `Mac8x8Tile` | (`w_l_n`, `w_l_p`, `vdd`, bias points) | autodiff via `DcBehavioral` + Adam (mirrors `spike-divider-block` 151-iter inverse-design) | analytic + FD + ngspice tt/mc + ORFS PEX |
| Outer (discrete) | full TinyConv graph | (`weight_bits ∈ {2,4,8}`, parallelism, pipeline depth) | DADO (mirrors `spike-dado-r2r`'s tabular categorical on chain JT; per-layer error is the natural decomposition) | full ORFS run on each candidate top config |

**Loss with accuracy gate** (mandatory):

```
loss = α·energy + β·delay + γ·area + λ·max(0, acc_drop_pp − ε)
```

Accuracy is evaluated by propagating the tile's calibrated noise model
(σ on MAC output) into the FPGA-speed inference path — re-run all 10k
MNIST images per candidate, no SPICE in the inner loop. SPICE
periodically validates the noise model itself.

This is also where the differentiability story pays off twice: the same
loss can drive **quantization-aware training** with the tile noise model
in the loop. Co-design the network and the silicon, not in two phases.
That is the actual research contribution; pure layout of a fixed network
is not.

## Bench harness layout

```
eda-bench-tinyconv/
├─ docker/
│  ├─ Dockerfile         # ORFS + sky130A + Yosys + OpenSTA + magic + netgen
│  └─ run_orfs.sh        # in: config.mk + verilog; out: metrics.json
├─ src/
│  ├─ backends/
│  │  ├─ inhouse.rs      # rlx-eda Block → GDS (klayout_io); parasitics via OpenRCX
│  │  ├─ orfs.rs         # SystemVerilog (rlx-fpga emit) → docker → parsed metrics.json
│  │  └─ fpga.rs         # rlx-fpga emit → nextpnr-ecp5 --report → parsed
│  ├─ metrics/
│  │  ├─ physical.rs     # area_um2, power_mw, max_freq_mhz, wns_ns, parasitic_cap_ff, peak_temp_c, energy_pj_per_inference
│  │  └─ functional.rs   # top1_acc, per_class_acc, divergence_first_layer, per_pvt_corner
│  ├─ manifest.rs        # PDK commit, ORFS digest, ngspice ver, weight hash, optimizer seed, Cargo.lock hash — every run
│  ├─ bisect.rs          # localize functional drift (image, layer, tile instance)
│  └─ report.rs          # markdown + PNG: backend × metric arm × PVT
└─ docs/ground_truth.md  # parity bands, what counts as agreement
```

**Two metric arms, not one.** Physical (area, power, freq, parasitics,
thermal). Functional (top-1 accuracy, per-class, per-PVT, yield). Both
required to pass before any candidate is reported.

**Three backends** because each plays a distinct role:

- **In-house** — the design under test. Differentiable, fast iteration.
- **ORFS** — physical ground truth. Industry-validated PnR, OpenSTA
  timing, OpenRCX parasitics, OpenROAD PDN. Slow (minutes).
- **FPGA** — functional ground truth at scale. Only backend that can run
  the full 10k test set under any candidate config in seconds.

**Parity bands** (release gate): area within ±15%, max freq within ±20%,
dynamic power within ±25% of ORFS on the same lowered IR. Functional
accuracy must match FPGA exactly on L1–L3 and within ε on L4–L5.

## Cross-cutting requirements (load-bearing)

1. **Reproducibility manifest** in every bench run — sky130 commit, ORFS
   image digest, ngspice version, weight blob hash, optimizer seed,
   Cargo.lock hash. Without this, "1.2 mm² Tuesday → 1.4 mm² Friday"
   isn't debuggable. ~50 LoC, do it before anything else.

2. **Parasitic extraction parity, not metric parity.** Don't pretend
   `klayout_geom::Region` is an independent extractor. Delegate to
   OpenRCX in the bench harness so both flows use the same RC stack.
   Document the residual.

3. **Yield gate, not mean accuracy.** Release condition is
   `P(top-1 ≥ 97% | PVT × MC) ≥ 99%`. Mean is informational; yield is
   the gate.

4. **PDN / IR-drop on the array.** Single-tile sim won't catch supply
   collapse when 256 MACs fire in lockstep. ORFS PDN analysis on the
   array; coarse current-density check on power straps in `eda-tile`.

5. **Failure-bisection.** When L5 reports an accuracy regression at a
   corner, the bench checkpoints per-layer activations from each backend
   and localizes the divergence to (image, layer, tile instance)
   automatically. Same shape as cicwave waveform diffs but on tensor
   outputs.

6. **Cross-backend energy definition.** One operational definition
   ("switching + leakage over one-image latency window, integrated"),
   every backend projects onto it, residuals documented. They do not
   compose otherwise.

## Strategy decisions baked into v1

- **Default MAC topology = `MacTopology::Digital`.** Synthesized
  integer MAC; matches `rlx-fpga`'s existing INT8×INT8 mac.rs so the
  IR lowering is direct, ORFS handles it as ground truth without
  mixed-signal trickery, and all four trait obligations have
  unambiguous bodies. Analog topologies (`ChargeRedistribution`,
  `CurrentMode`) are declared as `MacTopology` variants in
  `spike-tinyconv-tile/src/topology.rs` but bodies stay
  `unimplemented!()` until someone signs up — switching is a
  one-line `Mac8x8Tile::with_topology(...)` call, no API churn.

- **Default PDK = sky130** via `eda-pdks::Sky130`. "Pick PDK later"
  is free: `Layout` and `Tile` impls on `Mac8x8Tile` are generic over
  any `MosfetPdk` (`spike-divider-block`), which already has sky130
  + gf180mcu impls. New foundries slot in by adding one `MosfetPdk`
  impl, no edits to `spike-tinyconv-tile`.

- **Single-shot, not streaming.** TinyConv-MNIST as a "classify one
  image" chip, matching the FPGA today. Streaming variant is a
  different floorplan and is out of scope for v1.

- **Hard area budget = 4 mm²** (sky130, placeholder — refine after
  first single-tile bench). Hard peak-power budget = 50 mW. The
  optimizer needs constraints, not just objectives.

- **External baselines anchored from day one.** Bench reports include
  the same units as ARM Ethos-U, ST NanoEdge, and MLPerf-Tiny entries,
  plus Eyeriss as an academic reference. Without anchors the chip
  exists in a vacuum.

## Build order

1. `eda-stdcells` — sky130_fd_sc_hd ingest + `StdCell` shim. (1–2 wk)
2. `eda-tile` — `Tile` trait + `tile_grid` + power-strap density check. (1 wk)
3. `spike-tinyconv-tile` — four HIR traits + analog pyramid + noise
   model. **First ML target.** Inner loop validates against accuracy
   from day one via a noise-injected wrapper around `rlx_fpga::reference`
   (full array exists yet). (3–4 wk)
4. `eda-bench-tinyconv` docker + ORFS backend, validated on a
   single-tile testbench. **Functional arm runs L1–L3 on golden subset
   before any physical numbers are reported.** Reproducibility manifest
   in place. (2 wk)
5. `spike-tinyconv-array` — IR lowering + controller FSM. (2–3 wk)
6. Co-design loop wired end-to-end (inner Adam + outer DADO + accuracy
   gate). (1–2 wk)
7. Full bench: three backends × two metric arms × PVT corners + yield
   computation + bisection + baseline anchors. (2–3 wk)

Steps 3+4 are the de-risking pair. Tile-level parity with ORFS before
committing to array work. If parity holds at tile scope the array is
mostly mechanical lowering. If it doesn't, we learn cheaply.

## v1 / v1.5 / deferred

**v1** (target: working single-tile flow + small-array proof):

- `eda-stdcells`, `eda-tile`, `spike-tinyconv-tile`
- Bench harness with all three backends and both metric arms
- Reproducibility manifest, yield gate, accuracy-gated co-design
- Single-shot TinyConv on a 4×4 tile array (not full MNIST resolution)
- LVS via magic+netgen in docker

**v1.5** (full MNIST silicon + production-grade gating):

- Full `spike-tinyconv-array` at MNIST resolution
- ORFS PDN analysis on the array
- Failure-bisection in the bench harness
- Published-baseline comparison table in the report

**Deferred** (called out so they don't surprise us):

- DFT (scan chains, BIST) — needed for tape-out, not for methodology
- Multi-clock / CTS — single domain in v1
- Adversarial / OOD test data (MNIST-C, rotated MNIST) — useful QAT
  regression catch but not before basic flow exists
- Photonic backend — `eda-pdks` already supports the PDKs and
  `spike-waveguide-block` is right there; flag for a separate research
  thread, do not build now
- Silicon-in-loop — design `Backend` trait so a `SiliconBackend` can
  drop in later, do not implement
- Throughput / streaming variant — different chip; revisit after v1.5

## Digital MAC tile floorplan (v1, `MacTopology::Digital`)

Sketch for the `Mac8x8Tile` body. Reviewable before any layout
code lands; once accepted, this section becomes the contract that
`spike-tinyconv-tile/src/{layout,behavioral,topology}.rs` implement.

### Architecture: weight-stationary INT8 × INT8 → INT32 MAC

Each tile owns **one weight** (loaded once from the weight bus at
configuration time) and processes a stream of activations. This is
the standard weight-stationary CNN dataflow: for a `(K×K)`
convolution kernel with `Cin → Cout` channels, you instantiate one
tile per `(oc, ic, ky, kx)` weight position. Activations skew
through the array; per-tile partial sums accumulate locally; final
results read out by the controller into the requantize block (which
lives at the array level, not per-tile).

The multiplier is INT8 × INT8 → INT16; the accumulator is INT32
(matches `rlx-fpga::quant` and `rlx-cortexm`'s convention).

### Internal blocks

| Block | Composition (sc_hd) | Approx. cells | Approx. area µm² |
|---|---|---|---|
| Weight register (8b) | 8 × `dfxtp_1` | 8 | ~50 |
| Multiplier 8×8 → 16b | 64 × `and2_1` + 56 × `fa_1` (carry-save array) | 120 | ~700 |
| Sign-extend 16 → 32b | combinational (routing only) | 0 | 0 |
| Adder 32b | 32 × `fa_1` (ripple-carry; CLA later if Fmax forces) | 32 | ~250 |
| Accumulator (32b) | 32 × `dfxtp_1` | 32 | ~200 |
| Output mux + control | ~10 × small gates | 10 | ~60 |
| **Tile total (cells)** | — | **~200** | **~1260 + routing** |

Wider weights (4b, 2b) shrink the multiplier proportionally; the
`weight_bits` knob picks the multiplier sub-array used. Same accumulator
in all three cases.

### Pitch + floorplan strategy

`sky130_fd_sc_hd` standard cell height = **2.72 µm**. Tiles use a
4-row floorplan (cell rows abut vertically, sharing power rails):

```
+──────────────────── tile pitch X ────────────────────+
│ row 3: control + output mux                           │  ← 2.72 µm
+───────────────────────────────────────────────────────+
│ row 2: accumulator high half + final ripple-carry top │  ← 2.72 µm
+───────────────────────────────────────────────────────+
│ row 1: multiplier rows 4-7 + accumulator low half     │  ← 2.72 µm
+───────────────────────────────────────────────────────+
│ row 0: weight register + multiplier rows 0-3          │  ← 2.72 µm
+──────────────────── tile pitch X ────────────────────+
   ↑
   pitch Y = 4 × 2.72 µm = 10.88 µm
```

Pitch X estimate: row 0 alone (8 × `dfxtp_1` + 32 × `and2_1` +
24 × `fa_1`) sums to ~148 µm of cell width, so pitch X is set to
**180 µm** for ~82 % cell utilization (the original 24 µm guess
was off by ~6×; adjusted on first cell-placement pass per the
PLAN's "refines after first layout" caveat). Pitch will be
reported by `Tile::pitch()` and verified at `tile_grid` compose
time.

### Edge ports (per `eda-tile::EdgePort`)

| Side | Port | Width | Notes |
|---|---|---|---|
| North | `weight_in[7:0]` | 8 | weight bus in (one row of the weight matrix) |
| North | `accum_out[31:0]` | 32 | optional adder-tree output (off in v1, used at v1.5) |
| South | `weight_pass[7:0]` | 8 | weight bus pass-through to next-row tile |
| West  | `act_in[7:0]` | 8 | activation in (skewed across columns) |
| East  | `act_pass[7:0]` | 8 | activation pass-through to next-column tile |
| any   | `clk`, `rst_n`, `enable`, `wload` | 1 each | global control wires |

Activations flow **W→E** across each row of the array; weights flow
**N→S** down each column (loaded once, then static). Result reads
out via the controller's dispatch bus (not an edge port — read by
`StdCell`-built mux at the array level).

### Trait body cuts

Concrete bodies the floorplan implies:

- **`Mac8x8Tile::layout(&self, lib, pdk)`** (`spike-tinyconv-tile/src/layout.rs`):
  1. Build per-row `CellBuilder`s (4 of them).
  2. For each row, instantiate the listed sc_hd cells (via `eda-stdcells::StdCell`)
     in left-to-right order at sc_hd cell-pitch increments.
  3. `eda_tile::tile_grid` style: row `CellBuilder`s abut vertically;
     final wrapper `CellBuilder` insertions yield the tile cell.
  4. Local routing (within-row): direct `klayout_route::ManhattanPlanner`
     calls between adjacent cells. Cross-row (multiplier↔accumulator):
     short metal2 jogs, hardcoded. ~30 routes per tile.
- **`Mac8x8Tile::pitch(&self)`**: returns `Vec2 { x: 24_000, y: 10_880 }`
  (DBU; sky130 dbu_per_um = 1000). Refines to actual after first layout.
- **`Mac8x8Tile::rails(&self)`**: VDD on `met1` at y = 0 + every 2.72 µm;
  GND on `met1` at y = 1.36 µm + every 2.72 µm. Width = 480 nm
  (sc_hd rail width). `dbu_per_um = 1000`.
- **`Mac8x8Tile::edge_ports(&self, side)`**: returns the table above
  with concrete y-offsets (port positions on `met2` / `met3`).
- **`Mac8x8Tile::add_to_dc(&self, graph)`** (`behavioral.rs`): a digital
  MAC's "DC behavior" is its static power, gate-input load, and
  output drive — not nonlinear. Returns a `NodeId` representing
  static + dynamic power as an analytic function of `(w_l_n, w_l_p,
  vdd, weight_bits, activity)`. Differentiable; exact closed form
  given sc_hd model card (αCV²f + Pleak(W/L, Vdd)).

### What's still up for review (please flag if any of these are wrong)

1. **Weight-stationary** is the right dataflow choice. (Alternatives:
   output-stationary tile = local 32b accum, weight bus continuous;
   row-stationary = both stream. WS is the simplest tile-as-MAC story.)
2. The **per-tile accumulator** stays in v1. Skipping it (multiplier-only
   tile + adder tree at array level) saves ~230 sq-µm per tile but
   forces a non-tileable adder-tree block. Worse for the "regular grid"
   premise.
3. **No requantize per tile** — Q31 srdhm/rdpot lives at the array
   level (one per output channel), shared across all tiles producing
   that channel. Matches `rlx-fpga::codegen::requant`.
4. **Ripple-carry adder, not CLA**, for v1. Latency is acceptable
   at single-cycle MAC throughput; CLA is a v1.5 swap if Fmax forces it.
5. **No DFT (scan chain)** in v1 per PLAN.md "Deferred" — but this
   means the per-tile DFFs (40 of them) won't be observable from the
   ORFS gate-level sim except via the output bus. If we want
   per-tile activation checkpoints for `bisect::bisect`, we need scan
   sooner than v1.5. Flag.

### Implementation order under this floorplan

Once accepted:

1. `liberty.rs` already lands the cell-area numbers; bench can
   sum tile area from sc_hd metadata.
2. `Mac8x8Tile::pitch` + `rails` + `edge_ports` (3 small const-ish
   bodies, no actual layout yet — unblocks `tile_grid` composability tests).
3. `Mac8x8Tile::layout` skeleton: build empty rows of correct size,
   no real cell instances yet. Verifies the abutment + pitch contract.
4. Populate row 0 (weight register + multiplier first half). Cell
   placement only, hand-routed.
5. Rows 1-3 in turn.
6. `Mac8x8Tile::add_to_dc` (closed-form αCV²f + Pleak; ngspice
   characterization for the constants).
7. `behavioral.rs` test bodies (`tests/{analytic,fd,ngspice}.rs`)
   consume the `add_to_dc` residual.

## Open questions (decide before step 1)

1. **Area / power budgets** — 4 mm² and 50 mW are placeholders. Need
   real numbers from the application context.
2. **Which 200 MNIST images** form the golden subset? Stratified across
   classes? Hard-classified by a baseline first?
3. **How aggressive is the QAT co-design** — do we accept that weight
   re-quantization tables shift per silicon configuration, requiring a
   re-run of the cortexm trainer in the loop? Or freeze quant first,
   silicon second?
4. **ORFS image versioning** — pin a specific digest, or track latest?
   Affects reproducibility vs feature uptake trade.

## Failure modes the plan is designed to prevent

- Shipping a chip that doesn't classify, because we measured the wrong
  things. *Fix: functional validation as a first-class bench arm.*
- Optimization landing on un-shippable corners. *Fix: accuracy gate in
  the loss.*
- Burning weeks on bespoke standard cells when the foundry library is
  right there. *Fix: ingest `sc_hd`.*
- Drifting from FPGA semantics over time as ML co-design evolves.
  *Fix: FPGA backend in CI as the fast accuracy oracle.*
- Numbers that don't reproduce across runs / contributors. *Fix:
  reproducibility manifest.*
- Mean-accuracy-passes / tail-accuracy-fails. *Fix: yield gate, not
  mean.*
- Apples-to-oranges parasitic comparison. *Fix: shared OpenRCX, not
  independent extractors.*
- Single-tile passes / array brownouts. *Fix: PDN at array scope.*

## What this is not

- Not a tape-out plan. No DFT, no signoff, no test program.
- Not a production silicon flow. ORFS is the reference, but the design
  path stays in Rust HIR.
- Not a replacement for `rlx-fpga`. The FPGA path is a peer backend and
  the functional accuracy oracle, kept in CI permanently.
- Not yet committed to throughput-mode TinyConv. Single-shot only in
  v1; revisit after v1.5.
