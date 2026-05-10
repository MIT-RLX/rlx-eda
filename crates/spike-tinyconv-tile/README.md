# spike-tinyconv-tile

The custom analog MAC tile that the TinyConv-MNIST silicon flow tiles
into an array.

Implements the full `eda-hir` trait family:

- `Block`              — stable parametric name
- `Layout<Sky130>`     — pitch-matched analog MAC + requantize cell
- `Schematic<Sky130>`  — symbolic schematic for `eda-viz`
- `DcBehavioral`       — differentiable model for the inner Adam loop
- `Tile<Sky130>` (from `eda-tile`) — abuts cleanly in the array

Plus the standard rlx-eda validation pyramid:

- **Tier 1** — analytic closed-form gain / delay
- **Tier 2** — finite-difference sensitivities (FD-on-AD parity)
- **Tier 3** — ngspice tt / Monte Carlo (using `mc_*_switch` overrides
  per the workspace memory note on sky130 MC composition)

And the **noise model** — closed-form `(mean, σ)` on MAC output as a
function of optimized parameters, calibrated against ngspice. The
bench harness's functional arm injects this into the FPGA inference
path so accuracy is gated per Adam step without invoking SPICE.

This crate is the **first ML target** in
[`../eda-bench-tinyconv/PLAN.md`](../eda-bench-tinyconv/PLAN.md).
Build-order step 3.
