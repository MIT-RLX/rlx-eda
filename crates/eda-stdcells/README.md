# eda-stdcells

Thin foundry standard-cell ingest. Reads `sky130_fd_sc_hd` GDS +
Liberty and exposes each cell as a `StdCell` that implements the
`eda-hir` traits (`Block + Layout<Sky130> + Schematic<Sky130>`).

**Not** a from-scratch parametric library. Justification: matches what
ORFS will use anyway, so any in-house ↔ ORFS divergence is in
floorplan, not in cell library. Saves 2-4 weeks vs hand-rolling.

Build-order step 1 in [`eda-bench-tinyconv/PLAN.md`](../eda-bench-tinyconv/PLAN.md).

## Status

Scaffolding. Public surface defined; ingest bodies are stubs. The
`sc-hd` feature gates anything that requires the foundry library to
be checked out (mirrors `eda-pdks`'s soft-skip pattern).
