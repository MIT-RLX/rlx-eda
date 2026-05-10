# spike-tinyconv-array

Full TinyConv-MNIST silicon block. Lowers
`rlx_fpga::model::Model` (the graph that `rlx-fpga::reference`
already validates and that `rlx-fpga::tune` already sweeps) into:

- a tiled instantiation of `spike-tinyconv-tile` (the analog/digital
  MAC tile, default `MacTopology::Digital`),
- abutted via `eda-tile::tile_grid` so PDN + edge-port contracts hold,
- with a controller FSM built from `eda-stdcells` (foundry
  `sky130_fd_sc_hd`).

Two lowerings now share `rlx_fpga::model`:

- `rlx-fpga` → SystemVerilog (already exists)
- `spike-tinyconv-array` → `Block` composition → klayout GDS

Build-order step 5 in
[`../eda-bench-tinyconv/PLAN.md`](../eda-bench-tinyconv/PLAN.md).
Mostly mechanical — depends on `Mac8x8Tile::layout` having a body
(the `Digital` topology variant from step 3).
