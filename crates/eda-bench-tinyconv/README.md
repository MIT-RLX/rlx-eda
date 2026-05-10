# eda-bench-tinyconv

Umbrella crate for the TinyConv-MNIST silicon initiative — full plan
in [`PLAN.md`](./PLAN.md).

Three backends, two metric arms, one IR (`rlx_fpga::ir`). The in-house
code-defined sky130 flow is the design under test; Yosys/OpenROAD in
docker is the physical ground truth (PnR, OpenSTA, OpenRCX, PDN); the
FPGA path is the functional ground truth at scale (only backend fast
enough to validate accuracy across the full MNIST test set under PVT).

Functional accuracy on MNIST is the load-bearing metric. Physical
metrics (area, power, freq, parasitics, thermal) ride on top.

Heavyweight backends are gated:

```sh
cargo test -p eda-bench-tinyconv                          # default, lightweight
cargo test -p eda-bench-tinyconv --features bench-orfs    # pulls ORFS docker
cargo test -p eda-bench-tinyconv --features bench-fpga    # pulls FPGA toolchain
just bench-tinyconv                                       # full three-backend run
```

Status: **scaffolding**. See `PLAN.md` build-order section for what
lands when.
