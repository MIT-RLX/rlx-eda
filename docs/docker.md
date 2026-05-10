# Docker images — centralized at `docker/`

Every container image rlx-eda ships lives under the top-level `docker/`
directory, one subdir per image. The shared `eda-container` crate owns
the registry (tags, env-var overrides, build paths) and the
`DockerRun` builder; consumer crates (`eda-extern-ngspice`,
`eda-bench-tinyconv`) route every `docker run` through it so docker
discovery, image auto-build, mounts, and error reporting stay in one
place. To add a new image, drop a Dockerfile under `docker/<name>/`,
register it in `eda_container::images`, and add a matching
`deps-docker-<name>` recipe in the Justfile.

| Image | Subdir | Default tag | Env override | Purpose |
| ----- | ------ | ----------- | ------------ | ------- |
| ngspice runtime | `docker/ngspice/` | `rlx-ngspice:local` | `RLX_NGSPICE_IMAGE` | SPICE simulation; consumed by `eda-extern-ngspice::DockerInvoker` |
| yosys synthesis | `docker/yosys/` | `rlx-yosys:local` | `RLX_YOSYS_IMAGE` | Standalone RTL→netlist (lighter than ORFS) |
| magic DRC/LVS | `docker/magic/` | `rlx-magic:local` | `RLX_MAGIC_IMAGE` | Standalone DRC, extraction, LVS via netgen — analog/full-custom flows |
| KLayout | `docker/klayout/` | `rlx-klayout:local` | `RLX_KLAYOUT_IMAGE` | Headless GDS I/O, scripted DRC, PNG renders; pairs with `klayout-rs` |
| ORFS bench (sky130) | `docker/orfs/` | `rlx-eda-orfs:local` | `RLX_ORFS_IMAGE` | Full ASIC flow (yosys + OpenROAD + OpenSTA + magic + netgen); `eda-bench-tinyconv::backends::orfs` behind `bench-orfs` |

`just deps-docker` builds every image (light → heavy);
`just deps-docker-<name>` builds one. `just deps-docker-check`
smoke-tests each by running its `--version`.

## ngspice via Docker

`eda-extern-ngspice` ships two `Invoker` impls: `LocalBinary` (host
ngspice) and `DockerInvoker`. The Docker path uses the pinned image
built from `docker/ngspice/Dockerfile` (debian-slim + apt ngspice).
Build it once with `just deps-docker-ngspice` and select it via the
`NGSPICE_BACKEND=docker` environment variable.
