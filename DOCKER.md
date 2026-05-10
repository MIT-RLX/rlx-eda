# Docker images in rlx-eda

Every container image rlx-eda ships lives under the top-level
`docker/` directory, one subdir per image. The `eda-container` crate
owns the registry (tags, env-var overrides, build paths) and the
`DockerRun` builder; all consumer crates route their `docker run`
calls through it so docker discovery, image auto-build, mounts, and
error reporting stay in one place.

```
docker/
  .dockerignore       (shared)
  ngspice/Dockerfile  → debian-slim + ngspice
  yosys/Dockerfile    → debian-slim + yosys
  magic/Dockerfile    → debian-slim + magic + netgen + tcl
  klayout/Dockerfile  → ubuntu:22.04 + klayout
  orfs/Dockerfile     → openroad/orfs + magic + netgen + jq
  orfs/run_orfs.sh    (entrypoint orchestrator for the bench)
crates/eda-container/  (shared Rust API — registry + DockerRun)
```

## Image inventory

| Image | Subdir | Default tag | Env override | Size | Purpose |
| ----- | ------ | ----------- | ------------ | ---- | ------- |
| ngspice | `docker/ngspice/` | `rlx-ngspice:local` | `RLX_NGSPICE_IMAGE` | 166 MB | SPICE simulation; consumed by `eda-extern-ngspice::DockerInvoker` |
| yosys | `docker/yosys/` | `rlx-yosys:local` | `RLX_YOSYS_IMAGE` | 576 MB | Standalone RTL→netlist (lighter than ORFS) |
| magic | `docker/magic/` | `rlx-magic:local` | `RLX_MAGIC_IMAGE` | 760 MB | DRC, parasitic extraction, LVS via netgen — analog/full-custom |
| klayout | `docker/klayout/` | `rlx-klayout:local` | `RLX_KLAYOUT_IMAGE` | 744 MB | Headless GDS I/O, scripted DRC, PNG renders; pairs with `klayout-rs` |
| orfs | `docker/orfs/` | `rlx-eda-orfs:local` | `RLX_ORFS_IMAGE` | 6.6 GB | Full ASIC flow (yosys 0.64 + OpenROAD 26Q2 + OpenSTA + magic + netgen + jq); used by `eda-bench-tinyconv::backends::orfs` behind `bench-orfs` |

Sizes are post-build local image sizes on macOS/Docker Desktop.

All images smoke-tested green via `just deps-docker-check`. ORFS's
`run_orfs.sh` ENTRYPOINT means `docker run rlx-eda-orfs:local <cmd>`
feeds `<cmd>` into the entrypoint (which expects `config.mk` +
verilog dir); pass `--entrypoint=bash` to invoke a tool ad-hoc.
`openroad` is at `/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad`,
not on `PATH`.

## Building & testing

```sh
just deps-docker            # Build every image (light → heavy ordering)
just deps-docker-ngspice    # …or one image: ngspice / yosys / magic / klayout / orfs
just deps-docker-check      # Smoke-test every image (--version round-trip)
just deps-docker-clean      # Remove every locally-built rlx-* image
```

Per-image variants exist for each: `deps-docker-<name>`,
`deps-docker-check-<name>`, `deps-docker-clean-<name>`.

### Smoke-test commands

| Image | Command (inside the container) | Why this form |
| ----- | ------------------------------ | ------------- |
| ngspice | `ngspice --version` | standard CLI |
| yosys | `yosys -V` | yosys's `--version` is `-V` |
| magic | `echo 'quit -noprompt' \| magic -dnull -noconsole` | magic has no `--version` flag |
| klayout | `klayout -zz -v` | `-v` alone tries to bind Qt and SIGTRAPs without a display; `-zz` selects no-GUI mode |
| orfs | (n/a — heavyweight; tested via the bench harness) | |

## Rust API — `eda-container`

```rust
use eda_container::{
    self as container,
    DockerRun, ContainerError,
    images,           // ImageSpec registry: NGSPICE, YOSYS, MAGIC, KLAYOUT, ORFS, ALL
    ensure_image,     // build if missing
    ensure_image_spec,
    image_exists,
    inspect_digest,   // for reproducibility manifests
    docker_available, // is `docker` on PATH?
    which,            // tiny dependency-free PATH lookup (also used by LTspice driver)
    workspace_root,
    docker_root,
};

// Stdin-piped run (the ngspice driver pattern):
let stdout = DockerRun::new(images::NGSPICE.tag())
    .interactive(true)
    .mount("/tmp", "/tmp")
    .args(["ngspice", "-b", "-n", "/dev/stdin"])
    .run_with_stdin(deck_bytes)?;

// Volume-only run, exit-status only (the ORFS bench pattern):
let status = DockerRun::new(images::ORFS.tag())
    .mount(work_dir, "/work")
    .arg("/work/config.mk").arg("/work/verilog")
    .status()?;

// Bench manifest digest record:
let digest = inspect_digest(&images::ORFS.tag()); // Option<String>
```

`ImageSpec::tag()` honors the env-var override; `context_dir()`
returns the absolute Dockerfile path. The workspace root is anchored
at compile time via `env!("CARGO_MANIFEST_DIR")` in `eda-container`,
so consumers don't need their own way to find `docker/`.

## Adding a new image

1. Drop a Dockerfile under `docker/<name>/` (and any helper scripts
   alongside it). Match the existing pattern: tiny base, no
   `ENTRYPOINT` so callers supply the full command line, default
   `CMD` is a `--version`-style smoke test.
2. Register an `ImageSpec` in `crates/eda-container/src/lib.rs`
   under `pub mod images`, append it to `images::ALL`. The
   `registered_images_have_context_dirs` test will then verify the
   Dockerfile is reachable from the workspace anchor.
3. Add three Justfile recipes — `deps-docker-<name>`,
   `deps-docker-check-<name>`, `deps-docker-clean-<name>` — and
   thread them through the aggregate `deps-docker` /
   `deps-docker-check` / `deps-docker-clean` recipes.
4. Update the inventory table in this file.
5. (Optional) Wire a Rust consumer using `DockerRun::new(spec.tag())`
   + `ensure_image_spec(spec)`.

## Why some tools aren't here

- **LTspice** — closed-source Analog Devices Windows app. There's no
  upstream Linux image; community wine-based images are 2GB+ and
  fragile. `eda-extern-ltspice` runs the host binary (env-var → PATH
  → macOS `.app` bundle); it shares `eda_container::which` for the
  PATH lookup but has no docker image.
- **External / vendored** — the five Dockerfiles under
  `external/carsten-lelo-ex/aicex/docker/` are vendored from an
  upstream submodule and stay untouched. Don't centralize them.

## Env-var overrides

Every `ImageSpec.env_var` overrides the default tag at runtime. Use
this to point a backend at a pre-built image with a different tag
without rebuilding:

```sh
RLX_NGSPICE_IMAGE=mycorp/ngspice@sha256:abc123 \
    cargo run -p spike-dado-sar
```

Setting an override does **not** automatically build the image — the
caller is responsible for making sure the tag resolves locally.

## Apple Silicon note

`openroad/orfs` (the ORFS base) only ships `linux/amd64` manifests —
its OpenROAD binaries are pre-built x86_64. On arm64 hosts (M-series
Macs) an unqualified pull fails with `no match for platform in
manifest`. `docker/orfs/Dockerfile` therefore pins
`FROM --platform=linux/amd64 openroad/orfs:…`, and the
`deps-docker-orfs` recipe uses `docker buildx build --load
--platform=linux/amd64` so the resulting image lands in the daemon's
runtime store (without `--load`, Docker Desktop with the containerd
image store puts the build artifact only in the buildx cache and
`docker run` then fails with "image not found"). Expect the bench
to run noticeably slower than on a native amd64 host because of
Rosetta emulation.

The four light images (ngspice, yosys, magic, klayout) build
natively on both amd64 and arm64.

## Reproducibility

`eda-bench-tinyconv::manifest::Manifest::capture` records each image
digest via `eda_container::inspect_digest(image)`; this lands in the
per-run manifest alongside the sky130 PDK commit, ngspice version,
and Cargo.lock SHA-256. ORFS is the load-bearing one here — pin the
image to a specific digest in `docker/orfs/Dockerfile` (currently
`FROM --platform=linux/amd64 openroad/orfs:latest`, tracked as a
TODO) so manifests stay meaningful across rebuilds.
