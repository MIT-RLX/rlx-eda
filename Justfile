# rlx-eda dev recipes. Cross-OS where it makes sense; everything also
# works as plain `cargo` / `docker` invocations. Install just from
# https://just.systems if you don't have it.
#
# Run `just` (no args) or `just --list` to see all recipes.

default:
    @just --list

# ── Host-dep install ──────────────────────────────────────────────────────
#
# `just deps` installs ngspice via the OS's native package manager
# (brew / apt / dnf / pacman). Skips itself on Windows — use Docker
# (`just deps-docker`) on platforms without a clean ngspice path.

[no-cd]
deps:
    #!/usr/bin/env bash
    set -euo pipefail
    case "$(uname -s)" in
        Darwin)
            command -v brew >/dev/null \
                || { echo "Homebrew not found. Install from https://brew.sh"; exit 1; }
            brew list ngspice >/dev/null 2>&1 || brew install ngspice
            ;;
        Linux)
            if command -v apt-get >/dev/null; then
                sudo apt-get update -qq
                sudo apt-get install -y -qq ngspice
            elif command -v dnf >/dev/null; then
                sudo dnf install -y ngspice
            elif command -v pacman >/dev/null; then
                sudo pacman -S --noconfirm ngspice
            elif command -v zypper >/dev/null; then
                sudo zypper install -y ngspice
            else
                echo "No supported Linux package manager found." >&2
                echo "Install ngspice manually, or use 'just deps-docker'." >&2
                exit 1
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "Windows host: native ngspice install isn't covered here." >&2
            echo "Use 'just deps-docker' to run ngspice in a container instead." >&2
            exit 1
            ;;
        *)
            echo "Unrecognised OS: $(uname -s). Use 'just deps-docker'." >&2
            exit 1
            ;;
    esac
    ngspice --version | head -1

# ── Centralized docker images ─────────────────────────────────────────────
#
# All Dockerfiles live under `docker/<name>/`. Tags + env-var overrides
# are mirrored in `eda_container::images` so Rust callers and these
# recipes stay in sync. Add a new image: drop a Dockerfile under
# `docker/<name>/`, register it in `eda_container::images`, then add a
# matching `deps-docker-<name>` recipe here.

# Build every rlx-eda docker image. Order matches
# `eda_container::images::ALL` (light → heavy); ORFS last because it
# pulls a multi-GB base.
deps-docker: deps-docker-ngspice deps-docker-yosys deps-docker-magic deps-docker-klayout deps-docker-orfs

# Smoke-test every image (each `--version` should round-trip).
deps-docker-check: deps-docker-check-ngspice deps-docker-check-yosys deps-docker-check-magic deps-docker-check-klayout deps-docker-check-orfs

# Remove every locally-built image — forces a rebuild on next deps-docker.
deps-docker-clean: deps-docker-clean-ngspice deps-docker-clean-yosys deps-docker-clean-magic deps-docker-clean-klayout deps-docker-clean-orfs

# ── ngspice image ────────────────────────────────────────────────────────

# Build the pinned ngspice runtime image (debian-slim + apt ngspice).
# Tag matches `eda_container::images::NGSPICE` / `RLX_NGSPICE_IMAGE`.
deps-docker-ngspice:
    docker build -t rlx-ngspice:local docker/ngspice

# Smoke-test: `ngspice --version` inside the image.
deps-docker-check-ngspice:
    docker run --rm rlx-ngspice:local ngspice --version

# Remove the locally-built ngspice image.
deps-docker-clean-ngspice:
    docker image rm -f rlx-ngspice:local

# ── yosys image ──────────────────────────────────────────────────────────
#
# Standalone synthesis (debian-slim + apt yosys). Tag matches
# `eda_container::images::YOSYS` / `RLX_YOSYS_IMAGE`.

deps-docker-yosys:
    docker build -t rlx-yosys:local docker/yosys

deps-docker-check-yosys:
    docker run --rm rlx-yosys:local yosys -V

deps-docker-clean-yosys:
    docker image rm -f rlx-yosys:local

# ── magic image ──────────────────────────────────────────────────────────
#
# Standalone DRC / extraction / LVS (debian-slim + magic + netgen).
# Tag matches `eda_container::images::MAGIC` / `RLX_MAGIC_IMAGE`.

deps-docker-magic:
    docker build -t rlx-magic:local docker/magic

# Magic has no --version flag; use a `quit -noprompt` to confirm the
# binary launches and exits cleanly under -dnull -noconsole.
deps-docker-check-magic:
    echo 'quit -noprompt' | docker run --rm -i rlx-magic:local magic -dnull -noconsole

deps-docker-clean-magic:
    docker image rm -f rlx-magic:local

# ── klayout image ────────────────────────────────────────────────────────
#
# Headless GDS / DRC / render (ubuntu:22.04 + apt klayout from
# universe). Tag matches `eda_container::images::KLAYOUT` /
# `RLX_KLAYOUT_IMAGE`.

deps-docker-klayout:
    docker build -t rlx-klayout:local docker/klayout

# `klayout -v` alone tries to bind Qt and crashes without a display.
# `-zz` selects the no-GUI mode that the rlx-eda flows actually use.
deps-docker-check-klayout:
    docker run --rm rlx-klayout:local klayout -zz -v

deps-docker-clean-klayout:
    docker image rm -f rlx-klayout:local

# ── ORFS image ───────────────────────────────────────────────────────────
#
# Heavy: openroad/orfs base + magic + netgen + jq. First build is slow
# (multi-GB pull). Driven by `eda-bench-tinyconv` behind the
# `bench-orfs` feature. Tag matches `eda_container::images::ORFS` /
# `RLX_ORFS_IMAGE`.

# Build the rlx-eda ORFS bench image. `--load` ensures the image
# lands in the docker daemon's runtime store (not just buildx
# cache); without it `docker run` fails with "image not found"
# when Docker Desktop has the containerd image store enabled.
deps-docker-orfs:
    docker buildx build --load --platform=linux/amd64 -t rlx-eda-orfs:local docker/orfs

# Smoke-test ORFS by bypassing the run_orfs.sh entrypoint and
# checking each bundled tool reports a version. openroad is at a
# fixed install path, not on PATH.
deps-docker-check-orfs:
    docker run --rm --entrypoint=bash rlx-eda-orfs:local -c \
        'set -e; \
         yosys -V | head -1; \
         /OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -version | head -1; \
         echo "quit -noprompt" | magic -dnull -noconsole 2>&1 | grep -m1 Magic; \
         echo "quit" | netgen -batch 2>&1 | head -1; \
         jq --version'

# Remove the locally-built ORFS image.
deps-docker-clean-orfs:
    docker image rm -f rlx-eda-orfs:local

# ── Build / test ──────────────────────────────────────────────────────────

# Build the whole workspace.
build:
    cargo build --workspace

# Build one crate in release mode.
build-rel crate:
    cargo build --release -p {{crate}}

# Run the workspace test suite.
test:
    cargo test --workspace

# Run tests for a single crate.
test-one crate:
    cargo test -p {{crate}}

# Lint with clippy, deny warnings.
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# ── Spike experiments ─────────────────────────────────────────────────────

# Run the R-2R DAC DADO experiment.
# Output: crates/spike-dado-r2r/docs/STORY.md and friends.
run-dado-r2r:
    cargo run --release -p spike-dado-r2r

# Run the SAR ADC DADO head-to-head experiment.
# Set NGSPICE_BACKEND=docker (or leave unset for native ngspice).
# Output: crates/spike-dado-sar/docs/STORY.md and friends.
run-dado-sar:
    cargo run --release -p spike-dado-sar

# Variant of run-dado-sar that pins the Docker backend.
run-dado-sar-docker:
    NGSPICE_BACKEND=docker cargo run --release -p spike-dado-sar

# ── Top-level CLI ─────────────────────────────────────────────────────────

# Wrapper around scripts/dado: `just dado r2r`, `just dado sar --docker`, etc.
dado +ARGS="help":
    ./scripts/dado {{ARGS}}
