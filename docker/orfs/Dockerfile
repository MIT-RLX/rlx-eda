# OpenROAD Flow Scripts + sky130A + Yosys + OpenSTA + magic + netgen.
#
# Pinning policy (PLAN.md cross-cutting #1 + open question #4):
# pin a specific digest, never `:latest`. The current digest sits in
# `image_digest` of the bench manifest so every run records exactly
# which ORFS produced the numbers.
#
# The base ORFS image already bundles Yosys, OpenROAD, OpenSTA, and
# the sky130 PDK (skywater130 + open_pdks build). We add magic +
# netgen for DRC + LVS, plus jq for metric post-processing.
#
# Build:
#   just deps-docker-orfs
#   # ↳ `docker build -t rlx-eda-orfs:local docker/orfs`
#
# Run (driven by `eda_bench_tinyconv::backends::orfs` via
# `eda_container::DockerRun`):
#   docker run --rm -v $PWD/work:/work \
#     rlx-eda-orfs:local \
#     /work/config.mk /work/verilog
#
# Output: /work/metrics.json (consumed by `backends::orfs`).
# Tag + env-var override defined in `eda_container::images::ORFS`.

# Pin --platform=linux/amd64: openroad/orfs ships amd64-only manifests
# (the OpenROAD binaries are pre-built x86_64). On Apple Silicon /
# arm64 hosts the unqualified pull fails with "no match for platform
# in manifest"; the explicit platform pin tells Docker Desktop to
# emulate via Rosetta.
#
# Replace <pinned-digest> when the image is first built; record in PLAN.md.
# FROM --platform=linux/amd64 openroad/orfs@sha256:<pinned-digest>
FROM --platform=linux/amd64 openroad/orfs:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        magic \
        netgen \
        jq \
    && rm -rf /var/lib/apt/lists/*

COPY run_orfs.sh /usr/local/bin/run_orfs.sh
RUN chmod +x /usr/local/bin/run_orfs.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/run_orfs.sh"]
