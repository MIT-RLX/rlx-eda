# rlx-eda magic runtime image — standalone DRC / LVS / extraction.
#
# Magic is bundled in the ORFS image already, but ORFS is multi-GB and
# its pipeline assumes the full sky130hd flow. This image is for
# analog / full-custom flows where you want to point magic at a GDS
# (or .mag) directly: DRC, parasitic extraction, or netlist-from-GDS.
# Pairs with `netgen` (also installed) for LVS.
#
# Magic runs headlessly via `-dnull -noconsole`; no Xvfb needed for
# scripted DRC.
#
# Built by `just deps-docker` (or `just deps-docker-magic` to target
# this image alone). Tag + env-var override defined in
# `eda_container::images::MAGIC`.
#
# Example:
#
#     docker run --rm -v $PWD:/work rlx-magic:local \
#         magic -dnull -noconsole -T sky130A.tech \
#               /work/drc.tcl

FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        magic \
        netgen \
        tcl \
 && rm -rf /var/lib/apt/lists/* \
 && echo 'quit -noprompt' | magic -dnull -noconsole 2>&1 | head -3 || true

WORKDIR /work

# Magic has no `--version`; default cmd just confirms the binary
# launches under -dnull -noconsole and exits on `quit`. Override
# with your own argv (e.g. a DRC tcl script).
ENTRYPOINT []
CMD ["sh", "-c", "echo 'quit -noprompt' | magic -dnull -noconsole"]
