# rlx-eda yosys runtime image — standalone synthesis.
#
# Lighter alternative to the full ORFS image when you only need
# RTL→gate-level netlist + tech-mapping (no PnR / STA / DRC). Targets
# digital flows in `eda-stdcells` / `eda-tile` and any future spike
# crate that wants to round-trip a Verilog netlist without paying for
# the multi-GB ORFS image.
#
# Built by `just deps-docker` (or `just deps-docker-yosys` to target
# this image alone). Tag + env-var override defined in
# `eda_container::images::YOSYS`.
#
# Caller routes through `eda_container::DockerRun`; example:
#
#     docker run --rm -i -v $PWD:/work rlx-yosys:local \
#         yosys -p 'read_verilog /work/top.v; synth; write_verilog /work/out.v'

FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends yosys \
 && rm -rf /var/lib/apt/lists/* \
 && yosys -V | head -1

WORKDIR /work

# No ENTRYPOINT — caller supplies the full yosys command line so a
# single image serves both `-p script` and `-s file.ys` flows.
ENTRYPOINT []
CMD ["yosys", "-V"]
