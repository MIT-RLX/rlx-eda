# rlx-eda ngspice runtime image — minimal, deterministic.
#
# Built and used by `eda_extern_ngspice::DockerInvoker`, which routes
# through `eda_container::DockerRun` and emits:
#
#     docker run --rm -i -v /tmp:/tmp <tag> ngspice -b -n /dev/stdin
#
# /tmp is bind-mounted so the deck's `write <tempfile>.raw` command
# lands on a host path the parser can read. Image is intentionally tiny
# (debian-slim + apt ngspice); first build takes ~30 s, subsequent
# `docker run`s start in well under a second.
#
# Built by `just deps-docker` (or `just deps-docker-ngspice` to target
# this image alone). Tag + env-var override defined in
# `eda_container::images::NGSPICE`.

FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ngspice \
 && rm -rf /var/lib/apt/lists/* \
 && ngspice --version | head -1

WORKDIR /tmp

# No ENTRYPOINT — caller supplies the full ngspice command line so the
# trait signature stays `[args, /dev/stdin]` shaped.
ENTRYPOINT []
CMD ["ngspice", "-b"]
