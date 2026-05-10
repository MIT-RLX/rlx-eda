# rlx-eda klayout runtime image — GDS / OASIS view + scripted DRC.
#
# Pairs with the `klayout-rs` integration (`klayout-core`,
# `klayout-io`, `klayout-drc` …) for any flow that needs:
#  - headless GDS read/write (`klayout -b -r script.py`),
#  - rule-deck DRC,
#  - LVS extraction via the KLayout DRC engine,
#  - or PNG renders of GDS (`klayout -zz -r render.py`).
#
# klayout is in Ubuntu's `universe`, not in Debian's main, so we use
# Ubuntu 22.04 as the base. Image is still tight (~600 MB) and the
# `apt install klayout` route avoids pinning to a third-party
# community image.
#
# Built by `just deps-docker` (or `just deps-docker-klayout` to
# target this image alone). Tag + env-var override defined in
# `eda_container::images::KLAYOUT`.
#
# Example:
#
#     docker run --rm -v $PWD:/work rlx-klayout:local \
#         klayout -b -r /work/drc.lydrc -rd input=/work/top.gds

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# klayout lives in `universe`. Enable it explicitly so the install
# works on minimal ubuntu base images that ship without universe.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
 && add-apt-repository universe \
 && apt-get update \
 && apt-get install -y --no-install-recommends klayout \
 && apt-get purge -y --auto-remove software-properties-common \
 && rm -rf /var/lib/apt/lists/* \
 && klayout -zz -v | head -1

WORKDIR /work

# Default to no-GUI mode so naked `docker run rlx-klayout:local`
# acts like a CLI without trying to bind Qt to a display. Caller
# can override with their own argv.
ENTRYPOINT []
CMD ["klayout", "-zz", "-v"]
