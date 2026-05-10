#!/usr/bin/env bash
# pin.sh — resolve the current digest of `openroad/orfs:latest` and
# write `digest.txt` so the bench can pin against it reproducibly.
#
# Run after `docker pull openroad/orfs:latest`. Idempotent — safe to
# re-run after any image upgrade.
#
# Output:
#   - `digest.txt`: the resolved `sha256:...` digest, one line, no
#     newline. Loaded by `Manifest::capture` (via the bench's
#     `ManifestInputs::orfs_image`) so every report is anchored to a
#     specific ORFS image.
#
# The bench harness does NOT build a Dockerfile here — `eda-container`
# wraps `docker run` against the pre-pulled image directly. If you
# need a custom build (magic + netgen on top of ORFS), add a
# Dockerfile back and a corresponding build step.

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
    echo "pin.sh: docker not on PATH" >&2
    exit 1
fi

RAW=$(docker image inspect openroad/orfs:latest \
    --format '{{range .RepoDigests}}{{.}}{{"\n"}}{{end}}' 2>/dev/null \
    | head -n1)

if [[ -z "$RAW" ]]; then
    echo "pin.sh: openroad/orfs:latest not pulled. Run:" >&2
    echo "  docker pull openroad/orfs:latest" >&2
    exit 1
fi

DIGEST=${RAW#*@}
if [[ "$DIGEST" != sha256:* ]]; then
    echo "pin.sh: unexpected digest format from docker: $RAW" >&2
    exit 1
fi

printf '%s' "$DIGEST" > digest.txt
echo "wrote digest.txt: $DIGEST"
echo "Commit digest.txt to record the pin."
