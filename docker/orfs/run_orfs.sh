#!/usr/bin/env bash
# run_orfs.sh — entrypoint inside the bench-tinyconv docker image.
#
# In:
#   $1 = path to config.mk (ORFS design configuration)
#   $2 = path to verilog source dir
# Out:
#   /work/metrics.json (mounted on host)
#
# Drives the full ORFS flow:
#   yosys synthesis → OpenROAD floorplan/place/CTS/route →
#   OpenSTA timing → OpenRCX parasitic extraction →
#   magic DRC → netgen LVS
# and assembles the resulting reports into one JSON consumed by
# `crate::backends::orfs`.
#
# PLAN.md "Bench harness layout" lists which Physical fields come
# from which tool. Cross-cutting #2 (parasitic extraction parity)
# requires that the in-house backend route through the same OpenRCX
# step, not roll its own extractor.

set -euo pipefail

CONFIG_MK=${1:?"usage: run_orfs.sh <config.mk> <verilog-dir>"}
VERILOG_DIR=${2:?"usage: run_orfs.sh <config.mk> <verilog-dir>"}
OUT=/work/metrics.json

cd /OpenROAD-flow-scripts/flow

# Run flow → STA → DRC/LVS. Each stage writes reports under
# `reports/<platform>/<design>/...` per ORFS convention.
make DESIGN_CONFIG="$CONFIG_MK" SDC_FILE=auto VERILOG_FILES="$VERILOG_DIR"/*.sv
make DESIGN_CONFIG="$CONFIG_MK" finish
make DESIGN_CONFIG="$CONFIG_MK" magic_drc
make DESIGN_CONFIG="$CONFIG_MK" netgen_lvs

# Stitch the reports we care about into one JSON. Schema mirrors
# `crate::metrics::Physical`. Missing fields → null so the Rust
# parser can detect partial flows.
REPORTS=reports/sky130hd/$(basename "$(dirname "$CONFIG_MK")")/base
jq -n \
    --arg area "$(grep '^Design area' "$REPORTS/6_report.log" | awk '{print $3}')" \
    --arg wns  "$(grep '^wns' "$REPORTS/6_finish.rpt" | awk '{print $2}')" \
    --arg fmax "$(grep 'clock period' "$REPORTS/6_finish.rpt" | awk '{print $NF}')" \
    --arg pwr  "$(grep '^Total' "$REPORTS/6_finish.power.rpt" | awk '{print $5}')" \
    --arg cap  "$(grep 'Total Cap' "$REPORTS/6_finish.parasitics.rpt" | awk '{print $3}')" \
    '{
        area_um2: ($area | tonumber? // null),
        max_freq_mhz: (if $fmax == "" then null else (1000.0 / ($fmax | tonumber)) end),
        wns_ns: ($wns | tonumber? // null),
        dynamic_power_mw: ($pwr | tonumber? // null),
        leakage_power_mw: null,
        parasitic_cap_ff: ($cap | tonumber? // null),
        peak_temp_c: null,
        energy_pj_per_inference: null
    }' > "$OUT"

echo "wrote $OUT" >&2
