#!/usr/bin/env bash
# Three-object test: baseball / soup can / mug placed randomly; the other two stay on the table as clutter.
# The first run calibrates, the others reuse its calibration.
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-$(date +%Y%m%d_%H%M%S)}
for k in baseball soup_can mug; do
  CAL=""; [ "$k" != baseball ] && CAL="--calibration runs/multi_${TAG}_baseball/calib/calibration.json"
  echo "##### $k start $(date +%T)"
  .venv/bin/python scripts/run.py --scene configs/scenes/multi_$k.json --scan configs/scans/multi_$k.json \
      --out runs/multi_${TAG}_$k $CAL 2>&1 | grep -E "===|Done|failed|Error"
  echo "##### $k end $(date +%T) rc=${PIPESTATUS[0]}"
done
