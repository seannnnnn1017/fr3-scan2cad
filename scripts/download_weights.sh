#!/usr/bin/env bash
# Detector / segmenter weights into <repo>/weights (gitignored).
# Ultralytics fetches its own release assets by name; SAM3 is gated on Hugging Face.
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p weights && cd weights
PY=${BRAIN_PYTHON:-../.venv/bin/python}
"$PY" - <<'PYEOF'
from ultralytics import SAM, YOLOE
for w in ("yoloe-26l-seg.pt", "yoloe-26s-seg.pt"):
    m = YOLOE(w)
    m.set_classes(["cup"])          # also pulls the MobileCLIP text encoder (mobileclip2_b.ts)
SAM("sam2.1_b.pt")
PYEOF
# SAM3: accept the licence at https://huggingface.co/facebook/sam3, run `hf auth login`, then:
[ -f sam3.pt ] || hf download facebook/sam3 sam3.pt --local-dir .
ls -la
