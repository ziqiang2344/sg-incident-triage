#!/usr/bin/env bash
# End-to-end check on synthetic data: every step, mock VLM, simulated labels. No network needed.
#   bash tests/run_pipeline_test.sh [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
S=${1:-/tmp/sgtriage_test}
rm -rf "$S"
python3 tests/make_synthetic.py "$S"
A="--raw $S/raw --work $S/work --static $S/static --site $S/docs"
python3 -m sgtriage static $A --static-zip "$S/static/TrafficLight_test.zip"
python3 -m sgtriage status $A
python3 -m sgtriage prepare $A --backend tiny
python3 tests/simulate_labels.py "$S/work" "$S/truth.csv"
python3 tests/mock_vlm.py 8791 > "$S/mock_vlm.log" 2>&1 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT
sleep 1
VLM_BASE_URL=http://127.0.0.1:8791/v1 VLM_MODEL=mock-vlm VLM_API_KEY=x python3 -m sgtriage vlm $A --workers 8
VLM_MODEL=mock-vlm python3 -m sgtriage evaluate $A
python3 -m sgtriage report $A
python3 -m sgtriage site $A
python3 - "$S" <<'PY'
import json, sys
from pathlib import Path
s = Path(sys.argv[1])
m = json.loads((s / "work" / "metrics.json").read_text())
fused = m["precision"]["fused"]["25"]["p"]
rand = m["prevalence"]["p"]
assert fused > 2 * rand, (fused, rand)
assert m["vlm"]["confusion"]["tp"] > 0
html = (s / "docs" / "index.html").read_text()
assert "/*DATA*/" not in html and len(list((s / "docs" / "img").glob("*.jpg"))) > 5
print(f"OK: fused top-25 {fused:.0%} vs random {rand:.1%}; VLM kappa {m['vlm']['kappa']:.2f}")
PY
