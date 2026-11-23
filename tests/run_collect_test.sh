#!/usr/bin/env bash
# Long-run + restart test of collect.py against the local mock server.
set -u
cd "$(dirname "$0")/.."
S=${1:-/tmp/collect_test}
rm -rf "$S"; mkdir -p "$S"
python3 tests/mock_server.py 8766 > "$S/mock.log" 2>&1 &
MOCK=$!
sleep 1.5
M="--datagov-url http://127.0.0.1:8766/v1/transport/traffic-images --datamall-base http://127.0.0.1:8766/ltaodataservice/"
F="--image-interval 4 --image-spacing 5 --incident-interval 5 --speed-interval 8 --faulty-interval 10 --status-every 10"
echo "=== run 1 (40 s) ==="
python3 collect.py --out "$S/raw" --lta-key testkey --hours 0.0111 $F $M | tail -4
echo "=== run 2, restart (15 s) ==="
python3 collect.py --out "$S/raw" --lta-key testkey --hours 0.0042 $F $M | tail -2
kill $MOCK
python3 - "$S/raw" <<'PY'
import csv, sys, gzip, glob, json
from pathlib import Path
from PIL import Image
raw = Path(sys.argv[1])
rows = list(csv.DictReader(open(raw/"images_index.csv")))
md5s = [r["md5"] for r in rows]
keys = [(r["camera_id"], r["image_ts"]) for r in rows]
print("index rows", len(rows), "unique md5", len(set(md5s)), "unique cam+ts", len(set(keys)))
missing = [r["path"] for r in rows if not (raw/r["path"]).exists()]
print("missing files", len(missing))
w = Image.open(raw/rows[0]["path"]).size
print("saved size", w, "orig", rows[0]["width"], rows[0]["height"])
inc = open(raw/"incidents.jsonl").read().splitlines()
polls = open(raw/"incident_polls.csv").read().splitlines()
print("incident lines", len(inc), "incident polls", len(polls) - 1)
sb = sorted(glob.glob(str(raw/"speedbands"/"*.csv.gz")))
n = sum(1 for _ in csv.DictReader(gzip.open(sb[0], "rt")))
print("speed files", len(sb), "rows in first", n)
print("faulty lines", len(open(raw/"faulty_lights.jsonl").read().splitlines()))
print("status", json.load(open(raw/"status.json")))
print("cameras.csv", len(open(raw/"cameras.csv").read().splitlines()) - 1)
PY
