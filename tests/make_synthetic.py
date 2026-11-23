"""Build a small synthetic dataset in the collector's format, with known ground truth.

python tests/make_synthetic.py OUT_DIR
Creates OUT_DIR/raw (as collect.py writes it), OUT_DIR/static/TrafficLight_test.zip and
OUT_DIR/truth.csv (frame_id -> visible incident), so the whole pipeline can be checked offline.
"""
import csv
import gzip
import io
import json
import struct
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SGT = timezone(timedelta(hours=8))
rng = np.random.default_rng(7)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth")
raw = out / "raw"
(raw / "images").mkdir(parents=True, exist_ok=True)
(raw / "speedbands").mkdir(exist_ok=True)
(out / "static").mkdir(exist_ok=True)

T0 = datetime(2026, 11, 23, 0, 0, tzinfo=SGT)
HOURS = 16
cams = {f"{1700 + i}": (1.33 + 0.002 * (i % 3), 103.70 + 0.025 * i) for i in range(10)}

# incidents: (camera or None, start minute, duration, visible, type)
events = []
for k in range(14):
    cam = list(cams)[k % 10]
    start = int(rng.integers(30, HOURS * 60 - 90))
    dur = int(rng.integers(20, 70))
    visible = k < 9
    typ = ["Accident", "Vehicle breakdown", "Obstacle"][k % 3]
    events.append({"cam": cam if k < 12 else None, "start": start, "dur": dur, "visible": visible and k < 12, "type": typ})
congestion = [{"cam": list(cams)[k], "start": int(rng.integers(420, 600)), "dur": 45, "type": "Heavy Traffic"} for k in (2, 5, 8)]
roadworks = [{"cam": None, "start": 0, "dur": 300, "type": "Roadwork"}]


def ev_latlon(e, k):
    if e["cam"] is None:
        return 1.45 + 0.001 * k, 104.00
    lat, lon = cams[e["cam"]]
    return lat + 0.0012, lon + 0.0008  # ~160 m from the camera


def active(e, minute):
    return e["start"] <= minute < e["start"] + e["dur"]


# ---- frames
idx_rows, truth = [], []
for cid, (lat, lon) in cams.items():
    base = rng.integers(60, 120, size=3)
    for f in range(HOURS * 20):
        minute = f * 3 + int(rng.integers(0, 2))
        t = T0 + timedelta(minutes=minute)
        hour = t.hour + t.minute / 60
        light = 1.0 if 7 <= hour < 19 else 0.35
        im = Image.new("RGB", (320, 180), tuple(int(c * light) for c in base))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 90, 320, 180], fill=tuple(int(70 * light) for _ in range(3)))
        for lane in (110, 135, 160):
            d.line([0, lane, 320, lane], fill=tuple(int(200 * light) for _ in range(3)), width=1)
        for _ in range(int(rng.integers(4, 14))):
            x, y = int(rng.integers(0, 300)), int(rng.integers(95, 170))
            col = tuple(int(c * light) for c in rng.integers(40, 230, size=3))
            d.rectangle([x, y, x + 14, y + 7], fill=col)
        vis = any(e["cam"] == cid and e["visible"] and active(e, minute) for e in events)
        if vis:
            d.rectangle([200, 140, 232, 156], fill=(255, 140, 0))
            d.ellipse([236, 142, 244, 150], fill=(255, 255, 255))
        if rng.random() < 0.01:
            im = Image.new("RGB", (320, 180), (128, 128, 128))  # blank camera frame
        rel = Path("images") / cid / t.strftime("%Y%m%d") / f"{t:%H%M%S}.jpg"
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        (raw / rel).write_bytes(buf.getvalue())
        md5 = f"{cid}-{f}"
        idx_rows.append([cid, t.isoformat(timespec="seconds"), t.isoformat(timespec="seconds"), lat, lon, 1920, 1080,
                         md5, "", rel.as_posix(), "datamall"])
        truth.append([f"{cid}_{t:%Y%m%d%H%M%S}", int(vis), int(hour < 7 or hour >= 19)])
with open(raw / "images_index.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["camera_id", "image_ts", "poll_ts", "lat", "lon", "width", "height", "md5", "url", "path", "source"])
    w.writerows(idx_rows)
with open(raw / "cameras.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["camera_id", "lat", "lon"])
    for cid, (lat, lon) in cams.items():
        w.writerow([cid, lat, lon])
with open(out / "truth.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_id", "visible", "night"])
    w.writerows(truth)

# ---- incident feed, polled every 2 minutes
all_events = events + congestion + roadworks
with open(raw / "incidents.jsonl", "w") as fi, open(raw / "incident_polls.csv", "w") as fp:
    fp.write("poll_ts,n_active\n")
    for minute in range(0, HOURS * 60, 2):
        t = T0 + timedelta(minutes=minute)
        act = [(k, e) for k, e in enumerate(all_events) if active(e, minute)]
        for k, e in act:
            lat, lon = ev_latlon(e, k)
            rep = T0 + timedelta(minutes=e["start"] + 1)
            fi.write(json.dumps({"poll_ts": t.isoformat(timespec="seconds"), "type": e["type"], "lat": lat, "lon": lon,
                                 "message": f"({rep:%d/%m}){rep:%H:%M} {e['type']} on PIE (towards Tuas) after Exit {k}."}) + "\n")
        fp.write(f"{t.isoformat(timespec='seconds')},{len(act)}\n")

# ---- speed bands every 5 minutes: 30 short expressway links per camera
links = []
for cid, (lat, lon) in cams.items():
    for j in range(30):
        a = lat + rng.normal(0, 0.0012)
        b = lon + rng.normal(0, 0.0012)
        links.append((f"{cid}{j:03d}", cid, a, b))
for minute in range(0, HOURS * 60, 5):
    t = T0 + timedelta(minutes=minute)
    rows = []
    for lid, cid, a, b in links:
        band = int(rng.choice([6, 7, 7, 8]))
        slow = any(e.get("cam") == cid and active(e, minute) for e in events + congestion)
        if slow and rng.random() < 0.6:
            band = int(rng.choice([2, 3]))
        rows.append([t.isoformat(timespec="seconds"), lid, "PIE", "1", band, 10 * band, 10 * band + 9, a, b, a + 0.0002, b + 0.0002])
    with gzip.open(raw / "speedbands" / f"{t:%Y%m%d_%H%M%S}.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["poll_ts", "LinkID", "RoadName", "RoadCategory", "SpeedBand", "MinimumSpeed", "MaximumSpeed",
                    "StartLat", "StartLon", "EndLat", "EndLon"])
        w.writerows(rows)

# ---- a tiny traffic-signal shapefile (SVY21, approximate inverse of the projection) near two incidents
def svy21(lat, lon):
    return 28001.642 + (lon - 103.8333333) * 111319.5, 38744.572 + (lat - 1.3666667) * 110574.3


pts = []
for k in (0, 1):
    lat, lon = ev_latlon(events[k], k)
    x, y = svy21(lat, lon)
    pts += [(x + dx, y + dy) for dx, dy in ((0, 0), (8, 5), (-6, 9), (12, -7))]
pts += [(30000.0, 30000.0), (30010.0, 30004.0)]
recs = b"".join(struct.pack(">2i", i + 1, 10) + struct.pack("<i2d", 1, x, y) for i, (x, y) in enumerate(pts))
xs, ys = [p[0] for p in pts], [p[1] for p in pts]
hdr = (struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", (100 + len(recs)) // 2)
       + struct.pack("<2i", 1000, 1) + struct.pack("<4d", min(xs), min(ys), max(xs), max(ys)) + b"\0" * 32)
with zipfile.ZipFile(out / "static" / "TrafficLight_test.zip", "w") as z:
    z.writestr("TrafficLight_test/TrafficSignalAspect.shp", hdr + recs)
print(f"synthetic data in {out}: {len(idx_rows)} frames, {len(events)} incidents ({sum(e['visible'] for e in events)} visible)")
