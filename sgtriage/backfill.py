"""Historical camera frames from data.gov.sg, around news-reported incidents.

Since about July 2026 LTA publishes live images for only 8 cameras, but data.gov.sg still serves
history for all ~90 expressway cameras up to mid-2026. For each incident in a CSV of news reports
(date, time, approximate location) this downloads every camera's frames from 20 minutes before
to 40 minutes after the reported time, plus the same window one week earlier as a matched
control (same cameras, same time of day, no known incident).

News reports are used only for *when and where*; the images are LTA's own camera frames.
Output is in the collector's format, so the rest of the pipeline runs unchanged on --raw data/raw_hist.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from .common import TZ

API = "https://api.data.gov.sg/v1/transport/traffic-images"
INDEX_FIELDS = ["camera_id", "image_ts", "poll_ts", "lat", "lon", "width", "height", "md5", "url", "path", "source",
                "window_id"]


def _session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                                                      allowed_methods=("GET",)), pool_maxsize=16))
    s.headers["User-Agent"] = "sg-incident-triage-backfill/1.0 (independent research prototype)"
    return s


def windows_for(incidents: pd.DataFrame, before: int, after: int, step: int, control_days: int):
    out = []
    for r in incidents.itertuples(index=False):
        t0 = datetime.fromisoformat(f"{r.date}T{r.time}:00").replace(tzinfo=TZ)
        for kind, shift in (("incident", 0), ("control", control_days)):
            if kind == "control" and control_days <= 0:
                continue
            base = t0 - timedelta(days=shift)
            ts = [base + timedelta(minutes=m) for m in range(-before, after + 1, step)]
            out.append({"window_id": f"{r.incident_id}-{kind[0]}", "incident_id": r.incident_id, "kind": kind,
                        "times": ts})
    return out


def run(incidents_csv: Path, out: Path, before: int = 20, after: int = 40, step: int = 5, control_days: int = 7,
        max_width: int = 1280, workers: int = 8, active_min: int = 40, api: str = API) -> None:
    inc = pd.read_csv(incidents_csv, dtype=str)
    inc["lat"], inc["lon"] = inc["lat"].astype(float), inc["lon"].astype(float)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    s = _session()
    index_path = out / "images_index.csv"
    seen = set()
    if index_path.exists():
        with open(index_path, newline="", encoding="utf-8") as f:
            seen = {r["md5"] for r in csv.DictReader(f)}
    lock = threading.Lock()
    cameras: dict[str, tuple[float, float]] = {}
    wins = windows_for(inc, before, after, step, control_days)
    n_calls = sum(len(w["times"]) for w in wins)
    print(f"{len(inc)} incidents -> {len(wins)} windows, {n_calls} snapshots of all cameras. Saving to {out}")

    def download(job):
        cam, win_id = job
        meta = cam.get("image_metadata") or {}
        md5 = meta.get("md5") or hashlib.md5(cam["image"].encode()).hexdigest()
        with lock:
            if md5 in seen:
                return None
            seen.add(md5)
        try:
            r = s.get(cam["image"], timeout=(10, 60))
            r.raise_for_status()
            ts = datetime.fromisoformat(cam["timestamp"]).astimezone(TZ)
            rel = Path("images") / str(cam["camera_id"]) / ts.strftime("%Y%m%d") / f"{ts:%H%M%S}.jpg"
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(io.BytesIO(r.content)) as im:
                w, h = im.size
                im = im.convert("RGB")
                if max_width and w > max_width:
                    im = im.resize((max_width, round(h * max_width / w)), Image.LANCZOS)
                im.save(dst, "JPEG", quality=85, optimize=True)
        except Exception as e:
            with lock:
                seen.discard(md5)
            return ("error", str(e)[:120])
        loc = cam.get("location") or {}
        return {"camera_id": str(cam["camera_id"]), "image_ts": ts.isoformat(timespec="seconds"),
                "poll_ts": ts.isoformat(timespec="seconds"), "lat": loc.get("latitude"), "lon": loc.get("longitude"),
                "width": w, "height": h, "md5": md5, "url": cam["image"], "path": rel.as_posix(),
                "source": "datagov-history", "window_id": win_id}

    done, errors, t_start = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for w in wins:
            for t in w["times"]:
                r = s.get(api, params={"date_time": t.strftime("%Y-%m-%dT%H:%M:%S")}, timeout=(10, 60))
                if r.status_code != 200:
                    print(f"  {t:%Y-%m-%d %H:%M}: HTTP {r.status_code}")
                    errors += 1
                    continue
                items = r.json().get("items") or []
                cams = items[0].get("cameras", []) if items else []
                for c in cams:
                    loc = c.get("location") or {}
                    cameras[str(c["camera_id"])] = (loc.get("latitude"), loc.get("longitude"))
                rows = [x for x in ex.map(download, [(c, w["window_id"]) for c in cams]) if x]
                good = [x for x in rows if isinstance(x, dict)]
                errors += len(rows) - len(good)
                if good:
                    new = not index_path.exists()
                    with open(index_path, "a", newline="", encoding="utf-8") as f:
                        wr = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
                        if new:
                            wr.writeheader()
                        wr.writerows(good)
                done += len(good)
                time.sleep(0.2)  # be polite to the API
            print(f"  {w['window_id']}: {done:,} frames so far, {errors} errors, {time.time() - t_start:,.0f}s")

    with open(out / "cameras.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["camera_id", "lat", "lon"])
        for cid, (lat, lon) in sorted(cameras.items()):
            wr.writerow([cid, lat, lon])

    # news incidents in the incident-feed format: sightings every 2 minutes while assumed active
    with open(out / "incidents.jsonl", "w", encoding="utf-8") as f:
        for r in inc.itertuples(index=False):
            t0 = datetime.fromisoformat(f"{r.date}T{r.time}:00").replace(tzinfo=TZ)
            msg = f"({t0:%d/%m}){t0:%H:%M} {r.type} on {r.road} (towards {r.direction}) {r.landmark}. [news report]"
            for m in range(0, active_min + 1, 2):
                f.write(json.dumps({"poll_ts": (t0 + timedelta(minutes=m)).isoformat(timespec="seconds"),
                                    "type": r.type, "lat": r.lat, "lon": r.lon, "message": msg,
                                    "source_url": r.source_url}) + "\n")
    (out / "settings.json").write_text(json.dumps({"camera_radius": 1000, "label_source": "news reports",
                                                   "kind": "history"}, indent=1), encoding="utf-8")
    (out / "windows.json").write_text(json.dumps([{**w, "times": [t.isoformat() for t in w["times"]]} for w in wins],
                                                 indent=1), encoding="utf-8")
    print(f"done: {done:,} frames from {len(cameras)} cameras in {time.time() - t_start:,.0f}s ({errors} errors)")
