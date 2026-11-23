#!/usr/bin/env python3
"""
Collector for Singapore open traffic data.

Runs four sources on independent timers:
  images        data.gov.sg traffic-images API (no key needed)
  incidents     LTA DataMall TrafficIncidents (needs LTA_ACCOUNT_KEY)
  speed bands   LTA DataMall TrafficSpeedBands (needs key; keeps only road links near cameras)
  faulty lights LTA DataMall FaultyTrafficLights (needs key)

Why it has to run live: the incident feed only shows what is happening now. There is
no public incident history, so incidents missed tonight are lost. Camera images are
less critical: data.gov.sg keeps image history, so gaps can be backfilled later.

Usage
  python collect.py --once        # one round of every source, prints what it got, then exits
  python collect.py --hours 16    # long run; Ctrl+C stops it cleanly; safe to restart

Put the key in .env next to this file as LTA_ACCOUNT_KEY=... (or export it).
Without a key, only images are collected.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency. Run: pip install -r requirements-collect.txt")

try:
    from PIL import Image
except ImportError:  # Pillow is optional; without it images are saved at full size
    Image = None

SGT = timezone(timedelta(hours=8))

DATAGOV_IMAGES_URL = "https://api.data.gov.sg/v1/transport/traffic-images"
DATAMALL_BASES = [
    "https://datamall2.mytransport.sg/ltaodataservice/",
    "http://datamall2.mytransport.sg/ltaodataservice/",
]
# DataMall has renamed and versioned endpoints over time; the first path that answers wins.
DATAMALL_PATHS = {
    "incidents": ["TrafficIncidents", "v3/TrafficIncidents", "v4/TrafficIncidents"],
    "speedbands": ["v4/TrafficSpeedBands", "v3/TrafficSpeedBands", "TrafficSpeedBandsv2", "TrafficSpeedBands"],
    "faultylights": ["FaultyTrafficLights", "v3/FaultyTrafficLights"],
    "images": ["Traffic-Imagesv2", "v3/Traffic-Imagesv2"],
}
PAGE_SIZE = 500  # DataMall returns at most 500 records per call
INDEX_FIELDS = ["camera_id", "image_ts", "poll_ts", "lat", "lon", "width", "height", "md5", "url", "path", "source"]
SPEED_FIELDS = ["poll_ts", "LinkID", "RoadName", "RoadCategory", "SpeedBand", "MinimumSpeed", "MaximumSpeed",
                "StartLat", "StartLon", "EndLat", "EndLon"]


# ----------------------------------------------------------------------------- helpers

def now() -> datetime:
    return datetime.now(SGT)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(SGT) if dt.tzinfo else dt.replace(tzinfo=SGT)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = "sg-incident-triage-collector/1.0 (independent research prototype)"
    return s


def save_image(content: bytes, dst: Path, max_width: int) -> tuple[int | None, int | None]:
    """Save a JPEG, downscaled to max_width. Returns the ORIGINAL (width, height)."""
    if Image is None:
        dst.write_bytes(content)
        return None, None
    with Image.open(io.BytesIO(content)) as im:
        w, h = im.size
        im = im.convert("RGB")
        if max_width and w > max_width:
            im = im.resize((max_width, round(h * max_width / w)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=85, optimize=True)
    return w, h


class Log:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, msg: str, echo: bool = True) -> None:
        line = f"[{now():%H:%M:%S}] {msg}"
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if echo:
            print(line, flush=True)


# ----------------------------------------------------------------------------- shared context

class Ctx:
    def __init__(self, args, key: str):
        self.args = args
        self.key = key
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.out / "collector.log")
        self.session = make_session()
        self.lock = threading.Lock()
        self.cameras: dict[str, tuple[float, float]] = {}
        self.stats = {
            "images_saved": 0, "image_bytes": 0, "image_polls": 0,
            "incident_polls": 0, "incidents_active": 0,
            "speed_polls": 0, "speed_links_kept": 0,
            "faulty_polls": 0, "faulty_active": 0,
            "errors": 0,
        }
        self.datamall_base: str | None = None
        self.datamall_paths: dict[str, str] = {}
        self.datamall_disabled = not bool(key)
        self.env_path = Path(__file__).resolve().with_name(".env")
        self.env_mtime = self.env_path.stat().st_mtime if self.env_path.exists() else None
        self._load_cameras()

    def refresh_key(self) -> None:
        """Pick up a key added to (or fixed in) .env while running, so no restart is needed."""
        if not self.env_path.exists():
            return
        mtime = self.env_path.stat().st_mtime
        if mtime == self.env_mtime:
            return
        self.env_mtime = mtime
        key = ""
        for line in self.env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("LTA_ACCOUNT_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if key and key != self.key:
            with self.lock:
                self.key = key
                self.datamall_disabled = False
                self.datamall_paths.clear()
                self.datamall_base = None
            self.log("LTA key loaded from .env: incidents, speed bands and faulty lights start now")

    # cameras ---------------------------------------------------------------
    def _load_cameras(self) -> None:
        p = self.out / "cameras.csv"
        if not p.exists():
            return
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    self.cameras[row["camera_id"]] = (float(row["lat"]), float(row["lon"]))
                except (KeyError, ValueError):
                    pass

    def update_cameras(self, cams: list[dict]) -> None:
        changed = False
        with self.lock:
            for c in cams:
                if c.get("lat") is None or c.get("lon") is None:
                    continue
                loc = (float(c["lat"]), float(c["lon"]))
                if self.cameras.get(c["camera_id"]) != loc:
                    self.cameras[c["camera_id"]] = loc
                    changed = True
            if changed:
                with open(self.out / "cameras.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["camera_id", "lat", "lon"])
                    for cid, (lat, lon) in sorted(self.cameras.items()):
                        w.writerow([cid, lat, lon])

    def cameras_snapshot(self) -> dict[str, tuple[float, float]]:
        with self.lock:
            return dict(self.cameras)

    def bump(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.stats[key] += n

    def set_stat(self, key: str, value) -> None:
        with self.lock:
            self.stats[key] = value

    # DataMall --------------------------------------------------------------
    def _dm_get(self, url: str) -> requests.Response:
        return self.session.get(url, headers={"AccountKey": self.key, "accept": "application/json"},
                                timeout=(10, 30))

    def _resolve(self, dataset: str) -> str | None:
        if dataset in self.datamall_paths:
            return self.datamall_paths[dataset]
        bases = [self.datamall_base] if self.datamall_base else list(self.args.datamall_base)
        statuses = []
        for base in bases:
            for path in DATAMALL_PATHS[dataset]:
                try:
                    r = self._dm_get(base + path)
                except requests.RequestException as e:
                    statuses.append(f"{base}{path}: {type(e).__name__}")
                    break  # this base is unreachable; try the next base
                statuses.append(f"{path}: HTTP {r.status_code}")
                if r.status_code == 200:
                    try:
                        if "value" in r.json():
                            self.datamall_base = base
                            self.datamall_paths[dataset] = path
                            self.log(f"{dataset}: using {base}{path}")
                            return path
                    except ValueError:
                        pass
        self.log(f"{dataset}: no working DataMall endpoint ({'; '.join(statuses)})")
        if statuses and all(("HTTP 401" in s or "HTTP 403" in s) for s in statuses):
            self.log("LTA DataMall rejected the key (401/403). Check LTA_ACCOUNT_KEY in .env. "
                     "Incidents, speed bands and faulty lights are paused; images continue.")
            self.datamall_disabled = True
        return None

    def datamall_all(self, dataset: str, max_pages: int = 400) -> list[dict] | None:
        """All records of a DataMall dataset (follows $skip paging). None on failure."""
        self.refresh_key()
        if self.datamall_disabled:
            return None
        path = self._resolve(dataset)
        if path is None:
            return None
        rows: list[dict] = []
        for page in range(max_pages):
            url = self.datamall_base + path + (f"?$skip={page * PAGE_SIZE}" if page else "")
            try:
                r = self._dm_get(url)
            except requests.RequestException as e:
                self.log(f"{dataset}: {type(e).__name__} on page {page}")
                self.bump("errors")
                return None
            if r.status_code != 200:
                self.log(f"{dataset}: HTTP {r.status_code} on page {page}")
                self.bump("errors")
                return None
            try:
                vals = r.json().get("value", [])
            except ValueError:
                self.log(f"{dataset}: response was not JSON on page {page}")
                self.bump("errors")
                return None
            rows.extend(vals)
            if len(vals) < PAGE_SIZE:
                break
        return rows


# ----------------------------------------------------------------------------- sources

class ImageSource:
    name = "images"

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.index_path = ctx.out / "images_index.csv"
        self.seen: set[str] = set()
        self.last_kept: dict[str, datetime] = {}
        self.lock = threading.Lock()
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        with open(self.index_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.seen.add(row.get("md5") or row.get("url") or "")
                ts = parse_ts(row.get("image_ts"))
                cid = row.get("camera_id")
                if ts and cid and (cid not in self.last_kept or ts > self.last_kept[cid]):
                    self.last_kept[cid] = ts

    def _list_datagov(self) -> list[dict] | None:
        headers = {}
        if os.environ.get("DATAGOV_API_KEY"):
            headers["x-api-key"] = os.environ["DATAGOV_API_KEY"]
        try:
            r = self.ctx.session.get(self.ctx.args.datagov_url, headers=headers, timeout=(10, 30))
        except requests.RequestException as e:
            self.ctx.log(f"images: data.gov.sg {type(e).__name__}")
            return None
        if r.status_code != 200:
            self.ctx.log(f"images: data.gov.sg HTTP {r.status_code}")
            return None
        try:
            items = r.json().get("items") or []
        except ValueError:
            self.ctx.log("images: data.gov.sg response was not JSON")
            return None
        if not items:
            return None
        cams = []
        for cam in items[0].get("cameras", []):
            loc = cam.get("location") or {}
            meta = cam.get("image_metadata") or {}
            if not cam.get("image"):
                continue
            cams.append({
                "camera_id": str(cam.get("camera_id")),
                "image_ts": parse_ts(cam.get("timestamp")) or now(),
                "lat": loc.get("latitude"), "lon": loc.get("longitude"),
                "url": cam["image"], "md5": meta.get("md5") or "",
                "width": meta.get("width") or "", "height": meta.get("height") or "",
                "source": "datagov",
            })
        return cams

    def _list_datamall(self) -> list[dict] | None:
        rows = self.ctx.datamall_all("images")
        if not rows:
            return None
        t = now()
        out = []
        for r in rows:
            link = r.get("ImageLink")
            if not link:
                continue
            # links look like .../1001_0904_20261123001607_ab12cd.jpg?X-Amz-...: the 14 digits are the capture time
            m = re.search(r"_(\d{14})_", link.split("?")[0])
            ts = t
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=SGT)
                except ValueError:
                    pass
            out.append({"camera_id": str(r.get("CameraID")), "image_ts": ts,
                        "lat": r.get("Latitude"), "lon": r.get("Longitude"),
                        "url": link, "md5": "", "width": "", "height": "", "source": "datamall"})
        return out

    def _download(self, c: dict) -> dict | None:
        try:
            r = self.ctx.session.get(c["url"], timeout=(10, 30))
        except requests.RequestException:
            self.ctx.bump("errors")
            return None
        if r.status_code != 200 or not r.content:
            self.ctx.bump("errors")
            return None
        md5 = c["md5"] or hashlib.md5(r.content).hexdigest()
        with self.lock:
            if md5 in self.seen:
                return None
            self.seen.add(md5)
        ts = c["image_ts"]
        rel = Path("images") / c["camera_id"] / ts.strftime("%Y%m%d") / f"{ts:%H%M%S}.jpg"
        dst = self.ctx.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            w, h = save_image(r.content, dst, self.ctx.args.max_width)
        except Exception as e:  # a corrupt image should not stop the run
            self.ctx.log(f"images: could not save camera {c['camera_id']}: {e!r}", echo=False)
            self.ctx.bump("errors")
            return None
        size = dst.stat().st_size
        self.ctx.bump("image_bytes", size)
        return {"md5": md5, "path": rel.as_posix(), "width": w or c["width"], "height": h or c["height"]}

    def poll(self) -> dict | None:
        # DataMall lists every LTA camera (needs the key); data.gov.sg lists only a few now, but
        # with permanent links. Use both; where a camera is in both, keep the data.gov.sg entry.
        dm = None if self.ctx.datamall_disabled else self._list_datamall()
        dg = self._list_datagov()
        merged = {c["camera_id"]: c for c in (dm or [])}
        merged.update({c["camera_id"]: c for c in (dg or [])})
        cams = list(merged.values())
        if not cams:
            return None
        self.ctx.update_cameras(cams)
        spacing = self.ctx.args.image_spacing
        todo = []
        for c in cams:
            key = c["md5"] or c["url"]
            if c["md5"] and key in self.seen:
                continue
            last = self.last_kept.get(c["camera_id"])
            if last and (c["image_ts"] - last).total_seconds() < spacing:
                continue
            todo.append(c)
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(self._download, todo))
        poll_ts = iso(now())
        rows = []
        for c, res in zip(todo, results):
            if not res:
                continue
            self.last_kept[c["camera_id"]] = c["image_ts"]
            rows.append({
                "camera_id": c["camera_id"], "image_ts": iso(c["image_ts"]), "poll_ts": poll_ts,
                "lat": c["lat"], "lon": c["lon"], "width": res["width"], "height": res["height"],
                "md5": res["md5"], "url": c["url"] if c["source"] == "datagov" else "",
                "path": res["path"], "source": c["source"],
            })
        if rows:
            new = not self.index_path.exists()
            with open(self.index_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
        self.ctx.bump("images_saved", len(rows))
        self.ctx.bump("image_polls")
        return {"listed": len(cams), "new": len(rows),
                "from_datamall": sum(c["source"] == "datamall" for c in cams),
                "from_datagov": sum(c["source"] == "datagov" for c in cams)}


class IncidentSource:
    needs_key = True
    name = "incidents"

    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    def poll(self) -> dict | None:
        rows = self.ctx.datamall_all("incidents")
        if rows is None:
            return None
        poll_ts = iso(now())
        with open(self.ctx.out / "incidents.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"poll_ts": poll_ts, "type": r.get("Type"), "lat": r.get("Latitude"),
                                    "lon": r.get("Longitude"), "message": r.get("Message")},
                                   ensure_ascii=False) + "\n")
        # one line per poll, even when empty: tells the pipeline when the feed was being watched
        new = not (self.ctx.out / "incident_polls.csv").exists()
        with open(self.ctx.out / "incident_polls.csv", "a", encoding="utf-8") as f:
            if new:
                f.write("poll_ts,n_active\n")
            f.write(f"{poll_ts},{len(rows)}\n")
        self.ctx.bump("incident_polls")
        self.ctx.set_stat("incidents_active", len(rows))
        types: dict[str, int] = {}
        for r in rows:
            types[r.get("Type") or "?"] = types.get(r.get("Type") or "?", 0) + 1
        return {"active": len(rows), "types": types}


class SpeedBandSource:
    needs_key = True
    name = "speed bands"

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.near: dict[str, bool] = {}
        self.n_cams_cached = 0
        (ctx.out / "speedbands").mkdir(exist_ok=True)

    @staticmethod
    def _coords(r: dict) -> tuple[float, float, float, float] | None:
        try:
            if "StartLat" in r:
                return float(r["StartLat"]), float(r["StartLon"]), float(r["EndLat"]), float(r["EndLon"])
            parts = str(r.get("Location", "")).split()  # older format: "lat1 lon1 lat2 lon2"
            return tuple(float(p) for p in parts[:4]) if len(parts) >= 4 else None
        except (TypeError, ValueError):
            return None

    def _is_near(self, link_id: str, coords, cams) -> bool:
        if link_id in self.near:
            return self.near[link_id]
        lat = (coords[0] + coords[2]) / 2
        lon = (coords[1] + coords[3]) / 2
        radius = self.ctx.args.speed_radius_m
        deg = radius / 111_000 * 1.2
        hit = False
        for clat, clon in cams.values():
            if abs(lat - clat) > deg or abs(lon - clon) > deg:
                continue
            if haversine_m(lat, lon, clat, clon) <= radius:
                hit = True
                break
        self.near[link_id] = hit
        return hit

    def poll(self) -> dict | None:
        radius = self.ctx.args.speed_radius_m
        cams = self.ctx.cameras_snapshot()
        if radius > 0 and not cams:
            self.ctx.log("speed bands: waiting for the camera list from the first image poll", echo=False)
            return None
        if len(cams) != self.n_cams_cached:
            self.near.clear()
            self.n_cams_cached = len(cams)
        t0 = time.time()
        rows = self.ctx.datamall_all("speedbands")
        if rows is None:
            return None
        poll_ts = iso(now())
        kept = []
        for r in rows:
            coords = self._coords(r)
            if coords is None:
                continue
            link = str(r.get("LinkID"))
            if radius <= 0 or self._is_near(link, coords, cams):
                kept.append({
                    "poll_ts": poll_ts, "LinkID": link, "RoadName": r.get("RoadName"),
                    "RoadCategory": r.get("RoadCategory"), "SpeedBand": r.get("SpeedBand"),
                    "MinimumSpeed": r.get("MinimumSpeed"), "MaximumSpeed": r.get("MaximumSpeed"),
                    "StartLat": coords[0], "StartLon": coords[1], "EndLat": coords[2], "EndLon": coords[3],
                })
        dst = self.ctx.out / "speedbands" / f"{now():%Y%m%d_%H%M%S}.csv.gz"
        with gzip.open(dst, "wt", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SPEED_FIELDS)
            w.writeheader()
            w.writerows(kept)
        self.ctx.bump("speed_polls")
        self.ctx.set_stat("speed_links_kept", len(kept))
        return {"fetched": len(rows), "kept": len(kept), "seconds": round(time.time() - t0, 1)}


class FaultyLightSource:
    needs_key = True
    name = "faulty lights"

    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    def poll(self) -> dict | None:
        rows = self.ctx.datamall_all("faultylights")
        if rows is None:
            return None
        poll_ts = iso(now())
        with open(self.ctx.out / "faulty_lights.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"poll_ts": poll_ts, "records": rows}, ensure_ascii=False) + "\n")
        self.ctx.bump("faulty_polls")
        self.ctx.set_stat("faulty_active", len(rows))
        return {"active": len(rows)}


# ----------------------------------------------------------------------------- running

def loop(src, interval: float, stop: threading.Event, ctx: Ctx) -> None:
    fails = 0
    while not stop.is_set():
        t0 = time.time()
        try:
            res = src.poll()
            waiting_for_key = res is None and getattr(src, "needs_key", False) and ctx.datamall_disabled
            fails = 0 if (res is not None or waiting_for_key) else fails + 1
        except Exception as e:  # keep the thread alive whatever happens
            fails += 1
            ctx.bump("errors")
            ctx.log(f"{src.name}: unexpected error {e!r}")
        wait = interval * (1 if fails < 3 else min(4, fails - 1))  # back off when a source keeps failing
        stop.wait(max(1.0, wait - (time.time() - t0)))


def status_line(ctx: Ctx, prev_images: int) -> tuple[str, int]:
    s = dict(ctx.stats)
    mb = s["image_bytes"] / 1e6
    line = (f"images {s['images_saved']:,} (+{s['images_saved'] - prev_images}) {mb:,.0f} MB | "
            f"incidents: {s['incident_polls']} polls, {s['incidents_active']} active | "
            f"speed bands: {s['speed_polls']} polls, {s['speed_links_kept']:,} links | "
            f"faulty lights: {s['faulty_active']} | errors {s['errors']}")
    return line, s["images_saved"]


def run_once(ctx: Ctx, sources) -> int:
    ok = True
    for src in sources:
        t0 = time.time()
        try:
            res = src.poll()
        except Exception as e:
            res = None
            ctx.log(f"{src.name}: unexpected error {e!r}")
        took = time.time() - t0
        if res is None:
            ok = ok and src.name != "images"
            if src.name != "images" and not ctx.key:
                print(f"  {src.name:<13} skipped: no LTA key")
            elif src.name != "images" and ctx.datamall_disabled:
                print(f"  {src.name:<13} skipped: LTA key rejected (check .env)")
            else:
                print(f"  {src.name:<13} FAILED ({took:.1f}s). Details in {ctx.out / 'collector.log'}")
        else:
            print(f"  {src.name:<13} OK ({took:.1f}s): {json.dumps(res, default=str)[:300]}")
    return 0 if ok else 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw", help="output folder (default: data/raw)")
    p.add_argument("--once", action="store_true", help="poll every source once, print results, exit")
    p.add_argument("--hours", type=float, default=0, help="stop automatically after this many hours (0 = run until Ctrl+C)")
    p.add_argument("--lta-key", default="", help="LTA DataMall AccountKey (default: LTA_ACCOUNT_KEY from env or .env)")
    p.add_argument("--image-interval", type=float, default=60, help="seconds between image list polls")
    p.add_argument("--image-spacing", type=float, default=170, help="min seconds between saved frames per camera")
    p.add_argument("--max-width", type=int, default=1280, help="downscale saved images to this width (0 = keep)")
    p.add_argument("--incident-interval", type=float, default=120, help="seconds between incident polls")
    p.add_argument("--speed-interval", type=float, default=300, help="seconds between speed band polls")
    p.add_argument("--speed-radius-m", type=float, default=0, help="keep only road links within this distance of a camera (0 = keep every link)")
    p.add_argument("--faulty-interval", type=float, default=600, help="seconds between faulty-light polls")
    p.add_argument("--status-every", type=float, default=60, help=argparse.SUPPRESS)
    p.add_argument("--datagov-url", default=DATAGOV_IMAGES_URL, help=argparse.SUPPRESS)
    p.add_argument("--datamall-base", nargs="+", default=DATAMALL_BASES, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    key = (args.lta_key or os.environ.get("LTA_ACCOUNT_KEY", "")).strip()
    ctx = Ctx(args, key)

    print(f"Output folder: {ctx.out.resolve()}")
    if not key:
        print("No LTA_ACCOUNT_KEY found: collecting camera images only. Incident labels need the key.")
    if Image is None:
        print("Pillow not installed: images are saved at full size (larger on disk).")

    image_src = ImageSource(ctx)
    sources = [image_src, IncidentSource(ctx), SpeedBandSource(ctx), FaultyLightSource(ctx)]
    if args.once:
        print("One round of every source:")
        return run_once(ctx, sources)

    stop = threading.Event()
    intervals = [args.image_interval, args.incident_interval, args.speed_interval, args.faulty_interval]
    threads = []
    image_src.poll()  # first image poll in the foreground, so speed bands know where the cameras are
    for src, iv in zip(sources, intervals):
        t = threading.Thread(target=loop, args=(src, iv, stop, ctx), daemon=True)
        t.start()
        threads.append(t)

    end = time.time() + args.hours * 3600 if args.hours > 0 else None
    ctx.log("collector started" + (f", will stop after {args.hours:g} h" if end else ", Ctrl+C to stop"))
    prev = 0
    try:
        while not stop.is_set():
            stop.wait(args.status_every)
            line, prev = status_line(ctx, prev)
            ctx.log(line)
            (ctx.out / "status.json").write_text(json.dumps({"updated": iso(now()), **ctx.stats}, indent=1))
            if end and time.time() >= end:
                stop.set()
    except KeyboardInterrupt:
        print("\nStopping...")
        stop.set()
    for t in threads:
        t.join(timeout=40)
    line, _ = status_line(ctx, prev)
    ctx.log("collector stopped. " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
