"""Shared paths, loaders and geo helpers."""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TZ = timezone(timedelta(hours=8))  # Singapore has no daylight saving, so a fixed offset is exact
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    raw: Path
    work: Path
    static: Path
    site: Path

    @classmethod
    def make(cls, raw="data/raw", work="data/work", static="data/static", site="docs") -> "Paths":
        p = cls(Path(raw), Path(work), Path(static), Path(site))
        p.work.mkdir(parents=True, exist_ok=True)
        p.static.mkdir(parents=True, exist_ok=True)
        return p


def load_env() -> None:
    """Read KEY=VALUE lines from .env at the repo root into os.environ (existing values win)."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def to_sgt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_convert(TZ)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ----------------------------------------------------------------------------- raw loaders

def load_frames(raw: Path) -> pd.DataFrame:
    """One row per saved camera frame, sorted by camera then time, with prev_frame_id."""
    df = pd.read_csv(raw / "images_index.csv", dtype={"camera_id": str, "md5": str})
    df = df.dropna(subset=["path"]).drop_duplicates("md5")
    df["image_ts"] = to_sgt(df["image_ts"])
    df = df[[(raw / p).exists() for p in df["path"]]]
    df = df.sort_values(["camera_id", "image_ts"]).reset_index(drop=True)
    df["frame_id"] = df["camera_id"] + "_" + df["image_ts"].dt.strftime("%Y%m%d%H%M%S")
    df = df.drop_duplicates("frame_id").reset_index(drop=True)
    prev = df.groupby("camera_id")["frame_id"].shift(1)
    gap = df.groupby("camera_id")["image_ts"].diff().dt.total_seconds() / 60
    df["prev_frame_id"] = prev.where(gap <= 10)  # only a real "previous frame" if it is close in time
    df["minutes_since_prev"] = gap
    df["hour"] = df["image_ts"].dt.hour + df["image_ts"].dt.minute / 60
    return df


def load_cameras(raw: Path, frames: pd.DataFrame | None = None) -> pd.DataFrame:
    p = raw / "cameras.csv"
    if p.exists():
        cams = pd.read_csv(p, dtype={"camera_id": str})
    elif frames is not None:
        cams = frames.groupby("camera_id")[["lat", "lon"]].median().reset_index()
    else:
        raise FileNotFoundError(p)
    return cams.drop_duplicates("camera_id").reset_index(drop=True)


def load_incident_sightings(raw: Path) -> pd.DataFrame:
    p = raw / "incidents.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["poll_ts", "type", "lat", "lon", "message"])
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["poll_ts", "type", "lat", "lon", "message"])
    df["poll_ts"] = to_sgt(df["poll_ts"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def load_incident_polls(raw: Path) -> pd.DataFrame:
    p = raw / "incident_polls.csv"
    if not p.exists():
        return pd.DataFrame(columns=["poll_ts", "n_active"])
    df = pd.read_csv(p)
    df["poll_ts"] = to_sgt(df["poll_ts"])
    return df.sort_values("poll_ts").reset_index(drop=True)


def load_speedbands(raw: Path) -> pd.DataFrame:
    files = sorted((raw / "speedbands").glob("*.csv.gz"))
    parts = []
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                parts.append(pd.read_csv(fh, dtype={"LinkID": str, "RoadCategory": str}))
        except (EOFError, OSError, pd.errors.EmptyDataError):
            continue  # a file cut off by Ctrl+C mid-write
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["poll_ts"] = to_sgt(df["poll_ts"])
    for c in ["SpeedBand", "StartLat", "StartLon", "EndLat", "EndLon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["SpeedBand", "StartLat", "StartLon", "EndLat", "EndLon"])
    df["mid_lat"] = (df["StartLat"] + df["EndLat"]) / 2
    df["mid_lon"] = (df["StartLon"] + df["EndLon"]) / 2
    return df.reset_index(drop=True)


def read_csv_if(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def require(path: Path, step: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python -m sgtriage {step}` first.")
    return path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with a Wilson 95% interval; behaves sensibly for small n and k=0."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
