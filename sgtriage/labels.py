"""Weak labels from LTA's incident feed.

The feed is polled every ~2 minutes and only shows incidents that are active right now.
Each poll repeats every active incident, so we collapse sightings into events, then match
events to nearby cameras and mark the camera frames inside each event's time window.

These labels are weak on purpose and we say so:
  - a reported incident 400 m away may be out of the camera's view;
  - the feed's time is when it was reported, not when it happened;
  - incidents the feed never reported are invisible to it.
That is why a human labels a sample, and why the feed is used to measure recall, not truth.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from .common import TZ, haversine_m

INCIDENT_TYPES = {"accident", "vehicle breakdown", "obstacle", "unattended vehicle", "road block",
                  "fire", "reverse flow"}
CONGESTION_TYPES = {"heavy traffic"}
_MSG_TIME = re.compile(r"^\s*\((\d{1,2})/(\d{1,2})\)\s*(\d{1,2}):(\d{2})")


def parse_reported(message: str, poll_ts: pd.Timestamp) -> pd.Timestamp:
    """'(23/11)08:12 Accident on PIE ...' -> 2026-11-23 08:12+08:00 (year taken from the poll)."""
    m = _MSG_TIME.match(message or "")
    if not m:
        return pd.NaT
    day, month, hour, minute = map(int, m.groups())
    try:
        ts = pd.Timestamp(year=poll_ts.year, month=month, day=day, hour=hour, minute=minute, tz=TZ)
    except ValueError:
        return pd.NaT
    if ts - poll_ts > pd.Timedelta(days=180):  # reported in December, polled in January
        ts = ts.replace(year=ts.year - 1)
    return ts


def category(type_norm: str) -> str:
    if type_norm in INCIDENT_TYPES:
        return "incident"
    if type_norm in CONGESTION_TYPES:
        return "congestion"
    return "other"


def build_events(sightings: pd.DataFrame, junctions: pd.DataFrame | None = None,
                 merge_m: float = 300.0, junction_m: float = 40.0) -> pd.DataFrame:
    """Collapse repeated sightings into one row per incident."""
    cols = ["event_id", "type", "category", "reported_at", "first_seen", "last_seen", "start", "end",
            "lat", "lon", "message", "n_sightings", "junction_id", "junction_dist_m", "at_junction"]
    if sightings.empty:
        return pd.DataFrame(columns=cols)
    s = sightings.copy()
    s["type_norm"] = s["type"].fillna("").str.strip().str.lower()
    s["reported_at"] = [parse_reported(m, p) for m, p in zip(s["message"], s["poll_ts"])]
    s["reported_at"] = s["reported_at"].fillna(s.groupby(["type_norm", "message"])["poll_ts"].transform("min"))

    rows = []
    for (type_norm, reported), g in s.groupby(["type_norm", "reported_at"], sort=False):
        # same type + same reported minute: usually one incident whose message was updated;
        # split only if the sightings are far apart (two incidents reported in the same minute)
        g = g.sort_values("poll_ts")
        groups: list[list[int]] = []
        centres: list[tuple[float, float]] = []
        for idx, r in g.iterrows():
            for gi, (clat, clon) in enumerate(centres):
                if haversine_m(r["lat"], r["lon"], clat, clon) <= merge_m:
                    groups[gi].append(idx)
                    break
            else:
                groups.append([idx])
                centres.append((r["lat"], r["lon"]))
        for members in groups:
            e = g.loc[members]
            first, last = e["poll_ts"].min(), e["poll_ts"].max()
            key = f"{type_norm}|{reported.isoformat()}|{e['lat'].median():.4f}|{e['lon'].median():.4f}"
            rows.append({
                "event_id": "E" + hashlib.sha1(key.encode()).hexdigest()[:8],
                "type": e["type"].iloc[-1], "category": category(type_norm),
                "reported_at": reported, "first_seen": first, "last_seen": last,
                "start": min(reported, first), "end": last,
                "lat": e["lat"].median(), "lon": e["lon"].median(),
                "message": e["message"].iloc[-1], "n_sightings": len(e),
            })
    ev = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

    if junctions is not None and len(junctions):
        d = haversine_m(ev["lat"].to_numpy()[:, None], ev["lon"].to_numpy()[:, None],
                        junctions["lat"].to_numpy()[None, :], junctions["lon"].to_numpy()[None, :])
        nearest = d.argmin(axis=1)
        ev["junction_id"] = junctions["junction_id"].to_numpy()[nearest]
        ev["junction_dist_m"] = d[np.arange(len(ev)), nearest].round(1)
        ev["at_junction"] = ev["junction_dist_m"] <= junction_m
    else:
        ev["junction_id"], ev["junction_dist_m"], ev["at_junction"] = "", np.nan, False
    return ev[cols]


def match_cameras(events: pd.DataFrame, cameras: pd.DataFrame, radius_m: float = 600.0) -> pd.DataFrame:
    """Every (event, camera) pair within radius_m."""
    if events.empty or cameras.empty:
        return pd.DataFrame(columns=["event_id", "camera_id", "dist_m"])
    d = haversine_m(events["lat"].to_numpy()[:, None], events["lon"].to_numpy()[:, None],
                    cameras["lat"].to_numpy()[None, :], cameras["lon"].to_numpy()[None, :])
    ei, ci = np.nonzero(d <= radius_m)
    return pd.DataFrame({"event_id": events["event_id"].to_numpy()[ei],
                         "camera_id": cameras["camera_id"].to_numpy()[ci],
                         "dist_m": d[ei, ci].round(1)}).sort_values(["event_id", "dist_m"]).reset_index(drop=True)


def weak_labels(frames: pd.DataFrame, events: pd.DataFrame, pairs: pd.DataFrame, polls: pd.DataFrame,
                pre_min: float = 5.0, post_min: float = 0.0, coverage_min: float = 6.0) -> pd.DataFrame:
    """Add weak-label columns to frames.

    covered     the incident feed was being polled within coverage_min of the frame
    weak_label  1 inside an incident window at a matched camera, 0 if covered and outside, NaN if not covered
    weak_congestion  same, for 'Heavy Traffic' reports (kept separate: congestion is not an incident)
    """
    f = frames.copy()
    f["covered"] = False
    if len(polls):
        pt = polls["poll_ts"].sort_values().to_numpy()
        ft = f["image_ts"].to_numpy()
        pos = np.searchsorted(pt, ft)
        lo = pt[np.clip(pos - 1, 0, len(pt) - 1)]
        hi = pt[np.clip(pos, 0, len(pt) - 1)]
        gap = np.minimum(np.abs(ft - lo), np.abs(hi - ft))
        f["covered"] = gap <= np.timedelta64(int(coverage_min * 60), "s")
    f["weak_label"] = np.where(f["covered"], 0.0, np.nan)
    f["weak_congestion"] = np.where(f["covered"], 0.0, np.nan)
    f["weak_event_id"] = ""
    f["weak_event_type"] = ""
    f["weak_dist_m"] = np.nan

    if events.empty or pairs.empty:
        return f
    ev = events.set_index("event_id")
    by_cam = {c: g.index.to_numpy() for c, g in f.groupby("camera_id")}
    pre, post = pd.Timedelta(minutes=pre_min), pd.Timedelta(minutes=post_min)
    for r in pairs.itertuples(index=False):
        e = ev.loc[r.event_id]
        if e["category"] not in ("incident", "congestion") or r.camera_id not in by_cam:
            continue
        idx = by_cam[r.camera_id]
        t = f.loc[idx, "image_ts"]
        inside = idx[((t >= e["start"] - pre) & (t <= e["end"] + post)).to_numpy()]
        if not len(inside):
            continue
        if e["category"] == "congestion":
            f.loc[inside, "weak_congestion"] = 1.0
            continue
        # if two incidents overlap at one camera, keep the closer one
        closer = inside[~(f.loc[inside, "weak_dist_m"] <= r.dist_m).to_numpy()]
        f.loc[inside, "weak_label"] = 1.0
        f.loc[closer, "weak_event_id"] = r.event_id
        f.loc[closer, "weak_event_type"] = e["type"]
        f.loc[closer, "weak_dist_m"] = r.dist_m
    return f
