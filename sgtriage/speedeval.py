"""Speed-drop incident detection, scored against LTA's live incident feed.

Two detectors, both deliberately simple and explainable:

  control chart   each road link against its own last hour: flag a drop of >= 2 speed bands.
                  Catches incidents, and also ordinary congestion.
  up/downstream   along the same road, compare a link with the links just after it. Congestion
                  slows neighbours together; a blockage leaves slow traffic before it and a nearly
                  empty road after it. Flag when "downstream minus upstream" jumps above its own
                  last-hour level. (The classic freeway incident-detection idea: a normal relationship
                  between neighbouring sensors breaks.)

Flagged links are grouped into alarms (nearby links in the same poll, tracked across polls).
An alarm matches a reported incident if it is within 1 km and overlaps the incident's active time.
Reported 'Heavy Traffic' is kept separate: an alarm matching it is congestion, not a miss or a hit.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, haversine_m, load_incident_polls, load_incident_sightings, load_speedbands, write_json
from .labels import build_events
from .static_layers import cluster_points, load_junctions

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(1.35))


def _xy(lat, lon):
    return np.asarray(lon) * M_PER_DEG_LON, np.asarray(lat) * M_PER_DEG_LAT


def band_matrix(speed: pd.DataFrame, categories: set[str]) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    sp = speed[speed["RoadCategory"].astype(str).str.strip().isin(categories)]
    links = (sp.drop_duplicates("LinkID")[["LinkID", "RoadName", "RoadCategory", "StartLat", "StartLon", "EndLat",
                                           "EndLon", "mid_lat", "mid_lon"]].reset_index(drop=True))
    m = sp.pivot_table(index="LinkID", columns="poll_ts", values="SpeedBand", aggfunc="first")
    m = m.reindex(links["LinkID"])
    return links, m.to_numpy(dtype=float), pd.DatetimeIndex(m.columns)


def rolling_median_prev(x: np.ndarray, times: pd.DatetimeIndex, minutes: int = 60) -> np.ndarray:
    """Median of each row over the previous `minutes` (current poll excluded)."""
    out = np.full_like(x, np.nan)
    t = (times - times[0]).total_seconds().to_numpy()
    for j in range(x.shape[1]):
        lo = np.searchsorted(t, t[j] - minutes * 60, side="left")
        if j - lo >= 3:  # need at least 3 earlier polls
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[:, j] = np.nanmedian(x[:, lo:j], axis=1)
    return out


def downstream_index(links: pd.DataFrame, join_m: float = 20.0, reach_m: float = 500.0) -> list[np.ndarray]:
    """For each link, the links reachable downstream on the same road within reach_m (by chaining ends to starts)."""
    sx, sy = _xy(links["StartLat"], links["StartLon"])
    ex, ey = _xy(links["EndLat"], links["EndLon"])
    length = np.hypot(ex - sx, ey - sy)
    roads = links["RoadName"].fillna("").to_numpy()
    cells: dict[tuple[str, int, int], list[int]] = {}
    for i in range(len(links)):
        cells.setdefault((roads[i], int(sx[i] // join_m), int(sy[i] // join_m)), []).append(i)
    nxt: list[list[int]] = [[] for _ in range(len(links))]
    for i in range(len(links)):
        cx, cy = int(ex[i] // join_m), int(ey[i] // join_m)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((roads[i], cx + dx, cy + dy), ()):
                    if j != i and np.hypot(sx[j] - ex[i], sy[j] - ey[i]) <= join_m:
                        nxt[i].append(j)
    down = []
    for i in range(len(links)):
        seen, frontier, acc = set(), [(j, length[j]) for j in nxt[i]], []
        while frontier:
            j, d = frontier.pop()
            if j in seen or d > reach_m:
                continue
            seen.add(j)
            acc.append(j)
            frontier.extend((k, d + length[k]) for k in nxt[j])
        down.append(np.array(sorted(acc), dtype=int))
    return down


def flags_control(x: np.ndarray, base: np.ndarray, drop: float = 2.0) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (base - x >= drop) & (x <= 4)


def flags_updown(x: np.ndarray, times: pd.DatetimeIndex, down: list[np.ndarray], jump: float = 3.0) -> np.ndarray:
    gap = np.full_like(x, np.nan)
    for i, d in enumerate(down):
        if len(d):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                gap[i] = np.nanmedian(x[d], axis=0) - x[i]
    base = rolling_median_prev(gap, times)
    with np.errstate(invalid="ignore"):
        return (gap - base >= jump) & (x <= 3)


def alarms_from_flags(flags: np.ndarray, links: pd.DataFrame, times: pd.DatetimeIndex, eps_m: float = 250.0,
                      min_links: int = 3, link_m: float = 600.0, min_polls: int = 2) -> pd.DataFrame:
    """Group flagged links into spatial clusters per poll, then chain clusters across polls into alarms."""
    x, y = _xy(links["mid_lat"], links["mid_lon"])
    cells = []
    for j in range(flags.shape[1]):
        idx = np.flatnonzero(flags[:, j])
        if len(idx) < min_links:
            continue
        lab = cluster_points(x[idx], y[idx], eps_m)
        for c in np.unique(lab):
            members = idx[lab == c]
            if len(members) >= min_links:
                cells.append((j, x[members].mean(), y[members].mean(), len(members)))
    alarms, open_ = [], []
    for j, cx, cy, n in sorted(cells):
        for a in open_:
            if a["last"] in (j - 1, j - 2) and np.hypot(a["x"] - cx, a["y"] - cy) <= link_m:
                a.update(last=j, x=cx, y=cy, n=max(a["n"], n), polls=a["polls"] + 1, pts=a["pts"] + [(cx, cy)])
                break
        else:
            a = {"first": j, "last": j, "x": cx, "y": cy, "n": n, "polls": 1, "pts": [(cx, cy)]}
            alarms.append(a)
            open_.append(a)
        open_ = [a for a in open_ if a["last"] >= j - 2]
    rows = []
    for k, a in enumerate(a for a in alarms if a["polls"] >= min_polls):
        lat = np.array([p[1] for p in a["pts"]]) / M_PER_DEG_LAT
        lon = np.array([p[0] for p in a["pts"]]) / M_PER_DEG_LON
        rows.append({"alarm_id": k, "onset": times[a["first"]], "end": times[a["last"]], "polls": a["polls"],
                     "max_links": a["n"], "lat": float(lat[0]), "lon": float(lon[0]),
                     "lats": lat.round(6).tolist(), "lons": lon.round(6).tolist()})
    return pd.DataFrame(rows)


def match(alarms: pd.DataFrame, events: pd.DataFrame, radius_m: float = 1000.0, pre_min: int = 15, post_min: int = 10):
    """Which alarms explain which reported events."""
    pairs = []
    for e in events.itertuples(index=False):
        a0, a1 = e.start - pd.Timedelta(minutes=pre_min), e.end + pd.Timedelta(minutes=post_min)
        for a in alarms.itertuples(index=False):
            if a.end < a0 or a.onset > a1:
                continue
            d = haversine_m(np.array(a.lats), np.array(a.lons), e.lat, e.lon).min()
            if d <= radius_m:
                pairs.append({"event_id": e.event_id, "alarm_id": a.alarm_id, "dist_m": round(float(d)),
                              "delay_min": round((a.onset - e.reported_at).total_seconds() / 60, 1)})
    return pd.DataFrame(pairs, columns=["event_id", "alarm_id", "dist_m", "delay_min"])


def evaluate(paths: Paths, categories=("1", "2"), observe_m: float = 300.0) -> dict:
    speed = load_speedbands(paths.raw)
    if speed.empty:
        raise SystemExit("No speed-band data in " + str(paths.raw))
    links, x, times = band_matrix(speed, set(categories))
    events = build_events(load_incident_sightings(paths.raw), load_junctions(paths.static))
    polls = load_incident_polls(paths.raw)
    t0, t1 = times.min(), times.max()
    events = events[(events["start"] >= t0 + pd.Timedelta(minutes=60)) & (events["start"] <= t1)]  # baseline first
    # an incident is only observable if speed data covers a link near it
    lx, ly = links["mid_lat"].to_numpy(), links["mid_lon"].to_numpy()
    if len(events):
        d = haversine_m(events["lat"].to_numpy()[:, None], events["lon"].to_numpy()[:, None], lx[None, :], ly[None, :])
        events = events.assign(observable=(d <= observe_m).any(axis=1))
    inc = events[(events["category"] == "incident") & events.get("observable", False)]
    cong = events[(events["category"] == "congestion") & events.get("observable", False)]

    base = rolling_median_prev(x, times)
    down = downstream_index(links)
    detectors = {"control_chart": flags_control(x, base), "up_downstream": flags_updown(x, times, down)}
    hours = (t1 - t0).total_seconds() / 3600 - 1
    res = {"window": {"start": t0.isoformat(), "end": t1.isoformat(), "hours_scored": round(hours, 1),
                      "polls": int(len(times)), "links": int(len(links)), "categories": list(categories),
                      "links_with_downstream": int(sum(len(d_) > 0 for d_ in down))},
           "events": {"incidents": int(len(inc)), "congestion_reports": int(len(cong)),
                      "by_type": inc["type"].value_counts().to_dict(),
                      "at_junction": int(inc["at_junction"].fillna(False).astype(bool).sum()) if len(inc) else 0,
                      "feed_polls": int(len(polls))},
           "detectors": {}}
    all_alarms = {}
    for name, fl in detectors.items():
        al = alarms_from_flags(fl, links, times)
        al = al[al["onset"] >= t0 + pd.Timedelta(minutes=60)] if len(al) else al
        mi = match(al, inc) if len(al) and len(inc) else pd.DataFrame(columns=["event_id", "alarm_id", "dist_m", "delay_min"])
        mc = match(al, cong) if len(al) and len(cong) else pd.DataFrame(columns=["event_id", "alarm_id", "dist_m", "delay_min"])
        hit = set(mi["event_id"])
        first = mi.sort_values("delay_min").drop_duplicates("event_id") if len(mi) else mi
        n_al = int(len(al))
        a_inc, a_cong = set(mi["alarm_id"]), set(mc["alarm_id"]) - set(mi["alarm_id"])
        res["detectors"][name] = {
            "alarms": n_al, "alarms_per_hour": round(n_al / hours, 1) if hours > 0 else None,
            "incidents_detected": len(hit), "incident_recall": round(len(hit) / len(inc), 3) if len(inc) else None,
            "median_delay_min": float(first["delay_min"].median()) if len(first) else None,
            "detected_before_report": int((first["delay_min"] < 0).sum()) if len(first) else 0,
            "alarms_matching_incident": len(a_inc), "alarms_matching_congestion": len(a_cong),
            "alarms_unexplained": n_al - len(a_inc) - len(a_cong),
            "precision_incident": round(len(a_inc) / n_al, 3) if n_al else None,
            "by_type_recall": {t: round(float(g["event_id"].isin(hit).mean()), 2) for t, g in inc.groupby("type")},
        }
        all_alarms[name] = al
        if len(al):
            al.assign(onset=al["onset"].astype(str), end=al["end"].astype(str)).to_json(
                paths.work / f"speed_alarms_{name}.json", orient="records")
    # incidents table for the map / report
    rows = []
    for e in inc.itertuples(index=False):
        rows.append({"event_id": e.event_id, "type": e.type, "lat": e.lat, "lon": e.lon, "message": e.message,
                     "reported_at": e.reported_at.isoformat(), "at_junction": bool(e.at_junction) if not pd.isna(e.at_junction) else False})
    res["incidents"] = rows
    res["timeline"] = _timeline(inc, all_alarms, t0, t1)
    write_json(paths.work / "speed_metrics.json", res)
    return res


def _timeline(inc: pd.DataFrame, alarms: dict, t0, t1, freq: str = "30min") -> dict:
    bins = pd.date_range(t0.floor(freq), t1.ceil(freq), freq=freq)
    out = {"bins": [b.isoformat() for b in bins[:-1]],
           "incidents": np.histogram(inc["reported_at"].astype("int64"), bins.astype("int64"))[0].tolist() if len(inc) else [0] * (len(bins) - 1)}
    for name, al in alarms.items():
        out[name] = (np.histogram(al["onset"].astype("int64"), bins.astype("int64"))[0].tolist() if len(al)
                     else [0] * (len(bins) - 1))
    return out


def figure(res: dict, out_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tl = res.get("timeline") or {}
    if not tl.get("bins"):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.to_datetime(tl["bins"])
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.bar(t, tl["incidents"], width=0.018, color="#D22B22", label="incidents reported")
    for name, col in (("control_chart", "#9aa3ad"), ("up_downstream", "#006B3F")):
        if name in tl:
            ax.plot(t, tl[name], color=col, lw=1.6, label={"control_chart": "alarms: control chart",
                                                          "up_downstream": "alarms: up/downstream"}[name])
    ax.set_ylabel("per 30 min")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=t.tz))
    fig.tight_layout()
    p = out_dir / "fig_speed_timeline.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
