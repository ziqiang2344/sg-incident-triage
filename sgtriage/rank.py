"""Turn signals into rankings, then build the pool a human labels.

Methods compared (each produces one score per frame):
  random    review frames in random order: the baseline, and the estimate of how rare incidents are
  speed     speed drop on nearby road links (time series only)
  novelty   how unusual the frame is for its camera (image only)
  zeroshot  CLIP similarity to incident descriptions (image only, no training)
  fused     average rank of the three above: multimodal, still label-free
  learned   logistic regression on all features, trained on the feed's weak labels,
            scored out-of-fold so a frame is never scored by a model that saw its incident

Rankings are over *windows*, not frames: once a camera-time is picked, frames from the same
camera within ±15 minutes are skipped, so one long incident cannot fill the whole top-k.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

METHODS = ["random", "speed", "novelty", "zeroshot", "fused", "learned"]
LEARNED_FEATURES = ["speed_max_drop", "speed_mean_drop", "speed_frac_drop2", "speed_min_band", "novelty",
                    "change", "zeroshot", "brightness", "hour_sin", "hour_cos"]


def _pct(s: pd.Series) -> pd.Series:
    r = s.rank(pct=True)
    return r.fillna(0.5)


def base_scores(f: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    f = f.copy()
    avail = {}
    rng = np.random.default_rng(seed)
    f["score_random"] = rng.random(len(f))
    avail["random"] = True

    # DataMall v4 speed links are short (often 10-50 m), so a camera has dozens of them. The max over
    # many noisy links is almost always high; the share of links that dropped 2+ bands is robust.
    has_speed = f["speed_frac_drop2"].notna()
    avail["speed"] = bool(has_speed.any())
    f["score_speed"] = np.where(has_speed, f["speed_frac_drop2"] + 0.1 * f["speed_mean_drop"].fillna(0), np.nan)

    f["score_novelty"] = f["novelty"]
    avail["novelty"] = bool(f["novelty"].notna().any())
    f["score_zeroshot"] = f["zeroshot"]
    avail["zeroshot"] = bool(f["zeroshot"].notna().any())

    parts = [_pct(f[f"score_{m}"]) for m in ("speed", "novelty", "zeroshot") if avail[m]]
    f["score_fused"] = np.mean(parts, axis=0) if parts else np.nan
    avail["fused"] = len(parts) >= 2

    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    return f, avail


def _time_groups(f: pd.DataFrame, events: pd.DataFrame, block_h: float = 2.0) -> np.ndarray:
    """CV groups = 2-hour blocks; every frame of an incident joins the block where the incident started."""
    t0 = f["image_ts"].min()
    g = ((f["image_ts"] - t0).dt.total_seconds() // (block_h * 3600)).astype(int).to_numpy().copy()
    if len(events):
        start_block = {r.event_id: int((r.start - t0).total_seconds() // (block_h * 3600))
                       for r in events.itertuples() if pd.notna(r.start)}
        ev = f["weak_event_id"].fillna("").to_numpy()
        for i, e in enumerate(ev):
            if e in start_block:
                g[i] = max(0, start_block[e])
    return g


def learned_scores(f: pd.DataFrame, events: pd.DataFrame, min_events: int = 5, n_splits: int = 4):
    """Out-of-fold logistic regression on weak labels. Returns (scores or None, info dict)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in LEARNED_FEATURES if c in f and f[c].notna().any()]
    train = f["weak_label"].notna().to_numpy()
    y = f["weak_label"].fillna(0).to_numpy().astype(int)
    pos_events = f.loc[train & (y == 1), "weak_event_id"].replace("", np.nan).dropna().unique()
    info = {"features": feats, "n_pos_frames": int((train & (y == 1)).sum()), "n_pos_events": int(len(pos_events)),
            "n_train_frames": int(train.sum())}
    if len(pos_events) < min_events:
        info["skipped"] = f"only {len(pos_events)} reported incidents near cameras (need {min_events})"
        return None, info
    if int((train & (y == 0)).sum()) < 50:
        info["skipped"] = "no reliable negatives: incident reports were not watched continuously"
        return None, info
    X = f[feats].astype(float)
    X = X.fillna(X.median()).to_numpy()
    groups = _time_groups(f, events)
    tr = np.flatnonzero(train)
    n_groups = len(np.unique(groups[tr]))
    k = min(n_splits, n_groups)
    if k < 2:
        info["skipped"] = "not enough time blocks for cross-validation"
        return None, info

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))

    oof = np.full(len(f), np.nan)
    untrained = np.flatnonzero(~train)
    other = np.zeros(len(untrained))
    folds_used = 0
    for a, b in GroupKFold(n_splits=k).split(X[tr], y[tr], groups[tr]):
        ia, ib = tr[a], tr[b]
        if y[ia].sum() == 0:
            continue
        m = model().fit(X[ia], y[ia])
        oof[ib] = m.predict_proba(X[ib])[:, 1]
        if len(untrained):
            other += m.predict_proba(X[untrained])[:, 1]
        folds_used += 1
    if folds_used == 0:
        info["skipped"] = "no fold had positives to train on"
        return None, info
    if len(untrained):
        oof[untrained] = other / folds_used
    full = model().fit(X[tr], y[tr])
    coefs = full[-1].coef_[0]
    info["coefficients"] = dict(sorted(zip(feats, map(float, coefs)), key=lambda kv: -abs(kv[1])))
    info["folds"] = folds_used
    return oof, info


def score_all(f: pd.DataFrame, events: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    f, avail = base_scores(f, seed)
    learned, info = learned_scores(f, events)
    f["score_learned"] = learned if learned is not None else np.nan
    avail["learned"] = learned is not None
    # flat grey/black frames, or frames CLIP finds far more like "no signal" than like any road scene
    junk = f["junk"].fillna(False).astype(bool) | (f["zeroshot_junk"].fillna(-9) > 1.5)
    f["junk"] = junk
    for m in METHODS:
        if m != "random":
            f.loc[junk, f"score_{m}"] = np.nan  # blank/black frames are never candidates
    return f, {"available": avail, "learned": info, "n_junk": int(junk.sum())}


def ranked_windows(f: pd.DataFrame, method: str, k: int, window_min: float = 15.0) -> list[str]:
    s = f[["frame_id", "camera_id", "image_ts", f"score_{method}"]].dropna()
    s = s.sort_values(f"score_{method}", ascending=False, kind="mergesort")
    picked, by_cam = [], {}
    win = pd.Timedelta(minutes=window_min)
    for fid, cam, t in zip(s["frame_id"], s["camera_id"], s["image_ts"]):
        if any(abs(t - u) <= win for u in by_cam.get(cam, ())):
            continue
        picked.append(fid)
        by_cam.setdefault(cam, []).append(t)
        if len(picked) >= k:
            break
    return picked


def feed_frames(f: pd.DataFrame, events: pd.DataFrame, pairs: pd.DataFrame, per_pair: int = 2) -> list[tuple[str, str]]:
    """For each reported incident near a camera: the frames ~2 and ~10 minutes after it was reported."""
    out = []
    if events.empty or pairs.empty:
        return out
    ev = events.set_index("event_id")
    by_cam = {c: g for c, g in f.groupby("camera_id")}
    for r in pairs.itertuples(index=False):
        e = ev.loc[r.event_id]
        if e["category"] != "incident" or r.camera_id not in by_cam:
            continue
        g = by_cam[r.camera_id]
        g = g[(g["image_ts"] >= e["start"] - pd.Timedelta(minutes=5)) & (g["image_ts"] <= e["end"])]
        if g.empty:
            continue
        for target in [2, 10][:per_pair]:
            t = e["start"] + pd.Timedelta(minutes=target)
            i = (g["image_ts"] - t).abs().idxmin()
            out.append((g.loc[i, "frame_id"], r.event_id))
    return list(dict.fromkeys(out))


def build_pool(f: pd.DataFrame, events: pd.DataFrame, pairs: pd.DataFrame, avail: dict,
               k: int = 50, n_random: int = 80, seed: int = 1) -> pd.DataFrame:
    ranks: dict[str, dict] = {}
    reasons: dict[str, set] = {}
    for m in METHODS:
        if not avail.get(m):
            continue
        for r, fid in enumerate(ranked_windows(f, m, n_random if m == "random" else k), 1):
            ranks.setdefault(fid, {})[m] = r
            reasons.setdefault(fid, set()).add(m)
    feed_event = {}
    for fid, eid in feed_frames(f, events, pairs):
        reasons.setdefault(fid, set()).add("feed")
        feed_event.setdefault(fid, eid)
    ids = list(reasons)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)  # review order is random: the labeller cannot tell which method picked a frame
    fx = f.set_index("frame_id")
    rows = []
    for n, fid in enumerate(ids, 1):
        r = fx.loc[fid]
        row = {"review_id": f"R{n:04d}", "frame_id": fid, "camera_id": r["camera_id"],
               "image_ts": r["image_ts"].isoformat(), "path": r["path"],
               "prev_frame_id": r["prev_frame_id"] if isinstance(r["prev_frame_id"], str) else "",
               "url": r.get("url", ""), "reasons": ";".join(sorted(reasons[fid])),
               "feed_event_id": feed_event.get(fid, ""), "weak_label": r["weak_label"],
               "weak_event_id": r["weak_event_id"]}
        for m in METHODS:
            row[f"rank_{m}"] = ranks.get(fid, {}).get(m, np.nan)
        rows.append(row)
    pool = pd.DataFrame(rows)
    prev_path = fx["path"]
    pool["prev_path"] = [prev_path.get(p, "") if p else "" for p in pool["prev_frame_id"]]
    return pool


def review_items(pool: pd.DataFrame) -> list[dict]:
    """What the labelling page is allowed to see: images, camera and time. No scores, no reasons."""
    items = []
    for r in pool.itertuples(index=False):
        t = pd.Timestamp(r.image_ts)
        items.append({"review_id": r.review_id, "img": r.path, "prev": r.prev_path or "",
                      "camera_id": r.camera_id, "time": t.strftime("%a %d %b %H:%M")})
    return items


def save_pool(pool: pd.DataFrame, work) -> None:
    pool.to_csv(work / "pool.csv", index=False)
    (work / "review_items.json").write_text(json.dumps(review_items(pool), indent=0), encoding="utf-8")
