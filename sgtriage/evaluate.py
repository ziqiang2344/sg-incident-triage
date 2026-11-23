"""Score every method against the human labels.

The question: if a person can only review k frames, which ranking finds the most real incidents?
  precision@k      share of the top-k windows a human marked as an incident (Wilson 95% interval)
  lift             precision@k divided by the random-sample incident rate
  feed recall@k    share of incidents LTA's feed reported near a camera that fall in the top-k
  VLM agreement    the VLM's verdict against the human's, and what filtering the top-k with it does
Frames labelled "can't tell" are left out of precision; the count is reported so nothing is hidden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import wilson, write_json
from .rank import METHODS

KS = (10, 25, 50)
DAY_HOURS = (7, 19)


def _labels(work: Path) -> pd.DataFrame:
    p = work / "labels.csv"
    if not p.exists():
        return pd.DataFrame(columns=["review_id", "label", "kind", "ms"])
    lab = pd.read_csv(p, dtype=str).drop_duplicates("review_id", keep="last")
    lab["ms"] = pd.to_numeric(lab["ms"], errors="coerce")
    return lab


def _prop(y: pd.Series) -> dict:
    y = y.dropna()
    k, n = int(y.sum()), int(len(y))
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def evaluate(work: Path, vlm_df: pd.DataFrame | None = None, settings: dict | None = None) -> dict:
    pool = pd.read_csv(work / "pool.csv", dtype={"camera_id": str})
    frames = pd.read_csv(work / "frames_scored.csv", dtype={"camera_id": str})
    frames["image_ts"] = pd.to_datetime(frames["image_ts"], format="ISO8601")
    events = pd.read_csv(work / "events.csv") if (work / "events.csv").exists() else pd.DataFrame()
    pairs = pd.read_csv(work / "event_cameras.csv", dtype={"camera_id": str}) if (work / "event_cameras.csv").exists() else pd.DataFrame()
    meta = json.loads((work / "rank_meta.json").read_text()) if (work / "rank_meta.json").exists() else {}
    lab = _labels(work)

    pool = pool.merge(lab[["review_id", "label", "kind", "ms"]], on="review_id", how="left")
    pool["y"] = pool["label"].map({"yes": 1.0, "no": 0.0})  # unsure and unlabelled -> NaN
    pool["image_ts"] = pd.to_datetime(pool["image_ts"], format="ISO8601")
    pool["hour"] = pool["image_ts"].dt.hour
    pool["is_day"] = pool["hour"].between(DAY_HOURS[0], DAY_HOURS[1] - 1)

    m: dict = {"counts": {
        "frames": int(len(frames)), "cameras": int(frames["camera_id"].nunique()),
        "hours": round((frames["image_ts"].max() - frames["image_ts"].min()).total_seconds() / 3600, 1),
        "start": frames["image_ts"].min().isoformat(), "end": frames["image_ts"].max().isoformat(),
        "pool": int(len(pool)), "labelled": int(pool["label"].notna().sum()),
        "unsure": int((pool["label"] == "unsure").sum()), "junk_frames": int(meta.get("n_junk", 0)),
    }, "available": meta.get("available", {}), "learned": meta.get("learned", {}),
         "label_source": (settings or {}).get("label_source", "LTA's live incident feed"),
         "kind": (settings or {}).get("kind", "live"),
         "camera_radius": (settings or {}).get("camera_radius", 600)}
    if "window_id" in frames:
        w = frames["window_id"].fillna("").astype(str)
        m["counts"]["windows"] = int(w.nunique())
        m["counts"]["incident_windows"] = int(w[w.str.endswith("-i")].nunique())

    ms = pool["ms"].dropna()
    ms = ms[(ms > 300) & (ms < 60000)]
    sec = float(ms.median() / 1000) if len(ms) else 4.0
    m["seconds_per_frame"] = round(sec, 2)

    # ---- base rate from the random sample
    rnd = pool[pool["rank_random"].notna()]
    m["prevalence"] = _prop(rnd["y"])
    p0 = m["prevalence"]["p"]

    # ---- precision@k per method
    prec = {}
    for meth in METHODS:
        col = f"rank_{meth}"
        if col not in pool or pool[col].isna().all():
            continue
        prec[meth] = {}
        for k in KS:
            top = pool[pool[col] <= k]
            if top.empty:
                continue
            d = _prop(top["y"])
            d.update({"k_requested": k, "in_top": int(len(top)), "unsure": int((top["label"] == "unsure").sum()),
                      "unlabelled": int(top["label"].isna().sum())})
            d["lift"] = (d["p"] / p0) if p0 and not np.isnan(p0) and p0 > 0 and not np.isnan(d["p"]) else None
            d["per_100"] = None if np.isnan(d["p"]) else round(100 * d["p"], 1)
            d["minutes_per_incident"] = (round(len(top) * sec / 60 / d["k"], 1) if d["k"] else None)
            prec[meth][str(k)] = d
    # random order has the same expected rate at every k, so use the whole random sample for its row
    if "random" in prec and m["prevalence"]["n"]:
        for k in list(prec["random"]):
            prec["random"][k] = dict(m["prevalence"], k_requested=int(k), in_top=m["prevalence"]["n"], unsure=0,
                                     unlabelled=0, lift=1.0, per_100=round(100 * p0, 1),
                                     minutes_per_incident=(round(m["prevalence"]["n"] * sec / 60 / m["prevalence"]["k"], 1)
                                                           if m["prevalence"]["k"] else None))
    m["precision"] = prec
    k_main = "25"
    cands = {meth: v[k_main]["p"] for meth, v in prec.items() if meth != "random" and k_main in v
             and not np.isnan(v[k_main]["p"])}
    m["best_method"] = max(cands, key=cands.get) if cands else None

    # ---- the incident feed: how many reports, how many visible on camera
    feed = {"events_total": 0}
    if len(events):
        inc = events[events["category"] == "incident"]
        feed["events_total"] = int(len(inc))
        feed["by_type"] = inc["type"].value_counts().to_dict()
        feed["at_junction"] = int(inc["at_junction"].fillna(False).astype(bool).sum()) if "at_junction" in inc else 0
        near = pairs[pairs["event_id"].isin(inc["event_id"])] if len(pairs) else pairs
        feed["events_near_camera"] = int(near["event_id"].nunique()) if len(near) else 0
        feed["event_camera_pairs"] = int(len(near))
        fp = pool[pool["reasons"].fillna("").str.contains("feed")]
        feed["frames_visible"] = _prop(fp["y"])
        vis_events = fp.groupby("feed_event_id")["y"].max()
        feed["events_labelled"] = int(vis_events.notna().sum())
        feed["events_visible"] = int((vis_events == 1).sum())
        # recall: does the top-k contain a frame from that camera during the reported window?
        recall = {}
        if len(near):
            evx = events.set_index("event_id")
            win = []
            for r in near.itertuples(index=False):
                e = evx.loc[r.event_id]
                win.append((r.event_id, r.camera_id, pd.Timestamp(e["start"]) - pd.Timedelta(minutes=5),
                            pd.Timestamp(e["end"])))
            for meth in prec:
                col = f"rank_{meth}"
                recall[meth] = {}
                for k in KS:
                    top = pool[pool[col] <= k][["camera_id", "image_ts"]]
                    hit_events, vis_hit = set(), set()
                    for eid, cam, a, b in win:
                        t = top[top["camera_id"] == cam]["image_ts"]
                        if ((t >= a) & (t <= b)).any():
                            hit_events.add(eid)
                    vis_ids = set(vis_events[vis_events == 1].index)
                    n_all = len({w[0] for w in win})
                    recall[meth][str(k)] = {
                        "all": round(len(hit_events) / n_all, 3) if n_all else None,
                        "visible": round(len(hit_events & vis_ids) / len(vis_ids), 3) if vis_ids else None,
                        "n_all": n_all, "n_visible": len(vis_ids)}
        feed["recall"] = recall
    m["feed"] = feed

    # ---- VLM against the human
    if vlm_df is not None and len(vlm_df):
        v = pool.merge(vlm_df[["frame_id", "incident", "type", "confidence", "evidence", "where",
                               "image_quality", "latency_s", "vlm_score"]], on="frame_id", how="left")
        both = v[v["incident"].notna()]
        m["vlm"] = _vlm_metrics(both, prec, k_main)
        m["vlm"]["model"] = str(vlm_df["model"].iloc[0]) if "model" in vlm_df else ""
        pool = v
    m["examples"] = _examples(pool)
    write_json(work / "metrics.json", m)
    return m


def _vlm_metrics(v: pd.DataFrame, prec: dict, k_main: str) -> dict:
    out: dict = {"n": int(len(v)), "abstain_rate": float((v["incident"] == "unsure").mean()) if len(v) else None,
                 "median_latency_s": float(v["latency_s"].median()) if len(v) else None}
    h = v[v["y"].notna() & v["incident"].isin(["yes", "no"])]
    tp = int(((h["incident"] == "yes") & (h["y"] == 1)).sum())
    fp = int(((h["incident"] == "yes") & (h["y"] == 0)).sum())
    fn = int(((h["incident"] == "no") & (h["y"] == 1)).sum())
    tn = int(((h["incident"] == "no") & (h["y"] == 0)).sum())
    out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    out["precision"] = tp / (tp + fp) if tp + fp else None
    out["recall"] = tp / (tp + fn) if tp + fn else None
    if len(h) >= 10 and h["y"].nunique() == 2:
        from sklearn.metrics import cohen_kappa_score
        out["kappa"] = float(cohen_kappa_score(h["y"].astype(int), (h["incident"] == "yes").astype(int)))
    else:
        out["kappa"] = None
    split = {}
    for name, g in (("day", h[h["is_day"]]), ("night", h[~h["is_day"]])):
        t = int(((g["incident"] == "yes") & (g["y"] == 1)).sum())
        f_ = int(((g["incident"] == "yes") & (g["y"] == 0)).sum())
        n_ = int(((g["incident"] == "no") & (g["y"] == 1)).sum())
        split[name] = {"n": int(len(g)), "precision": t / (t + f_) if t + f_ else None,
                       "recall": t / (t + n_) if t + n_ else None}
    out["day_night"] = split
    # cascade: send only what the VLM does not rule out ("yes" or "unsure") to the human
    casc = {}
    for meth, d in prec.items():
        if meth == "random" or k_main not in d:
            continue
        top = v[v[f"rank_{meth}"] <= int(k_main)]
        keep = top[top["incident"].isin(["yes", "unsure"]) | top["incident"].isna()]
        y_top, y_keep = top["y"].dropna(), keep["y"].dropna()
        casc[meth] = {"reviewed_before": int(len(top)), "reviewed_after": int(len(keep)),
                      "found_before": int(y_top.sum()), "found_after": int(y_keep.sum()),
                      "precision_after": float(y_keep.mean()) if len(y_keep) else None}
    out["cascade"] = casc
    return out


def _examples(pool: pd.DataFrame, n: int = 8) -> dict:
    def pack(df):
        cols = ["review_id", "frame_id", "camera_id", "image_ts", "path", "prev_path", "label", "kind",
                "reasons", "weak_event_id"] + [c for c in pool.columns if c.startswith("rank_")]
        cols += [c for c in ("incident", "type", "confidence", "evidence", "where") if c in df]
        out = df[cols].copy()
        out["image_ts"] = out["image_ts"].astype(str)
        return json.loads(out.head(n).to_json(orient="records"))

    rank_cols = [c for c in pool.columns if c.startswith("rank_") and c != "rank_random"]
    pool = pool.copy()
    pool["best_rank"] = pool[rank_cols].min(axis=1) if rank_cols else np.nan
    ex = {"found": pack(pool[(pool["y"] == 1) & (pool["best_rank"] <= 25)].sort_values("best_rank"))}
    if "incident" in pool:
        ex["vlm_false_alarm"] = pack(pool[(pool["incident"] == "yes") & (pool["y"] == 0)].sort_values("confidence", ascending=False))
        ex["vlm_missed"] = pack(pool[(pool["incident"] == "no") & (pool["y"] == 1)])
    ex["feed_not_visible"] = pack(pool[pool["reasons"].fillna("").str.contains("feed") & (pool["y"] == 0)])
    return ex


# ----------------------------------------------------------------------------- figures (for the PDF)

def figures(m: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    made = []
    names = {"random": "Random", "speed": "Speed drop", "novelty": "Image novelty", "zeroshot": "CLIP zero-shot",
             "fused": "Fused (label-free)", "learned": "Learned (weak labels)"}
    prec = m.get("precision", {})
    rows = [(names.get(k, k), v["25"]) for k, v in prec.items() if "25" in v and v["25"]["n"]]
    if rows:
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        labels_ = [r[0] for r in rows]
        p = np.array([r[1]["p"] for r in rows]) * 100
        lo = p - np.array([r[1]["lo"] for r in rows]) * 100
        hi = np.array([r[1]["hi"] for r in rows]) * 100 - p
        colors = ["#9aa3ad" if l == "Random" else "#006B3F" for l in labels_]
        ax.barh(labels_, p, xerr=[lo, hi], color=colors, error_kw={"elinewidth": 1, "capsize": 3, "ecolor": "#444"})
        ax.invert_yaxis()
        ax.set_xlabel("Incidents per 100 frames reviewed (top 25 windows; random: whole sample; 95% interval)")
        for i, val in enumerate(p):
            ax.text(val + 1, i, f"{val:.0f}", va="center", fontsize=8)
        fig.tight_layout()
        path = out_dir / "fig_precision.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        made.append(path)
    v = m.get("vlm")
    if v and v.get("confusion"):
        c = v["confusion"]
        fig, ax = plt.subplots(figsize=(2.6, 2.3))
        mat = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=11,
                    color="white" if val > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["VLM: yes", "VLM: no"])
        ax.set_yticks([0, 1], ["Human: yes", "Human: no"])
        ax.spines[:].set_visible(False)
        fig.tight_layout()
        path = out_dir / "fig_vlm_confusion.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        made.append(path)
    return made
