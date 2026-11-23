"""Cheap per-frame signals used to rank frames before anyone looks at them.

Two independent sources, the same shape as sensor anomaly detection:
  speed   - road links near the camera slowing down against their own recent baseline
            (incidents cause speed drops; so does ordinary congestion, which is the main
            false alarm and the reason speed alone is not enough)
  image   - how unusual a frame looks for *that* camera, from CLIP embeddings:
            novelty = distance to the camera's most similar frames from other times of the night/day
            zero-shot = CLIP's similarity to incident descriptions minus normal-traffic descriptions
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common import haversine_m

EXPRESSWAY_CATEGORIES = {"A", "1"}  # DataMall RoadCategory for expressways (letter or number form)

INCIDENT_PROMPTS = [
    "a traffic camera photo of a car crash on a highway",
    "a traffic camera photo of a vehicle stopped on the road shoulder with hazard lights",
    "a traffic camera photo of a broken down lorry stopped in a highway lane",
    "a traffic camera photo of debris lying on the highway",
    "a traffic camera photo of police or ambulance vehicles stopped on the highway",
    "a traffic camera photo of people standing on the highway next to a stopped car",
]
NORMAL_PROMPTS = [
    "a traffic camera photo of normal traffic flowing on a highway",
    "a traffic camera photo of an empty highway at night",
    "a traffic camera photo of a traffic jam on a highway",
    "a traffic camera photo of a highway in the rain",
    "a traffic camera photo of cars driving on an expressway in daylight",
]
JUNK_PROMPTS = [
    "a blank grey image",
    "a black screen with no signal",
    "a camera image that is completely blurred",
]


# ----------------------------------------------------------------------------- speed

def camera_links(cameras: pd.DataFrame, speed: pd.DataFrame, radius_m: float = 500.0) -> pd.DataFrame:
    """Road links whose midpoint lies within radius_m of each camera (expressway links preferred)."""
    links = speed.drop_duplicates("LinkID")[["LinkID", "RoadName", "RoadCategory", "mid_lat", "mid_lon"]]
    d = haversine_m(cameras["lat"].to_numpy()[:, None], cameras["lon"].to_numpy()[:, None],
                    links["mid_lat"].to_numpy()[None, :], links["mid_lon"].to_numpy()[None, :])
    ci, li = np.nonzero(d <= radius_m)
    pairs = pd.DataFrame({"camera_id": cameras["camera_id"].to_numpy()[ci],
                          "LinkID": links["LinkID"].to_numpy()[li], "dist_m": d[ci, li]})
    pairs = pairs.merge(links[["LinkID", "RoadName", "RoadCategory"]], on="LinkID")
    pairs["is_expressway"] = pairs["RoadCategory"].astype(str).str.strip().isin(EXPRESSWAY_CATEGORIES)
    has_xw = pairs.groupby("camera_id")["is_expressway"].transform("any")
    return pairs[pairs["is_expressway"] | ~has_xw].reset_index(drop=True)


def speed_features(frames: pd.DataFrame, cameras: pd.DataFrame, speed: pd.DataFrame,
                   radius_m: float = 500.0, baseline_min: int = 60, max_lag_min: int = 12) -> pd.DataFrame:
    """Per-frame speed-drop features. Speed bands: 1 = slowest ... 8 = fastest."""
    out = pd.DataFrame({"frame_id": frames["frame_id"]})
    cols = ["speed_max_drop", "speed_mean_drop", "speed_frac_drop2", "speed_min_band", "speed_n_links"]
    if speed is None or speed.empty:
        for c in cols:
            out[c] = np.nan
        return out
    sp = speed[["poll_ts", "LinkID", "SpeedBand"]].sort_values(["LinkID", "poll_ts"]).reset_index(drop=True)
    # baseline = median band over the previous hour, excluding the current poll
    base = (sp.set_index("poll_ts").groupby("LinkID")["SpeedBand"]
            .rolling(f"{baseline_min}min", closed="left").median())
    sp["baseline"] = base.to_numpy()
    sp["drop"] = (sp["baseline"] - sp["SpeedBand"]).clip(lower=0)

    pairs = camera_links(cameras, speed, radius_m)
    m = pairs[["camera_id", "LinkID"]].merge(sp, on="LinkID")
    if m.empty:
        for c in cols:
            out[c] = np.nan
        return out
    agg = m.groupby(["camera_id", "poll_ts"]).agg(
        speed_max_drop=("drop", "max"), speed_mean_drop=("drop", "mean"),
        speed_frac_drop2=("drop", lambda x: float((x >= 2).mean())),
        speed_min_band=("SpeedBand", "min"), speed_n_links=("LinkID", "nunique")).reset_index()
    agg = agg.sort_values("poll_ts")
    f = frames[["frame_id", "camera_id", "image_ts"]].sort_values("image_ts")
    agg["poll_ts"] = agg["poll_ts"].astype(f["image_ts"].dtype)
    merged = pd.merge_asof(f, agg, left_on="image_ts", right_on="poll_ts", by="camera_id",
                           direction="backward", tolerance=pd.Timedelta(minutes=max_lag_min))
    return out.merge(merged[["frame_id"] + cols], on="frame_id", how="left")


# ----------------------------------------------------------------------------- images

def _load(path: Path, size=None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im.resize(size) if size else im


def image_stats(frames: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """Brightness and contrast per frame; flags frames that are blank, black or frozen."""
    rows = []
    for fid, p in zip(frames["frame_id"], frames["path"]):
        try:
            g = np.asarray(_load(raw / p, (160, 90)).convert("L"), dtype=np.float32)
            rows.append((fid, float(g.mean()), float(g.std())))
        except Exception:
            rows.append((fid, np.nan, np.nan))
    st = pd.DataFrame(rows, columns=["frame_id", "brightness", "contrast"])
    st["junk"] = (st["contrast"] < 3) | st["brightness"].isna()  # flat grey/black = camera offline
    return st


class TinyEmbedder:
    """Dependency-free fallback (for tests or no torch): small grey thumbnail + colour histogram."""
    name = "tiny"
    zero_shot = False

    def encode(self, paths: list[Path]) -> np.ndarray:
        feats = []
        for p in paths:
            im = _load(p, (32, 18))
            g = np.asarray(im.convert("L"), dtype=np.float32).ravel() / 255.0
            h = np.asarray(im.convert("HSV"), dtype=np.float32).reshape(-1, 3)
            hist = np.concatenate([np.histogram(h[:, i], bins=8, range=(0, 255))[0] for i in range(3)]).astype(np.float32)
            v = np.concatenate([g - g.mean(), hist / hist.sum()])
            feats.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(feats)


class ClipEmbedder:
    """OpenCLIP ViT-B/32. Uses Apple-silicon GPU (MPS) or CUDA when available."""
    zero_shot = True

    def __init__(self, model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str | None = None):
        import open_clip
        import torch
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.name = f"clip-{model}-{pretrained}"

    def encode(self, paths: list[Path]) -> np.ndarray:
        x = self.torch.stack([self.preprocess(_load(p)) for p in paths]).to(self.device)
        with self.torch.no_grad():
            e = self.model.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.float().cpu().numpy()

    def text(self, prompts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            t = self.model.encode_text(self.tokenizer(prompts).to(self.device))
            t = t / t.norm(dim=-1, keepdim=True)
        return t.float().cpu().numpy()

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())


def embed(frames: pd.DataFrame, raw: Path, work: Path, backend: str = "clip", batch: int = 64) -> pd.DataFrame:
    """Embed every frame (cached in data/work/embeddings_<backend>.npz); returns zero-shot scores."""
    enc = TinyEmbedder() if backend == "tiny" else ClipEmbedder()
    cache = work / f"embeddings_{backend}.npz"
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["frame_id"].tolist(), z["emb"]))
    todo = frames[~frames["frame_id"].isin(have.keys())]
    print(f"embedding {len(todo):,} new frames with {enc.name} ({len(have):,} cached)")
    ids, paths = todo["frame_id"].tolist(), [raw / p for p in todo["path"]]
    for i in range(0, len(paths), batch):
        e = enc.encode(paths[i:i + batch])
        for fid, v in zip(ids[i:i + batch], e):
            have[fid] = v.astype(np.float16)
        if (i // batch) % 20 == 0:
            print(f"  {min(i + batch, len(paths)):,}/{len(paths):,}")
    fid_all = frames["frame_id"].tolist()
    emb = np.stack([have[f] for f in fid_all]).astype(np.float32)
    np.savez(cache, frame_id=np.array(fid_all), emb=emb.astype(np.float16))

    out = pd.DataFrame({"frame_id": fid_all})
    if enc.zero_shot:
        s = enc.logit_scale
        inc, nor, junk = enc.text(INCIDENT_PROMPTS), enc.text(NORMAL_PROMPTS), enc.text(JUNK_PROMPTS)

        def lse(m):  # log-sum-exp over prompts, numerically stable
            mx = m.max(axis=1, keepdims=True)
            return (mx + np.log(np.exp(m - mx).sum(axis=1, keepdims=True))).ravel()
        li, ln, lj = lse(s * emb @ inc.T), lse(s * emb @ nor.T), lse(s * emb @ junk.T)
        out["zeroshot"] = li - ln
        out["zeroshot_junk"] = lj - np.logaddexp(li, ln)
    else:
        out["zeroshot"] = np.nan
        out["zeroshot_junk"] = np.nan
    return out, emb


def novelty(frames: pd.DataFrame, emb: np.ndarray, exclude_min: float = 45.0, k: int = 5) -> pd.DataFrame:
    """How unlike a frame is to the same camera's other frames.

    Frames within ±exclude_min are not allowed as neighbours: an incident that lasts 40 minutes
    would otherwise match its own frames and look normal. The score is ranked within each camera,
    because some views are naturally busier than others.
    """
    nov = np.full(len(frames), np.nan)
    change = np.full(len(frames), np.nan)
    t_min = ((frames["image_ts"] - frames["image_ts"].min()).dt.total_seconds() / 60).to_numpy()  # unit-safe
    for _, idx in frames.groupby("camera_id").indices.items():
        idx = np.sort(idx)
        e = emb[idx]
        d = 1 - e @ e.T
        tt = t_min[idx]
        near = np.abs(tt[:, None] - tt[None, :]) <= exclude_min
        d_masked = np.where(near, np.inf, d)
        kk = min(k, max(1, len(idx) - 1))
        part = np.partition(d_masked, kk - 1, axis=1)[:, :kk]
        with np.errstate(invalid="ignore"):
            vals = np.where(np.isinf(part), np.nan, part)
            nov[idx] = np.nanmean(vals, axis=1) if vals.size else np.nan
        # change from the previous frame (camera moves and sudden events both show up here)
        gaps = np.diff(tt, prepend=np.nan)
        prev_d = np.r_[np.nan, d[np.arange(1, len(idx)), np.arange(0, len(idx) - 1)]]
        change[idx] = np.where(gaps <= 10, prev_d, np.nan)
    out = pd.DataFrame({"frame_id": frames["frame_id"], "camera_id": frames["camera_id"],
                        "novelty_raw": nov, "change": change})
    out["novelty"] = out.groupby("camera_id")["novelty_raw"].rank(pct=True)
    return out.drop(columns="camera_id")
