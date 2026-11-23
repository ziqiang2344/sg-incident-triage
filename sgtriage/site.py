"""Build the static dashboard (docs/) for GitHub Pages.

Everything is precomputed: the page makes no API calls and holds no keys. Example frames are
copied in at reduced size. Settings (author name, links) come from site_config.json at the repo root.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from .common import ROOT, Paths, load_cameras, require

KIND_NAMES = {"collision": "Collision", "stopped_vehicle": "Stopped vehicle", "obstacle": "Obstacle on the road",
              "emergency_response": "Emergency response", "people_on_road": "People on the carriageway"}
METHOD_NAMES = {"speed": "speed drop", "novelty": "image novelty", "zeroshot": "CLIP zero-shot",
                "fused": "the fused ranking", "learned": "the learned ranking"}


def _copy_img(src, dst, max_w: int = 960) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=82, optimize=True)


def _limits(m: dict) -> list[str]:
    c, f = m["counts"], m.get("feed", {})
    if m.get("kind") == "history":
        out = [f"Frames come from windows around {f.get('events_total', 0)} news-reported crashes and the same windows a "
               f"week earlier. Incidents the news never covered are found only through the ranking and the labels.",
               f"News locations are approximate (taken from article text), so incidents are matched to cameras within "
               f"{m.get('camera_radius', 1000)} m.",
               "LTA now publishes live images for only 8 cameras, so live road-speed data and camera frames could not be "
               "combined. Speed data is not included in this test."]
    else:
        out = [f"One collection window: {c['hours']:g} hours from {c['cameras']} cameras. Rates will move with weather, "
               f"time of day and traffic volume."]
    if f.get("events_labelled"):
        out.append(f"A reported location is not a view: {f['events_visible']} of {f['events_labelled']} reported "
                   f"incidents near a camera were actually visible in its frames, so reports alone are weak labels.")
    out.append("Labels are one person's judgement on still frames about 3 minutes apart. "
               "Frames marked “can't tell” are left out of the rates and counted separately.")
    out.append("Rankings were fixed before labelling and not tuned on the labels; intervals are Wilson 95%.")
    v = m.get("vlm") or {}
    dn = v.get("day_night") or {}
    if dn.get("night", {}).get("recall") is not None and dn.get("day", {}).get("recall") is not None:
        out.append(f"The VLM is a general model with no fine-tuning. Recall by day {dn['day']['recall']:.0%}, "
                   f"at night {dn['night']['recall']:.0%}.")
    out.append("Public cameras cover expressways and checkpoints only, so junction behaviour is not tested here.")
    return out


def build_site(paths: Paths) -> None:
    m = json.loads(require(paths.work / "metrics.json", "evaluate").read_text())
    cfg = {"author": "", "author_url": "", "repo_url": ""}
    if (ROOT / "site_config.json").exists():
        cfg.update(json.loads((ROOT / "site_config.json").read_text()))
    site = paths.site
    img_dir = site / "img"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)

    events = pd.read_csv(paths.work / "events.csv") if (paths.work / "events.csv").exists() else pd.DataFrame()
    msg = dict(zip(events["event_id"], events["message"])) if len(events) else {}

    examples = {}
    for key, items in (m.get("examples") or {}).items():
        out = []
        for e in items:
            src = paths.raw / e["path"]
            if not src.exists():
                continue
            name = f"{e['frame_id']}.jpg"
            if not (img_dir / name).exists():
                _copy_img(src, img_dir / name)
            e = dict(e, img=f"img/{name}", feed=msg.get(e.get("weak_event_id") or "", ""))
            out.append(e)
        examples[key] = out

    hero = None
    found = examples.get("found", [])
    if found:
        both = [e for e in found if e.get("incident") == "yes"]
        h = (both or found)[0]
        ranks = {k[5:]: v for k, v in h.items() if k.startswith("rank_") and v is not None and k != "rank_random"}
        bm = min(ranks, key=ranks.get) if ranks else None
        cap = []
        if bm:
            cap.append(f"Ranked #{int(ranks[bm])} by {METHOD_NAMES.get(bm, bm)}.")
        if h.get("evidence"):
            cap.append(f"VLM: “{h['evidence']}”")
        hero = {"img": h["img"], "camera_id": h["camera_id"], "image_ts": h["image_ts"],
                "kind": KIND_NAMES.get(h.get("kind") or h.get("type") or "", "Incident"),
                "title": KIND_NAMES.get(h.get("kind") or h.get("type") or "", "Incident") + ".", "caption": " ".join(cap)}

    cams = load_cameras(paths.raw)
    ev_pts = []
    if len(events):
        inc = events[events["category"] == "incident"]
        ev_pts = [{"lat": r.lat, "lon": r.lon, "type": r.type, "message": r.message,
                   "at_junction": bool(r.at_junction) if not pd.isna(r.at_junction) else False}
                  for r in inc.itertuples()]
    data = {"metrics": m, "examples": examples, "hero": hero, "limits": _limits(m),
            "cameras": cams[["camera_id", "lat", "lon"]].to_dict("records"), "events": ev_pts,
            "author": cfg["author"], "author_url": cfg["author_url"], "repo_url": cfg["repo_url"],
            "has_pdf": (site / "report.pdf").exists()}
    tpl = (ROOT / "sgtriage" / "site_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
    payload = payload.replace("</", "<\\/")  # never close the <script> tag early
    (site / "index.html").write_text(tpl.replace("/*DATA*/", payload), encoding="utf-8")
    (site / ".nojekyll").write_text("")
    n_img = sum(len(v) for v in examples.values())
    print(f"site -> {site / 'index.html'} ({n_img} example frames). Preview: python -m http.server -d {site} 8010")
