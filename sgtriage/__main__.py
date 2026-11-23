"""python -m sgtriage <step> [options]

Steps, in order:
  backfill   download historical frames around news-reported incidents (data/raw_hist)
  status     what has been collected so far
  static     download LTA traffic-light locations -> signalised junctions
  labels     incident feed -> events -> weak labels on frames
  embed      CLIP embeddings, zero-shot scores, image novelty, blank-frame flags
  features   speed-band drop features near each camera
  rank       score every frame with every method (incl. out-of-fold learned ranker)
  pool       build the shuffled review set (top-k per method + feed reports + random sample)
  prepare    labels + embed + features + rank + pool in one go
  review     open the blind labelling page
  models     list the models your VLM endpoint offers (vision-capable ones marked)
  vlmtest    send 2 recent frames to the VLM and print the replies (checks the endpoint)
  vlm        VLM triage of the review set (needs VLM_* in .env)
  evaluate   metrics + figures
  speed      speed-drop detectors scored against the live incident feed (no labelling needed)
  site       build the static dashboard in docs/
  report     build docs/report.html (and report.pdf if Playwright is installed)
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import evaluate as ev
from . import features as ft
from . import labels as lb
from . import rank as rk
from .common import Paths, load_cameras, load_env, load_frames, load_incident_polls, load_incident_sightings, \
    load_speedbands, require, to_sgt, write_json
from .static_layers import build_junctions, load_junctions


def _frames_labels(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(require(paths.work / "frames_labels.csv", "labels"), dtype={"camera_id": str, "md5": str})
    df["image_ts"] = to_sgt(df["image_ts"])
    return df


def _events(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = pd.read_csv(paths.work / "events.csv")
    for c in ("reported_at", "first_seen", "last_seen", "start", "end"):
        if len(e):
            e[c] = to_sgt(e[c])
    p = pd.read_csv(paths.work / "event_cameras.csv", dtype={"camera_id": str})
    return e, p


def step_status(paths: Paths, a) -> None:
    f = load_frames(paths.raw)
    sightings = load_incident_sightings(paths.raw)
    polls = load_incident_polls(paths.raw)
    sb = load_speedbands(paths.raw)
    print(f"frames: {len(f):,} from {f['camera_id'].nunique()} cameras, "
          f"{f['image_ts'].min():%a %H:%M} -> {f['image_ts'].max():%a %H:%M}")
    print(f"frames per hour: {f.groupby(f['image_ts'].dt.floor('h')).size().to_dict()}")
    print(f"incident polls: {len(polls):,}; sightings: {len(sightings):,}")
    if len(sightings):
        events = lb.build_events(sightings, load_junctions(paths.static))
        print("events by type:", events["type"].value_counts().to_dict())
        r = _radius(paths, a)
        pairs = lb.match_cameras(events[events["category"] == "incident"], load_cameras(paths.raw, f), r)
        print(f"incidents within {r:.0f} m of a camera: {pairs['event_id'].nunique()}")
    print(f"speed-band rows: {len(sb):,}" + (f", polls {sb['poll_ts'].nunique()}" if len(sb) else ""))


def step_static(paths: Paths, a) -> None:
    from pathlib import Path
    build_junctions(paths.static, zip_path=Path(a.static_zip) if a.static_zip else None)


def _settings(paths: Paths) -> dict:
    p = paths.raw / "settings.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _radius(paths: Paths, a) -> float:
    return a.camera_radius or _settings(paths).get("camera_radius", 600)


def step_labels(paths: Paths, a) -> None:
    frames = load_frames(paths.raw)
    cams = load_cameras(paths.raw, frames)
    events = lb.build_events(load_incident_sightings(paths.raw), load_junctions(paths.static))
    a.camera_radius = _radius(paths, a)
    pairs = lb.match_cameras(events, cams, a.camera_radius)
    f = lb.weak_labels(frames, events, pairs, load_incident_polls(paths.raw))
    events.to_csv(paths.work / "events.csv", index=False)
    pairs.to_csv(paths.work / "event_cameras.csv", index=False)
    f.to_csv(paths.work / "frames_labels.csv", index=False)
    inc = events[events["category"] == "incident"]
    near = pairs[pairs["event_id"].isin(inc["event_id"])]
    print(f"{len(f):,} frames | {len(inc)} incidents reported ({int(inc['at_junction'].sum()) if len(inc) else 0} at "
          f"signalised junctions), {near['event_id'].nunique()} within {a.camera_radius:.0f} m of a camera | "
          f"weak positives: {int((f['weak_label'] == 1).sum())} frames | feed coverage: {f['covered'].mean():.0%}")


def step_embed(paths: Paths, a) -> None:
    f = _frames_labels(paths)
    zs, emb = ft.embed(f, paths.raw, paths.work, backend=a.backend, batch=a.batch)
    nov = ft.novelty(f, emb)
    st = ft.image_stats(f, paths.raw)
    out = zs.merge(nov, on="frame_id").merge(st, on="frame_id")
    out.to_csv(paths.work / "features_image.csv", index=False)
    print(f"image features for {len(out):,} frames; blank/black frames: {int(out['junk'].sum())}")


def step_features(paths: Paths, a) -> None:
    f = _frames_labels(paths)
    sp = ft.speed_features(f, load_cameras(paths.raw, f), load_speedbands(paths.raw), radius_m=a.speed_radius)
    sp.to_csv(paths.work / "features_speed.csv", index=False)
    print(f"speed features: {sp['speed_max_drop'].notna().mean():.0%} of frames have nearby speed data")


def step_rank(paths: Paths, a) -> None:
    f = _frames_labels(paths)
    img = pd.read_csv(require(paths.work / "features_image.csv", "embed"))
    sp = pd.read_csv(require(paths.work / "features_speed.csv", "features"))
    f = f.merge(img, on="frame_id", how="left").merge(sp, on="frame_id", how="left")
    events, _ = _events(paths)
    scored, meta = rk.score_all(f, events)
    scored.to_csv(paths.work / "frames_scored.csv", index=False)
    write_json(paths.work / "rank_meta.json", meta)
    print("methods available:", {k: v for k, v in meta["available"].items()})
    li = meta["learned"]
    print("learned ranker:", li.get("skipped") or f"{li['n_pos_events']} incidents, {li['n_pos_frames']} positive frames, "
          f"top weights {list(li['coefficients'].items())[:4]}")


def step_pool(paths: Paths, a) -> None:
    f = pd.read_csv(require(paths.work / "frames_scored.csv", "rank"), dtype={"camera_id": str})
    f["image_ts"] = to_sgt(f["image_ts"])
    events, pairs = _events(paths)
    meta = json.loads((paths.work / "rank_meta.json").read_text())
    pool = rk.build_pool(f, events, pairs, meta["available"], k=a.k, n_random=a.n_random)
    rk.save_pool(pool, paths.work)
    print(f"review set: {len(pool)} frames (top {a.k} per method, {a.n_random} random, feed reports). "
          f"At ~4 s each that is ~{len(pool) * 4 / 60:.0f} minutes. Next: python -m sgtriage review")


def step_prepare(paths: Paths, a) -> None:
    for fn in (step_labels, step_embed, step_features, step_rank, step_pool):
        print(f"--- {fn.__name__.replace('step_', '')}")
        fn(paths, a)


def step_review(paths: Paths, a) -> None:
    from .review import serve
    serve(paths.raw, paths.work, port=a.port, open_browser=not a.no_browser)


def step_vlm(paths: Paths, a) -> None:
    from . import vlm
    base, key, model = vlm.settings_from_env()
    pool = pd.read_csv(require(paths.work / "pool.csv", "pool"), dtype={"camera_id": str})
    res = vlm.run(pool, paths.raw, paths.work, base, key, model, workers=a.workers, limit=a.limit,
                  fullres=not a.no_fullres, max_side=a.max_side, two_images=not a.one_image)
    if len(res):
        print(res["incident"].value_counts().to_dict())


def step_models(paths: Paths, a) -> None:
    import os
    from . import vlm
    base = os.environ.get("VLM_BASE_URL", "")
    if not base:
        raise SystemExit("Set VLM_BASE_URL (and VLM_API_KEY) in .env first.")
    for mid, vision in vlm.list_models(base, os.environ.get("VLM_API_KEY", "")):
        print(("  [vision?] " if vision else "            ") + mid)


def step_vlmtest(paths: Paths, a) -> None:
    from . import vlm
    base, key, model = vlm.settings_from_env()
    if a.probe:
        return vlm.probe(paths.raw, base, key, model, max_side=a.max_side)
    vlm.smoke_test(paths.raw, base, key, model, n=a.limit or 2, max_side=a.max_side, two_images=not a.one_image,
                   compare=a.compare)


def step_evaluate(paths: Paths, a) -> None:
    from . import vlm
    import os
    res = vlm.load_results(paths.work, os.environ.get("VLM_MODEL") or None)
    m = ev.evaluate(paths.work, res if len(res) else None, settings=_settings(paths))
    figs = ev.figures(m, paths.work / "figs")
    p = m["prevalence"]
    print(f"labelled {m['counts']['labelled']}/{m['counts']['pool']} | random-sample incident rate "
          f"{p['k']}/{p['n']} ({p['p']:.1%})" if p["n"] else "no random-sample labels yet")
    for meth, d in m["precision"].items():
        if "25" in d:
            x = d["25"]
            print(f"  {meth:<9} top-25: {x['k']}/{x['n']} incidents ({x['p']:.0%}, 95% {x['lo']:.0%}-{x['hi']:.0%})"
                  + (f", lift x{x['lift']:.1f}" if x.get("lift") else ""))
    if m.get("vlm"):
        v = m["vlm"]
        print(f"  VLM vs human: precision {v['precision']}, recall {v['recall']}, kappa {v['kappa']}, "
              f"abstain {v['abstain_rate']}")
    print(f"metrics -> {paths.work / 'metrics.json'}; figures: {[str(x) for x in figs]}")


def step_backfill(paths: Paths, a) -> None:
    from pathlib import Path
    from .backfill import run
    run(Path(a.incidents), paths.raw, before=a.before, after=a.after, step=a.step_min, control_days=a.control_days)


def step_speed(paths: Paths, a) -> None:
    from . import speedeval
    res = speedeval.evaluate(paths, categories=tuple(a.road_categories.split(",")))
    fig = speedeval.figure(res, paths.work / "figs")
    w, e = res["window"], res["events"]
    print(f"{w['links']:,} links x {w['polls']} polls, {w['hours_scored']} h scored | {e['incidents']} incidents, "
          f"{e['congestion_reports']} heavy-traffic reports observable")
    for name, d in res["detectors"].items():
        print(f"  {name:<14} recall {d['incident_recall']} | median delay {d['median_delay_min']} min | "
              f"{d['alarms']} alarms ({d['alarms_per_hour']}/h): {d['alarms_matching_incident']} incident, "
              f"{d['alarms_matching_congestion']} congestion, {d['alarms_unexplained']} unexplained")
    print(f"-> {paths.work / 'speed_metrics.json'}" + (f", {fig}" if fig else ""))


def step_site(paths: Paths, a) -> None:
    from .site import build_site
    build_site(paths)


def step_report(paths: Paths, a) -> None:
    from .report import build_report
    build_report(paths)


STEPS = {"vlmtest": step_vlmtest, "models": step_models, "backfill": step_backfill, "speed": step_speed, "status": step_status, "static": step_static, "labels": step_labels, "embed": step_embed,
         "features": step_features, "rank": step_rank, "pool": step_pool, "prepare": step_prepare,
         "review": step_review, "vlm": step_vlm, "evaluate": step_evaluate, "site": step_site,
         "report": step_report}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sgtriage", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=list(STEPS))
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--work", default="data/work")
    p.add_argument("--static", default="data/static")
    p.add_argument("--site", default="docs")
    p.add_argument("--static-zip", default=None, help="use a local traffic-light zip instead of downloading")
    p.add_argument("--backend", default="clip", choices=["clip", "tiny"], help="image embedder (tiny = no torch)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--camera-radius", type=float, default=None,
                   help="metres: incident-to-camera match radius (default 600; 1000 for news-based history)")
    p.add_argument("--speed-radius", type=float, default=500, help="metres: road links counted as near a camera")
    p.add_argument("--k", type=int, default=50, help="top windows per method in the review set")
    p.add_argument("--n-random", type=int, default=80, help="random frames in the review set")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="parallel VLM calls")
    p.add_argument("--limit", type=int, default=None, help="only send this many frames to the VLM (testing)")
    p.add_argument("--no-fullres", action="store_true", help="send our saved copy instead of the original")
    p.add_argument("--max-side", type=int, default=1280, help="VLM: longest image side in pixels")
    p.add_argument("--one-image", action="store_true", help="VLM: send only the current frame (faster)")
    p.add_argument("--compare", action="store_true", help="vlmtest: time 3 settings on one frame")
    p.add_argument("--probe", action="store_true", help="vlmtest: check hidden reasoning and parallel throughput")
    p.add_argument("--incidents", default="data_inputs/news_incidents.csv", help="backfill: news-reported incidents")
    p.add_argument("--before", type=int, default=20, help="backfill: minutes before each incident")
    p.add_argument("--after", type=int, default=40, help="backfill: minutes after each incident")
    p.add_argument("--step-min", type=int, default=5, help="backfill: minutes between snapshots")
    p.add_argument("--control-days", type=int, default=7, help="backfill: matched control window N days earlier (0 = none)")
    p.add_argument("--road-categories", default="1,2", help="speed: DataMall road categories to score (1 = expressway)")
    a = p.parse_args(argv)
    load_env()
    paths = Paths.make(a.raw, a.work, a.static, a.site)
    STEPS[a.step](paths, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
