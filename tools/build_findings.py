"""Build the findings page (docs/index.html) and two-page note (docs/report.html/.pdf) for the
historical camera test. Numbers come from data/work/metrics.json; the frames shown are listed in
data_inputs/findings_examples.json; the model's words are quoted from data/work/vlm_cache.jsonl.

python tools/build_findings.py [--raw data/raw_hist] [--work data/work] [--site docs]
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sgtriage.common import wilson  # noqa: E402

E = html.escape


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def load(work: Path):
    m = json.loads((work / "metrics.json").read_text())
    pool = pd.read_csv(work / "pool.csv", dtype={"camera_id": str})
    lab = pd.read_csv(work / "labels.csv", dtype=str).drop_duplicates("review_id", keep="last")
    pool = pool.merge(lab[["review_id", "label", "kind"]], on="review_id", how="left")
    vl = {}
    for line in (work / "vlm_cache.jsonl").read_text().splitlines():
        r = json.loads(line)
        vl[r["frame_id"]] = r["parsed"]
    pool["vlm"] = pool["frame_id"].map(lambda f: (vl.get(f) or {}).get("incident"))
    pool["vlm_evidence"] = pool["frame_id"].map(lambda f: (vl.get(f) or {}).get("evidence", ""))
    return m, pool, vl


def copy_img(raw: Path, rel: str, dst: Path, max_w: int = 1100) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw / rel) as im:
        im = im.convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        elif im.width < 640:  # old 320x240 cameras: enlarge for display, softly
            im = im.resize((640, round(im.height * 640 / im.width)), Image.BICUBIC)
        im.save(dst, "JPEG", quality=84, optimize=True)


def stamp(row) -> str:
    t = pd.Timestamp(row["image_ts"])
    return f"Camera {row['camera_id']}, {t:%-d %b %Y %H:%M}"


def build(raw: Path, work: Path, site: Path, examples_path: Path, repo_url: str = "") -> None:
    m, pool, vl = load(work)
    ex = json.loads(examples_path.read_text())
    by_rid = pool.set_index("review_id")
    img_dir = site / "img"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)

    def img(rid: str) -> tuple[str, pd.Series]:
        r = by_rid.loc[rid]
        name = f"{r['frame_id']}.jpg"
        if not (img_dir / name).exists():
            copy_img(raw, r["path"], img_dir / name)
        return f"img/{name}", r

    c, prev, feed, v = m["counts"], m["prevalence"], m["feed"], m["vlm"]
    n_yes = int((pool["label"] == "yes").sum())
    n_pool = int(len(pool))
    conf = v["confusion"]
    n_flag = conf["tp"] + conf["fp"]
    p_lo, p_hi = wilson(conf["tp"], n_flag)[1:] if n_flag else (0, 0)
    fv = feed["frames_visible"]
    nov50 = m["precision"]["novelty"]["50"]
    novelty_rank = by_rid.loc[ex["found"]["review_id"], "rank_novelty"]

    # ---- contact sheet: every labelled frame, in review order
    tiles = []
    for r in pool.sort_values("review_id").itertuples(index=False):
        cls = "hit" if r.label == "yes" else ""
        if r.vlm == "yes":
            cls += " flag"
        tiles.append(f'<i class="{cls.strip()}"></i>')

    def fig(rid: str, caption_html: str, extra_cls: str = "") -> str:
        src, r = img(rid)
        return (f'<figure class="{extra_cls}"><div class="frame"><img loading="lazy" src="{src}" alt="{E(stamp(r))}">'
                f'<span class="stamp">{E(stamp(r))}</span></div><figcaption>{caption_html}</figcaption></figure>')

    hero_rid = ex["hero"]["review_id"]
    hero_src, hero_row = img(hero_rid)
    hero_said = vl[hero_row["frame_id"]]["evidence"]

    not_in_view = "".join(fig(e["review_id"], E(e["caption"])) for e in ex["not_in_view"])
    errors = "".join(
        fig(e["review_id"], f'<b>{E(e["pattern"])}.</b> Model: “{E(vl[by_rid.loc[e["review_id"], "frame_id"]]["evidence"])}”'
                            f'<br>On review: {E(e["on_review"])}') for e in ex["vlm_errors"])
    response = "".join(fig(e["review_id"], E(e["caption"])) for e in ex["response"])
    found = fig(ex["found"]["review_id"], E(ex["found"]["caption"]), "wide")

    rows = [("Random sample", prev["n"], prev["k"]),
            ("Frames at the reported crashes", fv["n"], fv["k"]),
            ("Image novelty, top 50 windows", 50, m["precision"]["novelty"]["50"]["k"]),
            ("CLIP zero-shot, top 50 windows", 50, m["precision"]["zeroshot"]["50"]["k"]),
            ("Both combined, top 50 windows", 50, m["precision"]["fused"]["50"]["k"]),
            ("Flagged by the model", n_flag, conf["tp"])]
    table = "".join(f"<tr><td>{E(a)}</td><td class='num'>{b}</td><td class='num'>{k}</td></tr>" for a, b, k in rows)
    code_link = f'<a href="{E(repo_url)}">Code and data pipeline</a>. ' if repo_url else ""

    page = TEMPLATE.format(
        frames=f"{c['frames']:,}", cameras=c["cameras"], n_pool=n_pool, n_yes=n_yes, n_flag=n_flag,
        tiles="".join(tiles), hero_src=hero_src, hero_stamp=E(stamp(hero_row)), hero_said=E(hero_said),
        hero_review=E(ex["hero"]["on_review"]),
        crashes_near=feed["events_near_camera"], fv_k=fv["k"], fv_n=fv["n"], prev_k=prev["k"], prev_n=prev["n"],
        prev_hi=pct(prev["hi"], 1), p_hi=pct(p_hi), not_in_view=not_in_view, errors=errors, response=response,
        found=found, table=table, code_link=code_link, novelty_rank=int(novelty_rank),
        median_s=round(v.get("median_latency_s") or 0))
    (site / "index.html").write_text(page, encoding="utf-8")
    (site / ".nojekyll").write_text("")

    # ---- two-page note
    report = REPORT.format(
        frames=f"{c['frames']:,}", cameras=c["cameras"], n_pool=n_pool, n_yes=n_yes, n_flag=n_flag,
        crashes_near=feed["events_near_camera"], fv_k=fv["k"], fv_n=fv["n"], prev_k=prev["k"], prev_n=prev["n"],
        prev_hi=pct(prev["hi"], 1), p_hi=pct(p_hi), table=table, novelty_rank=int(novelty_rank),
        tiles="".join(tiles), hero_src=hero_src, hero_said=E(hero_said), hero_review=E(ex["hero"]["on_review"]),
        nv_src=img(ex["not_in_view"][0]["review_id"])[0], nv_cap=E(ex["not_in_view"][0]["caption"]),
        rs_src=img(ex["response"][0]["review_id"])[0], rs_cap=E(ex["response"][0]["caption"]),
        median_s=round(v.get("median_latency_s") or 0))
    (site / "report.html").write_text(report, encoding="utf-8")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page()
            pg.goto((site / "report.html").resolve().as_uri())
            pg.wait_for_timeout(500)
            pg.pdf(path=str(site / "report.pdf"), format="A4", print_background=True, prefer_css_page_size=True)
            b.close()
        print("report ->", site / "report.pdf")
    except Exception as e:
        print(f"report.html written; PDF not made ({type(e).__name__}). Print it to PDF from a browser.")
    print("page ->", site / "index.html")


CSS = """
:root { --bg:#F1F3F2; --surface:#FFFFFF; --text:#1D2125; --muted:#5B646B; --rule:#C5CBCF; --green:#006B3F;
        --red:#D22B22; --tile:#DCE1E3; --measure:68ch; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#15181B; --surface:#1E2226; --text:#E5E8EA;
        --muted:#A0A9B0; --rule:#384047; --green:#39A86B; --red:#FF5B4F; --tile:#2C3237; } }
:root[data-theme="dark"] { --bg:#15181B; --surface:#1E2226; --text:#E5E8EA; --muted:#A0A9B0; --rule:#384047;
        --green:#39A86B; --red:#FF5B4F; --tile:#2C3237; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:400 16px/1.55 "Overpass", system-ui, -apple-system,
       "Segoe UI", sans-serif; font-variant-numeric:tabular-nums; }
a { color:var(--green); text-underline-offset:2px; }
a:focus-visible { outline:3px solid var(--green); outline-offset:2px; }
.wrap { max-width:1080px; margin:0 auto; padding:0 16px; }
header.top { display:flex; justify-content:space-between; gap:16px; padding:18px 0; border-bottom:1px solid var(--rule); flex-wrap:wrap; }
header.top b { font-weight:800; }
header.top nav { display:flex; gap:18px; font-size:15px; flex-wrap:wrap; }
section { padding:40px 0; border-bottom:1px solid var(--rule); }
h1 { font-size:clamp(30px,5vw,44px); line-height:1.08; font-weight:800; margin:0 0 16px; letter-spacing:-.01em; max-width:20ch; }
h2 { font-size:24px; line-height:1.2; font-weight:800; margin:0 0 12px; }
p { margin:0 0 12px; max-width:var(--measure); }
.lede { font-size:18px; }
.result { font-size:21px; font-weight:600; max-width:40ch; }
.muted { color:var(--muted); } .small { font-size:14px; }
.hero { display:grid; grid-template-columns:1fr 1.1fr; gap:32px; align-items:start; }
@media (max-width:860px) { .hero { grid-template-columns:1fr; } }
figure { margin:0; }
.frame { position:relative; background:#000; border-radius:4px; overflow:hidden; }
.frame img { display:block; width:100%; height:auto; }
.stamp { position:absolute; left:0; right:0; bottom:0; padding:6px 10px; font-size:13px; color:#fff;
         background:linear-gradient(to top, rgba(0,0,0,.72), rgba(0,0,0,0)); }
figcaption { font-size:14px; color:var(--muted); margin-top:8px; }
figcaption b, .quote b { color:var(--text); font-weight:600; }
.quote { font-size:15px; margin-top:10px; }
.quote .said { color:var(--red); }
.tiles { display:grid; grid-template-columns:repeat(47, 1fr); gap:3px; margin:16px 0 8px; }
@media (max-width:700px) { .tiles { grid-template-columns:repeat(24, 1fr); } }
.tiles i { display:block; aspect-ratio:16/9; background:var(--tile); border-radius:1px; }
.tiles i.flag { background:transparent; box-shadow:inset 0 0 0 2px var(--red); }
.tiles i.hit { background:var(--red); }
.key { display:flex; gap:18px; font-size:13px; color:var(--muted); flex-wrap:wrap; }
.key span::before { content:""; display:inline-block; width:16px; height:9px; margin-right:6px; vertical-align:middle; background:var(--tile); }
.key .k-hit::before { background:var(--red); }
.key .k-flag::before { background:transparent; box-shadow:inset 0 0 0 2px var(--red); }
.gallery { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:22px; margin-top:16px; }
.gallery figure.wide { grid-column:1 / -1; max-width:720px; }
ol.steps { counter-reset:s; list-style:none; padding:0; margin:16px 0 0; display:grid; grid-template-columns:repeat(4,1fr); gap:22px; }
@media (max-width:860px) { ol.steps { grid-template-columns:1fr 1fr; } }
@media (max-width:560px) { ol.steps { grid-template-columns:1fr; } }
ol.steps li { border-top:3px solid var(--green); padding-top:10px; font-size:15px; }
ol.steps li::before { counter-increment:s; content:counter(s); display:block; font-weight:800; font-size:26px; color:var(--green); line-height:1; margin-bottom:8px; }
table { border-collapse:collapse; width:100%; max-width:620px; font-size:15px; margin:12px 0; }
th, td { text-align:left; padding:8px; border-bottom:1px solid var(--rule); }
th { color:var(--muted); font-weight:600; font-size:14px; } td.num, th.num { text-align:right; }
ul.plain { margin:0; padding-left:18px; max-width:var(--measure); } ul.plain li { margin:6px 0; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:32px; } @media (max-width:760px) { .cols { grid-template-columns:1fr; } }
footer { padding:28px 0 48px; font-size:14px; color:var(--muted); }
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rare incident labels from traffic cameras</title>
<meta name="description" content="A test on 21,885 real LTA camera frames: how often incidents are visible, how an off-the-shelf vision-language model does, and what a labelling pipeline should look like instead.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Overpass:wght@400;600;800&display=swap" rel="stylesheet">
<style>""" + CSS.replace("{", "{{").replace("}", "}}") + """</style></head>
<body><div class="wrap">
<header class="top"><b>Rare incident labels from traffic cameras</b>
<nav><a href="#findings">Findings</a><a href="#instead">What I'd build</a><a href="#method">Method</a><a href="report.pdf">Two-page note (PDF)</a></nav></header>

<section class="hero">
  <div>
    <h1>What it takes to label rare traffic incidents from camera data</h1>
    <p class="lede">Incident labels are rare, so I tested a cheap way to find them: rank camera frames with signals that need no labels, have a vision-language model check the candidates, and confirm by hand. I ran it on {frames} real frames from LTA's {cameras} expressway cameras, around 10 news-reported crashes and the same hours a week earlier, and labelled {n_pool} frames without seeing the model's answers.</p>
    <p class="result">{n_yes} of the {n_pool} frames showed an incident. The model flagged {n_flag}, and none of them held up.</p>
  </div>
  <figure>
    <div class="frame"><img src="{hero_src}" alt="{hero_stamp}"><span class="stamp">{hero_stamp}</span></div>
    <p class="quote"><b>The model:</b> <span class="said">“{hero_said}”</span><br><b>On review:</b> {hero_review}</p>
  </figure>
</section>

<section>
  <h2>All {n_pool} frames I labelled</h2>
  <p>Each tile is one frame, in the order I labelled them. Nearly every frame shows ordinary traffic, and the model's alarms land on it too.</p>
  <div class="tiles" aria-hidden="true">{tiles}</div>
  <div class="key"><span class="k-hit">incident, confirmed by me</span><span class="k-flag">model said incident</span><span>no incident</span></div>
</section>

<section id="findings">
  <h2>The cameras didn't see the crashes</h2>
  <p>None of the {crashes_near} crashes reported within 1 km of a camera was visible in that camera's frames 1 and 10 minutes after the report ({fv_k} of {fv_n} frames). The random sample had none either ({prev_k} of {prev_n}; 95% interval 0 to {prev_hi}). A camera sees only a few hundred metres of road, and a crash takes longer than 10 minutes to clear, so these crashes were out of the camera's view, not between snapshots.</p>
  <div class="gallery">{not_in_view}</div>
</section>

<section>
  <h2>An off-the-shelf model over-flags</h2>
  <p>Qwen3-VL 32B saw each frame with the one before it, was told to report only what it could see, and was allowed to say it couldn't tell. It flagged {n_flag} of {n_pool} frames. On review none showed an incident (95% interval 0 to {p_hi}), and it missed the one frame that did. Its errors follow patterns:</p>
  <div class="gallery">{errors}</div>
</section>

<section>
  <h2>The response shows even when the crash doesn't</h2>
  <p>In three of the reported crashes, frames 1 to 31 minutes later show police, SCDF or recovery vehicles with lights on, in the queue the crash caused. I labelled them "no incident" because the crash itself is out of the camera's view, but they are the clearest visual trace of one.</p>
  <div class="gallery">{response}</div>
</section>

<section>
  <h2>The one incident</h2>
  <div class="gallery">{found}</div>
</section>

<section id="instead">
  <h2>What I'd build instead</h2>
  <p>The ranking and the model both need work, but the first problem is coverage: the cameras see a small part of the road, and most incidents happen out of their view. So I would turn the order around:</p>
  <ol class="steps">
    <li>Find where and when to look from a signal that covers every road, all the time: speed bands, loop detectors, the incident feed.</li>
    <li>Pull 30 to 60 seconds of video from the nearest cameras, not single stills. Stopped vehicles and responders are clear in motion and ambiguous in one frame.</li>
    <li>Measure the model's precision on a small labelled set before trusting it, and point it at the cues that showed up here: vehicles that don't move, and emergency or recovery vehicles.</li>
    <li>Send what's left to a person. Confirmed labels retrain the cheap ranker, so each pass needs fewer human checks.</li>
  </ol>
</section>

<section id="method" class="cols">
  <div>
    <h2>How I tested it</h2>
    <ul class="plain">
      <li><b>Frames.</b> LTA has published live images for only 8 cameras since mid-2026, but data.gov.sg still keeps the history of all 90. For each of 10 crashes reported in the news between February 2025 and May 2026: every camera, every 5 minutes, from 20 minutes before to 40 minutes after, plus the same hour a week earlier as a control.</li>
      <li><b>Rankings, no labels needed.</b> Image novelty (CLIP embedding distance to the same camera's other frames, ignoring those within 45 minutes), CLIP zero-shot similarity to incident descriptions, and the two combined. The top 50 camera-time windows from each, 80 random frames and 28 frames at the reported crashes made a shuffled review set of {n_pool}.</li>
      <li><b>Labels.</b> One person, me, blind to the rankings and the model. Incident means a crash, a stopped or broken-down vehicle, debris, responders attending, or people on the carriageway. Congestion alone isn't.</li>
      <li><b>Model.</b> Qwen3-VL 32B through NUS SoC's API, temperature 0, a fixed JSON reply with the option to abstain, about {median_s} seconds per frame.</li>
    </ul>
  </div>
  <div>
    <h2>The numbers</h2>
    <table><tr><th>Frames reviewed</th><th class="num">Frames</th><th class="num">Incidents</th></tr>{table}</table>
    <p class="small muted">The one incident was ranked #{novelty_rank} by image novelty. With counts this small the intervals are wide; the point is the order of magnitude.</p>
    <h2 style="margin-top:28px">Limits</h2>
    <ul class="plain">
      <li>235 frames and 10 crashes: enough to show the pattern, not to rank models precisely.</li>
      <li>One labeller, single frames 5 minutes apart. Some incidents are only clear on video.</li>
      <li>One general-purpose model and one prompt, untuned. A better prompt or a fine-tuned model may do better; the point is to measure before trusting.</li>
      <li>Public cameras only: one still every few minutes. Video from an operational camera network would show far more.</li>
      <li>No speed or loop-detector data: it exists only live, so it couldn't be matched to these past crashes.</li>
    </ul>
  </div>
</section>

<footer>
  <p>{code_link}Built by Lim Zi Qiang. Independent prototype, not affiliated with or endorsed by the Land Transport Authority. Camera images from LTA via data.gov.sg under the <a href="https://data.gov.sg/open-data-licence">Singapore Open Data Licence</a>. Crash times and places come from the news reports cited in the code.</p>
</footer>
</div></body></html>
"""

REPORT = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Rare incident labels from traffic cameras</title>
<style>
@page {{ size:A4; margin:15mm 15mm 13mm; }}
body {{ font:9.6pt/1.42 "Helvetica Neue", Arial, sans-serif; color:#1D2125; margin:0; }}
h1 {{ font-size:17pt; line-height:1.15; margin:0 0 2mm; }} h2 {{ font-size:11pt; color:#006B3F; margin:4.5mm 0 1.5mm; }}
p {{ margin:0 0 2mm; }} .sub {{ color:#5B646B; }} .result {{ font-weight:700; font-size:11pt; }}
table {{ border-collapse:collapse; width:100%; font-size:9pt; margin:1mm 0 2mm; }}
th, td {{ border-bottom:1px solid #C5CBCF; padding:1.2mm 1.5mm; text-align:left; }} th {{ color:#5B646B; font-weight:600; }}
td.num, th.num {{ text-align:right; }}
.tiles {{ display:grid; grid-template-columns:repeat(47,1fr); gap:1.6px; margin:2mm 0 1mm; }}
.tiles i {{ display:block; aspect-ratio:16/9; background:#DCE1E3; }} .tiles i.flag {{ background:#fff; box-shadow:inset 0 0 0 1.2px #D22B22; }}
.tiles i.hit {{ background:#D22B22; }} .key {{ font-size:8pt; color:#5B646B; }}
.imgs {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:3mm; margin:2mm 0; }} .imgs img {{ width:100%; display:block; }}
.cap {{ font-size:8pt; color:#5B646B; margin-top:1mm; }}
ol, ul {{ margin:0 0 2mm; padding-left:5mm; }} li {{ margin:.7mm 0; }} .page {{ page-break-after:always; }}
</style></head><body>
<div class="page">
<h1>What it takes to label rare traffic incidents from camera data</h1>
<p class="sub">A weekend test on {frames} real frames from LTA's {cameras} expressway cameras. Lim Zi Qiang.</p>
<h2>Question</h2>
<p>Incident labels are rare. Can a cheap pipeline find them in camera data: rank frames with signals that need no labels, let a vision-language model check the candidates, and confirm by hand?</p>
<h2>Result</h2>
<p class="result">{n_yes} of {n_pool} frames I labelled showed an incident. The model flagged {n_flag}; none held up.</p>
<div class="tiles">{tiles}</div>
<p class="key">Each tile is one labelled frame. Filled red: incident. Red outline: the model said incident.</p>
<ul>
<li><b>The cameras didn't see the crashes.</b> None of the {crashes_near} crashes reported within 1 km of a camera was visible 1 and 10 minutes after the report ({fv_k} of {fv_n} frames): they were out of the camera's view, not between snapshots. The random sample had none either ({prev_k} of {prev_n}; 95% interval 0 to {prev_hi}).</li>
<li><b>An off-the-shelf model over-flags.</b> Qwen3-VL 32B flagged {n_flag} frames, none correct (95% interval 0 to {p_hi}), and missed the one real incident. It read motion blur as an overturned car, brake lights as stopped vehicles and checkpoint queues as incidents.</li>
<li><b>The response shows even when the crash doesn't.</b> In three reported crashes, later frames show police, SCDF or recovery vehicles with lights on in the resulting queue.</li>
</ul>
<table><tr><th>Frames reviewed</th><th class="num">Frames</th><th class="num">Incidents</th></tr>{table}</table>
<div class="imgs">
<div><img src="{hero_src}"><div class="cap">Model: “{hero_said}” On review: {hero_review}</div></div>
<div><img src="{nv_src}"><div class="cap">{nv_cap}</div></div>
<div><img src="{rs_src}"><div class="cap">{rs_cap}</div></div>
</div>
</div>
<h2>What I would build instead</h2>
<p>The ranking and the model both need work, but the first problem is coverage: the cameras see a small part of the road, and most incidents happen out of their view. So turn the order around:</p>
<ol>
<li>Find where and when to look from a signal that covers every road, all the time: speed bands, loop detectors, the incident feed.</li>
<li>Pull 30 to 60 seconds of video from the nearest cameras rather than single stills; stopped vehicles and responders are clear in motion and ambiguous in one frame.</li>
<li>Measure the model's precision on a small labelled set before trusting it, and point it at the cues seen here: vehicles that don't move, and emergency or recovery vehicles.</li>
<li>Send the rest to a person; confirmed labels retrain the cheap ranker, so each pass needs fewer human checks.</li>
</ol>
<h2>Method</h2>
<ul>
<li><b>Frames.</b> LTA has published live images for only 8 cameras since mid-2026; data.gov.sg keeps the history of all 90. For 10 crashes reported in the news (Feb 2025 to May 2026): every camera every 5 minutes from 20 minutes before to 40 minutes after, plus the same hour a week earlier as a control.</li>
<li><b>Rankings.</b> Image novelty (CLIP embedding distance to the same camera's other frames, ignoring those within 45 minutes), CLIP zero-shot similarity to incident descriptions, and both combined. The top 50 camera-time windows from each, 80 random frames and 28 frames at the reported crashes formed a shuffled review set of {n_pool}. The one incident was ranked #{novelty_rank} by image novelty.</li>
<li><b>Labels.</b> One labeller (me), blind to rankings and model output. Incident: a crash, a stopped or broken-down vehicle, debris, responders attending, or people on the carriageway; congestion alone is not.</li>
<li><b>Model.</b> Qwen3-VL 32B via NUS SoC's API, temperature 0, fixed JSON reply with an abstain option, about {median_s} s per frame.</li>
</ul>
<h2>Limits</h2>
<ul>
<li>235 frames and 10 crashes: enough to show the pattern, not to rank models precisely.</li>
<li>One labeller; single frames 5 minutes apart; some incidents are only clear on video.</li>
<li>One general-purpose model and prompt, untuned: the lesson is to measure before trusting, not that VLMs cannot do this.</li>
<li>Public cameras only, one still every few minutes; video would show far more. No speed or loop data: it exists only live.</li>
</ul>
<p class="cap">Independent prototype, not affiliated with LTA. Images: LTA via data.gov.sg, Singapore Open Data Licence.</p>
</body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_hist")
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--site", default="docs")
    ap.add_argument("--examples", default=str(ROOT / "data_inputs" / "findings_examples.json"))
    ap.add_argument("--repo-url", default="")
    a = ap.parse_args()
    Path(a.site).mkdir(parents=True, exist_ok=True)
    build(Path(a.raw), Path(a.work), Path(a.site), Path(a.examples), a.repo_url)
