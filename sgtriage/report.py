"""Two-page technical note (docs/report.html -> docs/report.pdf).

The text is generated from metrics.json so every number in it is the computed one.
PDF: uses Playwright if installed (pip install playwright && playwright install chromium);
otherwise open report.html in a browser and print to PDF (A4, no headers/footers).
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from .common import Paths, require
from .evaluate import figures

NAMES = {"random": "Random order", "speed": "Speed drop", "novelty": "Image novelty", "zeroshot": "CLIP zero-shot",
         "fused": "Fused (no labels)", "learned": "Learned (feed labels)"}


def _img64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def _pct(x, d=0):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def build_report(paths: Paths) -> Path:
    m = json.loads(require(paths.work / "metrics.json", "evaluate").read_text())
    figs = figures(m, paths.work / "figs")
    c, p, feed, v = m["counts"], m.get("prevalence", {}), m.get("feed", {}), m.get("vlm") or {}
    best = m.get("best_method")
    bp = (m["precision"].get(best, {}) or {}).get("25") if best else None
    e = html.escape

    rows = ""
    for meth, d in m["precision"].items():
        x = d.get("25") or d.get("10")
        if not x:
            continue
        r = (feed.get("recall", {}).get(meth, {}) or {}).get("25", {})
        lift = "1×" if meth == "random" else (f"{x['lift']:.1f}×" if x.get("lift") else "–")
        surfaced = _pct(r.get("all")) if r.get("all") is not None else "–"
        rows += (f"<tr><td>{NAMES.get(meth, meth)}</td><td>{x['k']}/{x['n']}</td><td>{_pct(x['p'])}</td>"
                 f"<td>{_pct(x['lo'])}–{_pct(x['hi'])}</td><td>{lift}</td><td>{surfaced}</td></tr>")

    headline = ""
    if p.get("n") and bp:
        headline = (f"In a random sample of {p['n']} frames, {p['k']} showed an incident ({_pct(p['p'], 1)}). "
                    f"Among the top {bp['n']} camera-time windows ranked by {NAMES.get(best, best).lower()}, "
                    f"{bp['k']} did ({_pct(bp['p'])}, 95% interval {_pct(bp['lo'])}–{_pct(bp['hi'])}): "
                    f"about {bp['lift']:.0f}× fewer frames reviewed per incident found." if bp.get("lift") else "")
    vlm_txt = ""
    if v.get("confusion"):
        cas = (v.get("cascade") or {}).get(best) or {}
        vlm_txt = (f"Against the human labels, the VLM ({e(v.get('model', ''))}) had precision {_pct(v.get('precision'))} "
                   f"and recall {_pct(v.get('recall'))}, and said it could not tell on {_pct(v.get('abstain_rate'))} of frames. ")
        if cas:
            vlm_txt += (f"As a filter on the top 25 it cut human review from {cas['reviewed_before']} to "
                        f"{cas['reviewed_after']} frames and kept {cas['found_after']} of {cas['found_before']} incidents.")
    hist = m.get("kind") == "history"
    radius = m.get("camera_radius", 600)
    scope = (f"{c['frames']:,} real frames from {c['cameras']} LTA expressway cameras around news-reported crashes, "
             f"plus the same windows one week earlier." if hist else
             f"{c['frames']:,} frames from {c['cameras']} LTA cameras over {c['hours']:g} hours, with LTA's live incident feed.")
    data_txt = ("Historical frames from data.gov.sg (all ~90 expressway cameras; LTA has published live images for only "
                "8 cameras since mid-2026), every 5 minutes from 20 minutes before to 40 minutes after each reported "
                "crash, and the same window 7 days earlier as a matched control. LTA's traffic-signal layer gives junction "
                "locations." if hist else
                "Camera frames every ~3 minutes (LTA DataMall, data.gov.sg) and Traffic Incidents polled every 2 minutes. "
                "LTA's traffic-signal layer gives junction locations.")
    label_txt = (f"News reports give each crash's time and approximate place (from article text); frames from cameras "
                 f"within {radius} m during the 40 minutes after the report are weak positives. The reports are used only "
                 f"for when and where; every image is LTA's." if hist else
                 f"Repeated feed sightings were merged into incidents; an incident labels frames from cameras within "
                 f"{radius} m during its active window.")
    speed_txt = ("Road speed bands would add an independent signal, but they exist only live and public live images now "
                 "cover 8 cameras, so speed is not part of this test." if hist else "")
    feed_txt = ""
    if feed.get("events_total"):
        src = "News reports gave" if hist else "LTA's feed reported"
        feed_txt = (f"{src} {feed['events_total']} incidents; {feed.get('events_near_camera', 0)} "
                    f"were within {radius} m of a camera and {feed.get('at_junction', 0)} within 40 m of a signalised junction. ")
        if feed.get("events_labelled"):
            feed_txt += (f"Only {feed['events_visible']} of {feed['events_labelled']} reported incidents near a camera were "
                         f"visible in its frames, which is why reports are used as weak labels and a recall check, not as truth.")

    fig_prec = _img64(paths.work / "figs" / "fig_precision.png")
    fig_conf = _img64(paths.work / "figs" / "fig_vlm_confusion.png")
    learned = m.get("learned", {})
    learned_txt = (f"Not run: {e(learned['skipped'])}." if learned.get("skipped") else
                   f"Logistic regression on {len(learned.get('features', []))} features, trained on the feed's weak labels "
                   f"({learned.get('n_pos_events', 0)} incidents) and scored out-of-fold by 2-hour blocks, so no frame is scored "
                   f"by a model that saw its incident.")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Rare incident triage: technical note</title>
<style>
@page {{ size: A4; margin: 16mm 16mm 14mm; }}
body {{ font: 10pt/1.45 "Overpass", "Helvetica Neue", Arial, sans-serif; color: #1D2125; margin: 0; }}
h1 {{ font-size: 18pt; margin: 0 0 2mm; line-height: 1.15; }} h2 {{ font-size: 11.5pt; margin: 5mm 0 1.5mm; color: #006B3F; }}
p {{ margin: 0 0 2.2mm; }} .sub {{ color: #5B646B; margin-bottom: 4mm; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 2mm 0; }}
th, td {{ border-bottom: 1px solid #C5CBCF; padding: 1.4mm 1.5mm; text-align: left; }} th {{ color: #5B646B; font-weight: 600; }}
img.fig {{ width: 100%; }} .two {{ display: grid; grid-template-columns: 1.6fr 1fr; gap: 5mm; align-items: start; }}
ul {{ margin: 0 0 2mm; padding-left: 5mm; }} li {{ margin: .8mm 0; }} .small {{ font-size: 8.5pt; color: #5B646B; }}
.page {{ page-break-after: always; }}
</style></head><body>
<div class="page">
<h1>Finding rare traffic incidents without reviewing every frame</h1>
<p class="sub">Technical note on a weekend prototype. {scope}</p>
<h2>Problem</h2>
<p>Incidents are rare in camera data, so a model that detects them, or any safety logic built on top, is starved of labels. Reviewing frames at random spends almost all human time on normal traffic. This prototype tests a cheaper route: rank frames with signals that need no labels, check the top candidates with a vision-language model (VLM), and give a person only what is left.</p>
<h2>Result</h2>
<p>{headline}</p>
<p>{vlm_txt}</p>
<div class="two"><div>{f'<img class="fig" src="{fig_prec}" alt="Incidents per 100 frames reviewed by ranking">' if fig_prec else ''}</div>
<div>{f'<img class="fig" src="{fig_conf}" alt="VLM against human labels">' if fig_conf else ''}</div></div>
<table><tr><th>Ranking (top 25 windows)</th><th>Incidents</th><th>Rate</th><th>95% interval</th><th>vs random</th><th>Known incidents surfaced</th></tr>{rows}</table>
<p class="small">A window is one camera within ±15 minutes, so a single long incident cannot fill the list. “Can't tell” labels ({c['unsure']}) are excluded from rates.</p>
</div>
<h2>Method</h2>
<ul>
<li><b>Data.</b> {data_txt}</li>
<li><b>Weak labels.</b> {label_txt} {e(feed_txt)}</li>
<li><b>Signals.</b> CLIP ViT-B/32 image embeddings. Novelty is the distance to the same camera's nearest frames from other times (frames within ±45 minutes excluded, so a long incident cannot match itself), ranked within camera; zero-shot compares the frame with incident and normal-traffic descriptions; fused is the average rank of the two. Blank or black frames are removed. {speed_txt}</li>
<li><b>Learned ranking.</b> {learned_txt}</li>
<li><b>VLM check.</b> The model sees the frame and the one before it, nothing else, and returns fixed JSON (incident yes/no/unsure, type, confidence, evidence). It never sees the feed, the scores or why a frame was picked. Responses are cached with model and prompt version.</li>
<li><b>Labelling.</b> A shuffled pool of the top 50 windows from each ranking, {c['pool']} frames in all, including feed-reported windows and a random sample. Labelled blind: the page shows only frames, camera and time.</li>
<li><b>Metrics.</b> Precision at k with Wilson 95% intervals; lift over the random-sample rate; recall of feed-reported incidents; VLM precision, recall and agreement (κ), by day and night.</li>
</ul>
<h2>Limits</h2>
<ul>
<li>{'Ten news-reported crashes and their control windows; incidents the news never covered are found only through the ranking.' if hist else 'One collection window; rates depend on weather, time of day and traffic.'}</li>
<li>One labeller, still frames about 3 minutes apart; some incidents are only clear in video.</li>
<li>Public cameras cover expressways and checkpoints, so junctions are not tested.</li>
<li>The VLM is general-purpose and not fine-tuned; errors are reported, not hidden.</li>
</ul>
<h2>Next: signalised junctions</h2>
<p>At junctions, crashes are rarer still, so the useful labels are conflicts and near-misses. The same three stages apply to junction CCTV: rank frames by unusual vehicle and pedestrian movement, check candidates with the VLM, confirm with a person. Confirmed conflicts could then be related to signal timing and turning volumes using LTA's on-request <i>Traffic Lights Traffic Plans</i> and <i>Indicative Traffic Counts at Junctions by Loop Detectors</i>, and the public <i>Traffic Count</i> turn-movement dataset.</p>
<p class="small">Independent prototype, not affiliated with LTA. Data under the Singapore Open Data Licence.</p>
</body></html>"""
    out = paths.site / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    pdf = paths.site / "report.pdf"
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page()
            pg.goto(out.resolve().as_uri())
            pg.pdf(path=str(pdf), format="A4", print_background=True, prefer_css_page_size=True)
            b.close()
        print(f"report -> {pdf}")
    except Exception as ex:  # Playwright missing or no browser
        print(f"report -> {out} (PDF not made: {type(ex).__name__}). Open it in a browser and print to PDF.")
    return out
