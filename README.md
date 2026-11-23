# Rare incident triage for Singapore traffic cameras

Traffic incidents are rare in camera data. Most frames show normal traffic, so a person reviewing frames at random spends nearly all their time confirming that nothing happened, and any model that should detect incidents is starved of labels.

This project tests a cheaper route on real LTA camera frames: rank every frame with signals that need no labels, have a vision-language model (VLM) check the top candidates, and give a person only what is left. Every number is measured against blind human labels.

**Dashboard:** `docs/` (GitHub Pages). **Technical note:** `docs/report.pdf`.

**Concept demo:** `docs/signals/` (Signal Timing Lab). A simulated four-junction corridor showing how signal timing (fixed, green wave, vehicle-actuated) changes delay, stops and collision risk. The traffic is simulated with the Intelligent Driver Model.

## Result

On 21,885 real frames from LTA's 90 expressway cameras (around 10 news-reported crashes, plus the same hours a week earlier), I labelled a shuffled review set of 235 frames blind:

- **1 of 235 frames showed an incident.** The random sample had none (0 of 80), and none of the 9 crashes reported within 1 km of a camera was visible in its frames (0 of 28).
- **An off-the-shelf VLM over-flags.** Qwen3-VL 32B flagged 32 frames and none held up; it missed the one real incident. It read motion blur as an overturned car, brake lights as stopped vehicles and checkpoint queues as incidents.
- **The response is the visible trace.** In three reported crashes, later frames show police, SCDF or recovery vehicles with lights on in the resulting queue.
- **Ranking:** image novelty surfaced the one incident (#28); CLIP zero-shot surfaced none.

So the first problem is coverage, not ranking: the crashes were out of the camera's view, not between snapshots. The pipeline I would build starts from a signal that covers every road (speed bands, loop detectors, the incident feed), pulls short video clips from the nearest cameras, and measures a model's precision before trusting it. Details: `docs/index.html` and `docs/report.pdf` (built by `tools/build_findings.py`).

## Data, and why it looks like this

- **Camera frames:** LTA has published live images for only 8 cameras since mid-2026 (checkpoints and Sentosa), but data.gov.sg still serves history for all ~90 expressway cameras up to then. The test uses that history: every camera, every 5 minutes, from 20 minutes before to 40 minutes after each of 10 news-reported crashes, plus the same window one week earlier as a matched control.
- **Incident reports:** news articles give each crash's time and approximate place. They are used only for *when and where*; every image is LTA's own. They act as weak labels and as a recall check, never as ground truth.
- **Traffic-signal layer:** LTA DataMall's signal-aspect points, clustered into signalised junctions, to mark incidents at traffic lights.
- **Not included:** road speed bands. They exist only live, and with 8 live cameras they cannot be paired with camera frames. `collect.py` and `python -m sgtriage speed` (speed-drop detectors scored against LTA's live incident feed) are in the repo for that next step.

## How it works

```
backfill        data.gov.sg history: all cameras around each reported crash + same window a week earlier
   |
labels          reports -> events -> weak labels on frames from cameras within 1 km
   |
embed           CLIP ViT-B/32 embeddings -> per-camera novelty, zero-shot score, blank-frame filter
   |
rank            random | novelty | zero-shot | fused (average rank)
   |
pool            top 50 windows per ranking + reported-incident frames + 80 random frames, shuffled
   |
review  (human, blind)        vlm  (one call per frame, fixed JSON, may abstain)
   \______________________________/
                 |
evaluate        incidents per 100 frames reviewed, lift over random, recall of reported crashes, VLM vs human
   |
report / site   docs/report.pdf and the static dashboard in docs/
```

## Design decisions

| Decision | Why |
|---|---|
| Rank camera-time **windows**, not frames | One long incident produces many near-identical frames. Without this, a single breakdown can fill the whole top-k and inflate precision. |
| Novelty ignores the same camera's frames within ±45 min, and is ranked within camera | An incident lasting 40 minutes would otherwise match its own frames and look normal. Some views are naturally busier than others. |
| Matched control windows one week earlier | Each camera gets normal frames from the same time of day, so "unusual" is not just "different lighting". |
| Fusion by average rank | Needs no labels or tuning, and is insensitive to each signal's scale. |
| The VLM sees only the frame and the one before it | Reports, scores and the reason a frame was picked would all leak the answer. |
| Fixed JSON output, "unsure" allowed | Structured output is checkable; abstaining is cheaper than a confident wrong answer. Responses are cached with model and prompt version. |
| Blind, shuffled labelling | The labelling page shows only frames, camera and time, so labels cannot be nudged by the system under test. |
| Reports are weak labels and a recall check, not truth | A crash reported near a camera is often out of the camera's view. |
| Wilson 95% intervals | Counts are small; normal-approximation intervals misbehave near 0. |

## Failure handling

- **Downloads:** retries with backoff, resume-safe de-duplication by image hash, polite pacing.
- **Images:** blank or black frames (camera offline) are filtered by contrast and by CLIP's similarity to "no signal" descriptions.
- **VLM:** works with any OpenAI-compatible endpoint (Chat Completions or Responses API); retries on rate limits, falls back when JSON mode is unsupported, treats unparseable replies as "unsure", and stops early if most calls fail.
- **Live collector (for the speed step):** endpoint fallbacks for renamed DataMall APIs, key reload without restart, and a log of every incident poll so gaps in coverage are known.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                        # VLM_BASE_URL, VLM_API_KEY, VLM_MODEL

R="--raw data/raw_hist"
python -m sgtriage backfill $R              # ~23k frames for the 10 incidents in data_inputs/news_incidents.csv
python -m sgtriage static                   # LTA traffic-signal layer -> signalised junctions
python -m sgtriage prepare $R               # labels, embeddings, rankings, review pool
python -m sgtriage review $R                # blind labelling page at http://localhost:8008
python -m sgtriage models                   # which models your endpoint offers
python -m sgtriage vlm $R
python -m sgtriage evaluate $R
python -m sgtriage report $R && python -m sgtriage site $R
```

Docker (CPU) runs every step except labelling:
`docker build -t sgtriage . && docker run --rm -v "$PWD/data:/app/data" -v "$PWD/docs:/app/docs" --env-file .env sgtriage prepare --raw data/raw_hist`

`bash tests/run_pipeline_test.sh` runs the whole pipeline offline on synthetic data with a mock VLM.

## What I would change next

- Pair live road-speed data with camera frames: speed flags where traffic broke down, the camera and VLM say what happened. Needs camera coverage beyond the public 8.
- Short video clips instead of stills 5 minutes apart: stopped vehicles and near-misses are much clearer in motion.
- Close the loop: retrain the ranker on confirmed labels and send the most uncertain frames to people first.
- Fine-tune or distil a small VLM on confirmed labels so the check runs cheaply, or at the edge.
- At signalised junctions, label conflicts and near-misses (far more frequent than crashes) and relate them to signal timing using LTA's on-request *Traffic Lights Traffic Plans* and *Indicative Traffic Counts at Junctions by Loop Detectors*.

## Data licence

LTA DataMall and data.gov.sg data are used under the [Singapore Open Data Licence](https://data.gov.sg/open-data-licence). News articles are cited in `data_inputs/news_incidents.csv` and used only for the time and place of each incident. This is an independent prototype, not affiliated with or endorsed by the Land Transport Authority.
