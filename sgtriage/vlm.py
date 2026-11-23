"""VLM triage: one vision-language-model call per candidate frame.

Design rules:
  - The model sees two frames (earlier + current) and nothing else. It never sees the incident
    feed, the scores or why the frame was picked: the feed is the answer key.
  - It must answer in a fixed JSON shape and may say "unsure". Abstaining is better than guessing.
  - Every response is cached with the model name and prompt version, so reruns cost nothing and
    results are reproducible and auditable.
Works with any OpenAI-compatible endpoint (VLM_BASE_URL, VLM_API_KEY, VLM_MODEL in .env).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

PROMPT_VERSION = "v1"
SYSTEM = ("You review frames from fixed traffic cameras on Singapore expressways and look for traffic incidents. "
          "Report only what is visible. Do not guess.")
USER_TWO = """Two frames from the same fixed camera. Image 1 is about 3 minutes earlier. Image 2 is the frame to judge.

Is a traffic incident visible in image 2?
Counts as an incident: a collision or crashed vehicle; a vehicle stopped in a live lane or on the shoulder; debris or an obstacle on the carriageway; police, ambulance, fire or recovery vehicles attending; people on the carriageway.
Not an incident: ordinary congestion or slow queues; planned roadworks with traffic moving; rain, glare, darkness or lens dirt.
Use image 1 to tell a stopped vehicle (same place in both) from moving traffic.
If image 2 is too dark, blurred or distant to tell, answer "unsure".

Reply with JSON only, no other text:
{"incident": "yes" | "no" | "unsure", "type": "collision" | "stopped_vehicle" | "obstacle" | "emergency_response" | "people_on_road" | "none", "confidence": <0.0-1.0>, "where": "<short location in image 2, or empty>", "evidence": "<one short sentence>", "image_quality": "good" | "poor" | "unusable"}"""
USER_ONE = USER_TWO.replace(
    "Two frames from the same fixed camera. Image 1 is about 3 minutes earlier. Image 2 is the frame to judge.",
    "One frame from a fixed camera (call it image 2).").replace(
    "Use image 1 to tell a stopped vehicle (same place in both) from moving traffic.\n", "")

VALID = {"incident": {"yes", "no", "unsure"},
         "type": {"collision", "stopped_vehicle", "obstacle", "emergency_response", "people_on_road", "none"},
         "image_quality": {"good", "poor", "unusable"}}


def _data_uri(img: Image.Image, max_side: int) -> str:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _json_objects(text: str) -> list[str]:
    """Every balanced top-level {...} in the text, in order (handles prose or reasoning around the JSON)."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def parse_reply(text: str) -> dict:
    out = {"incident": "unsure", "type": "none", "confidence": 0.0, "where": "", "evidence": "",
           "image_quality": "poor", "parse_error": True}
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # reasoning models
    j = None
    for cand in reversed(_json_objects(text)):  # the answer is usually the last JSON object
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "incident" in obj:
            j = obj
            break
    if j is None:
        return out
    for k in ("incident", "type", "image_quality"):
        v = str(j.get(k, "")).strip().lower()
        if v in VALID[k]:
            out[k] = v
    try:
        out["confidence"] = min(1.0, max(0.0, float(j.get("confidence", 0))))
    except (TypeError, ValueError):
        pass
    out["where"] = str(j.get("where", ""))[:200]
    out["evidence"] = str(j.get("evidence", ""))[:300]
    out["parse_error"] = False
    return out


def vlm_score(p: dict) -> float:
    """One number for ranking: P(incident) as the model states it."""
    c = p.get("confidence", 0.5)
    return {"yes": c, "no": 1 - c}.get(p.get("incident"), 0.5)


class Client:
    """OpenAI-compatible client. Uses /chat/completions, or /responses if the gateway only offers that."""

    def __init__(self, base_url: str, api_key: str, model: str, max_side: int = 1280, timeout: int = 120):
        self.base = base_url.rstrip("/")
        self.key, self.model, self.max_side, self.timeout = api_key, model, max_side, timeout
        self.json_mode = True
        self.mode = "chat"
        self.s = requests.Session()

    def _payload(self, images: list[str], text: str, json_mode: bool) -> tuple[str, dict]:
        if self.mode == "chat":
            content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in images]
            body = {"model": self.model, "temperature": 0, "max_tokens": 1200,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            return self.base + "/chat/completions", body
        content = [{"type": "input_text", "text": text}] + [{"type": "input_image", "image_url": u} for u in images]
        return self.base + "/responses", {"model": self.model, "instructions": SYSTEM, "temperature": 0,
                                          "max_output_tokens": 1200, "input": [{"role": "user", "content": content}]}

    @staticmethod
    def _text(j: dict) -> str:
        if "choices" in j:
            text = j["choices"][0]["message"].get("content") or ""
            if isinstance(text, list):  # some gateways return content parts
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            return text
        if j.get("output_text"):
            return j["output_text"]
        return "".join(c.get("text", "") for o in j.get("output", []) if isinstance(o, dict)
                       for c in (o.get("content") or []) if isinstance(c, dict))

    def ask(self, cur: Image.Image, prev: Image.Image | None) -> tuple[dict, str, dict, float]:
        images = ([_data_uri(prev, self.max_side)] if prev is not None else []) + [_data_uri(cur, self.max_side)]
        text = USER_TWO if prev is not None else USER_ONE
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        delay = 5.0
        for attempt in range(8):
            sent_json, sent_mode = self.json_mode, self.mode
            url, body = self._payload(images, text, sent_json)
            t0 = time.time()
            r = self.s.post(url, json=body, headers=headers, timeout=self.timeout)
            latency = time.time() - t0
            if r.status_code == 400 and sent_json and sent_mode == "chat" and "response_format" in r.text:
                self.json_mode = False  # endpoint does not support JSON mode; the prompt still asks for JSON
                continue
            if r.status_code in (404, 405) and sent_mode == "chat":
                self.mode = "responses"  # gateway only speaks the Responses API
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", delay))
                time.sleep(min(wait, 60))
                delay *= 2
                continue
            r.raise_for_status()
            j = r.json()
            self.last_json = j
            raw = self._text(j)
            return parse_reply(raw), raw, j.get("usage", {}), latency
        raise RuntimeError(f"VLM endpoint kept failing (last HTTP {r.status_code}): {r.text[:200]}")


VISION_HINTS = ("vl", "vision", "omni", "4o", "gpt-4.1", "gpt-5", "gemini", "claude", "glm-4v", "glm-4.5v",
                "llava", "pixtral", "internvl", "minicpm-v", "doubao-seed", "seed-1", "kimi-vl", "gemma-3")


def list_models(base_url: str, api_key: str) -> list[tuple[str, bool]]:
    r = requests.get(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                     timeout=30)
    r.raise_for_status()
    ids = sorted(m.get("id", "") for m in r.json().get("data", []))
    return [(i, any(h in i.lower() for h in VISION_HINTS)) for i in ids]


def _fullres(row, raw: Path, cache_dir: Path, session: requests.Session) -> Image.Image:
    """Prefer the original resolution from data.gov.sg (small stopped cars matter); fall back to our copy."""
    url = row.get("url") if isinstance(row.get("url"), str) else ""
    if url.startswith("http"):
        p = cache_dir / (hashlib.md5(url.encode()).hexdigest() + ".jpg")
        if not p.exists():
            try:
                r = session.get(url, timeout=30)
                if r.status_code == 200 and r.content:
                    p.write_bytes(r.content)
            except requests.RequestException:
                pass
        if p.exists():
            try:
                return Image.open(p)
            except Exception:
                pass
    return Image.open(raw / row["path"])


def run(pool: pd.DataFrame, raw: Path, work: Path, base_url: str, api_key: str, model: str,
        workers: int = 4, limit: int | None = None, max_side: int = 1280, fullres: bool = True,
        two_images: bool = True) -> pd.DataFrame:
    cache = work / "vlm_cache.jsonl"
    done = set()
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    client = Client(base_url, api_key, model, max_side=max_side)
    dl = requests.Session()
    fr_dir = work / "fullres"
    fr_dir.mkdir(exist_ok=True)
    lock = threading.Lock()
    rows = pool.to_dict("records")
    todo = [r for r in rows if f"{r['frame_id']}|{model}|{PROMPT_VERSION}" not in done]
    if limit:
        todo = todo[:limit]
    print(f"VLM triage with {model}: {len(todo)} frames to send ({len(rows) - len(todo)} cached)")

    def one(r):
        key = f"{r['frame_id']}|{model}|{PROMPT_VERSION}"
        cur = _fullres(r, raw, fr_dir, dl) if fullres else Image.open(raw / r["path"])
        prev = (Image.open(raw / r["prev_path"]) if two_images and isinstance(r.get("prev_path"), str) and r["prev_path"]
                else None)
        parsed, text, usage, latency = client.ask(cur, prev)
        rec = {"key": key, "frame_id": r["frame_id"], "model": model, "prompt_version": PROMPT_VERSION,
               "parsed": parsed, "raw": text, "usage": usage, "latency_s": round(latency, 2),
               "n_images": 2 if prev is not None else 1, "at": pd.Timestamp.now(tz="UTC").isoformat()}
        with lock:
            with open(cache, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                fu.result()
            except Exception as e:
                failures += 1
                print(f"  failed: {e}")
                if failures >= 5 and failures > n // 2:
                    print("Too many failures; stopping. Check VLM_BASE_URL / VLM_MODEL / VLM_API_KEY.")
                    ex.shutdown(cancel_futures=True)
                    break
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")
    return load_results(work, model)


def load_results(work: Path, model: str | None = None) -> pd.DataFrame:
    cache = work / "vlm_cache.jsonl"
    if not cache.exists():
        return pd.DataFrame()
    recs = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    if model:
        recs = [r for r in recs if r["model"] == model]
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame([{"frame_id": r["frame_id"], "model": r["model"], "prompt_version": r["prompt_version"],
                        "latency_s": r["latency_s"], **r["parsed"]} for r in recs])
    df = df.drop_duplicates(["frame_id", "model", "prompt_version"], keep="last")
    df["vlm_score"] = [vlm_score(p) for p in df.to_dict("records")]
    return df


def settings_from_env() -> tuple[str, str, str]:
    base, key, model = (os.environ.get("VLM_BASE_URL", ""), os.environ.get("VLM_API_KEY", ""),
                        os.environ.get("VLM_MODEL", ""))
    if not base or not model:
        raise SystemExit("Set VLM_BASE_URL and VLM_MODEL (and VLM_API_KEY if needed) in .env first.")
    return base, key, model


def smoke_test(raw: Path, base_url: str, api_key: str, model: str, n: int = 2, max_side: int = 1280,
               two_images: bool = True, compare: bool = False) -> None:
    """Send recent frames and print replies, latency and token use. compare=True times 3 settings on one frame."""
    from .common import load_frames
    f = load_frames(raw).sort_values("image_ts", ascending=False)
    paths = dict(zip(f["frame_id"], f["path"]))
    rows = list(f.head(1 if compare else n).itertuples(index=False))
    settings = ([(1280, True), (896, True), (896, False)] if compare else [(max_side, two_images)])
    for r in rows:
        for side, two in settings:
            client = Client(base_url, api_key, model, max_side=side)
            prev = (Image.open(raw / paths[r.prev_frame_id]) if two and isinstance(r.prev_frame_id, str)
                    and r.prev_frame_id in paths else None)
            parsed, text, usage, latency = client.ask(Image.open(raw / r.path), prev)
            msg = (getattr(client, "last_json", {}).get("choices") or [{}])[0].get("message", {})
            extra = [k for k in msg if k not in ("role", "content")]
            print(f"--- camera {r.camera_id} at {r.image_ts:%H:%M}: {'2 images' if prev is not None else '1 image'} "
                  f"at {side}px -> {latency:.1f}s via {client.mode}"
                  f"{', JSON mode' if client.json_mode and client.mode == 'chat' else ''}")
            print("tokens:", usage, ("| extra fields: " + ", ".join(extra)) if extra else "")
            print("parsed:", parsed["incident"], parsed["type"], parsed["confidence"], "|", parsed["evidence"][:160])


def probe(raw: Path, base_url: str, api_key: str, model: str, max_side: int = 1280) -> None:
    """Why is a call slow? Checks hidden reasoning, a no-thinking switch, and whether the server runs calls in parallel."""
    from .common import load_frames
    f = load_frames(raw).sort_values("image_ts", ascending=False)
    paths = dict(zip(f["frame_id"], f["path"]))
    r = f.iloc[0]
    imgs = []
    if isinstance(r["prev_frame_id"], str) and r["prev_frame_id"] in paths:
        imgs.append(_data_uri(Image.open(raw / paths[r["prev_frame_id"]]), max_side))
    imgs.append(_data_uri(Image.open(raw / r["path"]), max_side))
    text = USER_TWO if len(imgs) == 2 else USER_ONE
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def call(suffix: str = "", extra: dict | None = None):
        content = [{"type": "text", "text": text + suffix}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
        body = {"model": model, "temperature": 0, "max_tokens": 1200, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
        body.update(extra or {})
        t0 = time.time()
        resp = requests.post(base_url.rstrip("/") + "/chat/completions", json=body, headers=headers, timeout=600)
        dt = time.time() - t0
        j = resp.json() if resp.ok else {}
        msg = (j.get("choices") or [{}])[0].get("message", {}) or {}
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        return {"seconds": round(dt, 1), "http": resp.status_code, "reasoning_chars": len(str(reasoning)),
                "completion_tokens": (j.get("usage") or {}).get("completion_tokens"),
                "answer": parse_reply(msg.get("content") or "")["incident"]}

    print("1 baseline          :", call())
    print("2 with /no_think    :", call("\n/no_think"))
    print("3 with think: false :", call(extra={"think": False}))
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda _: call(), range(4)))
    print(f"4 four calls at once: {time.time() - t0:.1f}s wall, each {[x['seconds'] for x in res]}s")
