"""Blind labelling page: python -m sgtriage review, then open http://localhost:8008

The page only gets images, camera id and time. It never sees scores, which method picked the
frame, the incident feed or the VLM's answer, so labels cannot be nudged by the system being
evaluated. Labels are saved on every keypress to data/work/labels.csv; closing and reopening
resumes where you stopped.
"""
from __future__ import annotations

import json
import mimetypes
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .common import append_csv

LABEL_FIELDS = ["review_id", "label", "kind", "ms", "at"]
PAGE = Path(__file__).with_name("review.html")


def serve(raw: Path, work: Path, port: int = 8008, open_browser: bool = True) -> None:
    items_path = work / "review_items.json"
    if not items_path.exists():
        raise SystemExit("No review items. Run `python -m sgtriage pool` first.")
    items = json.loads(items_path.read_text(encoding="utf-8"))
    labels_path = work / "labels.csv"
    raw_root = raw.resolve()

    def current_labels() -> dict:
        out = {}
        if labels_path.exists():
            import csv
            with open(labels_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    out[r["review_id"]] = {"label": r["label"], "kind": r.get("kind", "")}
        return out

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/items":
                return self._send(200, json.dumps({"items": items, "labels": current_labels()}).encode(),
                                  "application/json")
            if path.startswith("/img/"):
                p = (raw_root / path[len("/img/"):]).resolve()
                if raw_root not in p.parents or not p.is_file():  # no path traversal outside data/raw
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/jpeg")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            try:
                j = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, b"bad json", "text/plain")
            if j.get("label") not in ("yes", "no", "unsure"):
                return self._send(400, b"bad label", "text/plain")
            append_csv(labels_path, {"review_id": j.get("review_id", ""), "label": j["label"],
                                     "kind": j.get("kind", ""), "ms": j.get("ms", ""),
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, LABEL_FIELDS)
            return self._send(200, b'{"ok":true}', "application/json")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://localhost:{port}"
    print(f"Labelling {len(items)} frames at {url}  (Ctrl+C to stop; progress is saved as you go)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
