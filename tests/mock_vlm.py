"""OpenAI-compatible stand-in for a VLM, for offline tests.

Looks for the synthetic 'stopped vehicle' (an orange block) in the last image; 10% of answers are flipped.
Rejects response_format on purpose so the client's fallback path is exercised.
python tests/mock_vlm.py 8790
"""
import base64
import io
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
random.seed(3)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "response_format" in body:
            msg = b'{"error": {"message": "response_format is not supported"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        parts = body["messages"][-1]["content"]
        img = [p for p in parts if p["type"] == "image_url"][-1]["image_url"]["url"]
        im = np.asarray(Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[1]))).convert("RGB")).astype(int)
        orange = ((im[..., 0] > 200) & (im[..., 1] > 100) & (im[..., 1] < 180) & (im[..., 2] < 60)).mean()
        dark = im.mean() < 45
        yes = orange > 0.002
        if random.random() < 0.10:
            yes = not yes
        if dark and random.random() < 0.3:
            ans = {"incident": "unsure", "type": "none", "confidence": 0.4, "where": "", "evidence": "Too dark to tell.",
                   "image_quality": "poor"}
        else:
            ans = {"incident": "yes" if yes else "no", "type": "stopped_vehicle" if yes else "none",
                   "confidence": 0.8, "where": "right lane" if yes else "",
                   "evidence": "Orange vehicle stationary in the right lane." if yes else "Traffic moving normally.",
                   "image_quality": "good"}
        out = json.dumps({"choices": [{"message": {"content": json.dumps(ans)}}],
                          "usage": {"prompt_tokens": 600, "completion_tokens": 60}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
