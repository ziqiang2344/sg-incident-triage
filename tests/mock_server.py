"""Local stand-in for data.gov.sg and LTA DataMall, used to test collect.py offline.

python tests/mock_server.py 8765
"""
import io
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

SGT = timezone(timedelta(hours=8))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
KEY = "testkey"
CAMS = {  # camera_id: (lat, lon)
    "1001": (1.29531, 103.87106), "1002": (1.31954, 103.87921), "2701": (1.44743, 103.77138),
    "4703": (1.32814, 103.86275), "6711": (1.28035, 103.83053), "9706": (1.31015, 103.74581),
}
N_LINKS = 1234


def frame_no():
    return int(time.time() // 3)  # every camera "updates" every 3 s


def jpeg(text, w=1920, h=1080):
    im = Image.new("RGB", (w, h), (40 + hash(text) % 120, 80, 120))
    ImageDraw.Draw(im).text((40, 40), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def links():
    out = []
    for i in range(N_LINKS):
        if i < 60:  # first 60 links sit next to cameras
            lat, lon = list(CAMS.values())[i % len(CAMS)]
            lat += 0.001 * ((i % 5) - 2)
        else:
            lat, lon = 1.35 + 0.0001 * i, 103.70 + 0.0002 * i
        out.append({"LinkID": str(100000 + i), "RoadName": f"ROAD {i % 40}", "RoadCategory": "1",
                    "SpeedBand": 1 + (i + frame_no()) % 8, "MinimumSpeed": "10", "MaximumSpeed": "19",
                    "StartLat": str(lat), "StartLon": str(lon), "EndLat": str(lat + 0.0005), "EndLon": str(lon + 0.0005)})
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/v1/transport/traffic-images":
            n = frame_no()
            ts = datetime.now(SGT).isoformat(timespec="seconds")
            cams = [{"timestamp": ts, "image": f"http://127.0.0.1:{PORT}/img/{cid}/{n}.jpg",
                     "location": {"latitude": lat, "longitude": lon}, "camera_id": cid,
                     "image_metadata": {"height": 1080, "width": 1920, "md5": f"{cid}-{n}"}}
                    for cid, (lat, lon) in CAMS.items()]
            return self.send(200, {"items": [{"timestamp": ts, "cameras": cams}], "api_info": {"status": "healthy"}})
        if u.path.startswith("/img/"):
            return self.send(200, jpeg(u.path), "image/jpeg")
        if u.path.startswith("/ltaodataservice/"):
            if self.headers.get("AccountKey") != KEY:
                return self.send(401, {"fault": "unauthorised"})
            name = u.path[len("/ltaodataservice/"):]
            skip = int(q.get("$skip", ["0"])[0])
            if name == "TrafficIncidents":
                return self.send(200, {"value": [
                    {"Type": "Accident", "Latitude": 1.2955, "Longitude": 103.8714,
                     "Message": "(22/11)23:41 Accident on ECP (towards City) after Fort Rd Exit. Avoid lane 1."},
                    {"Type": "Vehicle breakdown", "Latitude": 1.40, "Longitude": 103.90,
                     "Message": "(22/11)23:30 Vehicle breakdown on TPE (towards SLE) after Punggol Rd Exit."},
                    {"Type": "Roadwork", "Latitude": 1.33, "Longitude": 103.86,
                     "Message": "(22/11)22:00 Roadworks on PIE (towards Tuas) at Kallang Way."}]})
            if name == "v3/TrafficSpeedBands":
                return self.send(200, {"value": links()[skip:skip + 500]})
            if name == "FaultyTrafficLights":
                return self.send(200, {"value": [{"AlarmID": "GL703034136", "NodeID": "3034136", "Type": 4,
                                                  "StartDate": "2026-11-22 23:00:00.0", "EndDate": "",
                                                  "Message": "(22/11)23:00 Traffic light blackout at Junction X"}]})
            if name == "Traffic-Imagesv2":
                n = frame_no()
                stamp = datetime.fromtimestamp(n * 3, SGT).strftime("%Y%m%d%H%M%S")
                extra = {"5794": (1.3533, 103.9184), "6703": (1.3283, 103.8410), "7793": (1.3726, 103.8999)}
                return self.send(200, {"value": [{"CameraID": cid, "Latitude": lat, "Longitude": lon,
                                                  "ImageLink": f"http://127.0.0.1:{PORT}/img/{cid}/{cid}_0904_{stamp}_ab12cd.jpg?X-Amz-Sig=1"}
                                                 for cid, (lat, lon) in {**CAMS, **extra}.items()]})
            return self.send(404, {"error": "not found"})
        return self.send(404, {"error": "not found"})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
