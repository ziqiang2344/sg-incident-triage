"""Static LTA layers: traffic-signal aspects -> signalised junctions.

LTA DataMall publishes every traffic-signal aspect (each signal head) as a point shapefile in
SVY21 (EPSG:3414). One junction has many aspects, so nearby aspects are clustered into one
signalised location. Pure Python: no GDAL, pyproj or pyshp needed.
"""
from __future__ import annotations

import io
import math
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TRAFFIC_LIGHT_URL = "https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/TrafficLight_Sep2026.zip"

# SVY21 (Singapore) transverse Mercator parameters, from the shapefile's .prj
_A = 6378137.0
_F = 1 / 298.257223563
_O_LAT, _O_LON = 1.366666666666667, 103.8333333333333
_O_N, _O_E, _K = 38744.572, 28001.642, 1.0


def _m(lat_rad):
    e2 = 2 * _F - _F * _F
    e4, e6 = e2 * e2, e2 * e2 * e2
    a0 = 1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256
    a2 = 3 / 8 * (e2 + e4 / 4 + 15 * e6 / 128)
    a4 = 15 / 256 * (e4 + 3 * e6 / 4)
    a6 = 35 * e6 / 3072
    return _A * (a0 * lat_rad - a2 * np.sin(2 * lat_rad) + a4 * np.sin(4 * lat_rad) - a6 * np.sin(6 * lat_rad))


def svy21_to_wgs84(n, e):
    """SVY21 northing/easting (metres) -> WGS84 lat/lon (degrees). Vectorised."""
    n, e = np.asarray(n, dtype=float), np.asarray(e, dtype=float)
    b = _A * (1 - _F)
    e2 = 2 * _F - _F * _F
    m_prime = _m(math.radians(_O_LAT)) + (n - _O_N) / _K
    nn = (_A - b) / (_A + b)
    n2, n3, n4 = nn ** 2, nn ** 3, nn ** 4
    g = _A * (1 - nn) * (1 - n2) * (1 + 9 * n2 / 4 + 225 * n4 / 64) * (math.pi / 180)
    sigma = m_prime * math.pi / (180 * g)
    lat_p = (sigma + (3 * nn / 2 - 27 * n3 / 32) * np.sin(2 * sigma) + (21 * n2 / 16 - 55 * n4 / 32) * np.sin(4 * sigma)
             + (151 * n3 / 96) * np.sin(6 * sigma) + (1097 * n4 / 512) * np.sin(8 * sigma))
    s2 = np.sin(lat_p) ** 2
    rho = _A * (1 - e2) / (1 - e2 * s2) ** 1.5
    v = _A / np.sqrt(1 - e2 * s2)
    psi = v / rho
    t = np.tan(lat_p)
    t2, t4, t6 = t ** 2, t ** 4, t ** 6
    e_p = e - _O_E
    x = e_p / (_K * v)
    lf = t / (_K * rho)
    lat = (lat_p - lf * (e_p * x / 2)
           + lf * (e_p * x ** 3 / 24) * (-4 * psi ** 2 + 9 * psi * (1 - t2) + 12 * t2)
           - lf * (e_p * x ** 5 / 720) * (8 * psi ** 4 * (11 - 24 * t2) - 12 * psi ** 3 * (21 - 71 * t2)
                                          + 15 * psi ** 2 * (15 - 98 * t2 + 15 * t4) + 180 * psi * (5 * t2 - 3 * t4) + 360 * t4)
           + lf * (e_p * x ** 7 / 40320) * (1385 - 3633 * t2 + 4095 * t4 + 1575 * t6))
    sec = 1 / np.cos(lat_p)
    lon = (math.radians(_O_LON) + x * sec - (x ** 3 * sec / 6) * (psi + 2 * t2)
           + (x ** 5 * sec / 120) * (-4 * psi ** 3 * (1 - 6 * t2) + psi ** 2 * (9 - 68 * t2) + 72 * psi * t2 + 24 * t4)
           - (x ** 7 * sec / 5040) * (61 + 662 * t2 + 1320 * t4 + 720 * t6))
    return np.degrees(lat), np.degrees(lon)


def read_point_shapefile(data: bytes) -> np.ndarray:
    """(n, 2) array of x, y from a point (.shp) file's bytes."""
    pts = []
    off = 100  # file header
    while off + 8 <= len(data):
        _, content_words = struct.unpack(">2i", data[off:off + 8])
        off += 8
        shape_type = struct.unpack("<i", data[off:off + 4])[0]
        if shape_type == 1:
            pts.append(struct.unpack("<2d", data[off + 4:off + 20]))
        off += content_words * 2
    return np.array(pts, dtype=float)


def cluster_points(x: np.ndarray, y: np.ndarray, eps_m: float = 40.0) -> np.ndarray:
    """Single-linkage clusters of points closer than eps_m (grid + union-find, O(n))."""
    n = len(x)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cells: dict[tuple[int, int], list[int]] = {}
    gx, gy = np.floor(x / eps_m).astype(int), np.floor(y / eps_m).astype(int)
    for i in range(n):
        cells.setdefault((gx[i], gy[i]), []).append(i)
    for i in range(n):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((gx[i] + dx, gy[i] + dy), ()):
                    if j > i and (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 <= eps_m ** 2:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def junctions_from_shp_bytes(shp: bytes, eps_m: float = 40.0) -> pd.DataFrame:
    xy = read_point_shapefile(shp)
    labels = cluster_points(xy[:, 0], xy[:, 1], eps_m)
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cluster": labels})
    j = df.groupby("cluster").agg(x=("x", "mean"), y=("y", "mean"), n_aspects=("x", "size")).reset_index(drop=True)
    j["lat"], j["lon"] = svy21_to_wgs84(j["y"].to_numpy(), j["x"].to_numpy())
    j.insert(0, "junction_id", [f"J{i:05d}" for i in range(len(j))])
    return j[["junction_id", "lat", "lon", "n_aspects"]]


def build_junctions(static_dir: Path, url: str = TRAFFIC_LIGHT_URL, zip_path: Path | None = None) -> pd.DataFrame:
    """Download (once) the traffic-light layer and write data/static/junctions.csv."""
    static_dir.mkdir(parents=True, exist_ok=True)
    zp = zip_path or static_dir / Path(url).name
    if not zp.exists():
        import requests
        print(f"Downloading {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zp.write_bytes(r.content)
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".shp"))
        shp = z.read(name)
    j = junctions_from_shp_bytes(shp)
    out = static_dir / "junctions.csv"
    j.to_csv(out, index=False)
    print(f"{len(j):,} signalised locations from {j['n_aspects'].sum():,} signal aspects -> {out}")
    return j


def load_junctions(static_dir: Path) -> pd.DataFrame | None:
    p = static_dir / "junctions.csv"
    return pd.read_csv(p) if p.exists() else None
