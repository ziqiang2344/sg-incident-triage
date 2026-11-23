"""Fill data/work/labels.csv from synthetic ground truth (3% label noise, some night frames 'unsure')."""
import csv
import random
import sys
from pathlib import Path

work, truth_path = Path(sys.argv[1]), Path(sys.argv[2])
random.seed(5)
truth = {r["frame_id"]: r for r in csv.DictReader(open(truth_path))}
pool = list(csv.DictReader(open(work / "pool.csv")))
with open(work / "labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["review_id", "label", "kind", "ms", "at"])
    w.writeheader()
    for r in pool:
        t = truth[r["frame_id"]]
        lab = "yes" if t["visible"] == "1" else "no"
        if random.random() < 0.03:
            lab = "no" if lab == "yes" else "yes"
        if t["night"] == "1" and random.random() < 0.08:
            lab = "unsure"
        w.writerow({"review_id": r["review_id"], "label": lab, "kind": "stopped_vehicle" if lab == "yes" else "",
                    "ms": random.randint(1800, 6000), "at": "2026-11-23T10:00:00"})
print(f"labelled {len(pool)} frames")
