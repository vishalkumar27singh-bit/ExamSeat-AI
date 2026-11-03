"""
eval_detector.py - precision / recall of a trained detector on the VALIDATION images only.

    python ml/eval_detector.py --weights runs/tables/weights/best.pt --data datasets/tables_yolo
A box is correct when IoU >= 0.5 with an unmatched ground-truth box of the same class.
Prints the counts it used so the numbers can be checked; with fewer than 5 validation photos it says so.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import NAMES, TableDetector


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def read_labels(path: str, W: int, H: int):
    out = []
    if os.path.exists(path):
        for ln in open(path).read().split("\n"):
            if ln.strip():
                c, cx, cy, w, h = ln.split()
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                out.append((NAMES[int(c)], [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "tables_yolo"))
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args(argv)
    det = TableDetector(a.weights)
    imgs = sorted(glob.glob(os.path.join(a.data, "images", "val", "*")))
    tp = fp = fn = 0
    for p in imgs:
        im = cv2.imread(p)
        H, W = im.shape[:2]
        gt = read_labels(os.path.join(a.data, "labels", "val", os.path.splitext(os.path.basename(p))[0] + ".txt"), W, H)
        used = set()
        for d in sorted(det.detect(im), key=lambda d: -d["score"]):
            hit = next((i for i, (c, b) in enumerate(gt) if i not in used and c == d["cls"] and iou(b, d["box"]) >= a.iou), None)
            if hit is None:
                fp += 1
            else:
                used.add(hit)
                tp += 1
        fn += len(gt) - len(used)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"validation images: {len(imgs)} | true positives {tp}, false positives {fp}, missed {fn}")
    print(f"precision {prec:.3f}  recall {rec:.3f}  (IoU >= {a.iou})")
    if len(imgs) < 5:
        print("WARNING: fewer than 5 validation photos - these numbers say very little.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
