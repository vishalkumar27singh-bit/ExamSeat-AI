"""
prepare_dataset.py - turn the project's ground-truth JSON files into a YOLO detection dataset.

Classes (detector #1: where are the tables?):
    0 table          the whole table body: header row A-F + all student rows (one box per table)
    1 room_heading   the printed "Room No: ATR-xxx" text  (optional; only if your JSON has "headings")

Why only these?  Reading the digits stays with the OCR; the row/column logic stays with seat_locator.py.
A detector is only useful for finding REGIONS (tables) so that the grid logic can run on a clean crop.

JSON keys used (see ground_truth/photo_001_clear.json):
    "tables":   [{"room": "ATR-205", "quad": [[x,y],[x,y],[x,y],[x,y]]}]       -> class 0
    "headings": [{"bbox": [x0,y0,x1,y1]}]                                     -> class 1   (optional)
    "group":    "photo_001"  (optional) photos that show the SAME paper must share a group

LEAKAGE RULE: train/val are split by GROUP, never by file.  Synthetic copies (blurry, tilted...) of one
photo are the same sheet; if one copy is in train and another in val the validation score is fake.
If the JSON has no "group", the file name up to the last "_" is the group (photo_001_clear -> photo_001).

    python ml/prepare_dataset.py                       # ground_truth/*.json -> datasets/tables_yolo
    python ml/prepare_dataset.py --val-fraction 0.2 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
NAMES = ["table", "room_heading"]


def group_of(gt: dict, path: str) -> str:
    if gt.get("group"):
        return str(gt["group"])
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def yolo_line(cls: int, x0, y0, x1, y1, W, H) -> str:
    x0, x1 = max(0.0, min(x0, x1)), min(float(W), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(H), max(y0, y1))
    return f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"


def labels_for(gt: dict, W: int, H: int):
    lines = []
    for t in gt.get("tables", []):
        q = np.array(t["quad"], dtype=float)
        lines.append(yolo_line(0, q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max(), W, H))
    for h in gt.get("headings", []):
        lines.append(yolo_line(1, *h["bbox"], W, H))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt", nargs="*", help="ground-truth JSON files (default ground_truth/*.json)")
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "tables_yolo"))
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-synthetic", action="store_true",
                    help="also use synthetic degradations (they stay in the same split as their source photo)")
    ap.add_argument("--allow-single-group", action="store_true",
                    help="SMOKE TEST ONLY: with one group, use it for both train and val (the score is meaningless)")
    a = ap.parse_args(argv)
    files = a.gt or sorted(p for p in glob.glob(os.path.join(ROOT, "ground_truth", "*.json"))
                           if not os.path.basename(p).startswith("_"))
    items = []
    for f in files:
        gt = json.load(open(f))
        if gt.get("synthetic") and not a.include_synthetic:
            continue
        if not gt.get("tables") and not gt.get("headings"):
            continue
        img_path = gt["image"] if os.path.isabs(gt["image"]) else os.path.join(ROOT, gt["image"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"skip {f}: cannot read {img_path}")
            continue
        items.append((group_of(gt, f), img_path, labels_for(gt, img.shape[1], img.shape[0]), gt))
    groups = defaultdict(list)
    for it in items:
        groups[it[0]].append(it)
    print(f"annotated images: {len(items)}   independent photos (groups): {len(groups)}   "
          f"table boxes: {sum(sum(l.startswith('0 ') for l in it[2]) for it in items)}")
    if not items:
        print("Nothing annotated yet. See ml/ANNOTATION_GUIDE.md")
        return 1
    names = sorted(groups)
    random.Random(a.seed).shuffle(names)
    if len(names) < 2:
        if not a.allow_single_group:
            print("Only ONE independent photo: a train/validation split is impossible without leakage.\n"
                  "Annotate more photos (ml/ANNOTATION_GUIDE.md) or pass --allow-single-group for a pipeline smoke test.")
            return 2
        val_g = train_g = set(names)
        print("WARNING: single group used for train AND val - any metric from this dataset is meaningless.")
    else:
        n_val = max(1, int(round(a.val_fraction * len(names))))
        val_g, train_g = set(names[:n_val]), set(names[n_val:])
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    for split, gs in (("train", train_g), ("val", val_g)):
        os.makedirs(os.path.join(a.out, "images", split))
        os.makedirs(os.path.join(a.out, "labels", split))
        for g in gs:
            for _, img_path, lines, _ in groups[g]:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                shutil.copy(img_path, os.path.join(a.out, "images", split, os.path.basename(img_path)))
                open(os.path.join(a.out, "labels", split, stem + ".txt"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES):
            fh.write(f"  {i}: {n}\n")
    n_tr = sum(len(groups[g]) for g in train_g)
    n_va = sum(len(groups[g]) for g in val_g)
    print(f"wrote {a.out}   train images {n_tr} ({len(train_g)} photos) | val images {n_va} ({len(val_g)} photos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
