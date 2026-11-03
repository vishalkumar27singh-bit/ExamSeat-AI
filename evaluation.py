"""
evaluation.py
=============
Measure how good the system really is, on photos that have a manually verified
ground-truth file.

Ground-truth file format  (one JSON per photo, see README "Collecting ground truth")
------------------------------------------------------------------------------------
{
  "image": "samples/photo_001_clear.jpeg",     # path relative to the project folder
  "condition": "clear",                        # clear | blurry | low_light | tilted | low_res | ...
  "synthetic": false,                          # true for artificially degraded copies
  "tables": [ {"room": "ATR-205", "quad": [[x,y],[x,y],[x,y],[x,y]]} ],   # optional, rough TL,TR,BR,BL
  "numbers": [ {"reg_no": "<registration-number>", "room": "<room>", "column": "A", "row": 1,
                "bbox": [x0, y0, x1, y1]} ]    # bbox optional (original photo pixels)
}

Metrics
-------
* OCR recall      : GT numbers that OCR read exactly (as a full 10-digit token)
* OCR precision   : of the 10-digit tokens inside annotated tables, how many are in the GT
* Exact search    : searching a GT number returns FOUND at all
* Room / column / row accuracy, end-to-end (a missed number counts as wrong) and when-found
* Confident errors: results labelled HIGH confidence that are wrong  <- the dangerous kind
* Localisation    : IoU>=0.5 with GT bbox when one exists; ALWAYS also "re-read OK" (the pixels
                    inside the highlighted polygon re-read as the requested number)
* False-found rate: searching numbers that are NOT on the chart must never return FOUND
Everything is broken down per photo condition.

Run:  python evaluation.py                      (all ground_truth/*.json)
      python evaluation.py ground_truth/photo_001_clear.json --backend paddle
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from image_processing import load_image, poly_iou
from ocr_engine import OCREngine, extract_tokens
from pipeline import add_enhanced_pass, analyze, search

ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _inside(quads: List[dict], pt) -> Optional[str]:
    for q in quads:
        if cv2.pointPolygonTest(np.array(q["quad"], dtype=np.float32).reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), False) >= 0:
            return q["room"]
    return None


def _negative_of(n: str, taken: set) -> str:
    """A 10-digit number guaranteed not to be on the chart (change the last digit)."""
    for d in "0123456789":
        cand = n[:-1] + d
        if cand != n and cand not in taken:
            return cand
    return n[:-2] + "99"


def evaluate_photo(gt_path: str, engine: OCREngine) -> Dict[str, object]:
    gt = json.load(open(gt_path))
    img = load_image(_resolve(gt["image"]))
    t0 = time.time()
    an = analyze(img, engine)
    t_ocr = time.time() - t0
    gt_nums = {n["reg_no"]: n for n in gt["numbers"]}

    # ---------------- OCR precision / recall --------------------------------
    toks = [t for t in extract_tokens(an.detections) if t.kind == "full"]
    exact_read = {t.digits for t in toks if not t.substituted}
    recall = len([n for n in gt_nums if n in exact_read]) / max(len(gt_nums), 1)
    precision = None
    quads = gt.get("tables", [])
    if quads:
        # map GT quads into the working image (analysis may have rescaled / rotated)
        wq = [dict(room=q["room"], quad=an.to_work.apply(np.array(q["quad"], dtype=np.float64)).tolist()) for q in quads]
        in_tab = [t for t in toks if _inside(wq, t.det.center)]
        precision = (sum(t.digits in gt_nums and not t.substituted for t in in_tab) / len(in_tab)) if in_tab else None

    # ---------------- per-query evaluation ----------------------------------
    rows = []
    queries = list(gt_nums.items())
    pending = []
    for reg, n in queries:
        t1 = time.time()
        r = search(an, engine, reg, retry_enhanced=False)
        rows.append(_row(gt, an, reg, n, r, time.time() - t1, "primary"))
        if r.status != "FOUND" or rows[-1]["room"] is None:
            pending.append(len(rows) - 1)
    if pending:  # one enhanced retry for everything that was missed
        add_enhanced_pass(an, engine)
        for i in pending:
            reg, n = rows[i]["reg_no"], gt_nums[rows[i]["reg_no"]]
            r = search(an, engine, reg, retry_enhanced=False)
            if r.status == "FOUND" and (rows[i]["status"] != "FOUND" or r.matches and r.matches[0].room is not None):
                rows[i] = _row(gt, an, reg, n, r, 0.0, "enhanced_retry")

    # ---------------- negative queries --------------------------------------
    taken = set(gt_nums) | {t.digits for t in toks}
    false_found = 0
    n_neg = 0
    for reg in list(gt_nums):
        neg = _negative_of(reg, taken)
        taken.add(neg)
        r = search(an, engine, neg, retry_enhanced=False)
        n_neg += 1
        false_found += r.status == "FOUND"

    return dict(gt=os.path.basename(gt_path), image=gt["image"], condition=gt.get("condition", "unknown"),
                synthetic=bool(gt.get("synthetic", False)), size=f"{img.shape[1]}x{img.shape[0]}",
                ocr_seconds=round(t_ocr, 2), n_gt=len(gt_nums), n_detections=len(an.detections),
                ocr_recall=recall, ocr_precision=precision, n_negative=n_neg, false_found=false_found,
                rows=rows)


def _row(gt: dict, an, reg: str, n: dict, r, secs: float, stage: str) -> dict:
    row = dict(reg_no=reg, gt_room=n["room"], gt_col=n["column"], gt_row=n["row"], status=r.status, stage=stage,
               room=None, col=None, row=None, level=None, confidence=None, verified_seat=None,
               reread_ok=None, iou=None, secs=round(secs, 3))
    if r.status == "FOUND":
        # if the number appears several times take the most confident match
        m = max(r.matches, key=lambda x: x.confidence)
        row.update(room=m.room, col=m.column, row=m.row, level=m.level, confidence=m.confidence,
                   verified_seat=m.seat.verified, reread_ok=m.verify.ok)
        if "bbox" in n or "poly" in n:
            g = np.array(n["poly"] if "poly" in n else
                         [[n["bbox"][0], n["bbox"][1]], [n["bbox"][2], n["bbox"][1]],
                          [n["bbox"][2], n["bbox"][3]], [n["bbox"][0], n["bbox"][3]]], dtype=np.float32)
            row["iou"] = round(poly_iou(m.poly_original, g), 3)
    return row


# ==========================================================================
def summarise(results: List[dict]) -> pd.DataFrame:
    recs = []
    for res in results:
        df = pd.DataFrame(res["rows"])
        found = df[df.status == "FOUND"]
        seat_ok = (found.room == found.gt_room) & (found.col == found.gt_col) & (found.row == found.gt_row)
        room_ok, col_ok, row_ok = found.room == found.gt_room, found.col == found.gt_col, found.row == found.gt_row
        high = found[found.level == "HIGH"]
        high_ok = ((high.room == high.gt_room) & (high.col == high.gt_col) & (high.row == high.gt_row))
        n = len(df)
        iou = found.iou.dropna()
        recs.append(dict(
            photo=res["gt"], condition=res["condition"], synthetic=res["synthetic"], size=res["size"], queries=n,
            ocr_recall=round(res["ocr_recall"], 3),
            ocr_precision=None if res["ocr_precision"] is None else round(res["ocr_precision"], 3),
            exact_found=round(len(found) / n, 3),
            room_acc_e2e=round(room_ok.sum() / n, 3), col_acc_e2e=round(col_ok.sum() / n, 3),
            row_acc_e2e=round(row_ok.sum() / n, 3), seat_acc_e2e=round(seat_ok.sum() / n, 3),
            seat_acc_when_found=round(seat_ok.mean(), 3) if len(found) else None,
            seat_verified_rate=round(found.verified_seat.astype(bool).mean(), 3) if len(found) else None,
            high_conf_results=len(high), confident_errors=int((~high_ok).sum()),
            wrong_but_marked_verified=int(((~seat_ok) & found.verified_seat.astype(bool)).sum()),
            reread_ok_rate=round(found.reread_ok.astype(bool).mean(), 3) if len(found) else None,
            iou_ge_0_5=(round((iou >= 0.5).mean(), 3) if len(iou) else None),
            false_found=f"{res['false_found']}/{res['n_negative']}", ocr_seconds=res["ocr_seconds"]))
    return pd.DataFrame(recs)


def overall(results: List[dict]) -> pd.DataFrame:
    """Weighted by number of queries, grouped by condition."""
    allrows = []
    for res in results:
        for r in res["rows"]:
            allrows.append(dict(r, condition=res["condition"], synthetic=res["synthetic"],
                                false_found=res["false_found"], n_neg=res["n_negative"]))
    df = pd.DataFrame(allrows)
    out = []
    for cond, g in df.groupby("condition"):
        f = g[g.status == "FOUND"]
        ok = (f.room == f.gt_room) & (f.col == f.gt_col) & (f.row == f.gt_row)
        out.append(dict(condition=cond, queries=len(g), found=f"{len(f)}/{len(g)}",
                        seat_correct_e2e=f"{int(ok.sum())}/{len(g)} ({ok.sum() / len(g):.1%})",
                        confident_errors=int(((f.level == 'HIGH') & ~ok).sum())))
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt", nargs="*", help="ground-truth json files (default: ground_truth/*.json)")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    a = ap.parse_args(argv)
    files = a.gt or sorted(p for p in glob.glob(os.path.join(ROOT, "ground_truth", "*.json"))
                           if not os.path.basename(p).startswith("_"))
    if not files:
        print("No ground-truth files found.")
        return 1
    os.makedirs(a.out, exist_ok=True)
    eng = OCREngine(a.backend)
    print(f"OCR backend used for this evaluation: {eng.name}" + (f"  ({eng.fallback_reason})" if eng.fallback_reason else ""))
    results = []
    for f in files:
        print(f"\n=== {os.path.basename(f)}")
        res = evaluate_photo(f, eng)
        print(f"   {res['n_gt']} GT numbers, OCR recall {res['ocr_recall']:.1%}, "
              f"found {sum(r['status'] == 'FOUND' for r in res['rows'])}/{res['n_gt']}, "
              f"negatives falsely found {res['false_found']}/{res['n_negative']}")
        results.append(res)
    summ = summarise(results)
    cond = overall(results)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n================ PER PHOTO ================")
    print(summ.to_string(index=False))
    print("\n================ PER CONDITION ================")
    print(cond.to_string(index=False))
    # ---- save ----
    pd.DataFrame([dict(r, photo=res["gt"], condition=res["condition"]) for res in results for r in res["rows"]]) \
        .to_csv(os.path.join(a.out, "eval_per_query.csv"), index=False)
    summ.to_csv(os.path.join(a.out, "eval_per_photo.csv"), index=False)
    with open(os.path.join(a.out, "eval_summary.md"), "w") as fh:
        fh.write(f"# ExamSeat AI evaluation\n\nOCR backend: **{eng.name}**\n\n")
        fh.write("## Per condition\n\n" + cond.to_markdown(index=False) + "\n\n## Per photo\n\n" + summ.to_markdown(index=False) + "\n")
    print("\nsaved:", os.path.join(a.out, "eval_per_query.csv"), "| eval_per_photo.csv | eval_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
