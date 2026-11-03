"""
diagnose_coordinates.py  -  STAGE 1: find out WHY a box lands on the wrong number
=================================================================================
Run on your Mac (inside your Anaconda environment):

    python diagnose_coordinates.py "<photo>" "<registration-number>"

It checks the six things you listed and prints a verdict:
  1. image file / EXIF orientation / size           (same image everywhere?)
  2. which coordinate system PaddleOCR answers in   (original or pre-processed image?)
  3. width/height mix-ups                           (shapes compared as (H,W) and (W,H))
  4. text <-> polygon alignment                     (list lengths, which list the polygon came from)
  5. visual proof: crops of every candidate polygon, re-read by OCR
  6. what the SAFE configuration returns            (pre-processing switched off)
Output pictures go to ./diagnostics/
"""
from __future__ import annotations

import hashlib
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps


def poly_of_box(b):
    x0, y0, x1, y1 = [float(v) for v in b]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def crop(img, poly, margin=0.6):
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    pad = max(6, margin * (y1 - y0))
    H, W = img.shape[:2]
    return img[max(0, int(y0 - pad)):min(H, int(y1 + pad)), max(0, int(x0 - pad)):min(W, int(x1 + pad))].copy()


def reread(ocr_safe, img_crop):
    """Read a crop on its own; returns the text lines found."""
    if img_crop.size == 0:
        return []
    c = cv2.copyMakeBorder(img_crop, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    res = ocr_safe.predict(c)
    return list(res[0]["rec_texts"]) if res else []


def run_config(name, kwargs, path, query):
    from paddleocr import PaddleOCR
    print(f"\n=== CONFIG {name}: PaddleOCR({', '.join(f'{k}={v}' for k, v in kwargs.items())})")
    ocr = PaddleOCR(**kwargs)
    res = ocr.predict(path)
    r = res[0]
    out = {}
    ms = r["model_settings"] if "model_settings" in r else {}
    print("   model_settings:", dict(ms))
    pre = r["doc_preprocessor_res"] if "doc_preprocessor_res" in r else None
    oimg = np.asarray(pre["output_img"]) if pre is not None and "output_img" in pre else None
    print("   image PaddleOCR actually processed (H,W,C):", None if oimg is None else oimg.shape)
    n_t, n_p, n_s = len(r["rec_texts"]), len(r["rec_polys"]), len(r["rec_scores"])
    n_b = len(r["rec_boxes"]) if "rec_boxes" in r else -1
    n_d = len(r["dt_polys"]) if "dt_polys" in r else -1
    print(f"   lengths: rec_texts={n_t} rec_scores={n_s} rec_polys={n_p} rec_boxes={n_b} dt_polys={n_d}")
    out.update(oimg=oimg, ocr=ocr, aligned=(n_t == n_p == n_s), dt_equal=(n_d == n_t), settings=dict(ms))
    hits = [i for i, t in enumerate(r["rec_texts"]) if "".join(t.split()) == query]
    print("   indices of the query in rec_texts:", hits)
    out["cands"] = []
    for i in hits:
        out["cands"].append((f"rec_polys[{i}]", np.asarray(r["rec_polys"][i])))
        if n_b > i:
            out["cands"].append((f"rec_boxes[{i}]", poly_of_box(r["rec_boxes"][i])))
        if n_d > i:
            out["cands"].append((f"dt_polys[{i}] (separate, unfiltered list!)", np.asarray(r["dt_polys"][i])))
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    path, query = argv[0], "".join(argv[1].split())
    os.makedirs("diagnostics", exist_ok=True)
    print("=== 1. THE IMAGE FILE")
    print("   absolute path :", os.path.abspath(path), "| exists:", os.path.exists(path))
    with open(path, "rb") as f:
        print("   sha1          :", hashlib.sha1(f.read()).hexdigest()[:16])
    pil = Image.open(path)
    print("   PIL size (W,H):", pil.size, "| EXIF orientation tag:", pil.getexif().get(274))
    print("   after exif_transpose (W,H):", ImageOps.exif_transpose(pil).size)
    cvimg = cv2.imread(path)
    print("   cv2.imread shape (H,W,C):", cvimg.shape, "  <- width and height are shape[1], shape[0]")
    H, W = cvimg.shape[:2]

    cfg_default = run_config("A (plain default - what most tutorials run)", {}, path, query)
    cfg_safe = run_config("B (SAFE: no unwarping / orientation / textline-orientation)",
                          dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                               use_textline_orientation=False), path, query)
    safe = cfg_safe["ocr"]

    print("\n=== 5. VISUAL PROOF  (each candidate polygon cropped from the ORIGINAL image and re-read by OCR)")
    verdicts = {}
    for tag, cfg in (("A", cfg_default), ("B", cfg_safe)):
        for name, poly in cfg["cands"]:
            tgt = [("original image", cvimg)]
            if cfg["oimg"] is not None and cfg["oimg"].shape[:2] != (H, W):
                tgt.append(("PaddleOCR's pre-processed image", cfg["oimg"]))
            elif cfg["oimg"] is not None and tag == "A":
                tgt.append(("PaddleOCR's pre-processed image", cfg["oimg"]))
            for tname, im in tgt:
                c = crop(im, poly)
                fn = f"diagnostics/{tag}_{name.split(' ')[0].replace('[', '_').replace(']', '')}_{'orig' if im is cvimg else 'preproc'}.png"
                cv2.imwrite(fn, c)
                vis = im.copy()
                cv2.polylines(vis, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
                cv2.imwrite(fn.replace(".png", "_overlay.jpg"), vis)
                txt = reread(safe, c)
                ok = query in ["".join(t.split()) for t in txt]
                verdicts[(tag, name, tname)] = ok
                print(f"   [{tag}] {name:<46} on {tname:<32} -> re-read {txt}  {'OK' if ok else 'WRONG'}   ({fn})")

    print("\n=== VERDICT")
    a_orig = [v for (t, n, tn), v in verdicts.items() if t == "A" and tn == "original image"]
    a_pre = [v for (t, n, tn), v in verdicts.items() if t == "A" and tn != "original image"]
    b_orig = [v for (t, n, tn), v in verdicts.items() if t == "B" and tn == "original image"]
    s = cfg_default["settings"]
    if (s.get("use_doc_unwarping") or s.get("use_doc_orientation_classify")) or \
            (cfg_default["oimg"] is not None and cfg_default["oimg"].shape[:2] != (H, W)):
        print(" * Default PaddleOCR PRE-PROCESSED the photo (orientation classifier / UVDoc unwarping).")
        print("   Its polygons live in THAT image, not in the file you loaded with OpenCV.")
        if a_orig and not any(a_orig) and any(a_pre):
            print("   CONFIRMED: polygons are wrong on the original but right on the pre-processed image.")
        elif a_orig and not any(a_orig):
            print("   Polygons are wrong on the original image (see the overlay pictures).")
    if not cfg_default["aligned"]:
        print(" * rec_texts / rec_scores / rec_polys have DIFFERENT lengths - never zip them with each other.")
    if not cfg_default["dt_equal"]:
        print(" * dt_polys is longer/shorter than rec_texts: using dt_polys[i] with rec_texts[i] picks the WRONG box.")
    if b_orig and all(b_orig):
        print(" * CONFIG B is correct: with pre-processing OFF the polygon encloses the number on the original image.")
        print("   FIX: create PaddleOCR with the three flags set to False (ocr_engine.py does this) and only use rec_polys.")
    elif b_orig:
        print(" * Even config B did not re-read the number from its polygon - send me the overlay images in ./diagnostics")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
