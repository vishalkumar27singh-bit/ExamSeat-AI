"""
make_variants.py
================
Create SYNTHETIC test photos (blurry, low-light, tilted, low-resolution) from a
real, annotated photo.  The ground truth (room / column / row of every number,
and the table quadrilaterals) is carried over - geometry is transformed together
with the image.

IMPORTANT: synthetic variants are a convenient robustness test, NOT a substitute
for real photos taken on different phones in different rooms.  They are flagged
`"synthetic": true` in their ground-truth files and reported separately.

Usage:  python make_variants.py ground_truth/photo_001_clear.json
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np


def _transform_geometry(gt: dict, M: np.ndarray) -> dict:
    """Apply a 3x3 matrix to every annotated polygon / bbox."""
    def tf(pts):
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, M).reshape(-1, 2).round(1).tolist()
    out = json.loads(json.dumps(gt))
    for t in out.get("tables", []):
        t["quad"] = tf(t["quad"])
    for n in out["numbers"]:
        if "bbox" in n:
            x0, y0, x1, y1 = n["bbox"]
            n["poly"] = tf([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            del n["bbox"]
    return out


def _tilt_matrix(w: int, h: int, deg: float, persp: float):
    """Rotation + mild perspective; returns (M, out_w, out_h) with all content kept."""
    c = np.array([w / 2, h / 2])
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rot = (corners - c) @ R.T + c
    # perspective: squeeze the top edge (camera looking up at the board)
    rot[0, 0] += persp * w; rot[1, 0] -= persp * w
    x0, y0 = rot.min(0)
    rot -= [x0, y0]
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), rot.astype(np.float32))
    return M, int(np.ceil(rot[:, 0].max())), int(np.ceil(rot[:, 1].max()))


def make_variants(gt_path: str, out_dir: str = "samples", gt_dir: str = "ground_truth") -> list:
    gt = json.load(open(gt_path))
    base = os.path.dirname(os.path.abspath(gt_path))
    img_path = gt["image"] if os.path.isabs(gt["image"]) else os.path.join(os.path.dirname(base), gt["image"])
    img = cv2.imread(img_path)
    assert img is not None, f"cannot read {img_path}"
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(gt["image"]))[0].replace("_clear", "")
    rng = np.random.default_rng(0)
    made = []

    def save(name: str, im: np.ndarray, M: np.ndarray, note: str):
        p_img = os.path.join(out_dir, f"{stem}_{name}.jpg")
        cv2.imwrite(p_img, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        g = _transform_geometry(gt, M)
        g.update(image=os.path.join(os.path.basename(out_dir), os.path.basename(p_img)), condition=name,
                 synthetic=True, variant_of=gt["image"], note=note)
        p_gt = os.path.join(gt_dir, f"{stem}_{name}.json")
        json.dump(g, open(p_gt, "w"), indent=1)
        made.append(p_gt)

    I = np.eye(3)
    # blurry: out-of-focus / hand shake
    save("blurry", cv2.GaussianBlur(img, (0, 0), 2.5), I, "Gaussian blur sigma=2.5")
    # low light: dark + sensor noise
    dark = np.clip(((img / 255.0) ** 1.8) * 0.55 * 255 + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    save("low_light", dark, I, "gamma 1.8, x0.55 brightness, gaussian noise sigma=5")
    # low resolution
    small = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    save("low_res", small, np.diag([0.6, 0.6, 1.0]), "downscaled to 60% (720x960)")
    # tilted: rotation + perspective
    M, tw, th = _tilt_matrix(w, h, 6.0, 0.03)
    tilted = cv2.warpPerspective(img, M, (tw, th), flags=cv2.INTER_CUBIC, borderValue=(50, 50, 50))
    save("tilted", tilted, M, "rotated 6 degrees + perspective squeeze (3%)")
    return made


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in make_variants(sys.argv[1]):
        print("wrote", p)
