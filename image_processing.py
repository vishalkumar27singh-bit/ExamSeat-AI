"""
image_processing.py
===================
Everything that touches pixels *before* or *around* OCR:

* loading an image the same way every time (EXIF rotation handled)
* quality checks (blur / brightness / contrast)
* same-size enhancement (CLAHE, gamma, sharpening) - same size means OCR
  coordinates stay valid
* geometry helpers: 90-degree rotation with an exact point transform,
  perspective warp of a quadrilateral, polygon cropping with a margin,
  drawing helpers

GOLDEN RULE OF THIS PROJECT
---------------------------
Every polygon that is ever drawn must be expressed in the coordinate system of
the image it is drawn on.  Whenever an image is rotated / resized we keep a
`Transform` so polygons can be mapped back to the ORIGINAL photo.
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

ImageSource = Union[str, os.PathLike, bytes, bytearray, io.IOBase, Image.Image, np.ndarray]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_image(source: ImageSource) -> np.ndarray:
    """Load any supported source as a BGR uint8 array (what OpenCV expects).

    EXIF orientation is applied, because phones often store a landscape pixel
    grid plus an "orientation" flag.  OpenCV's imread also applies EXIF in recent
    versions, but PIL's behaviour is explicit, so we always use PIL and make the
    result deterministic.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        return source.copy()
    if isinstance(source, Image.Image):
        pil = source
    elif isinstance(source, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(bytes(source)))
    elif hasattr(source, "read"):
        pil = Image.open(io.BytesIO(source.read()))
    else:
        pil = Image.open(source)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def image_fingerprint(img: np.ndarray) -> str:
    """Short hash of pixel content - used to cache OCR results per image."""
    h = hashlib.sha1()
    h.update(str(img.shape).encode())
    h.update(np.ascontiguousarray(img[:: max(1, img.shape[0] // 64), :: max(1, img.shape[1] // 64)]).tobytes())
    return h.hexdigest()[:16]


def limit_size(img: np.ndarray, max_side: int = 3600) -> Tuple[np.ndarray, float]:
    """Down-scale very large photos.  Returns (image, scale) with
    work_coords = original_coords * scale."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    s = max_side / float(m)
    out = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return out, s


# --------------------------------------------------------------------------
# Quality assessment and enhancement
# --------------------------------------------------------------------------
def image_quality(img: np.ndarray) -> Dict[str, object]:
    """Cheap numbers that tell the user *why* an OCR result may be unreliable."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # normalise resolution so the blur metric is comparable between phones
    s = 1200.0 / max(h, w)
    small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    warnings: List[str] = []
    if blur < 60:
        warnings.append("Photo looks blurry - hold the phone steady and tap to focus.")
    if brightness < 80:
        warnings.append("Photo is dark - use better lighting (avoid shadows on the chart).")
    if brightness > 215:
        warnings.append("Photo is over-exposed - avoid glare / flash on the chart.")
    if contrast < 35:
        warnings.append("Low contrast - text may not be readable.")
    if min(h, w) < 900:
        warnings.append("Low resolution - move closer so each number is at least ~20 px tall.")
    return dict(width=w, height=h, blur=round(blur, 1), brightness=round(brightness, 1),
                contrast=round(contrast, 1), warnings=warnings)


def enhance(img: np.ndarray, mode: str = "clahe+sharpen") -> np.ndarray:
    """Same-size enhancement.  The output has EXACTLY the input's width/height,
    so polygons found on it are valid on the input as well."""
    out = img.copy()
    if "clahe" in mode:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean_l = float(l.mean())
        if mean_l < 110:  # dark photo -> brighten with gamma first
            gamma = np.clip(np.log(128.0 / 255.0) / np.log(max(mean_l, 1.0) / 255.0), 0.4, 1.0)
            lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
            l = cv2.LUT(l, lut)
        l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if "sharpen" in mode:
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    assert out.shape == img.shape, "enhance() must never change the image size"
    return out


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
@dataclass
class Transform:
    """3x3 homogeneous matrix M with  dst = M @ src  (points as columns)."""
    M: np.ndarray

    def apply(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        ph = np.hstack([p, np.ones((len(p), 1))])
        q = (self.M @ ph.T).T
        q = q[:, :2] / q[:, 2:3]
        return q.reshape(np.asarray(pts).shape)

    def inverse(self) -> "Transform":
        return Transform(np.linalg.inv(self.M))

    def then(self, other: "Transform") -> "Transform":
        """self first, then other."""
        return Transform(other.M @ self.M)

    @staticmethod
    def identity() -> "Transform":
        return Transform(np.eye(3))

    @staticmethod
    def scale(s: float) -> "Transform":
        return Transform(np.diag([s, s, 1.0]))


def rotate_quarter_turns(img: np.ndarray, k: int) -> Tuple[np.ndarray, Transform]:
    """Rotate by k * 90 degrees clockwise.  Returns (image, Transform original->rotated).
    Uses continuous pixel-edge coordinates (pixel i covers [i, i+1))."""
    k %= 4
    h, w = img.shape[:2]
    if k == 0:
        return img, Transform.identity()
    if k == 1:   # x' = h - y ; y' = x
        out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        M = np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    elif k == 2:  # x' = w - x ; y' = h - y
        out = cv2.rotate(img, cv2.ROTATE_180)
        M = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float64)
    else:         # x' = y ; y' = w - x
        out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        M = np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64)
    return out, Transform(M)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = p.sum(1)
    d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


def warp_quad(img: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Perspective-correct the quadrilateral `quad` (TL,TR,BR,BL) to an
    out_w x out_h rectangle.  Returns (warped, H) with H: image -> warped."""
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), H


def poly_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two polygons (rasterised, so works for slanted quads too)."""
    ax0, ay0, ax1, ay1 = poly_bbox(a)
    bx0, by0, bx1, by1 = poly_bbox(b)
    x0, y0 = int(np.floor(min(ax0, bx0))) - 1, int(np.floor(min(ay0, by0))) - 1
    x1, y1 = int(np.ceil(max(ax1, bx1))) + 1, int(np.ceil(max(ay1, by1))) + 1
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 0.0
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.round(np.asarray(a) - [x0, y0]).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(np.asarray(b) - [x0, y0]).astype(np.int32)], 1)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    return inter / union if union else 0.0


def crop_polygon(img: np.ndarray, poly: np.ndarray, margin: float = 0.6,
                 rectify: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop around a polygon with a margin (fraction of the polygon's height,
    plus a few pixels).  Returns (crop, (x_offset, y_offset)) so callers can
    map crop coordinates back.

    rectify=False -> plain axis-aligned crop (what a human sees on the photo)
    rectify=True  -> perspective-straightened crop of the (slanted) polygon
    """
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    h_img, w_img = img.shape[:2]
    if rectify:
        q = order_quad(cv2.boxPoints(cv2.minAreaRect(poly)) if len(poly) != 4 else poly)
        w = float(np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = float(np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        pad = max(4.0, margin * h)
        ex = (q[1] - q[0]) / max(np.linalg.norm(q[1] - q[0]), 1e-6)  # unit along text
        ey = (q[3] - q[0]) / max(np.linalg.norm(q[3] - q[0]), 1e-6)  # unit across text
        q2 = np.array([q[0] - ex * pad - ey * pad, q[1] + ex * pad - ey * pad,
                       q[2] + ex * pad + ey * pad, q[3] - ex * pad + ey * pad], dtype=np.float32)
        out_w, out_h = int(round(w + 2 * pad)), int(round(h + 2 * pad))
        warped, _ = warp_quad(img, q2, max(out_w, 8), max(out_h, 8))
        return warped, (int(q2[:, 0].min()), int(q2[:, 1].min()))
    x0, y0, x1, y1 = poly_bbox(poly)
    pad = max(6.0, margin * (y1 - y0))
    X0, Y0 = max(0, int(np.floor(x0 - pad))), max(0, int(np.floor(y0 - pad)))
    X1, Y1 = min(w_img, int(np.ceil(x1 + pad))), min(h_img, int(np.ceil(y1 + pad)))
    return img[Y0:Y1, X0:X1].copy(), (X0, Y0)


def draw_polygon(img: np.ndarray, poly: np.ndarray, color=(0, 200, 0), thickness: int = 3,
                 label: Optional[str] = None) -> np.ndarray:
    """Draw a polygon (in place) and an optional label above it."""
    pts = np.round(np.asarray(poly, dtype=np.float64).reshape(-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    if label:
        x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
        scale = max(0.5, img.shape[1] / 1800)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ty = max(th + 4, y - 8)
        cv2.rectangle(img, (x, ty - th - 4), (x + tw + 6, ty + bl), color, -1)
        cv2.putText(img, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
