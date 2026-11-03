"""
ocr_engine.py
=============
OCR inference + coordinate safety + registration-number search.

Why this file exists (the bug it prevents)
------------------------------------------
PaddleOCR 3.x, when created with its DEFAULT settings, first runs a
"document preprocessor" (orientation classifier + UVDoc *unwarping*).  The text
polygons it returns (`rec_polys`, `rec_boxes`) are then expressed in the
coordinate system of the UNWARPED image, not of the photo you loaded with
OpenCV.  Drawing them on the original photo boxes the wrong cell.

This engine therefore
  1. switches those preprocessing stages OFF,
  2. refuses to return results if the library says it processed a different
     image size / used unwarping,
  3. only reads `rec_texts`, `rec_scores`, `rec_polys` (which are built
     together, so index i always means the same detection),
  4. offers `verify_polygon()` - re-reads the pixels INSIDE a polygon and checks
     they really say the number we claim.  A wrong box can no longer be shown
     silently.
"""
from __future__ import annotations

import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from image_processing import (crop_polygon, draw_polygon, enhance, load_image, order_quad,
                              poly_bbox, poly_iou)

REG_LEN = 10  # registration numbers at this university are 10 digits
logger = logging.getLogger(__name__)


# ==========================================================================
# Data classes
# ==========================================================================
@dataclass
class Detection:
    """One OCR text line: the string and ITS OWN polygon (never separated)."""
    text: str
    score: float
    poly: np.ndarray            # (4,2) float32, TL,TR,BR,BL, in the coordinates of the image that was OCR'd
    index: int = -1             # position in the OCR library's output
    tag: str = ""               # which pass produced it ("orig", "enhanced", "rot90" ...)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return poly_bbox(self.poly)

    @property
    def center(self) -> np.ndarray:
        return self.poly.mean(axis=0)

    @property
    def height(self) -> float:
        p = self.poly
        return float((np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2)

    @property
    def width(self) -> float:
        p = self.poly
        return float((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2)


@dataclass
class RegToken:
    """A detection interpreted as (part of) a registration number."""
    det: Detection
    digits: str                 # only digits
    substituted: int            # how many letters were turned into digits (O->0 ...). 0 = text was pure digits
    kind: str                   # 'full' | 'partial' | 'merged'
    raw_to_digit_pos: List[float] = field(default_factory=list)  # fractional x-position (0..1) of each digit's left edge
    raw_to_digit_end: List[float] = field(default_factory=list)  # ... and right edge


@dataclass
class Match:
    token: RegToken
    poly: np.ndarray            # polygon of exactly the matched digits
    score: float
    kind: str                   # 'exact' | 'exact_in_merged'


@dataclass
class Candidate:
    text: str
    distance: int
    score: float                # OCR score of that detection
    confidence: float           # score * similarity
    poly: np.ndarray
    reason: str


@dataclass
class VerifyResult:
    ok: bool
    reread_text: str
    reread_score: float
    crop: Optional[np.ndarray] = None
    note: str = ""


# ==========================================================================
# OCR backends
# ==========================================================================
def _quad(poly) -> np.ndarray:
    try:
        p = np.asarray(poly, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OCR polygon coordinates must be numeric") from exc
    if p.ndim == 1 and p.size >= 6 and p.size % 2 == 0:
        p = p.reshape(-1, 2)
    elif p.ndim >= 2 and p.shape[-1] == 2:
        p = p.reshape(-1, 2)
    else:
        raise ValueError("OCR polygon must contain at least three (x, y) points")
    if len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("OCR polygon must contain at least three finite points")
    if len(p) != 4:  # curved / polygon text -> minimum-area rectangle
        p = cv2.boxPoints(cv2.minAreaRect(p))
    quad = order_quad(p)
    if cv2.contourArea(quad) <= 1e-3:
        raise ValueError("OCR polygon has no measurable area")
    return quad


class _PaddleBackend:
    name = "paddleocr"

    def __init__(self, lang: str = "en", det_limit_side_len: Optional[int] = None, **_):
        import paddleocr  # noqa: F401  (ImportError propagates to the caller)
        from paddleocr import PaddleOCR
        self.version = getattr(paddleocr, "__version__", "?")
        major = int(str(self.version).split(".")[0]) if str(self.version)[0].isdigit() else 3
        self.api = 3 if major >= 3 else 2
        if self.api == 3:
            kw = dict(use_doc_orientation_classify=False,  # <- these three switches are the whole fix
                      use_doc_unwarping=False,             #    (defaults are True!)
                      use_textline_orientation=False,
                      lang=lang)
            if det_limit_side_len:
                kw.update(text_det_limit_side_len=int(det_limit_side_len), text_det_limit_type="max")
            self.ocr = PaddleOCR(**kw)
        else:
            self.ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)

    def run(self, img: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        if self.api == 3:
            res = self.ocr.predict(img)
            if not res:
                return []
            r = res[0]
            try:
                texts, scores, polys = list(r["rec_texts"]), list(r["rec_scores"]), list(r["rec_polys"])
            except (KeyError, TypeError) as exc:
                raise RuntimeError("PaddleOCR returned an incomplete recognition result") from exc
            if not (len(texts) == len(scores) == len(polys)):
                raise RuntimeError(f"PaddleOCR returned misaligned lists: texts={len(texts)} "
                                   f"scores={len(scores)} polys={len(polys)}")
            # --- coordinate-system guards -----------------------------------
            ms = r["model_settings"] if "model_settings" in r else {}
            if ms.get("use_doc_unwarping") or ms.get("use_doc_orientation_classify"):
                raise RuntimeError("PaddleOCR used document unwarping/orientation: polygons are NOT in the "
                                   "coordinates of the input image.")
            pre = r["doc_preprocessor_res"] if "doc_preprocessor_res" in r else None
            if pre is not None and "output_img" in pre:
                oh, ow = np.asarray(pre["output_img"]).shape[:2]
                if (oh, ow) != tuple(img.shape[:2]):
                    raise RuntimeError(f"PaddleOCR processed a {ow}x{oh} image but input was "
                                       f"{img.shape[1]}x{img.shape[0]} - polygons would be in the wrong space.")
            return list(zip(polys, texts, scores))
        # PaddleOCR 2.x
        res = self.ocr.ocr(img, cls=False)
        lines = res[0] if res and res[0] else []
        out = []
        for line in lines:
            try:
                polygon, recognition = line
                text, score = recognition
            except (TypeError, ValueError):
                out.append(line)
                continue
            out.append((polygon, text, score))
        return out


class _RapidBackend:
    """RapidOCR = the same PP-OCR detection/recognition model family run through
    ONNX.  Used as automatic fallback when PaddleOCR cannot start (for example
    no internet to download its models)."""
    name = "rapidocr"

    def __init__(self, **_):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.eng, self.new_api = RapidOCR(), False
        except ImportError:
            from rapidocr import RapidOCR  # newer package name
            self.eng, self.new_api = RapidOCR(), True

    def run(self, img: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        if self.new_api:
            r = self.eng(img)
            if r is None or r.boxes is None or r.txts is None or r.scores is None:
                return []
            if not (len(r.boxes) == len(r.txts) == len(r.scores)):
                raise RuntimeError("RapidOCR returned misaligned boxes, texts, and scores")
            return list(zip(r.boxes, r.txts, r.scores))
        res, _ = self.eng(img)
        return [(p, t, s) for p, t, s in (res or [])]


class OCREngine:
    """Thin wrapper: backend='auto' tries PaddleOCR first, then RapidOCR."""

    def __init__(self, backend: str = "auto", lang: str = "en", det_limit_side_len: Optional[int] = None):
        self.requested = backend
        self.fallback_reason: Optional[str] = None
        b = backend.lower()
        if b in ("auto", "paddle", "paddleocr"):
            try:
                self._b = _PaddleBackend(lang=lang, det_limit_side_len=det_limit_side_len)
            except Exception as e:  # missing models, no internet, import problems ...
                if b != "auto":
                    raise
                self.fallback_reason = f"PaddleOCR could not start ({type(e).__name__}: {str(e)[:160]})"
                print("[ocr_engine] WARNING:", self.fallback_reason, "-> falling back to RapidOCR", file=sys.stderr)
                self._b = _RapidBackend()
        elif b in ("rapid", "rapidocr"):
            self._b = _RapidBackend()
        else:
            raise ValueError(f"unknown OCR backend '{backend}'")

    @property
    def name(self) -> str:
        return self._b.name

    def run(self, img_bgr: np.ndarray, tag: str = "") -> List[Detection]:
        """OCR `img_bgr`.  Returned polygons are in THIS array's pixel coordinates."""
        if not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            raise ValueError("OCR input must be a non-empty image array")
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
            raise ValueError("OCR input must be a uint8 BGR or grayscale image")
        out = []
        raw = self._b.run(img_bgr)
        if raw is None:
            logger.warning("OCR backend %s returned no result object", self.name)
            return out
        for i, item in enumerate(raw):
            try:
                poly, text, score = item
                if text is None or not str(text).strip():
                    raise ValueError("OCR text is empty")
                score = float(score)
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("OCR confidence must be finite and between 0 and 1")
                out.append(Detection(text=str(text), score=score, poly=_quad(poly), index=i, tag=tag))
            except (TypeError, ValueError, OverflowError, cv2.error) as exc:
                logger.warning("Ignoring malformed OCR result %d from %s: %s", i, self.name, exc)
        return out


# ==========================================================================
# Text normalisation  (careful: never silently change a number)
# ==========================================================================
_CONFUSABLE = {"O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "|": "1", "i": "1", "!": "1",
               "Z": "2", "z": "2", "S": "5", "s": "5", "G": "6", "B": "8"}
_STRIP_RE = re.compile(r"[\s\-_.,:;'`\"]")


def _clean_with_positions(text: str) -> Tuple[str, List[int], int]:
    """NFKC-normalise, drop spaces/punctuation. Returns (clean, kept raw indices, len(raw))."""
    raw = unicodedata.normalize("NFKC", text)
    kept = [i for i, c in enumerate(raw) if not _STRIP_RE.match(c)]
    return "".join(raw[i] for i in kept), kept, max(len(raw), 1)


def normalize_query(q: str) -> Tuple[Optional[str], str]:
    """Normalise what the student typed.  Returns (digits or None, message)."""
    clean, _, _ = _clean_with_positions(q or "")
    if not clean:
        return None, "Please type a registration number."
    if not clean.isdigit():
        return None, ("The registration number must contain digits only (found other characters). "
                      "Letters such as O or l are NOT converted automatically - please retype it.")
    if len(clean) != REG_LEN:
        return clean, f"Note: registration numbers here have {REG_LEN} digits, you typed {len(clean)}."
    return clean, ""


def to_token(det: Detection, reg_len: int = REG_LEN, min_partial: int = 5) -> Optional[RegToken]:
    """Interpret a detection as digits (or None if it is ordinary text)."""
    clean, kept, n_raw = _clean_with_positions(det.text)
    if not clean:
        return None
    out, subs = [], 0
    for c in clean:
        if c.isdigit():
            out.append(c)
        elif c in _CONFUSABLE:
            out.append(_CONFUSABLE[c]); subs += 1
        else:
            return None
    if sum(c.isdigit() for c in clean) < 0.6 * len(clean):
        return None
    digits = "".join(out)
    if len(digits) < min_partial:
        return None
    kind = "full" if len(digits) == reg_len else ("partial" if len(digits) < reg_len else "merged")
    starts = [kept[i] / n_raw for i in range(len(kept))]
    ends = [(kept[i] + 1) / n_raw for i in range(len(kept))]
    return RegToken(det, digits, subs, kind, starts, ends)


def extract_tokens(dets: Sequence[Detection], reg_len: int = REG_LEN) -> List[RegToken]:
    return [t for t in (to_token(d, reg_len) for d in dets) if t is not None]


def _sub_polygon(poly: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """Part of a text polygon between fractions f0..f1 of its length (TL,TR,BR,BL)."""
    tl, tr, br, bl = [np.asarray(p, dtype=np.float64) for p in order_quad(poly)]
    top0, top1 = tl + (tr - tl) * f0, tl + (tr - tl) * f1
    bot0, bot1 = bl + (br - bl) * f0, bl + (br - bl) * f1
    return np.array([top0, top1, bot1, bot0], dtype=np.float32)


# ==========================================================================
# Search
# ==========================================================================
def find_exact(tokens: Sequence[RegToken], query: str) -> List[Match]:
    """EXACT matches only.  A token whose letters had to be 'fixed' (O->0 ...)
    is never accepted here - it goes to the candidate list instead."""
    out: List[Match] = []
    for t in tokens:
        if t.substituted:
            continue
        if t.digits == query:
            out.append(Match(t, t.det.poly.copy(), t.det.score, "exact"))
        elif len(t.digits) > len(query) and query in t.digits:
            i = t.digits.find(query)
            f0, f1 = t.raw_to_digit_pos[i], t.raw_to_digit_end[i + len(query) - 1]
            out.append(Match(t, _sub_polygon(t.det.poly, f0, f1), t.det.score, "exact_in_merged"))
    return _dedupe_matches(out)


def _dedupe_matches(ms: List[Match]) -> List[Match]:
    """The same number found by two OCR passes at the same place = one match."""
    ms = sorted(ms, key=lambda m: -m.score)
    keep: List[Match] = []
    for m in ms:
        if all(poly_iou(m.poly, k.poly) < 0.3 for k in keep):
            keep.append(m)
    return keep


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def rank_candidates(tokens: Sequence[RegToken], query: str, max_dist: int = 2, k: int = 6) -> List[Candidate]:
    """Near-misses to show the user (NEVER reported as 'found')."""
    cands: List[Candidate] = []
    for t in tokens:
        d = t.det
        if t.kind == "merged":
            # look at every query-length window inside the merged string
            windows = [(t.digits[i:i + len(query)], i) for i in range(len(t.digits) - len(query) + 1)]
        else:
            windows = [(t.digits, 0)]
        for text, i in windows:
            if text == query and not t.substituted:
                continue  # that is an exact match, handled elsewhere
            if t.kind == "partial" and len(text) >= 5 and (text in query):
                dist, reason = len(query) - len(text), "fragment of your number (cut off at the photo edge?)"
            else:
                dist = _levenshtein(text, query)
                reason = "same length, differs in " + ", ".join(
                    f"digit {p + 1}" for p, (x, y) in enumerate(zip(text, query)) if x != y) if len(text) == len(query) \
                    else f"{len(text)} digits read instead of {len(query)}"
                if t.substituted:
                    reason += f"; {t.substituted} letter(s) were read as digits (e.g. O/l)"
            if dist > max_dist and not (t.kind == "partial" and text in query):
                continue
            sim = max(0.0, 1.0 - dist / max(len(query), 1))
            poly = d.poly if t.kind != "merged" else _sub_polygon(d.poly, t.raw_to_digit_pos[i],
                                                                  t.raw_to_digit_end[min(i + len(query) - 1, len(t.digits) - 1)])
            cands.append(Candidate(text, dist, d.score, round(d.score * sim, 3), poly, reason))
    cands.sort(key=lambda c: (c.distance, -c.score))
    seen, out = set(), []
    for c in cands:
        key = (c.text, tuple(np.round(c.poly.mean(0) / 8).astype(int)))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out[:k]


# ==========================================================================
# Polygon verification  ("does this box really contain the number?")
# ==========================================================================
def verify_polygon(engine: OCREngine, img: np.ndarray, poly: np.ndarray, expected: str,
                   margin: float = 0.35) -> VerifyResult:
    """Cut the pixels inside `poly` out of `img`, read them again on their own and
    compare with `expected`.  If the polygon belongs to a neighbouring cell, the
    re-read text differs and verification FAILS."""
    crop, _ = crop_polygon(img, poly, margin=margin, rectify=True)
    if crop.size == 0:
        return VerifyResult(False, "", 0.0, crop, "empty crop")
    h = crop.shape[0]
    if h < 64:  # tiny crops confuse detectors -> upscale
        f = 64.0 / h
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    pad = 24
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    dets = engine.run(padded, tag="verify")
    cx, cy = padded.shape[1] / 2, padded.shape[0] / 2
    best, best_d = None, 1e9
    for d in dets:
        t = to_token(d, min_partial=3)
        if t is None:
            continue
        dist = float(np.hypot(*(d.center - [cx, cy])))
        if dist < best_d:
            best, best_d = t, dist
    if best is None:
        return VerifyResult(False, "", 0.0, crop, "no digits could be read inside the highlighted polygon")
    ok = (best.digits == expected) and best.substituted == 0
    return VerifyResult(ok, best.digits, best.det.score, crop,
                        "" if ok else f"the box actually contains '{best.digits}', not '{expected}'")


# ==========================================================================
# Merging several OCR passes
# ==========================================================================
def merge_detections(passes: Sequence[Sequence[Detection]], iou_thr: float = 0.5) -> List[Detection]:
    """Union of detections from several passes (same coordinate system).
    Two detections with the SAME text at the same place are merged (best score kept).
    Detections with different text at the same place are both kept - we do not
    decide silently which one is 'right'."""
    out: List[Detection] = []
    for dets in passes:
        for d in dets:
            dup = None
            for o in out:
                if o.text == d.text and poly_iou(o.poly, d.poly) >= iou_thr:
                    dup = o
                    break
            if dup is None:
                out.append(d)
            elif d.score > dup.score:
                out[out.index(dup)] = d
    return out


# ==========================================================================
# Stage-2 command line tool:  python ocr_engine.py <image> [registration_no]
# ==========================================================================
def _main(argv: Sequence[str]) -> int:
    import argparse, os
    ap = argparse.ArgumentParser(description="Stage 2: run OCR and show every box on the image.")
    ap.add_argument("image")
    ap.add_argument("query", nargs="?")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    img = load_image(a.image)
    print(f"image: {a.image}\nshape (H,W,C) = {img.shape}")
    eng = OCREngine(a.backend)
    print("OCR backend:", eng.name, ("| " + eng.fallback_reason) if eng.fallback_reason else "")
    dets = eng.run(img, "orig")
    toks = extract_tokens(dets)
    print(f"{len(dets)} text lines, {sum(t.kind == 'full' for t in toks)} full registration-number tokens")
    vis = img.copy()
    for d in dets:
        t = to_token(d)
        draw_polygon(vis, d.poly, (0, 160, 0) if (t and t.kind == "full") else (0, 0, 220), 1)
    cv2.imwrite(os.path.join(a.out, "ocr_all_boxes.jpg"), vis)
    print("wrote", os.path.join(a.out, "ocr_all_boxes.jpg"), "(green = 10-digit number, red = other / partial)")
    if a.query:
        q, msg = normalize_query(a.query)
        print("query:", q, msg)
        ms = find_exact(toks, q) if q else []
        for m in ms:
            v = verify_polygon(eng, img, m.poly, q)
            print("EXACT MATCH", m.kind, "score=%.3f" % m.score, "poly=", np.round(m.poly).astype(int).tolist(),
                  "| re-read:", v.reread_text, "ok=" + str(v.ok))
            crop, _ = crop_polygon(img, m.poly, margin=0.6)
            cv2.imwrite(os.path.join(a.out, f"crop_{q}.png"), crop)
            hi = img.copy()
            draw_polygon(hi, m.poly, (0, 200, 0), 3, q)
            cv2.imwrite(os.path.join(a.out, f"highlight_{q}.jpg"), hi)
        if not ms:
            print("NOT FOUND exactly. Candidates:")
            for c in rank_candidates(toks, q):
                print(f"  {c.text}  dist={c.distance}  conf={c.confidence}  ({c.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
