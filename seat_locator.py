"""
seat_locator.py
===============
Room, column and row detection from the PHOTO'S OWN GRID LINES.

How a seat is found (no reading-order guessing)
-----------------------------------------------
1. OpenCV finds the long near-vertical lines of every table (morphology + Hough).
2. At the height of the number, the two vertical lines on its left and right are
   the cell walls.  Walking left/right along evenly spaced lines gives the whole
   table; exactly 6 columns must be found (A-F) or the result is "unverified".
3. In a narrow band at the number's own x, OpenCV finds the horizontal lines.
   Going UP from the number we look for the header row (a thin row holding one
   tiny letter A-F).  Row 1 = first row below that header.
4. A table belongs to the nearest "Room No: ..." heading ABOVE it, as long as no
   other table lies between them.
5. Every step has a sanity check.  If one fails the result is reported as
   unverified instead of guessed.

All geometry is measured in the working image's coordinates; slanted/perspective
tables are handled because every measurement is local (line x at this y, line y
at this x).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from image_processing import Transform, draw_polygon, order_quad, warp_quad
from ocr_engine import Detection, RegToken, extract_tokens

COLUMN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
logger = logging.getLogger(__name__)


# ==========================================================================
# Room headings
# ==========================================================================
# Labels and optional alphanumeric prefixes are kept separate so a plain "Room: 204"
# does not need a configured room prefix.
_ROOM_RE = re.compile(
    r"R[O0Q]{2}M\s*(?:(?:N[O0Q]\.?|NUMBER|#)\s*)?[:;.\-–—_ ]*\s*"
    r"(?:(?P<prefix>[A-Z]{2,6})\s*[-–—_]?\s*)?(?P<num>[0-9OoIlSB]{2,4})(?!\d)",
    re.I,
)
# "Room No: ATR-3RD FLOOR LIB-A" (rooms that are not numbered)
_ROOM_FREE_RE = re.compile(r"R[O0Q]{2}M\s*N[O0Q]\.?\s*[:;.\-]*\s*([A-Z]{2,6})\s*[-–—_]\s*(\d[A-Z]{2})\s*FL[O0]{2}R\s*([A-Z]{2,5})\s*[-–—_]?\s*([A-Z0-9])\s*$", re.I)
_ROOM_LABEL_RE = re.compile(r"^\s*R[O0Q]{2}M\s*(?:(?:N[O0Q]\.?|NUMBER|#)\s*)?[:;.\-–—_ ]*\s*$", re.I)
_ROOM_CODE_RE = re.compile(
    r"^\s*(?:(?P<prefix>[A-Z]{2,6})\s*[-–—_]?\s*)?(?P<num>[0-9OoIlSB]{2,4})\s*$",
    re.I,
)
_NUM_FIX = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"}


@dataclass
class RoomHeading:
    room: str
    det: Detection
    note: str = ""

    @property
    def center(self) -> np.ndarray:
        return self.det.center


def _room_code(letters: Optional[str], num: str) -> Tuple[str, str]:
    fixed = "".join(_NUM_FIX.get(c, c) for c in num)
    note = f"room digits '{num}' read as '{fixed}'" if fixed != num else ""
    return (f"{letters.upper()}-{fixed}" if letters else fixed), note


def find_room_headings(dets: Sequence[Detection], prefixes: Sequence[str] = ("ATR",)) -> List[RoomHeading]:
    """Find 'Room No: XXX-nnn' headings.  Course codes such as BCA-214 are NOT
    mistaken for rooms because a room must follow the words 'Room No'."""
    out: List[RoomHeading] = []
    used = set()
    for i, d in enumerate(dets):
        mf = _ROOM_FREE_RE.search(d.text)
        if mf:
            name = f"{mf.group(1)}-{mf.group(2)} FLOOR {mf.group(3)}-{mf.group(4)}".upper()
            out.append(RoomHeading(name, d, "room name is not a number (read as written)"))
            used.add(i)
            continue
        m = _ROOM_RE.search(d.text)
        if m:
            code, note = _room_code(m.group("prefix"), m.group("num"))
            out.append(RoomHeading(code, d, note))
            used.add(i)
    # label and code in two separate detections ("Room No:" ... "ATR-206")
    for i, d in enumerate(dets):
        if i in used or not _ROOM_LABEL_RE.match(d.text):
            continue
        best, bd = None, 1e9
        for j, e in enumerate(dets):
            if j == i or not _ROOM_CODE_RE.match(e.text):
                continue
            dx = e.center[0] - d.center[0]
            dy = abs(e.center[1] - d.center[1])
            if 0 < dx < 6 * d.width and dy < d.height and dx < bd:
                best, bd = e, dx
        if best is not None:
            mm = _ROOM_CODE_RE.match(best.text)
            code, note = _room_code(mm.group("prefix"), mm.group("num"))
            out.append(RoomHeading(code, best, note))
    # configured prefixes written without the 'Room No' label (last resort)
    if not out and prefixes:
        pat = re.compile(r"^\s*(" + "|".join(map(re.escape, prefixes)) + r")\s*[-–—_]?\s*(\d{2,4})\s*$", re.I)
        for d in dets:
            m = pat.match(d.text)
            if m:
                out.append(RoomHeading(f"{m.group(1).upper()}-{m.group(2)}", d, "heading label 'Room No' not read"))
    return out


# ==========================================================================
# Line / table data structures
# ==========================================================================
@dataclass
class VLine:
    id: int
    a: float
    b: float              # x = a*y + b
    y0: float
    y1: float
    covered: float        # total y-length covered by Hough segments
    virtual: bool = False  # True = not seen in the photo, placed by _complete_chain

    def x_at(self, y: float) -> float:
        return self.a * y + self.b


@dataclass
class Table:
    key: Tuple[int, ...]
    lines: List[VLine]
    top: float
    bottom: float
    heading: Optional[RoomHeading] = None
    heading_note: str = ""
    n_numbers: int = 0

    def x_left(self, y: float) -> float:
        return self.lines[0].x_at(y)

    def x_right(self, y: float) -> float:
        return self.lines[-1].x_at(y)


@dataclass
class SeatResult:
    verified: bool
    room: Optional[str]
    column: Optional[str]
    row: Optional[int]
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    grid_confidence: float = 0.0
    cell_quad: Optional[np.ndarray] = None
    table_key: Optional[Tuple[int, ...]] = None
    heading: Optional[RoomHeading] = None


def _polygon(value) -> np.ndarray:
    """Return finite polygon points in (N, 2) form or raise a clear input error."""
    if value is None:
        raise ValueError("polygon is missing")
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("polygon coordinates must be numeric") from exc
    if points.ndim == 1 and points.size >= 6 and points.size % 2 == 0:
        points = points.reshape(-1, 2)
    elif points.ndim >= 2 and points.shape[-1] == 2:
        points = points.reshape(-1, 2)
    else:
        raise ValueError("polygon must contain at least three (x, y) points")
    if len(points) < 3:
        raise ValueError("polygon must contain at least three (x, y) points")
    if not np.isfinite(points).all():
        raise ValueError("polygon coordinates must be finite")
    if cv2.contourArea(points.astype(np.float32)) <= 1e-3:
        raise ValueError("polygon has no measurable area")
    return points


# ==========================================================================
# Helpers: skew estimation and ridge filters
# ==========================================================================
def _gauss(s: float) -> np.ndarray:
    n = int(6 * s) | 1
    x = np.arange(n) - n // 2
    k = np.exp(-x * x / (2 * s * s))
    return (k / k.sum()).astype(np.float32)


def _gauss_d2(s: float) -> np.ndarray:
    n = int(6 * s) | 1
    x = np.arange(n) - n // 2
    k = (x * x / s ** 4 - 1 / s ** 2) * np.exp(-x * x / (2 * s * s))
    k -= k.mean()
    return (k / np.abs(k).sum()).astype(np.float32)


def ridge_map(gray: np.ndarray, u: float, vertical: bool) -> np.ndarray:
    """Response of an elongated dark-line detector.  Long thin dark lines give a strong
    response even when blurred; short text strokes are smoothed away."""
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray).astype(np.float32)
    sx, sy = max(1.2, 0.10 * u), max(3.0, 0.8 * u)
    r = cv2.sepFilter2D(g, -1, _gauss_d2(sx), _gauss(sy)) if vertical else cv2.sepFilter2D(g, -1, _gauss(sy), _gauss_d2(sx))
    r = np.maximum(r, 0)
    return r / max(float(np.percentile(r, 99.7)), 1e-6)


def estimate_skew(img: np.ndarray, max_deg: float = 20.0) -> float:
    """Angle (degrees, cv2 convention) that makes table lines/text horizontal+vertical:
    maximises the sharpness of the row/column ink profiles."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 800.0 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bw = cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 8)
    h, w = bw.shape
    c = (w / 2, h / 2)

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
        return float(r.sum(1).var() + r.sum(0).var())

    coarse = np.arange(-max_deg, max_deg + 0.01, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.01, 0.25)
    return float(max(fine, key=score))


# ==========================================================================
# The locator
# ==========================================================================
class SeatLocator:
    """All inputs/outputs are in the coordinates of `img` (the working image).  Internally the
    layout is analysed on a de-skewed copy; results are mapped back."""

    def __init__(self, img: np.ndarray, detections: Sequence[Detection], n_cols: int = 6,
                 room_prefixes: Sequence[str] = ("ATR",)):
        self.work = img
        self.n_cols = n_cols
        # ---- de-skew ----------------------------------------------------
        self.skew = estimate_skew(img)
        h0, w0 = img.shape[:2]
        if abs(self.skew) > 0.8:
            M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), self.skew, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h0 * sin + w0 * cos), int(h0 * cos + w0 * sin)
            M[0, 2] += nw / 2 - w0 / 2
            M[1, 2] += nh / 2 - h0 / 2
            self.img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            self.T = Transform(np.vstack([M, [0, 0, 1]]))
        else:
            self.img, self.T = img, Transform.identity()
        self.H, self.W = self.img.shape[:2]
        self.dets: List[Detection] = []
        for i, detection in enumerate(detections):
            try:
                if not isinstance(detection, Detection):
                    raise ValueError("detection has an unsupported type")
                if not isinstance(detection.text, str) or not detection.text.strip():
                    raise ValueError("detection text is empty")
                score = float(detection.score)
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("detection confidence is invalid")
                polygon = _polygon(detection.poly)
                if len(polygon) != 4:
                    polygon = order_quad(
                        cv2.boxPoints(cv2.minAreaRect(polygon.astype(np.float32)))
                    ).astype(np.float64)
                transformed = self.T.apply(polygon).astype(np.float32)
                if not np.isfinite(transformed).all():
                    raise ValueError("transformed coordinates are not finite")
            except (TypeError, ValueError, OverflowError, cv2.error) as exc:
                logger.warning("Ignoring malformed OCR detection %d in seat locator: %s", i, exc)
                continue
            self.dets.append(Detection(detection.text, score, transformed, detection.index, detection.tag))
        self.tokens = extract_tokens(self.dets)
        full = [t for t in self.tokens if t.kind == "full"]
        # u = text height in pixels; every length below is a multiple of u (works for any resolution)
        self.u = float(np.median([t.det.height for t in full])) if full else 24.0
        self.wn = float(np.median([t.det.width for t in full])) if full else 4.4 * self.u
        self.headings = find_room_headings(self.dets, room_prefixes)
        # "NIST UNIVERSITY" is printed centred on every sheet: it tells where a sheet's middle is
        self.titles = [d.center for d in self.dets if re.search(r"N[I1l]ST\s*UNIV", d.text, re.I)]
        self._build_maps()
        self.vlines = self._find_vertical_lines()
        self.tables: Dict[Tuple[int, ...], Table] = {}
        self.table_alias: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        self._cells: Optional[Dict[tuple, set]] = None
        self._build_tables()

    # ------------------------------------------------------------------ maps
    def _build_maps(self) -> None:
        u = self.u
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), (3, 3), 0)
        block = int(max(15, round(1.05 * u))) | 1
        self.bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 8)
        rv = ridge_map(gray, u, True)
        kh, kw = max(15, int(round(1.25 * u))), max(3, int(round(0.12 * u)))
        self.vopen = cv2.morphologyEx((rv > 0.30).astype(np.uint8) * 255, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
        self.hor = (ridge_map(gray, u, False) > 0.30).astype(np.uint8) * 255

    # ------------------------------------------------------- vertical lines
    def _find_vertical_lines(self) -> List[VLine]:
        u = self.u
        segs = cv2.HoughLinesP(self.vopen, 1, np.pi / 720, threshold=int(1.6 * u),
                               minLineLength=int(3.3 * u), maxLineGap=int(1.0 * u))
        if segs is None:
            return []
        segs = segs[:, 0, :].astype(np.float64)
        ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
        segs = segs[np.abs(ang - 90) < 12]
        lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        segs = segs[np.argsort(-lens)]
        tol, gap_max = 0.25 * u, 3.0 * u
        groups: List[dict] = []
        for x0, y0, x1, y1 in segs:
            if abs(y1 - y0) < 1e-6:
                continue
            placed = False
            for g in groups:
                if abs(x0 - (g["a"] * y0 + g["b"])) < tol and abs(x1 - (g["a"] * y1 + g["b"])) < tol:
                    lo, hi = min(y0, y1), max(y0, y1)
                    gap = max(0.0, max(lo - g["y1"], g["y0"] - hi))
                    if gap <= gap_max:
                        pts2 = g["pts"] + [(x0, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]
                        p = np.array(pts2)
                        a2, b2 = np.polyfit(p[:, 1], p[:, 0], 1)
                        # two sheets can share a collinear edge but not the same tilt: refuse a merge
                        # that makes the straight-line fit wrong for any of its points
                        if np.abs(p[:, 0] - (a2 * p[:, 1] + b2)).max() > 1.5 * tol:
                            continue
                        g["pts"] = pts2
                        g["iv"].append((lo, hi))
                        g["y0"], g["y1"] = min(g["y0"], lo), max(g["y1"], hi)
                        g["a"], g["b"] = a2, b2
                        placed = True
                        break
            if not placed:
                a = (x1 - x0) / (y1 - y0)
                groups.append(dict(a=a, b=x0 - a * y0, pts=[(x0, y0), (x1, y1)], iv=[(min(y0, y1), max(y0, y1))],
                                   y0=min(y0, y1), y1=max(y0, y1)))
        lines: List[VLine] = []
        for g in groups:
            iv = sorted(g["iv"])
            cov, cur0, cur1 = 0.0, iv[0][0], iv[0][1]
            for a0, a1 in iv[1:]:
                if a0 <= cur1:
                    cur1 = max(cur1, a1)
                else:
                    cov += cur1 - cur0
                    cur0, cur1 = a0, a1
            cov += cur1 - cur0
            if cov >= 4 * u:
                lines.append(VLine(0, g["a"], g["b"], g["y0"], g["y1"], cov))
        lines.sort(key=lambda L: L.x_at((L.y0 + L.y1) / 2))
        for i, L in enumerate(lines):
            L.id = i
        return lines

    # -------------------------------------------------------- column chain
    def _chain_at(self, cx: float, cy: float, xmin: Optional[float] = None, xmax: Optional[float] = None) -> Optional[dict]:
        """Vertical lines on the scan-line through (., cy), grouped into the table containing the
        number whose horizontal extent is [xmin, xmax].  A wall must lie clear of the digits."""
        u, wn = self.u, self.wn
        xmin = cx - wn / 2 if xmin is None else xmin
        xmax = cx + wn / 2 if xmax is None else xmax
        lo, hi = 1.15 * wn, 3.2 * wn            # allowed column width
        # a printed grid line is straight, so a line detected over part of its length may be
        # extended by a few rows (5u) - but not across the gap between two sheets.
        # very short lines are text strokes, not grid lines.
        cand = [(L.x_at(cy), L) for L in self.vlines
                if L.y0 - 5 * u <= cy <= L.y1 + 5 * u and L.covered >= 5 * u]
        cand.sort(key=lambda t: t[0])
        left = [(x, L) for x, L in cand if x <= xmin - 0.15 * u]
        right = [(x, L) for x, L in cand if x >= xmax + 0.15 * u]
        if not left or not right:
            return None
        xl, Ll = left[-1]
        xr, Lr = right[0]
        w0 = xr - xl
        if not (lo <= w0 <= hi):
            return {"lines": [Ll, Lr], "idx": 0, "total": 1, "note": f"cell width {w0:.0f}px looks wrong"}

        centers = np.array([t.det.center for t in self.tokens if t.kind == "full"]) if self.tokens else np.zeros((0, 2))
        near = centers[np.abs(centers[:, 1] - cy) <= 10 * u] if len(centers) else centers

        def has_text(xa: float, xb: float) -> bool:
            """a real column contains registration numbers; the strip between two sheets does not"""
            lo_, hi_ = min(xa, xb), max(xa, xb)
            return bool(len(near) and ((near[:, 0] > lo_) & (near[:, 0] < hi_)).any())

        def walk(start: float, sign: int) -> List[Tuple[float, VLine]]:
            out, pos, w = [], start, w0
            while True:
                win = [(x, L) for x, L in cand
                       if 0.7 * w <= sign * (x - pos) <= 1.4 * w and has_text(pos, x)]
                if not win:
                    return out
                best_cov = max(L.covered for _, L in win)
                strong = [(x, L) for x, L in win if L.covered >= 0.5 * best_cov]
                x, L = min(strong, key=lambda t: abs(abs(t[0] - pos) - w))   # closest to the expected width
                out.append((x, L))
                w = abs(x - pos)          # columns may differ in width (edge columns are narrower)
                pos = x
        chain_l = [(xl, Ll)] + walk(xl, -1)
        chain_r = [(xr, Lr)] + walk(xr, +1)
        lines = [L for _, L in reversed(chain_l)] + [L for _, L in chain_r]
        out = {"lines": lines, "idx": len(chain_l) - 1, "total": len(lines) - 1, "note": ""}
        if out["total"] != self.n_cols:
            out = self._complete_chain(out, cx, cy)
        return out

    def _complete_chain(self, ch: dict, cx: float, cy: float) -> dict:
        """A sheet's OUTER walls are often not found (faint, shaded, cut by glare) or a neighbouring
        sheet's column is attached.  When the columns found are evenly spaced, the sheet's own
        centred title decides which 6 of them (or 5 + one inferred wall) form this table.  Anything
        less clear stays 'not verified'."""
        lines, idx, n = ch["lines"], ch["idx"], ch["total"]
        if n not in (4, 5, 7) or any(L.virtual for L in lines):
            return ch
        xs = np.array([L.x_at(cy) for L in lines])
        g = np.diff(xs)
        if len(g) == 0:
            return ch
        med = float(np.median(g))
        rel = np.abs(g / med - 1.0)
        if not (1.3 * self.wn <= med <= 1.9 * self.wn):
            return ch                      # columns of this chart are 1.3-1.9 number-widths wide
        # Keep the check tolerant for a single missing wall or one slightly skewed seam while still
        # rejecting obviously wrong line groups.  The title alignment check below is the real safety.
        if np.median(rel) > 0.25 and np.max(rel) > 0.45:
            return ch
        above = [t for t in self.titles if cy - 45 * self.u < t[1] < cy]
        if not above:
            return ch
        tx = min(above, key=lambda t: abs(t[0] - cx))[0]          # the sheet whose title is nearest in x
        cands = []                                              # (|centre - title|, lines, idx)
        if n == 7:
            for s0 in (0, 1):
                cands.append((abs((xs[s0] + xs[s0 + 6]) / 2 - tx), lines[s0:s0 + 7], idx - s0))
        else:
            for side in (-1, 1):
                ref = lines[0] if side < 0 else lines[-1]
                for extra in range(1, self.n_cols - n + 1):
                    steps = range(extra, 0, -1) if side < 0 else range(1, extra + 1)
                    virtual = []
                    for step in steps:
                        x = xs[0] - med * step if side < 0 else xs[-1] + med * step
                        ident = -(abs(ref.id) + 1) * 1000 - step * 2 - (1 if side < 0 else 2)
                        virtual.append(VLine(ident, ref.a, x - ref.a * cy, ref.y0, ref.y1, 0.0, True))
                    new = virtual + lines if side < 0 else lines + virtual
                    xx = np.array([L.x_at(cy) for L in new])
                    cands.append((abs((xx[0] + xx[-1]) / 2 - tx), new, idx + (extra if side < 0 else 0)))
        if not cands:
            return {**ch, "note": "columns found do not line up with the sheet's centred title"}
        cands.sort(key=lambda c: c[0])
        d0, ln, ix = cands[0]
        if d0 > 0.5 * med or len(ln) != self.n_cols + 1 or not (0 <= ix < self.n_cols):
            return {**ch, "note": "columns found do not line up with the sheet's centred title"}
        return {"lines": ln, "idx": ix, "total": self.n_cols, "note": "", "inferred": any(L.virtual for L in ln)}

    @staticmethod
    def _key(lines: Sequence[VLine]) -> Tuple[int, ...]:
        # ALL line ids: two stacked sheets can share a (merged) border line but never their inner lines
        return tuple(L.id for L in lines)

    def _build_tables(self) -> None:
        counts: Dict[Tuple[int, ...], int] = {}
        for t in self.tokens:
            if t.kind != "full":
                continue
            c = t.det.center
            x0_, _, x1_, _ = t.det.bbox
            ch = self._chain_at(c[0], c[1], x0_, x1_)
            if not ch or ch["total"] != self.n_cols:
                continue
            ls = ch["lines"]
            key = self._key(ls)
            counts[key] = counts.get(key, 0) + 1
            if key not in self.tables:
                inner = ls[1:-1]
                self.tables[key] = Table(key, ls, float(np.median([L.y0 for L in inner])),
                                         float(np.median([L.y1 for L in inner])))
        for k, n in counts.items():
            self.tables[k].n_numbers = n
        self._merge_duplicate_tables()
        for tb in self.tables.values():
            tb.heading, tb.heading_note = self._heading_for(tb)

    def _merge_duplicate_tables(self) -> None:
        """One physical table can be found twice when a line was detected in two pieces."""
        u = self.u
        keys = sorted(self.tables, key=lambda k: -self.tables[k].n_numbers)
        canon: List[Tuple[int, ...]] = []
        for k in keys:
            tb = self.tables[k]
            hit = None
            for c in canon:
                cb = self.tables[c]
                colw = (cb.x_right(cb.top) - cb.x_left(cb.top)) / self.n_cols
                if (abs(tb.top - cb.top) < 2 * u and abs(tb.x_left(tb.top) - cb.x_left(cb.top)) < 0.5 * colw
                        and abs(tb.x_right(tb.top) - cb.x_right(cb.top)) < 0.5 * colw):
                    hit = c
                    break
            if hit is None:
                canon.append(k)
                self.table_alias[k] = k
            else:
                self.table_alias[k] = hit
                self.tables[hit].n_numbers += tb.n_numbers
        self.tables = {k: self.tables[k] for k in canon}

    # ----------------------------------------------------- room association
    def _heading_for(self, tb: Table) -> Tuple[Optional[RoomHeading], str]:
        u = self.u
        y_top = tb.top
        xl, xr = tb.x_left(y_top), tb.x_right(y_top)
        colw = (xr - xl) / self.n_cols
        above = [h for h in self.headings
                 if h.center[1] < y_top - u and xl - 2 * colw <= h.center[0] <= xr + 2 * colw]
        if not above:
            return None, "no 'Room No' heading above this table (cut off in the photo?)"
        h = max(above, key=lambda h: h.center[1])        # closest one above
        for other in self.tables.values():                 # another table between => heading is theirs
            if other is tb:
                continue
            ox0, ox1 = other.x_left(other.top), other.x_right(other.top)
            overlap = min(xr, ox1) - max(xl, ox0)
            if overlap > 0.5 * (xr - xl) and h.center[1] < other.top < y_top - 2 * u:
                return None, ("the nearest heading above belongs to another table; "
                              "this table's own heading is not visible")
        if y_top - h.center[1] > 0.7 * (xr - xl):
            return None, "nearest heading is too far above the table"
        return h, h.note

    # ---------------------------------------------------------- row lines
    def _band_peaks(self, x0: float, x1: float, y0: float, y1: float) -> List[float]:
        x0, x1 = max(0, int(x0)), min(self.W, int(x1))
        y0, y1 = max(0, int(y0)), min(self.H, int(y1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return []
        prof = (self.hor[y0:y1, x0:x1] > 0).mean(axis=1)
        idx = np.where(prof >= 0.6)[0]
        runs: List[List[int]] = []
        for i in idx:
            if runs and i - runs[-1][-1] <= 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        centers = [float(np.average(r, weights=prof[r])) + y0 for r in runs]
        merged: List[List[float]] = []
        for c in centers:
            if merged and c - merged[-1][-1] <= 0.3 * self.u:
                merged[-1].append(c)
            else:
                merged.append([c])
        return [float(np.mean(m)) for m in merged]

    def _row_lines(self, lL: VLine, lR: VLine, cy: float, y0: float, y1: float):
        """Horizontal grid lines of a cell column.  A line only counts if it is visible in BOTH
        text-free margins of the cell (digits never reach the margins).  Returns a list of
        ((xa, ya), (xb, yb)) sorted top to bottom."""
        xl, xr = lL.x_at(cy), lR.x_at(cy)
        colw = xr - xl
        w = max(6.0, 0.11 * colw)
        xa0, xa1 = xl + 0.05 * colw, xl + 0.05 * colw + w
        xb1 = xr - 0.05 * colw
        xb0 = xb1 - w
        PL = self._band_peaks(xa0, xa1, y0, y1)
        PR = self._band_peaks(xb0, xb1, y0, y1)
        if len(PL) < 2 or len(PR) < 2:
            return []
        d = np.diff(PL)
        d = d[(d > 0.6 * self.u) & (d < 3 * self.u)]
        pitch = float(np.median(d)) if len(d) else 1.45 * self.u
        # the right margin sees each line shifted by the local tilt: find that common shift
        best_delta, best_n = 0.0, -1
        for delta in np.arange(-0.5 * pitch, 0.5 * pitch + 0.01, 1.0):
            n = sum(any(abs(pr - (pl + delta)) <= 0.2 * pitch for pr in PR) for pl in PL)
            if n > best_n:
                best_delta, best_n = float(delta), n
        tol = 0.2 * pitch
        pairs, used = [], set()
        for pl in PL:
            best, bd = None, tol
            for i, pr in enumerate(PR):
                if i not in used and abs(pr - (pl + best_delta)) <= bd:
                    best, bd = i, abs(pr - (pl + best_delta))
            if best is not None:
                used.add(best)
                pairs.append(((xa0 + xa1) / 2, pl, (xb0 + xb1) / 2, PR[best]))
        return pairs

    @staticmethod
    def _y_at(line, x: float) -> float:
        xa, ya, xb, yb = line
        return ya + (yb - ya) * (x - xa) / (xb - xa)

    def _is_header_cell(self, xc: float, colw: float, ya: float, yb: float) -> bool:
        """A header cell holds ONE tiny glyph (A-F), unlike a number cell (many digits)
        or an empty cell (nothing)."""
        u = self.u
        hh = yb - ya
        if hh < 0.4 * u or hh > 2.6 * u:
            return False
        xa, xb = int(xc - 0.2 * colw), int(xc + 0.2 * colw)
        ya_, yb_ = int(ya + 0.12 * hh + 1), int(yb - 0.12 * hh - 1)
        if xb - xa < 8 or yb_ - ya_ < 3:
            return False
        sub = self.bw[ya_:yb_, xa:xb].copy()
        sub[self.hor[ya_:yb_, xa:xb] > 0] = 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        comps = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(4, int(0.04 * u * u))]
        if not (1 <= len(comps) <= 3):
            return False
        ink = float(sub.sum()) / 255.0 / sub.size
        return ink < 0.2 and all(c[cv2.CC_STAT_HEIGHT] <= 0.8 * hh + 2 for c in comps)

    # ------------------------------------------------------------- locate
    def locate(self, det_or_poly) -> SeatResult:
        """Room / column / row of a registration number (see _locate_raw), plus a cross-check:
        two DIFFERENT numbers can never sit in the same seat.  If they do, neither is trusted."""
        res = self._locate_raw(det_or_poly)
        if not res.verified:
            return res
        if self._cells is None:
            self._cells = {}
            for t in self.tokens:
                if t.kind != "full":
                    continue
                r = self._locate_raw(self.T.inverse().apply(np.asarray(t.det.poly, dtype=np.float64)))
                if r.verified:
                    self._cells.setdefault((r.room, r.column, r.row), set()).add(t.digits)
        poly0 = _polygon(det_or_poly.poly if isinstance(det_or_poly, Detection) else det_or_poly)
        mine = {t.digits for t in self.tokens if t.kind == "full" and
                np.linalg.norm(t.det.center - self.T.apply(poly0).mean(axis=0)) < 0.5 * self.u}
        others = self._cells.get((res.room, res.column, res.row), set()) - mine
        res.checks["unique_cell"] = not others
        if others:
            res.verified = False
            res.grid_confidence = round(res.grid_confidence * 0.5, 3)
            res.warnings.append("Another registration number was mapped to the same seat - the grid reading is "
                                "inconsistent here, so this seat is NOT verified.")
        return res

    def _locate_raw(self, det_or_poly) -> SeatResult:
        """Room / column / row of a registration number.  Accepts a Detection or a polygon
        in WORKING-image coordinates."""
        u = self.u
        res = SeatResult(False, None, None, None)
        try:
            poly0 = _polygon(det_or_poly.poly if isinstance(det_or_poly, Detection) else det_or_poly)
            poly = self.T.apply(poly0).astype(np.float32)
        except (TypeError, ValueError, OverflowError, cv2.error) as exc:
            res.checks["valid_polygon"] = False
            res.warnings.append(f"Seat could not be located because OCR returned invalid coordinates: {exc}.")
            logger.warning("Skipping invalid seat polygon: %s", exc)
            return res
        if not np.isfinite(poly).all():
            res.checks["valid_polygon"] = False
            res.warnings.append("Seat could not be located because transformed OCR coordinates are not finite.")
            logger.warning("Skipping seat polygon with non-finite transformed coordinates")
            return res
        res.checks["valid_polygon"] = True
        cx, cy = poly.mean(axis=0)
        chk, w = res.checks, res.warnings

        # ---- column -------------------------------------------------------
        ch = self._chain_at(cx, cy, float(poly[:, 0].min()), float(poly[:, 0].max()))
        if ch is None:
            w.append("No vertical grid lines were found around this number - the table grid could not be detected. "
                     "Please retake the photo (whole table visible, in focus, good light).")
            return res
        chk["six_columns"] = ch["total"] == self.n_cols
        valid_chain = (len(ch["lines"]) == self.n_cols + 1 and 0 <= ch["idx"] < self.n_cols)
        if not chk["six_columns"] or not valid_chain:
            w.append(f"Expected {self.n_cols} table columns but detected {ch['total']} "
                     f"{('(' + ch['note'] + ') ') if ch['note'] else ''}- cannot name the column reliably.")
            if not valid_chain:
                w.append("The detected grid boundaries were inconsistent; no column was assigned.")
            res.grid_confidence = 0.2
            return res
        lines, idx = ch["lines"], ch["idx"]
        if ch.get("inferred"):
            w.append("One outer wall of this table was not visible; its position was inferred from the evenly "
                     "spaced columns and the sheet's centred title - double-check the column letter.")
        key = self.table_alias.get(self._key(lines), self._key(lines))
        res.table_key = key
        tb = self.tables.get(key)
        res.column = COLUMN_LETTERS[idx]
        lL, lR = lines[idx], lines[idx + 1]
        tol = 0.3 * u
        chk["inside_column"] = bool(all(lL.x_at(y) - tol <= x <= lR.x_at(y) + tol for x, y in poly))
        if not chk["inside_column"]:
            w.append("The number's box crosses a column line - the box may belong to a neighbouring cell.")
        xl, xr = lL.x_at(cy), lR.x_at(cy)
        xc_cell, colw = (xl + xr) / 2, xr - xl

        # ---- row ----------------------------------------------------------
        top_hint = (tb.top if tb else cy) - 7 * u
        rl = self._row_lines(lL, lR, cy, top_hint, cy + 3 * u)
        ys = [self._y_at(l, cx) for l in rl]
        order = np.argsort(ys)
        rl, ys = [rl[i] for i in order], [ys[i] for i in order]
        j = next((k for k in range(len(ys) - 1) if ys[k] < cy <= ys[k + 1]), None)
        if j is None:
            chk["row_lines"] = False
            w.append("Could not find the horizontal grid lines above and below this number.")
        else:
            chk["row_lines"] = True
            ref = ys[j + 1] - ys[j]
            mid = (ys[j] + ys[j + 1]) / 2
            chk["inside_row"] = bool(abs(cy - mid) <= 0.3 * ref and poly[:, 1].min() >= ys[j] - 0.4 * u
                                     and poly[:, 1].max() <= ys[j + 1] + 0.4 * u)
            if not chk["inside_row"]:
                w.append("The number's box is not centred in its row - it may belong to a neighbouring row.")
            # walk UP: bands of the same height are data rows; the first clearly thinner band must be
            # the header row (one tiny letter); anything else = cannot verify
            hdr = None
            for k in range(j - 1, max(-1, j - 60), -1):
                hk = ys[k + 1] - ys[k]
                if abs(hk - ref) <= 0.25 * ref:
                    continue
                if hk < 0.85 * ref and self._is_header_cell(xc_cell, colw, ys[k], ys[k + 1]):
                    hdr = k
                break
            chk["header_found"] = hdr is not None
            if hdr is None:
                w.append("The column-header row (A-F) was not found directly above the first data row - "
                         "the row number cannot be verified.")
            else:
                res.row = j - hdr
                res.cell_quad = self._cell_quad(lL, lR, rl[j], rl[j + 1])

        # ---- room ---------------------------------------------------------
        if tb and tb.heading:
            res.room, res.heading = tb.heading.room, tb.heading
            chk["room_found"] = True
            if tb.heading_note:
                w.append("Room heading note: " + tb.heading_note)
        else:
            chk["room_found"] = False
            w.append("Room not determined: " + (tb.heading_note if tb else "table not recognised"))

        res.verified = all(chk.values()) and res.row is not None
        conf = 1.0
        for k, v in chk.items():
            if not v:
                conf *= 0.7 if k == "room_found" else 0.5
        if ch.get("inferred"):
            conf *= 0.8
        res.grid_confidence = round(conf, 3)
        return res

    def _cell_quad(self, lL: VLine, lR: VLine, top, bot) -> np.ndarray:
        """Cell corners TL,TR,BR,BL in WORKING-image coordinates."""
        def inter(vl: VLine, ln) -> Tuple[float, float]:
            xa, ya, xb, yb = ln
            m = (yb - ya) / (xb - xa + 1e-9)
            y = ya
            for _ in range(4):
                x = vl.x_at(y)
                y = ya + m * (x - xa)
            return (vl.x_at(y), y)
        q = np.array([inter(lL, top), inter(lR, top), inter(lR, bot), inter(lL, bot)], dtype=np.float64)
        return self.T.inverse().apply(q).astype(np.float32)

    # ------------------------------------------------------------- views
    def rectified_view(self, seat: SeatResult, poly: np.ndarray, n_rows: int = 2, n_cols: int = 1,
                       cell_px: Tuple[int, int] = (150, 64)) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Perspective-corrected neighbourhood of the seat (±n_rows, ±n_cols cells) cut from the
        working image.  Returns (warped image, warped cell polygon, warped number polygon)."""
        if seat.cell_quad is None:
            return None
        q = seat.cell_quad.astype(np.float64)
        ex, ey = q[1] - q[0], q[3] - q[0]
        big = np.array([q[0] - n_cols * ex - n_rows * ey, q[1] + n_cols * ex - n_rows * ey,
                        q[2] + n_cols * ex + n_rows * ey, q[3] - n_cols * ex + n_rows * ey], dtype=np.float32)
        W, Hh = cell_px[0] * (2 * n_cols + 1), cell_px[1] * (2 * n_rows + 1)
        warped, Hm = warp_quad(self.work, big, W, Hh)

        def tf(p):
            return cv2.perspectiveTransform(np.asarray(p, dtype=np.float64).reshape(-1, 1, 2), Hm).reshape(-1, 2)
        return warped, tf(q), tf(poly)

    def draw_layout(self, img: Optional[np.ndarray] = None) -> np.ndarray:
        """Debug picture on the working image: grid lines (blue), numbers (green), headings (magenta)."""
        vis = (self.work if img is None else img).copy()
        Ti = self.T.inverse()
        for L in self.vlines:
            p = Ti.apply(np.array([[L.x_at(L.y0), L.y0], [L.x_at(L.y1), L.y1]]))
            cv2.line(vis, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), (255, 80, 0), 2)
        for t in self.tokens:
            if t.kind == "full":
                draw_polygon(vis, Ti.apply(t.det.poly), (0, 170, 0), 1)
        for tb in self.tables.values():
            p = Ti.apply(np.array([[tb.x_left(tb.top), tb.top], [tb.x_right(tb.top), tb.top]]))
            label = tb.heading.room if tb.heading else "room ?"
            cv2.putText(vis, label, (int(p[0][0]) + 8, int(p[0][1]) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 140, 255), 2, cv2.LINE_AA)
        for h in self.headings:
            draw_polygon(vis, Ti.apply(h.det.poly), (200, 0, 200), 2, h.room)
        return vis
