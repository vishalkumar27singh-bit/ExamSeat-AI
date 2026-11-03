"""
pipeline.py
===========
Glue code: photo -> OCR -> search -> verify -> room/row/column -> highlight.

Typical use
-----------
    engine   = OCREngine("auto")
    analysis = analyze(load_image("photo.jpg"), engine)      # slow part, once per photo
    result   = search(analysis, engine, "<registration-number>")  # fast, once per query
    img      = render_highlight(analysis, result.matches[0])

Command line:   python pipeline.py photo.jpg "<registration-number>"
"""
from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd

from image_processing import (Transform, bgr_to_rgb, crop_polygon, draw_polygon, enhance, image_fingerprint,
                              image_quality, limit_size, load_image, rotate_quarter_turns)
from ocr_engine import (Candidate, Detection, Match, OCREngine, VerifyResult, extract_tokens, find_exact,
                        merge_detections, normalize_query, rank_candidates, verify_polygon)
from seat_locator import SeatLocator, SeatResult, find_room_headings

logger = logging.getLogger(__name__)

# ==========================================================================
# Analysis of one photo (everything that does not depend on the query)
# ==========================================================================
@dataclass
class Analysis:
    original: np.ndarray                 # exactly what the user uploaded (EXIF-rotated)
    work: np.ndarray                     # image the OCR actually ran on (maybe rescaled / rotated)
    to_work: Transform                   # original -> work coordinates
    detections: List[Detection]          # in WORK coordinates
    engine_name: str
    engine_note: Optional[str]
    quality: Dict[str, object]
    notes: List[str] = field(default_factory=list)
    enhanced_done: bool = False
    fingerprint: str = ""
    _locator: Optional[SeatLocator] = None

    @property
    def locator(self) -> SeatLocator:
        if self._locator is None:
            self._locator = SeatLocator(self.work, self.detections)
        return self._locator

    def poly_to_original(self, poly: np.ndarray) -> np.ndarray:
        return self.to_work.inverse().apply(np.asarray(poly, dtype=np.float64)).astype(np.float32)

    def add_detections(self, dets: Sequence[Detection]) -> None:
        self.detections = merge_detections([self.detections, dets])
        self._locator = None  # layout must be rebuilt with the new detections


def _n_full(dets: Sequence[Detection]) -> int:
    return sum(t.kind == "full" for t in extract_tokens(dets))


def analyze(img_bgr: np.ndarray, engine: OCREngine, auto_rotate: bool = True, max_side: int = 3600) -> Analysis:
    """Run the one-time, query-independent work on a photo."""
    notes: List[str] = []
    img_bgr = load_image(img_bgr)
    work, s = limit_size(img_bgr, max_side)
    T = Transform.scale(s)
    if s != 1.0:
        notes.append(f"Large photo scaled by {s:.2f} for OCR (highlight is mapped back to the original).")
    dets = engine.run(work, "orig")
    n = _n_full(dets)
    if auto_rotate and n < 6:
        best = (n, 0, work, dets, Transform.identity())
        for k in (1, 3, 2):
            rot, Tr = rotate_quarter_turns(work, k)
            d = engine.run(rot, f"rot{90 * k}")
            nk = _n_full(d)
            if nk > best[0] + 5:
                best = (nk, k, rot, d, Tr)
        if best[1] != 0:
            n, k, work, dets, Tr = best
            T = T.then(Tr)
            notes.append(f"Photo was rotated by {90 * k} degrees automatically (found {n} registration numbers).")
    analysis = Analysis(original=img_bgr, work=work, to_work=T, detections=dets, engine_name=engine.name,
                        engine_note=engine.fallback_reason, quality=image_quality(img_bgr), notes=notes,
                        fingerprint=image_fingerprint(img_bgr))
    quality = analysis.quality
    if (quality["blur"] < 60 or quality["contrast"] < 35 or quality["brightness"] < 80
            or quality["brightness"] > 215 or min(work.shape[:2]) < 900):
        add_enhanced_pass(analysis, engine)
    return analysis


def add_enhanced_pass(analysis: Analysis, engine: OCREngine) -> None:
    """Compare a contrast-enhanced OCR pass and keep it only when its evidence improves."""
    if analysis.enhanced_done:
        return
    enh = enhance(analysis.work, "clahe+sharpen")
    if enh.shape != analysis.work.shape:
        raise ValueError("enhancement changed image dimensions; OCR coordinates would no longer align")
    candidate = engine.run(enh, "enhanced")
    analysis.enhanced_done = True
    base_tokens = [t for t in extract_tokens(analysis.detections) if t.kind == "full"]
    candidate_tokens = [t for t in extract_tokens(candidate) if t.kind == "full"]
    base_rooms = len(find_room_headings(analysis.detections))
    candidate_rooms = len(find_room_headings(candidate))
    base_mean = float(np.mean([t.det.score for t in base_tokens])) if base_tokens else 0.0
    candidate_mean = float(np.mean([t.det.score for t in candidate_tokens])) if candidate_tokens else 0.0
    improved = (len(candidate_tokens) > len(base_tokens)
                or candidate_rooms > base_rooms
                or (len(candidate_tokens) == len(base_tokens) and candidate_rooms == base_rooms
                    and candidate_mean >= base_mean + 0.02))
    if improved:
        analysis.add_detections(candidate)
        analysis.notes.append(
            "Contrast-enhanced OCR was retained after it improved roll-number or room-heading evidence."
        )
    else:
        analysis.notes.append(
            "Contrast-enhanced OCR was compared but did not improve roll-number or room-heading evidence; "
            "the original OCR results were retained."
        )


# ==========================================================================
# Search
# ==========================================================================
@dataclass
class MatchReport:
    reg_no: str
    kind: str                            # exact | exact_in_merged
    poly_work: np.ndarray
    poly_original: np.ndarray
    ocr_score: float
    verify: VerifyResult
    seat: SeatResult
    reg_confidence: float
    seat_confidence: float
    confidence: float
    level: str                           # HIGH | MEDIUM | LOW
    warnings: List[str]

    @property
    def room(self): return self.seat.room
    @property
    def column(self): return self.seat.column
    @property
    def row(self): return self.seat.row


@dataclass
class SearchResult:
    status: str                          # FOUND | NOT_FOUND | INVALID_QUERY
    query: Optional[str]
    message: str = ""
    matches: List[MatchReport] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)


def _report(analysis: Analysis, engine: OCREngine, m: Match, query: str) -> MatchReport:
    work = analysis.work
    ver = verify_polygon(engine, work, m.poly, query)
    # locate() needs the detection for its default polygon; we pass the (possibly sub-) polygon explicitly
    seat = analysis.locator.locate(m.poly)
    reg_conf = float(np.sqrt(max(m.score, 1e-3) * max(ver.reread_score, 1e-3))) if ver.ok else 0.4 * m.score
    if m.kind == "exact_in_merged":
        reg_conf *= 0.9
    conf = round(reg_conf * seat.grid_confidence, 3)
    level = "HIGH" if (ver.ok and seat.verified and conf >= 0.7) else ("MEDIUM" if conf >= 0.45 and ver.ok else "LOW")
    warns: List[str] = []
    if not ver.ok:
        warns.append("Highlight could not be verified: " + (ver.note or "re-read failed") +
                     ". Do not rely on this box.")
    if m.kind == "exact_in_merged":
        warns.append("OCR read this number glued to its neighbour; the box was split by position - double-check it.")
    warns += seat.warnings
    if analysis.engine_note:
        warns.append(analysis.engine_note)
    warns += list(analysis.quality.get("warnings", []))
    return MatchReport(query, m.kind, m.poly, analysis.poly_to_original(m.poly), m.score, ver, seat,
                       round(reg_conf, 3), seat.grid_confidence, conf, level, warns)


def search(analysis: Analysis, engine: OCREngine, query_raw: str, retry_enhanced: bool = True) -> SearchResult:
    q, msg = normalize_query(query_raw)
    if q is None:
        return SearchResult("INVALID_QUERY", None, msg)
    matches = find_exact(extract_tokens(analysis.detections), q)
    if not matches and retry_enhanced and not analysis.enhanced_done:
        add_enhanced_pass(analysis, engine)
        matches = find_exact(extract_tokens(analysis.detections), q)
    if matches:
        reports = [_report(analysis, engine, m, q) for m in matches]
        # room heading unreadable (blur, glare)? try once more on the enhanced copy
        if retry_enhanced and not analysis.enhanced_done and any(not r.seat.checks.get("room_found", False) for r in reports):
            add_enhanced_pass(analysis, engine)
            matches = find_exact(extract_tokens(analysis.detections), q) or matches
            reports = [_report(analysis, engine, m, q) for m in matches]
        if len(reports) > 1:
            for r in reports:
                r.warnings.insert(0, f"This number appears {len(reports)} times in the photo - check which one is yours.")
        return SearchResult("FOUND", q, msg, reports)
    cands = rank_candidates(extract_tokens(analysis.detections), q)
    text = (f"'{q}' was NOT found exactly in this photo. " +
            ("These similar numbers were read - please confirm one explicitly:" if cands
             else "No similar number was read either - try a sharper photo."))
    return SearchResult("NOT_FOUND", q, text, [], cands)


# ==========================================================================
# Rendering
# ==========================================================================
def render_highlight(analysis: Analysis, rep: MatchReport, color=(0, 200, 0)) -> np.ndarray:
    """Original photo with the matched number's polygon drawn on it."""
    img = analysis.original.copy()
    th = max(3, int(round(img.shape[1] / 400)))
    label = f"{rep.reg_no}" + (f"  {rep.room} {rep.column}{rep.row}" if rep.seat.verified else "")
    return draw_polygon(img, rep.poly_original, color, th, label)


def render_crop(analysis: Analysis, rep: MatchReport, margin: float = 0.6) -> np.ndarray:
    crop, _ = crop_polygon(analysis.original, rep.poly_original, margin=margin)
    return crop


def render_seat_view(analysis: Analysis, rep: MatchReport) -> Optional[np.ndarray]:
    """Perspective-corrected neighbourhood of the seat with the cell and number outlined."""
    rv = analysis.locator.rectified_view(rep.seat, rep.poly_work)
    if rv is None:
        return None
    warped, cell, num = rv
    out = warped.copy()
    cv2.polylines(out, [np.round(cell).astype(np.int32).reshape(-1, 1, 2)], True, (255, 120, 0), 3, cv2.LINE_AA)
    cv2.polylines(out, [np.round(num).astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 0), 2, cv2.LINE_AA)
    return out


def all_numbers_table(analysis: Analysis) -> pd.DataFrame:
    """Every 10-digit number read in the photo with its seat (for export / debugging)."""
    rows = []
    for t in extract_tokens(analysis.detections):
        if t.kind != "full":
            continue
        s = analysis.locator.locate(t.det)
        rows.append(dict(reg_no=t.digits, letters_converted=t.substituted, ocr_score=round(t.det.score, 3),
                         room=s.room, column=s.column, row=s.row, seat_verified=s.verified,
                         x=int(t.det.center[0]), y=int(t.det.center[1]), warnings="; ".join(s.warnings)))
    columns = ["reg_no", "letters_converted", "ocr_score", "room", "column", "row", "seat_verified",
               "x", "y", "warnings"]
    return (pd.DataFrame(rows, columns=columns)
            .sort_values(["room", "row", "column"], na_position="last")
            .reset_index(drop=True))


# ==========================================================================
def _main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Find a registration number's room and seat in a photo.")
    ap.add_argument("image")
    ap.add_argument("reg_no")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    eng = OCREngine(a.backend)
    img = load_image(a.image)
    an = analyze(img, eng)
    print(f"OCR engine: {eng.name} | image {img.shape[1]}x{img.shape[0]} | {len(an.detections)} text lines | notes: {an.notes}")
    r = search(an, eng, a.reg_no)
    print(r.status, "-", r.message)
    for m in r.matches:
        print(f"  {m.reg_no}  room={m.room}  column={m.column}  row={m.row}  confidence={m.confidence} ({m.level})")
        print(f"  polygon (original photo coordinates): {np.round(m.poly_original).astype(int).tolist()}")
        print(f"  re-read of the highlighted pixels: '{m.verify.reread_text}'  verified={m.verify.ok}")
        for w in m.warnings:
            print("  WARNING:", w)
        cv2.imwrite(os.path.join(a.out, f"result_{m.reg_no}.jpg"), render_highlight(an, m))
        cv2.imwrite(os.path.join(a.out, f"crop_{m.reg_no}.png"), render_crop(an, m))
        sv = render_seat_view(an, m)
        if sv is not None:
            cv2.imwrite(os.path.join(a.out, f"seatview_{m.reg_no}.png"), sv)
    for c in r.candidates:
        print(f"  candidate {c.text} dist={c.distance} confidence={c.confidence} ({c.reason})")
    return 0 if r.status == "FOUND" else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
