"""
multi_search.py
===============
Search ONE registration number across MANY seating-chart photos.

Every photo is processed independently by pipeline.analyze / pipeline.search (OCR, grid, verification),
so a broken, blank or unreadable file only produces an ERROR row for that file; the others go on.
Nothing is invented: a photo is reported FOUND only when the number was read exactly in it.

    engine = OCREngine("auto")
    cache  = {}                                   # keep between searches: OCR runs once per photo
    batch  = search_many([("a.jpg", bytes_a), ("b.jpg", bytes_b)], "<registration-number>", engine, cache=cache)
    for o in batch.outcomes: print(o.name, o.status, [(m.room, m.column, m.row) for m in o.matches])

Command line:  python multi_search.py "<registration-number>" photo1.jpg photo2.jpg ...
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from image_processing import load_image
from ocr_engine import Candidate, OCREngine, normalize_query
from pipeline import Analysis, MatchReport, analyze, search

Source = Tuple[str, Union[bytes, str]]          # (display name, file bytes or a path)

FOUND, NOT_FOUND, ERROR = "FOUND", "NOT_FOUND", "ERROR"


@dataclass
class ImageOutcome:
    name: str
    status: str                                  # FOUND | NOT_FOUND | ERROR
    message: str = ""
    matches: List[MatchReport] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    analysis: Optional[Analysis] = None
    fingerprint: str = ""
    seconds: float = 0.0
    n_numbers_read: int = 0
    flags: List[str] = field(default_factory=list)   # cross-image warnings (duplicates ...)
    ocr_ran: bool = False                            # True = this call ran OCR; False = cached analysis reused


@dataclass
class BatchResult:
    query: Optional[str]
    outcomes: List[ImageOutcome]
    message: str = ""

    @property
    def found(self) -> List[ImageOutcome]:
        return [o for o in self.outcomes if o.status == FOUND]

    @property
    def best(self) -> Optional[Tuple[ImageOutcome, MatchReport]]:
        """The one verified match to show first - only when all verified appearances agree on the seat.
        Otherwise None: the caller must show every match and let the person decide."""
        ok = [(o, m) for o in self.outcomes if o.status == FOUND for m in o.matches if m.seat.verified and m.verify.ok]
        if ok and len({(m.room, m.column, m.row) for _, m in ok}) == 1:
            return max(ok, key=lambda t: t[1].confidence)
        return None

    @property
    def status(self) -> str:
        if self.query is None:
            return "INVALID_QUERY"
        return FOUND if self.found else NOT_FOUND

    def table(self) -> pd.DataFrame:
        """One row per match (and one row per photo without a match) - nothing is filled in by guessing."""
        rows = []
        for o in self.outcomes:
            if o.matches:
                for m in o.matches:
                    v = bool(m.seat.verified)            # room / column / row are shown ONLY when verified
                    rows.append(dict(image=o.name, status=o.status, reg_no=m.reg_no, room=m.room if v else None,
                                     column=m.column if v else None, row=m.row if v else None, seat_verified=v, level=m.level, confidence=m.confidence,
                                     ocr_score=round(m.ocr_score, 3), reread_ok=m.verify.ok,
                                     flags="; ".join(o.flags)))
            else:
                rows.append(dict(image=o.name, status=o.status, reg_no=None, room=None, column=None, row=None,
                                 seat_verified=None, level=None, confidence=None, ocr_score=None, reread_ok=None,
                                 flags="; ".join(o.flags + ([o.message] if o.status == ERROR else []))))
        df = pd.DataFrame(rows)
        if "row" in df:
            df["row"] = df["row"].astype("Int64")
        return df


def _read_bytes(src: Union[bytes, str]) -> bytes:
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    with open(src, "rb") as fh:
        return fh.read()


_last_ocr_ran: List[str] = []          # names for which OCR actually ran in the current search_many call


def _analysis_for(name: str, data: bytes, engine: OCREngine, cache: Optional[dict], auto_rotate: bool) -> Tuple[Analysis, str]:
    """OCR once per (file content, engine); later searches reuse it."""
    fp = hashlib.sha1(data).hexdigest()[:16]
    key = (fp, engine.name, auto_rotate)
    if cache is not None and key in cache:
        return cache[key], fp
    _last_ocr_ran.append(name)
    img = load_image(data)                       # raises on a file that is not an image
    an = analyze(img, engine, auto_rotate=auto_rotate)
    if cache is not None:
        cache[key] = an
    return an, fp


def search_many(sources: Sequence[Source], query_raw: str, engine: OCREngine,
                cache: Optional[dict] = None, auto_rotate: bool = True,
                progress: Optional[Callable[[int, int, str, str], None]] = None) -> BatchResult:
    """Search one registration number in every photo.  `progress(i, n, name, state)` is called before
    ('reading') and after ('done' / 'error') each photo."""
    q, msg = normalize_query(query_raw)
    if q is None:
        return BatchResult(None, [], msg)
    outcomes: List[ImageOutcome] = []
    n = len(sources)
    _last_ocr_ran.clear()
    for i, (name, src) in enumerate(sources):
        t0 = time.time()
        if progress:
            progress(i, n, name, "reading")
        try:
            data = _read_bytes(src)
            if not data:
                raise ValueError("the file is empty")
            before = len(_last_ocr_ran)
            an, fp = _analysis_for(name, data, engine, cache, auto_rotate)
            ran = len(_last_ocr_ran) > before
            res = search(an, engine, q)
            out = ImageOutcome(name, FOUND if res.status == "FOUND" else NOT_FOUND, res.message, res.matches,
                               res.candidates, an, fp, 0.0, sum(1 for d in an.detections if d.text))
            out.ocr_ran = ran
            if res.status != "FOUND":
                out.message = "Registration Number Not Found"
        except Exception as e:                    # one bad photo must never stop the others
            out = ImageOutcome(name, ERROR, f"{type(e).__name__}: {e}")
            if os.environ.get("EXAMSEAT_DEBUG"):
                traceback.print_exc()
        out.seconds = round(time.time() - t0, 2)
        outcomes.append(out)
        if progress:
            progress(i + 1, n, name, "error" if out.status == ERROR else "done")
    _flag_duplicates(outcomes, q)
    return BatchResult(q, outcomes, msg)


def _flag_duplicates(outcomes: List[ImageOutcome], q: str) -> None:
    """Cross-image checks.  The same number in several photos is shown, never silently merged."""
    # 1. identical files uploaded twice
    seen: Dict[str, str] = {}
    for o in outcomes:
        if o.fingerprint:
            if o.fingerprint in seen:
                o.flags.append(f"identical file to '{seen[o.fingerprint]}' (uploaded twice?)")
            else:
                seen[o.fingerprint] = o.name
    # 2. the number appears more than once.  Overlapping photos of neighbouring sheets make this common,
    #    so say exactly what is known instead of a vague warning.
    appearances = [(o, m) for o in outcomes if o.status == FOUND for m in o.matches]
    total = len(appearances)
    if total <= 1:
        return
    ok = [(o, m) for o, m in appearances if m.seat.verified and m.verify.ok]
    seats = {(m.room, m.column, m.row) for _, m in ok}
    n_photos = len({o.name for o, _ in appearances})
    head = f"possible duplicate: {q} was found {total} times in {n_photos} photo(s)"
    if not ok:
        msg = head + " - none of them has a verified seat, so none can be trusted yet"
    elif len(seats) == 1:
        room, col, row = next(iter(seats))
        rest = total - len(ok)
        msg = (head + f" - verified at the same seat ({room} {col}{row}) in {len(ok)} appearance(s): the same sheet "
               "photographed more than once or overlapping photos"
               + (f"; the other {rest} appearance(s) are not verified (probably a cut-off view)" if rest else ""))
    else:
        msg = (head + " - CONFLICT: verified at different seats (" + ", ".join(sorted(f"{r} {c}{w}" for r, c, w in seats))
               + "). Two sheets may print the same number; check the paper")
    for o in {id(o): o for o, _ in appearances}.values():
        o.flags.append(msg)


def candidates_across(batch: BatchResult, limit: int = 6) -> List[Tuple[str, Candidate]]:
    """Near-misses (never reported as matches) from photos where the number was not found exactly."""
    allc = [(o.name, c) for o in batch.outcomes for c in o.candidates]
    allc.sort(key=lambda t: -t[1].confidence)
    return allc[:limit]


# ==========================================================================
def _main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Find a registration number in several photos.")
    ap.add_argument("reg_no")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args(argv)
    eng = OCREngine(a.backend)
    print(f"OCR engine: {eng.name}")
    b = search_many([(os.path.basename(p), p) for p in a.images], a.reg_no, eng,
                    progress=lambda i, n, nm, st: print(f"  [{i}/{n}] {nm}: {st}"))
    if b.status == "INVALID_QUERY":
        print(b.message)
        return 2
    print(b.table().to_string(index=False))
    if not b.found:
        print("Registration Number Not Found")
    return 0 if b.found else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
