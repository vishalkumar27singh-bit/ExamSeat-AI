"""
app.py - ExamSeat AI (Streamlit), multi-image version
Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_processing import bgr_to_rgb, load_image
from multi_search import ERROR, FOUND, NOT_FOUND, candidates_across, search_many
from ocr_engine import OCREngine
from pipeline import all_numbers_table, render_crop, render_highlight, render_seat_view

APP_VERSION = "multi-image 2026-10-19"      # shown on the page: proves WHICH copy of the app is running
ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(ROOT, "samples", "photo_001_clear.jpeg")

st.set_page_config(page_title="ExamSeat AI", page_icon="🪑", layout="wide")


@st.cache_resource(show_spinner="Loading OCR models (first time only)…")
def get_engine(backend: str) -> OCREngine:
    return OCREngine(backend)


@st.cache_data(show_spinner=False, max_entries=64)
def _thumb(data: bytes, side: int = 220):
    """Small preview; an unreadable file gives None (it is reported later, not fatal)."""
    try:
        im = load_image(data)
    except Exception:
        return None
    h, w = im.shape[:2]
    s = side / max(h, w)
    return bgr_to_rgb(cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA))


def fit(fn, *args, **kw):
    """Works on old and new Streamlit: width='stretch' (new) or use_container_width=True (old)."""
    try:
        return fn(*args, width="stretch", **kw)
    except TypeError:
        return fn(*args, use_container_width=True, **kw)


def _how(o) -> str:
    """debug text: was this photo OCR'd now, reused from the cache, or skipped because it is unreadable?"""
    return "(skipped: unreadable)" if o.status == ERROR else ("(OCR ran)" if o.ocr_ran else "(cached analysis)")


def level_box(level: str, text: str) -> None:
    {"HIGH": st.success, "MEDIUM": st.warning, "LOW": st.error}[level](text)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("OCR engine", ["auto", "paddle", "rapid"],
                           help="auto = PaddleOCR, falling back to RapidOCR (same model family) if Paddle cannot start.")
    auto_rotate = st.checkbox("Auto-rotate sideways photos", True)
    use_sample = st.checkbox("Also add the bundled sample photo", False, disabled=not os.path.exists(SAMPLE))
    show_debug = st.checkbox("Show debug info (temporary)", True,
                             help="Shows how many files were received and that each one was processed.")
    st.caption("Tip: photograph each whole table straight-on, in focus, without glare. "
               "Every photo is processed on its own.")

st.title("🪑 ExamSeat AI")
st.caption("Find an examination seat across several seating-chart photos - exact matching, verified highlighting, "
           "and honest warnings when something cannot be verified.")

st.caption(f"App version: **{APP_VERSION}** · running from `{os.path.abspath(__file__)}`")

# ---- upload: ALL files are kept in a list; nothing below ever uses only the first one -----------------
uploaded_files = st.file_uploader("Upload all seating chart photos", type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True, key="seat_photos")
sources = [(f.name, f.getvalue()) for f in (uploaded_files or [])]      # one (name, bytes) per uploaded file
if use_sample and os.path.exists(SAMPLE):
    sources.append(("sample_photo_001_clear.jpeg", open(SAMPLE, "rb").read()))
if not sources:
    st.info("Upload one or more photos (JPG / PNG) - or tick the bundled sample in the sidebar.")
    st.stop()

# files with the same name get a number so every result stays attributable
seen: dict = {}
uniq = []
for nm, data in sources:
    seen[nm] = seen.get(nm, 0) + 1
    uniq.append((nm if seen[nm] == 1 else f"{nm} ({seen[nm]})", data))
sources = uniq

st.subheader(f"{len(sources)} photo(s) uploaded")
st.write("Files: " + ", ".join(f"**{i + 1}. {nm}**" for i, (nm, _) in enumerate(sources)))
cols = st.columns(min(len(sources), 6))
for i, (nm, data) in enumerate(sources):
    th = _thumb(data)
    with cols[i % len(cols)]:
        if th is None:
            st.error(f"⚠ {nm}\nnot a readable image")
        else:
            st.image(th, caption=nm)

if show_debug:
    with st.expander("🛠 Debug: files received (temporary)", expanded=True):
        st.write(f"`len(uploaded_files)` = **{len(uploaded_files or [])}** · bundled sample added = **{bool(use_sample)}** · "
                 f"photos that will be searched = **{len(sources)}**")
        import hashlib
        st.dataframe([dict(n=i + 1, filename=nm, bytes=len(d), sha1=hashlib.sha1(d).hexdigest()[:10])
                      for i, (nm, d) in enumerate(sources)])

st.divider()
if "confirmed" not in st.session_state:
    st.session_state.confirmed = ""
reg = st.text_input("Registration number", value=st.session_state.confirmed, max_chars=20,
                    placeholder="10-digit registration number")
go = st.button("Search all photos", type="primary")
if st.session_state.confirmed and reg == st.session_state.confirmed:
    go = True

cache = st.session_state.setdefault("ocr_cache", {})        # OCR runs once per photo, many searches
if reg and (go or st.session_state.get("last_batch_reg") == reg):
    engine = get_engine(backend)
    if engine.fallback_reason:
        st.warning(f"OCR engine in use: **{engine.name}**. {engine.fallback_reason}")
    keyset = (tuple((n, len(d)) for n, d in sources), reg, backend, auto_rotate)
    if st.session_state.get("batch_key") != keyset:
        bar = st.progress(0.0, text="Starting…")
        log = st.empty()
        lines: list = []

        def on_progress(i, n, name, state):
            bar.progress(i / n, text=f"{i}/{n} photos processed")
            if state == "reading":
                lines.append(f"⏳ {name}: reading…")
            else:
                lines[-1] = {"done": f"✅ {name}: done", "error": f"⚠️ {name}: could not be processed"}[state]
            log.markdown("\n\n".join(lines))

        st.session_state.batch = search_many(sources, reg, engine, cache=cache, auto_rotate=auto_rotate,
                                             progress=on_progress)
        st.session_state.batch_key = keyset
        st.session_state.last_batch_reg = reg
        bar.empty()
        log.empty()
    batch = st.session_state.batch

    if batch.status == "INVALID_QUERY":
        st.error(batch.message)
    else:
        n_err = sum(o.status == ERROR for o in batch.outcomes)
        st.subheader(f"Search status - {len(batch.outcomes)} of {len(sources)} photo(s) searched for {batch.query}")
        label = {FOUND: "✅ Found", NOT_FOUND: "➖ Not found in this photo", ERROR: "⚠️ Could not be processed"}
        st.dataframe([dict(photo=o.name, status=label[o.status], numbers_read=o.n_numbers_read,
                           matches=len(o.matches), seconds=o.seconds) for o in batch.outcomes])
        if show_debug:
            msg = (f"[ExamSeat debug] uploaded={len(uploaded_files or [])} searched={len(batch.outcomes)} "
                   + " | ".join(f"{o.name}:{o.status}{_how(o)}" for o in batch.outcomes))
            print(msg, file=sys.stderr, flush=True)
            st.caption("🛠 " + " · ".join(f"{o.name}: {o.status.lower()} {_how(o)}" for o in batch.outcomes))
        if batch.found:
            names = ", ".join(f"**{o.name}**" for o in batch.found)
            st.success(f"Found in {len(batch.found)} of {len(batch.outcomes)} photo(s): {names}"
                       + (f"  ({n_err} photo(s) could not be processed.)" if n_err else ""))
        else:
            st.error(f"Not found in any uploaded photo - searched {len(batch.outcomes) - n_err} readable photo(s). "
                     "Registration Number Not Found."
                     + (f"  {n_err} photo(s) could not be processed - the number might be in one of them." if n_err else ""))
            cands = candidates_across(batch)
            if cands:
                st.caption("Similar numbers were read. Nothing below is reported as your seat until you confirm it.")
                for i, (nm, c) in enumerate(cands):
                    a, b, cc = st.columns([2, 5, 2])
                    a.code(c.text)
                    b.write(f"in **{nm}** · {c.reason} · confidence {c.confidence:.2f}")
                    if cc.button("That is me", key=f"cand{i}"):
                        st.session_state.confirmed = c.text
                        st.rerun()

        best = batch.best
        if best is not None:
            bo, bm = best
            st.info(f"**Best verified match:** {bm.reg_no} · room **{bm.room}** · column **{bm.column}** · row **{bm.row}** "
                    f"· photo *{bo.name}* · confidence {bm.confidence:.2f}")
        elif len([m for o in batch.outcomes for m in o.matches]) > 1:
            st.warning("The number was found more than once and the results do not agree on one verified seat - "
                       "every match is listed below; check which one is yours.")

        df = batch.table()
        st.subheader("Summary")
        fit(st.dataframe, df)
        st.download_button("Download summary (CSV)", df.to_csv(index=False), f"seat_search_{batch.query}.csv", "text/csv")

        for oi, o in enumerate(batch.outcomes):
            icon = {"FOUND": "✅", "NOT_FOUND": "➖", "ERROR": "⚠️"}[o.status]
            with st.expander(f"{icon} {o.name} - {o.status.replace('_', ' ').title()} ({o.seconds}s)",
                             expanded=(o.status == FOUND)):
                if o.status == ERROR:
                    st.error(f"This photo could not be processed and was skipped: {o.message}")
                    continue
                for f_ in o.flags:
                    st.warning(f_)
                if o.status == NOT_FOUND:
                    st.write(f"Registration Number Not Found in this photo ({o.n_numbers_read} text lines were read).")
                for k, m in enumerate(o.matches):
                    st.markdown(f"**📄 {o.name} — Match {k + 1}**" if len(o.matches) > 1 else f"**📄 {o.name} — Result**")
                    if m.level == "HIGH":
                        level_box("HIGH", f"Found and verified - confidence {m.confidence:.2f} (high)")
                    elif m.level == "MEDIUM":
                        level_box("MEDIUM", f"Found, but check the details below - confidence {m.confidence:.2f}")
                    else:
                        level_box("LOW", f"The number was read, but its seat could NOT be verified - confidence {m.confidence:.2f}")
                    k0, k1, k2, k3, k4 = st.columns([2, 2, 1, 1, 2])
                    k0.metric("Registration no.", m.reg_no)
                    ver = bool(m.seat.verified)        # room / column / row only when the seat is verified
                    k1.metric("Room", (m.room or "unknown") if ver else "not verified")
                    k2.metric("Column", (m.column or "unknown") if ver else "not verified")
                    k3.metric("Row", (m.row if m.row else "unknown") if ver else "not verified")
                    k4.metric("OCR confidence", f"{m.ocr_score:.2f}")
                    st.caption(f"Photo: **{o.name}** · re-read of the highlighted pixels: **{m.verify.reread_text or '-'}** → "
                               + ("matches ✅" if m.verify.ok else "DOES NOT MATCH ❌"))
                    for w in m.warnings:
                        st.warning(w)
                    a, b = st.columns([3, 2])
                    hi = render_highlight(o.analysis, m)
                    fit(a.image, bgr_to_rgb(hi), caption="Highlighted on the original photo")
                    with b:
                        st.image(bgr_to_rgb(render_crop(o.analysis, m)), caption="Crop of the highlighted box")
                        sv = render_seat_view(o.analysis, m)
                        if sv is not None:
                            st.image(bgr_to_rgb(sv), caption="Perspective-corrected view: blue = seat cell, green = number")
                    ok, buf = cv2.imencode(".jpg", hi)
                    st.download_button("Download highlighted image", buf.tobytes(), f"seat_{m.reg_no}_{oi}_{k}.jpg",
                                       "image/jpeg", key=f"dl{oi}_{k}")
                    with st.expander("How was this decided?"):
                        st.write({k_: bool(v) for k_, v in m.seat.checks.items()})
                        st.write(f"OCR score {m.ocr_score:.2f} · registration confidence {m.reg_confidence:.2f} · "
                                 f"seat-grid confidence {m.seat_confidence:.2f}")
                        st.write("Polygon in original-photo pixels:", np.round(m.poly_original).astype(int).tolist())

        ok_out = [o for o in batch.outcomes if o.analysis is not None]
        if ok_out:
            with st.expander("Developer view: grid detection and every number read"):
                pick = st.selectbox("Photo", [o.name for o in ok_out])
                an = next(o.analysis for o in ok_out if o.name == pick)
                fit(st.image, bgr_to_rgb(an.locator.draw_layout()),
                    caption="blue = detected grid lines, green = numbers, magenta = room headings")
                tbl = all_numbers_table(an)
                fit(st.dataframe, tbl, height=300)
                st.download_button("Download table (CSV)", tbl.to_csv(index=False), "all_numbers.csv", "text/csv")
else:
    st.caption("Enter a registration number and press **Search all photos**.")
