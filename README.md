# ExamSeat AI — Multi-Image Examination Seat Finder

Upload **one or many** phone photos of university sitting charts, type one registration number, and the app searches **every photo independently**. For each match it shows the photo name, **room, seat column (A–F), seat row, OCR confidence, verification status and the number highlighted on the original photo** — or "Registration Number Not Found", or a clear warning when something can't be verified. One unreadable photo never stops the others.

```bash
streamlit run app.py                                      # multi-photo app
python multi_search.py "<registration-number>" a.jpg b.jpg c.jpg  # same search from the terminal
python -m pytest -q                                       # full regression suite
```
Setup for macOS: `docs/SETUP_MACOS.md` · Architecture: `docs/architecture.png` · Report: `docs/PROJECT_REPORT.md` · Viva: `docs/VIVA_QUESTIONS.md` · Custom ML: `ml/ANNOTATION_GUIDE.md`

## Public deployment

The app is a single Streamlit service; it has no separate API or database. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for deployment to Streamlit Community Cloud,
upload limits, privacy notes, and verification steps.

## 0. What was actually tested (read this first)

| Item | Status |
|---|---|
| Automated regression tests | 50 tests passed locally on Python 3.12.14, including exact matching, invalid OCR geometry, image validation, room labels, tilted-photo mapping, and multi-image handling |
| OCR robustness comparison | On the bundled synthetic blurry image, the original RapidOCR pass found fewer full roll tokens than the accepted contrast-enhanced pass; the test checks a relative gain, not an accuracy percentage |
| Full ground-truth evaluation | The older numbers below are historical and have not been recomputed for these changes; collect and verify local ground truth before quoting current accuracy |
| YOLO tooling (`ml/`) | Smoke-tested only (1 epoch from scratch, meaningless weights, deleted). **No model has been trained**: only one annotated real photo exists (see `ml/ANNOTATION_GUIDE.md`) |
| Local walkthrough notebook | Not included because it contains embedded exam data |
| OCR used for those runs | **RapidOCR** (PP-OCR models through ONNX). PaddleOCR's models could not be downloaded in the build sandbox |
| PaddleOCR code path (`_PaddleBackend`) | Written against PaddleOCR 3.7's source code and tested only with a **mock** that mimics its result structure. **Run `diagnose_coordinates.py` on your Mac first** — it exercises the real thing |
| Accuracy numbers | Local evaluation data is not included; collect and verify your own ground truth before quoting accuracy |

## 1. The coordinate bug — root cause

PaddleOCR 3.x, created with plain `PaddleOCR()`, **turns on a document pre-processor by default** (`OCR.yaml`: `use_doc_orientation_classify: True`, `use_doc_unwarping: True` → the UVDoc model). The polygons `rec_polys` / `rec_boxes` are then in the coordinates of the **pre-processed (unwarped) image**, `result["doc_preprocessor_res"]["output_img"]`, not of the file you loaded with OpenCV. Drawing them on the original photo boxes a neighbour — exactly your symptom (right column, wrong row, crop showing half of another number).

Evidence gathered:
* On the original test photo, a highlighted polygon was displaced vertically and landed across neighbouring entries. A width/height swap or uniform resize did not explain the error; geometric remapping from document unwarping did.
* It is an exact axis-aligned rectangle, i.e. `rec_boxes`-style output.
* Second trap in the same library: `dt_polys` is a **separate, unfiltered** list; only `rec_texts/rec_scores/rec_polys/rec_boxes` are built together. Mixing them shifts boxes onto other numbers.

**Fix (in `ocr_engine.py`)**: create PaddleOCR with `use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False`; read only `rec_texts/rec_scores/rec_polys`; raise an error if Paddle reports it processed a different image size or used unwarping; and **re-read every highlighted polygon** (`verify_polygon`) so a wrong box can never be shown silently.

A separate tilted-photo crash was reproduced in the bundled sample. `_complete_chain()` could report six columns after inferring multiple missing walls while inserting only one wall. The locator now inserts each inferred wall, checks that the line count and column index agree before indexing, and returns an unverified result for malformed OCR polygons rather than failing the page. The Cloud `TypeError` excerpt supplied with the report did not include enough frames or values to reproduce that exact exception, so it is not claimed to be the same failure.

**Confirm locally** (with PaddleOCR installed):
```bash
cd examseat_ai
python diagnose_coordinates.py "<photo>" "<registration-number>"
```
It prints image facts, the config PaddleOCR used, list lengths, and crops of every candidate polygon (saved in `diagnostics/`) re-read by OCR, then a verdict.

## 2. Setup on macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

RapidOCR is the hosted default. To use PaddleOCR locally, additionally install
`paddlepaddle>=3.0` and `paddleocr>=3.0`. See [`docs/SETUP_MACOS.md`](docs/SETUP_MACOS.md).

## 3. Files

| File | Purpose |
|---|---|
| `diagnose_coordinates.py` | Stage 1: proves why a box is wrong |
| `image_processing.py` | loading (EXIF), quality checks, enhancement, rotation/perspective helpers, cropping/drawing |
| `ocr_engine.py` | OCR wrapper with coordinate guards, exact search, candidates, polygon verification |
| `seat_locator.py` | grid lines → column/row; room heading association |
| `pipeline.py` | glue; `python pipeline.py photo.jpg "<registration-number>"` |
| `multi_search.py` | **many photos, one number**: per-photo error isolation, OCR cache, duplicate/conflict flags |
| `app.py` | Streamlit interface (multi-file upload, thumbnails, per-photo progress, results, editable review/export table) |
| `tests/` | pytest: matching, coordinate transforms, multi-image behavior, and malformed-input/tilted-photo regressions |
| `ml/` | optional YOLO table-region tooling: dataset builder, trainer, detector, evaluator, annotation guide |
| `docs/` | architecture diagram, project report, viva questions, macOS setup |
| `evaluation.py`, `make_variants.py` | metrics, synthetic photo conditions |
| Local evaluation data | Excluded from the public repository because it contains exam photos and student identifiers |
| Local notebook | Not included in this repository because it contains embedded exam data |

## 4. Running it, stage by stage

```bash
python diagnose_coordinates.py "<photo>" "<registration-number>" # 1. coordinate diagnosis
python ocr_engine.py "<photo>" "<registration-number>"           # 2-3. OCR boxes, exact search, crop + highlight in outputs/
python pipeline.py "<photo>" "<registration-number>"             # 4-6. + room, column, row, confidence
streamlit run app.py                                    # 7. web app (opens in your browser)
python evaluation.py                                    # 8. metrics on ground_truth/*.json
```
The local walkthrough notebook is not included because it contains embedded exam data.
The evaluation command requires locally maintained files under `ground_truth/`, which are
intentionally excluded from this public repository.

## 4a. "Is the app really using all my photos?" (and how to be sure you run THIS version)

* The page shows **App version: multi-image 2026-11-03**. If you do not see that line, you are running an old copy of `app.py` (another folder, or an old `streamlit run` still open): stop it with Ctrl+C and start it from this folder.
* The uploader is `st.file_uploader("Upload all seating chart photos", ..., accept_multiple_files=True)`. Click **Browse files** and select several files at once (Cmd/Shift-click), or drag several in; you can also add files one after another.
* There is no photo-count limit. Uploads are limited to 10 MB per photo, 30 MB total, and 20 megapixels per image to keep public-hosting resource use bounded.
* Debug details are off by default. Use **Clear this session's photos and results** in the sidebar to discard temporary session data.
* Room, column and row are shown **only when the seat is verified**; otherwise the app says "not verified".

## 4b. Multi-image search — how it behaves

* Every photo goes through `pipeline.analyze` + `search` on its own; the only shared things are the typed number and an OCR cache (a second search on the same photos does not re-run OCR).
* Corrupt/empty/non-image files give an `ERROR` row with the reason; blank or unrelated photos give `NOT_FOUND`; nothing is invented for them.
* The same number in several photos: **all** matches are shown. The app states what is known — same verified seat in several photos (same sheet photographed twice / overlapping photos), a verified match plus unverified cut-off views, or a **conflict** (verified at different seats). A "best match" banner appears only when verified appearances agree.
* Near-miss numbers from any photo are offered only as candidates the user must confirm.

## 5. How it works

1. **Load** the photo with PIL (EXIF-rotated) → BGR array. One array is used for OCR *and* drawing.
2. **OCR** → list of `Detection(text, score, polygon)`; text and polygon are never separated. Sideways photos are rotated automatically (transform remembered, polygons mapped back).
3. **Search** is exact on digits. Letters like `O`/`l` are **never** silently turned into digits: such tokens only appear as candidates. If nothing matches exactly: near-misses with confidence, and the app asks you to confirm explicitly. A match glued to a neighbour is split by position and flagged.
4. **Verify**: the polygon's pixels are cropped, read again alone, and must say the same number.
5. **Seat** (OpenCV, no reading-order guessing): de-skew → ridge-filter + Hough find vertical grid lines → the walls on both sides of the number and the chain to the table edges must give **exactly 6 columns** (A–F) → horizontal lines are accepted only if visible in **both text-free margins** of the cell → walking up from the number, same-height bands are rows, the first thinner band must hold one tiny glyph = header row → row 1 is just below it. Grid-chain length and index are checked before use.
6. **Room**: labels such as "Room No: ATR-xxx", "Room Number: 204", "Room: 204", and "ROOM-204" are recognized; a room value split into adjacent OCR detections is also supported. The room is attached to the nearest heading **above** the table, unless another table lies between them (then the room is reported unknown).
7. **Confidence** = OCR × re-read score × grid checks; HIGH / MEDIUM / LOW with explicit warnings. Whenever a check fails the seat is "unverified" instead of guessed.

## 6. Limitations (be upfront with your supervisor)

* Assumes this university's layout: 6 columns A–F, a "Room No: …" heading above each table, a thin header row. Other layouts need `n_cols` / regex changes.
* Needs the whole table width and the header row in the photo, sharp enough for ~20 px digits and visible grid lines.
* Cut-off tables (like the one at the very top of the sample) are deliberately reported "room unknown / unverified".
* A photo too blurry for the room heading gives the column/row but "room unknown".
* Handwritten text is not a separately trained OCR target; recognition depends on what the selected text OCR backend can read. The editable review table lets a user correct room/roll fields for export, but edits do not change automatic search results or mark a seat as OCR-verified.

## 7. Historical evaluation results (RapidOCR stand-in; not rerun for the changes above)

These figures are from the prior evaluation recorded in `outputs/eval_summary.md`; they are not current accuracy claims. Per-query end-to-end: every number on the chart is searched; a miss counts as wrong. 96 numbers per photo. Re-run `python evaluation.py` against your verified ground truth before relying on or publishing accuracy figures.

| Condition | Real? | Exact number found | Room+column+row all correct | Confident (HIGH) wrong answers |
|---|---|---|---|---|
| clear | real photo | 96/96 | 96/96 | 0 |
| low light | synthetic | 96/96 | 95/96 | 0 |
| tilted (~10° total) | synthetic | 96/96 | 96/96 | 0 |
| low resolution (60 %) | synthetic | 95/96 | 85/96 | 0 |
| blurry (σ = 2.5) | synthetic | 82/96 | 0/96 (grid col/row found, **room unreadable → reported unverified**) | 0 |

Also: OCR recall 100 % on clear; **0 of 96** invented numbers were returned as "found"; no wrong answer was ever labelled verified. Failures degrade to "unverified". The localization check "re-read of the highlighted pixels equals the requested number" passed for every found result on clear/low-light/tilted photos (19 % on blurry — correctly refusing to vouch).

## 7b. 13 additional real photos (no ground truth yet)

Run with the RapidOCR stand-in. "Verified" = every grid check passed. Because there is no hand-made ground truth, correctness is only checked indirectly: no two numbers map to the same seat, and rows/columns are monotonic in image position for every verified number (0 violations).

| Photo(s) | Verified / numbers read | Main reason for the rest |
|---|---|---|
| n02 (original), n08, n12 | 96/104, 88/99, 95/104 | cut-off tables at the photo edge |
| n05 | 91/120 | neighbouring sheets, cut-off tables |
| n03, n06, n07, n13 | 68/104, 70/95, 36/47, 30/37 | double-line row borders, header row not found |
| n09, n11 | 48/104, 46/76 | missing outer walls, side sheets |
| n04 | 36/87 | missing column walls |
| n01, n10 | 0/56, 0/49 | shaded cells / curved, folded page: grid not trusted |
| **Total** | **704 of 1,000+ read numbers** | refused, not guessed |

Changes made because of these photos: (1) when the column walls found are evenly spaced but 5 or 7 instead of 6, the sheet's centred "NIST UNIVERSITY" title decides which 6 form the table (one inferred wall lowers confidence and is flagged); (2) room names like `ATR-3RD FLOOR LIB-A`; (3) a global check that two different numbers can never share a seat; the original photo's results did not change.

## 8. Collecting and annotating more ground truth

1. Photograph real charts: ≥ 20 photos, several phones, rooms, lighting, angles; include some bad ones. Save in `samples/`.
2. Copy `ground_truth/photo_001_clear.json`, change `image` and `condition` (`clear|tilted|blurry|low_light|...`), `synthetic: false`.
3. Fill `numbers`: for every seat you can read, `reg_no`, `room`, `column` (A–F), `row` (1 = first student row under the A–F header). **Transcribe from the paper/photo yourself, not from the system's output.** Tip: the course counts in each chart (e.g. 1+33+4+…=48) should equal the number of seats you annotate.
4. Optional: `tables` (rough quadrilateral per table, for OCR precision) and a `bbox` `[x0,y0,x1,y1]` per number (pixel coordinates of the **original** photo) for true IoU localization.
5. `python evaluation.py` → `outputs/eval_summary.md`. For robustness checks on a new photo: `python make_variants.py ground_truth/<yourfile>.json`.

## 8b. Custom YOLO model — decision

Not trained. The only annotated real data is one photo (2 table boxes) and four synthetic copies of it; training on that would memorise one sheet. `ml/` contains everything needed once you annotate 30+ independent photos (dataset builder that splits by photo to prevent leakage, trainer that refuses tiny datasets, detector wrapper, precision/recall evaluator). A detector is meant only to propose table regions; OCR keeps reading numbers and `seat_locator.py` keeps mapping seats.

## 9. Troubleshooting

* *PaddleOCR can't download models / offline*: engine "auto" falls back to RapidOCR and says so in the UI.
* *Result LOW with "6 columns" warning*: the photo is too tilted/cropped/blurry; retake straight-on with the full table.
* *Streamlit warning about `use_container_width`*: already handled in `app.py` for old and new versions.
