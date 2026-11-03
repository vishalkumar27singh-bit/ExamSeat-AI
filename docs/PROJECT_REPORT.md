# ExamSeat AI - Multi-Image Examination Seat Finder
B.Tech CSE-AIML major project report (draft: fill in names, guide, dates)

## 1. Abstract
Students find their examination seat on printed sitting charts pinned in corridors. ExamSeat AI takes several phone
photographs of such charts and one registration number, and reports in which photo, room, seat column (A-F) and seat
row the number is printed, with the number highlighted on the original photograph. The system favours *refusing* over
*guessing*: every answer is re-verified by cropping the highlighted region and reading it again, and a seat is called
verified only if independent grid checks agree. Text recognition (PaddleOCR / RapidOCR), seat mapping (OpenCV line
geometry) and optional object detection (YOLO tooling) are kept as separate modules.

## 2. Problem and objectives
* Find an exact 10-digit registration number in many photos at once; never silently alter it (O/0, l/1).
* Return room, column, row, OCR confidence and a verification status; "Registration Number Not Found" otherwise.
* Draw the box on the ORIGINAL image so that it never lands on a neighbouring number.
* Survive bad inputs: one corrupt photo must not stop the others.
* Measure honestly, including on photos the rules were not tuned on.

## 3. Background (what each technique is for)
* **PP-OCR (PaddleOCR)**: text detection + recognition; returns a polygon per text line.
* **Hough transform (Duda & Hart, 1972)** on a ridge-filtered image: finds the straight printed table lines.
* **Ridge filtering** (second derivative of an anisotropic Gaussian; compare Frangi et al., 1998): responds to thin dark lines.
* **YOLO (Redmon et al., 2016)**: one-pass object detector; here only proposed for table regions (section 8).

## 4. Architecture
See `docs/architecture.png` (source `docs/architecture.dot`).
`app.py` -> `multi_search.py` (per-photo try/except, OCR cache, duplicate flags) -> `pipeline.py` for each photo:
`image_processing.py` (loading, scaling, rotation, coordinate `Transform`) -> `ocr_engine.py` (text + own polygon, exact search,
candidates, re-read verification) -> `seat_locator.py` (grid and room mapping).

## 5. Key design decisions
1. **Original-image coordinates everywhere.** Every resize/rotation is a `Transform`; OCR runs on a working copy and
   its polygons are mapped back. Tests rotate and enlarge the real photo and check the box centre against the known transform.
2. **The coordinate bug (root cause).** PaddleOCR 3.x enables a document pre-processor by default (orientation classifier and
   unwarping), so `rec_polys` live in the *unwarped* image, not in the file loaded by OpenCV. Fix: switch those stages off,
   use only the aligned `rec_*` lists, raise on any size mismatch, and re-read each highlighted polygon.
   Status: the diagnosis is supported by the geometry of the user's polygon and Paddle's source code; it has **not yet been
   confirmed on a real PaddleOCR run** (see section 9).
3. **Exact matching only.** A token in which a letter had to be "fixed" is never an exact match; near-misses are offered as
   candidates that the user must confirm.
4. **Seat position from geometry, not reading order.** Six columns must be found between walls with registration numbers
   in them; row lines must be visible in both text-free margins of a cell; the header row is recognised by being thinner
   and holding one small glyph; row 1 is the first row under it.
5. **Cross-checks.** Two different numbers can never share a seat; evenly spaced 5- or 7-column results are completed only if the
   sheet's centred "NIST UNIVERSITY" title agrees (an inferred wall lowers confidence and is flagged).
6. **Multi-image semantics.** Photos are independent. If a number appears in several photos the program says whether the
   appearances agree on one verified seat, include unverified partial views, or conflict; it picks a "best match" only when
   verified appearances agree.

## 6. Testing (actual results)
Environment of these runs: OCR = RapidOCR (PP-OCR via ONNX) because PaddleOCR's models cannot be downloaded in the build
sandbox; PaddleOCR itself was exercised only through a mock of its result structure.

| Test set | Result |
|---|---|
| `python -m pytest tests` | **23 passed** (matching rules, rotation/scale coordinate mapping on the real photo, highlight never overlapping a neighbour, multi-image behaviour, caching, error isolation, duplicate/conflict logic) |
| `python tests/e2e_streamlit_upload.py` (real Streamlit app in Chromium, 4 files incl. one corrupt) | 7 of 7 checks passed |
| `ml/` smoke test | prepare -> train 1 epoch from scratch -> detect -> evaluate ran without error; P = R = 0 as expected; weights deleted |

## 7. Accuracy (what can and cannot be claimed)
Single real photo (96 numbers) and four synthetic degradations of it (`evaluation.py`):

| Condition | Real? | Exact number found | Room+column+row correct | Confident (HIGH) wrong answers |
|---|---|---|---|---|
| clear | real | 96/96 | 96/96 | 0 |
| low light | synthetic | 96/96 | 95/96 | 0 |
| tilted | synthetic | 96/96 | 96/96 | 0 |
| low resolution | synthetic | 95/96 | 85/96 | 0 |
| blurry | synthetic | 82/96 | 0/96 (room unreadable -> reported unverified) | 0 |

The grid rules were first tuned on that one photo. Thirteen further real photos (one of them the same photo as above)
have **no hand-made ground truth**, so no accuracy is claimed for them. What was measured: 704 of 1,000+ numbers read
received a verified seat; no two numbers were assigned the same seat; rows/columns were monotonic with image position
for all verified numbers. Two photos (shaded cells; a curved, folded page) got no verified seat at all.

Overlap between photos (measured on 14 distinct files): 974 distinct numbers were read, **189 of them appear in more than
one photo** because neighbouring sheets overlap in the frame (88 of those are because n02 is byte-identical to photo_001_clear.jpeg, included twice). Duplicate
handling is therefore a real requirement, not a corner case.

## 8. Machine-learning component
Decision: **no custom YOLO model was trained.** Labelled data = one real photo (2 table boxes) plus four copies of it.
The tooling is complete and tested (`ml/`): annotation guide, dataset builder that splits by photo group to avoid leakage, a
trainer that refuses to run on fewer than 30 training images, an inference wrapper and a precision/recall evaluator.
No training result is reported because none exists.

## 9. Limitations and open items
* Needs the whole table width and header row in the photo; cut-off tables are reported unverified.
* Tuned for this layout (6 columns A-F, "Room No:" heading, centred title).
* PaddleOCR on the real machine has not been run by the author of this report: run `python diagnose_coordinates.py` and keep its output.
* Accuracy on unseen photos is unknown until ground truth for them exists (`ground_truth/` format in the README).

## 10. Future work
Annotate 30+ independent photos; evaluate a YOLO table detector against the grid code on held-out photos; perspective
rectification of curved pages; photo-capture guidance in the app.

## References
PaddleOCR / PP-OCR (PaddlePaddle project). Duda & Hart, "Use of the Hough transformation to detect lines and curves in
pictures", CACM 1972. Frangi et al., "Multiscale vessel enhancement filtering", MICCAI 1998. Redmon et al., "You Only Look
Once", CVPR 2016.
