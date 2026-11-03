# Annotating data for the custom detector

## Should you train a YOLO model at all? (honest answer for this project)

**Not yet.** What exists today: ONE real annotated photo (2 tables, 96 numbers) and 4 degraded copies of that same photo.
A detector trained on that would memorise one sheet, and validating on its copies would give a fantasy score.

The weak spots seen on 13 extra real photos are in the *grid logic* (a missing outer wall, shaded cells, a curved page),
not in finding numbers - the OCR already detects number text well. A table detector could help by cropping each table
before the grid code runs, but that is a hypothesis. **Test it with `ml/eval_detector.py` on photos the model never
saw, and compare against the current grid code, before wiring it in.**

Separate jobs stay separate: OCR reads digits (`ocr_engine.py`), the seat mapper turns grid lines into A-F / row
(`seat_locator.py`), a detector would only propose regions (`ml/detector.py`).

## Classes

| id | name | draw a box around | notes |
|---|---|---|---|
| 0 | `table` | the whole table body: from the top of the A-F header row to the bottom of the last row, left wall to right wall | one box per table; include partly visible tables only if at least the header row and 3 rows are visible |
| 1 | `room_heading` | the printed text "Room No: ATR-xxx" | optional |

Do **not** annotate individual numbers or seats for the detector: OCR does that, and thousands of tiny boxes cost hours.

## Collecting photos (what makes the data useful)

* Independent photos of *different* papers, different rooms, different phones, lighting, angles, some with glare, curved
  paper, pins, shaded cells. Several photos of the same paper count as ONE photo for splitting (set the same `"group"`).
* Suggested minimum before trusting any number: **30+ independent photos** for training, **10+ different photos held out**
  for testing. (`ml/train_yolo.py` refuses to train below 30 training images.) These are working guidelines, not results.
* Keep the original file; never annotate a screenshot or a resized copy.

## Annotating, step by step

Option A - the project's own JSON (simplest, already used by `evaluation.py`):

1. Copy `ground_truth/photo_001_clear.json`, set `image`, `condition`, `"synthetic": false`.
2. `tables`: for each table give `room` and `quad` = 4 corner points `[x, y]` in ORIGINAL photo pixels
   (top-left, top-right, bottom-right, bottom-left). Any image viewer that shows pixel coordinates works.
3. Optional `headings`: `[{"bbox": [x0, y0, x1, y1]}]`.
4. For the seat-mapping evaluation also fill `numbers` (`reg_no`, `room`, `column`, `row`) **from the paper, not from
   the program's output**.

Option B - a labelling tool (CVAT, Label Studio, Roboflow): draw boxes with the two class names above, export "YOLO
detection" format, and put the files in `datasets/tables_yolo/{images,labels}/{train,val}/` yourself (keep photos of
one paper in one split).

## Build the dataset, train, evaluate

```bash
python ml/prepare_dataset.py                      # JSON -> YOLO files; splits by photo group, not by file
python ml/train_yolo.py --model yolov8n.pt --epochs 100     # downloads the small pretrained weights once
python ml/eval_detector.py --weights runs/tables/weights/best.pt
```

`prepare_dataset.py` prints how many independent photos and boxes it found and stops if there is only one group.
`train_yolo.py --smoke` trains one epoch of a tiny model from scratch **only to check the plumbing**; its weights are useless.

## What to report honestly

Number of independent training / validation photos, number of boxes, precision and recall at IoU 0.5 on held-out
photos (from `eval_detector.py`), and - the real question - whether seat answers on held-out photos improved compared
with the grid code alone.
