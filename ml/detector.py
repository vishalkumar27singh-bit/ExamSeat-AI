"""
detector.py - optional table-region detector (YOLO) kept SEPARATE from OCR and seat mapping.

    det = TableDetector("runs/tables/weights/best.pt")
    boxes = det.detect(img_bgr)          # [{"box": [x0,y0,x1,y1], "score": 0.9, "cls": "table"}, ...] in ORIGINAL pixels

Nothing in pipeline.py needs this module: without weights the project uses the classical grid detection
in seat_locator.py.  A detector is only worth wiring in after ml/eval_detector.py shows it beats that baseline
on photos it was NOT trained on.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

NAMES = {0: "table", 1: "room_heading"}


class TableDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 960):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"weights not found: {weights} (train with ml/train_yolo.py)")
        from ultralytics import YOLO
        self.model, self.conf, self.imgsz = YOLO(weights), conf, imgsz

    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        res = self.model.predict(img_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False, device="cpu")[0]
        out = []
        for b in res.boxes:
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]      # ultralytics returns original-image pixels
            out.append(dict(box=[x0, y0, x1, y1], score=float(b.conf[0]), cls=NAMES.get(int(b.cls[0]), str(int(b.cls[0])))))
        return out
