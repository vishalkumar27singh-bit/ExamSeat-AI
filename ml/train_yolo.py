"""
train_yolo.py - fine-tune a YOLO detector for table regions (needs `pip install ultralytics`).

It REFUSES to train on too little data, because a model trained on a handful of photos and validated on
copies of them would report a fantasy score.  Defaults: at least 30 independent photos.

    python ml/prepare_dataset.py
    python ml/train_yolo.py --data datasets/tables_yolo/data.yaml --model yolov8n.pt --epochs 100
    python ml/train_yolo.py --smoke        # 1 epoch on a tiny model from scratch: checks the plumbing ONLY
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count_photos(data_yaml: str):
    base = os.path.dirname(data_yaml)
    tr = glob.glob(os.path.join(base, "images", "train", "*"))
    va = glob.glob(os.path.join(base, "images", "val", "*"))
    return len(tr), len(va)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "tables_yolo", "data.yaml"))
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained weights to fine-tune (downloaded once), or a .yaml to train from scratch")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960, help="table borders are thin: keep the image large")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--min-train-images", type=int, default=30)
    ap.add_argument("--project", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--name", default="tables")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, tiny model from scratch, no data-size check; metrics are meaningless")
    a = ap.parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed:  pip install ultralytics")
        return 2
    if not os.path.exists(a.data):
        print(f"{a.data} not found - run ml/prepare_dataset.py first")
        return 2
    n_tr, n_va = count_photos(a.data)
    print(f"dataset: {n_tr} training images, {n_va} validation images")
    if a.smoke:
        a.model, a.epochs, a.imgsz, a.batch, a.name = "yolov8n.yaml", 1, 320, 2, "smoke"
        print("SMOKE TEST: checking that training runs. The resulting weights are useless.")
    elif n_tr < a.min_train_images:
        print(f"Only {n_tr} training images (< {a.min_train_images}). Annotate more photos first "
              f"(ml/ANNOTATION_GUIDE.md), or lower --min-train-images knowing the result will not be trustworthy.")
        return 3
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=a.project, name=a.name,
                exist_ok=True, device="cpu" if not _has_gpu() else 0, workers=0, plots=False, verbose=False,
                degrees=5, perspective=0.0005, hsv_v=0.4, fliplr=0.0, mosaic=0.5)
    best = os.path.join(a.project, a.name, "weights", "best.pt")
    print("weights:", best if os.path.exists(best) else "(not written)")
    return 0


def _has_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
