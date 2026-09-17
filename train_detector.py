# -*- coding: utf-8 -*-
"""Train YOLOv8 / YOLOv8-P2 detector from point-derived pseudo-boxes, then count boxes."""
from __future__ import print_function
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BASELINE_ROOT", HERE)
YOLO26 = os.path.join(ROOT, "YOLO26")
P2_YAML = os.path.join(HERE, "configs", "yolov8s-p2.yaml")
COUNT = os.environ.get("COUNT_DATA", os.path.join(HERE, "data"))
DATA = os.environ.get("YOLO_DATA_YAML", os.path.join(COUNT, "yolo", "pest.yaml"))
SRC = COUNT


def swan(name, cfg):
    key = os.environ.get("SWANLAB_API_KEY", "")
    if not key:
        return None
    try:
        import swanlab

        swanlab.login(api_key=key)
        return swanlab.init(project="pest-count-yolo", experiment_name=name, config=cfg)
    except Exception as e:
        print("swanlab failed", e, flush=True)
        return None


def import_yolo():
    try:
        from ultralytics import YOLO

        return YOLO
    except Exception:
        sys.path.insert(0, YOLO26)
        from ultralytics import YOLO

        return YOLO


def write_p2_nc1(src, dst):
    text = open(src).read().replace("nc: 80", "nc: 1", 1)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        f.write(text)
    return dst


def gt_count(split):
    lab = os.path.join(SRC, "labels", split)
    out = {}
    for n in os.listdir(lab):
        if n.endswith(".txt"):
            with open(os.path.join(lab, n)) as f:
                out[os.path.splitext(n)[0]] = sum(1 for line in f if len(line.split()) >= 2)
    return out


def eval_count(model, split, conf=0.25):
    img_d = os.path.join(SRC, "images", split)
    gts = gt_count(split)
    names = sorted(n for n in os.listdir(img_d) if n.lower().endswith((".jpg", ".png")))
    mae = mse = 0.0
    for name in names:
        pred = model.predict(os.path.join(img_d, name), conf=conf, verbose=False)
        npred = 0
        if pred:
            boxes = getattr(pred[0], "boxes", None)
            npred = int(len(boxes)) if boxes is not None else 0
        gt = gts.get(os.path.splitext(name)[0], 0)
        mae += abs(npred - gt)
        mse += (npred - gt) ** 2
    n = max(1, len(names))
    return mae / n, (mse / n) ** 0.5


def build_yolov8s(YOLO, out):
    """Official 3-scale YOLOv8s (P3/P4/P5), same width as YOLOv8s-P2 without the P2 head."""
    src = os.path.join(YOLO26, "ultralytics", "cfg", "models", "v8", "yolov8.yaml")
    dst = os.path.join(out, "yolov8s.yaml")
    if os.path.isfile(src):
        return YOLO(write_p2_nc1(src, dst))
    print("local yolov8.yaml missing, use package yolov8s.yaml", flush=True)
    return YOLO("yolov8s.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("yolo26", "yolov8p2", "yolov8s"), required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    name_map = {"yolo26": "YOLO26-SUM", "yolov8p2": "YOLOv8-P2-SUM", "yolov8s": "YOLOv8s-SUM"}
    name = os.environ.get("SWANLAB_RUN") or name_map[args.which]
    out = args.out or os.path.join(os.environ.get("RUNS_DIR", os.path.join(HERE, "runs")), name)
    os.makedirs(out, exist_ok=True)
    run = swan(name, vars(args))
    YOLO = import_yolo()
    if args.which == "yolo26":
        yaml = os.path.join(YOLO26, "ultralytics", "cfg", "models", "26", "yolo26s.yaml")
        if not os.path.isfile(yaml):
            yaml = "yolo26s.yaml"
        model = YOLO(yaml)
    elif args.which == "yolov8s":
        try:
            model = build_yolov8s(YOLO, out)
        except Exception as e:
            print("yolov8s yaml failed, fallback yolov8s.yaml", e, flush=True)
            model = YOLO("yolov8s.yaml")
        print("arch yolov8s no-P2 Detect P3/P4/P5", flush=True)
    else:
        yaml = write_p2_nc1(P2_YAML, os.path.join(out, "yolov8s-p2-nc1.yaml"))
        try:
            model = YOLO(yaml)
        except Exception as e:
            print("p2 yaml failed, fallback yolov8s.yaml", e, flush=True)
            model = YOLO("yolov8s.yaml")

    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=out,
        name="train",
        exist_ok=True,
        workers=4,
        pretrained=True,
        amp=False,
    )
    best = os.path.join(out, "train", "weights", "best.pt")
    if os.path.isfile(best):
        model = YOLO(best)
    vmae, vrmse = eval_count(model, "val")
    tmae, trmse = eval_count(model, "test")
    rec = {
        "which": args.which,
        "val_mae": float(vmae),
        "val_rmse": float(vrmse),
        "test_mae": float(tmae),
        "test_rmse": float(trmse),
        "best": best if os.path.isfile(best) else "",
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print("YOLO COUNT", rec, flush=True)
    if run:
        try:
            import swanlab

            swanlab.log({k: v for k, v in rec.items() if isinstance(v, float)})
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
