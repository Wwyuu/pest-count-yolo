# -*- coding: utf-8 -*-
"""Eval dual-head YOLO+density: boxes and D from one forward, then blob routing."""
from __future__ import print_function
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch


from yolo_den.decode import DecodeCfg, decode_boxes  # noqa: E402
from yolo_den.dual_yolo import DualYoloDen, upsample_density  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

COUNT = os.environ.get("COUNT_DATA", os.path.join(HERE, "data"))
ROOT = os.environ.get("BASELINE_ROOT", HERE)
YOLO26 = os.path.join(ROOT, "YOLO26")
RUNS = os.environ.get("RUNS_DIR", os.path.join(HERE, "runs"))
P2_BEST = os.environ.get(
    "DET_CKPT",
    os.path.join(RUNS, "YOLOv8-P2-SUM", "train", "weights", "best.pt"),
)

def import_yolo():
    try:
        from ultralytics import YOLO

        return YOLO
    except Exception:
        sys.path.insert(0, YOLO26)
        from ultralytics import YOLO

        return YOLO


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(float(new_shape) / h, float(new_shape) / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw = (new_shape - nw) / 2.0
    dh = (new_shape - nh) / 2.0
    if (w, h) != (nw, nh):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, left, top


def scale_boxes(boxes, r, padw, padh, h0, w0):
    if boxes is None or boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.copy()
    out[:, [0, 2]] -= padw
    out[:, [1, 3]] -= padh
    out[:, :4] /= max(r, 1e-12)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w0)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h0)
    return out


def gt_n(path):
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if len(line.split()) >= 2:
                n += 1
    return n


def cfg_from_env(mode):
    return DecodeCfg(
        box_side=float(os.environ.get("DENNMS_BOX", "20")),
        nt0=float(os.environ.get("DENNMS_NT0", "0.5")),
        tau_m=float(os.environ.get("DENNMS_TAU_M", "3.0")),
        blob_tmean=float(os.environ.get("DENNMS_BLOB_TMEAN", "2.5")),
        blob_tfloor=float(os.environ.get("DENNMS_BLOB_TFLOOR", "1e-5")),
        blob_close=int(os.environ.get("DENNMS_BLOB_CLOSE", "11")),
        blob_dilate=int(os.environ.get("DENNMS_BLOB_DILATE", "3")),
        blob_tau_miss=float(os.environ.get("DENNMS_BLOB_MISS", "0.5")),
        blob_drop_out=os.environ.get("DENNMS_BLOB_DROP", "1") != "0",
        mode=mode,
    )


def load_dual(dual_ckpt, device, imgsz):
    YOLO = import_yolo()
    blob = torch.load(dual_ckpt, map_location="cpu")
    src = blob.get("yolo_src") or P2_BEST
    kind = blob.get("head_type") or "p2p3"
    yolo = YOLO(src)
    dual = DualYoloDen.from_detection_model(yolo.model, imgsz=imgsz, device="cpu", head=kind)
    dual.det.load_state_dict(blob["det"], strict=True)
    dual.head.load_state_dict(blob["head"], strict=True)
    dual.to(device).eval()
    print("loaded dual", dual_ckpt, "head", kind, "from", src, "epoch", blob.get("epoch"), flush=True)
    return dual


def import_nms():
    try:
        from ultralytics.utils.nms import non_max_suppression

        return non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

        return non_max_suppression


def infer_numpy(dual, bgr, imgsz, conf, iou_raw, iou_def, max_det):
    non_max_suppression = import_nms()

    h0, w0 = bgr.shape[:2]
    lb, r, padw, padh = letterbox(bgr, imgsz)
    rgb = lb[:, :, ::-1].copy()
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(next(dual.parameters()).device)
    with torch.no_grad():
        preds, den = dual(x)
    if isinstance(preds, (list, tuple)):
        pred = preds[0]
    else:
        pred = preds
    nc = 1
    raw = non_max_suppression(pred, conf, iou_raw, max_det=max_det, nc=nc)[0]
    defb = non_max_suppression(pred, conf, iou_def, max_det=max_det, nc=nc)[0]

    def unpack(t):
        if t is None or t.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        xy = t[:, :4].detach().cpu().numpy().astype(np.float32)
        sc = t[:, 4].detach().cpu().numpy().astype(np.float32)
        return scale_boxes(xy, r, padw, padh, h0, w0), sc

    braw, sraw = unpack(raw)
    bdef, sdef = unpack(defb)
    up = upsample_density(den[0], h0, w0)[0, 0].detach().cpu().numpy()
    return braw, sraw, bdef, sdef, up


def summarize(rows, key, gt_key="gt"):
    xs = [abs(r[key] - r[gt_key]) for r in rows if r.get(gt_key) is not None]
    if not xs:
        return None
    return float(np.mean(xs))


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis in ("0", "1", "0,1"):
        raise SystemExit("refusing GPU 0/1, got %s" % vis)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(RUNS, "YOLO-DenHead-SUM", "best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-raw", type=float, default=0.99)
    ap.add_argument("--iou-def", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RUNS, "eval-yolo-denhead"))
    ap.add_argument("--tag", default="YOLOv8-P2")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not os.path.isfile(args.ckpt):
        raise SystemExit("missing " + args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual = load_dual(args.ckpt, device, args.imgsz)
    img_d = os.path.join(COUNT, "images", args.split)
    lab_d = os.path.join(COUNT, "labels", args.split)
    names = sorted(n for n in os.listdir(img_d) if n.lower().endswith((".jpg", ".png", ".jpeg")))
    if args.limit:
        names = names[: args.limit]
    modes = [m.strip() for m in os.environ.get("DENNMS_MODES", "det,nms_fixed,blob,blob_open,blob_keep,den").split(",") if m.strip()]
    rows = []
    dump_rows = []
    for i, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        bgr = cv2.imread(os.path.join(img_d, name))
        if bgr is None:
            print("skip", name, flush=True)
            continue
        braw, sraw, bdef, sdef, den = infer_numpy(
            dual, bgr, args.imgsz, args.conf, args.iou_raw, args.iou_def, args.max_det
        )
        gt = gt_n(os.path.join(lab_d, stem + ".txt"))
        row = {
            "stem": stem,
            "gt": gt,
            "yolo_default": float(bdef.shape[0]),
            "yolo_raw": float(braw.shape[0]),
            "count_den": float(den.sum()),
        }
        for m in modes:
            out = decode_boxes(braw, sraw, den, cfg_from_env(m))
            row[m] = float(out["count"])
            row["det_%s" % m] = float(out.get("det_n", 0) or 0)
            row["den_%s" % m] = float(out.get("den_n", 0) or 0)
            row["drop_%s" % m] = int(out.get("n_drop", 0) or 0)
            row["ncc_%s" % m] = int(out.get("n_cc", 0) or 0)
        rows.append(row)
        dump_rows.append(
            {
                "stem": stem,
                "name": name,
                "n_default": int(bdef.shape[0]),
                "n_raw": int(braw.shape[0]),
                "boxes_raw": braw.tolist(),
                "scores_raw": sraw.tolist(),
                "boxes_default": bdef.tolist(),
                "scores_default": sdef.tolist(),
            }
        )
        if (i + 1) % 40 == 0:
            print("eval", i + 1, "/", len(names), "gt", gt, "den", row["count_den"], "blob", row.get("blob"), flush=True)
    keys = ["yolo_default", "yolo_raw", "count_den"] + list(modes)
    summary = {"ckpt": args.ckpt, "split": args.split, "n": len(rows), "one_forward": True}
    for k in keys:
        summary["mae_" + k] = summarize(rows, k)
    bins = [(0, 20, "0-20"), (21, 50, "21-50"), (51, 80, "51-80"), (81, 10 ** 9, "81+")]
    summary["mae_by_bin"] = {}
    for lo, hi, name in bins:
        sub = [r for r in rows if lo <= r["gt"] <= hi]
        if not sub:
            continue
        summary["mae_by_bin"][name] = {k: float(np.mean([abs(r[k] - r["gt"]) for r in sub])) for k in keys}
        summary["mae_by_bin"][name]["n"] = len(sub)
    os.makedirs(args.out, exist_ok=True)
    json.dump(
        {"summary": summary, "rows": rows},
        open(os.path.join(args.out, "%s_%s_dual.json" % (args.tag, args.split)), "w"),
        indent=2,
    )
    json.dump(
        {"tag": "%s-dual" % args.tag, "ckpt": args.ckpt, "split": args.split, "rows": dump_rows},
        open(os.path.join(args.out, "%s_%s_boxes.json" % (args.tag, args.split)), "w"),
        indent=2,
    )
    print("RESULT", json.dumps({k: summary[k] for k in summary if k != "mae_by_bin"}), flush=True)
    print("BINS", json.dumps(summary.get("mae_by_bin", {})), flush=True)


if __name__ == "__main__":
    main()
