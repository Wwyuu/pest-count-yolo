# -*- coding: utf-8 -*-
"""Joint train: YOLOv8-P2 box head + neck density head (P2/P3).

Init from YOLOv8-P2-SUM best.pt. pest_yolo env. GPU via CUDA_VISIBLE_DEVICES.
"""
from __future__ import print_function
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.utils.data as data


from yolo_den.dual_yolo import DualYoloDen, ensure_criterion, joint_density_losses  # noqa: E402
from yolo_den.gaussian import gaussian_density_yx  # noqa: E402

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


def list_split(split):
    img_d = os.path.join(COUNT, "images", split)
    yolo_lab = os.path.join(COUNT, "yolo", "labels", split)
    pt_lab = os.path.join(COUNT, "labels", split)
    names = sorted(
        n for n in os.listdir(img_d) if n.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    return img_d, yolo_lab, pt_lab, names


def read_points_and_boxes(stem, yolo_lab, pt_lab, w0, h0, box_side=20.0):
    """Return (cls, xywh_norm_orig, points_xy_orig)."""
    pts = []
    boxes = []
    ptxt = os.path.join(pt_lab, stem + ".txt")
    ytxt = os.path.join(yolo_lab, stem + ".txt")
    if os.path.isfile(ptxt):
        with open(ptxt) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
    if os.path.isfile(ytxt):
        with open(ytxt) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    cls, xc, yc, bw, bh = map(float, p[:5])
                    boxes.append((cls, xc, yc, bw, bh))
                elif len(p) >= 3:
                    cls = float(p[0])
                    xc, yc = float(p[1]), float(p[2])
                    boxes.append((cls, xc, yc, box_side / w0, box_side / h0))
    if not pts and boxes:
        for cls, xc, yc, bw, bh in boxes:
            pts.append((xc * w0, yc * h0))
    if not boxes and pts:
        for x, y in pts:
            boxes.append((0.0, x / w0, y / h0, box_side / w0, box_side / h0))
    return boxes, pts


class PestDetDen(data.Dataset):
    def __init__(self, split, imgsz=640, augment=True, box_side=20.0, sigma=4.0):
        self.imgsz = int(imgsz)
        self.augment = bool(augment)
        self.box_side = float(box_side)
        self.sigma = float(sigma)
        self.img_d, self.yolo_lab, self.pt_lab, self.names = list_split(split)
        if not self.names:
            raise SystemExit("no images in %s" % self.img_d)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        stem = os.path.splitext(name)[0]
        im = cv2.imread(os.path.join(self.img_d, name))
        if im is None:
            raise RuntimeError("bad image " + name)
        h0, w0 = im.shape[:2]
        boxes, pts = read_points_and_boxes(stem, self.yolo_lab, self.pt_lab, w0, h0, self.box_side)
        flip = self.augment and random.random() < 0.5
        if flip:
            im = np.ascontiguousarray(np.fliplr(im))
            boxes = [(c, 1.0 - xc, yc, bw, bh) for c, xc, yc, bw, bh in boxes]
            pts = [(w0 - 1.0 - x, y) for x, y in pts]
        lb, r, padw, padh = letterbox(im, self.imgsz)
        xywh = []
        cls = []
        pts_lb = []
        for c, xc, yc, bw, bh in boxes:
            x = xc * w0 * r + padw
            y = yc * h0 * r + padh
            ww = bw * w0 * r
            hh = bh * h0 * r
            xywh.append([x / self.imgsz, y / self.imgsz, ww / self.imgsz, hh / self.imgsz])
            cls.append([c])
        for x, y in pts:
            pts_lb.append((y * r + padh, x * r + padw))
        den = gaussian_density_yx(pts_lb, self.imgsz, self.imgsz, sigma=self.sigma)
        img = lb[:, :, ::-1].copy()
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        n = len(xywh)
        if n:
            bboxes = torch.tensor(xywh, dtype=torch.float32)
            clst = torch.tensor(cls, dtype=torch.float32)
        else:
            bboxes = torch.zeros(0, 4, dtype=torch.float32)
            clst = torch.zeros(0, 1, dtype=torch.float32)
        return {
            "img": img,
            "cls": clst,
            "bboxes": bboxes,
            "den": torch.from_numpy(den),
            "n": n,
            "stem": stem,
            "ori_hw": (h0, w0),
            "ratio_pad": (r, padw, padh),
        }


def collate(batch):
    imgs = torch.stack([b["img"] for b in batch], 0)
    dens = torch.stack([b["den"] for b in batch], 0)
    cls, boxes, bidx = [], [], []
    for i, b in enumerate(batch):
        n = int(b["cls"].shape[0])
        if n:
            cls.append(b["cls"])
            boxes.append(b["bboxes"])
            bidx.append(torch.full((n,), i, dtype=torch.int64))
    if cls:
        cls = torch.cat(cls, 0)
        boxes = torch.cat(boxes, 0)
        bidx = torch.cat(bidx, 0)
    else:
        cls = torch.zeros(0, 1, dtype=torch.float32)
        boxes = torch.zeros(0, 4, dtype=torch.float32)
        bidx = torch.zeros(0, dtype=torch.int64)
    return {
        "img": imgs,
        "cls": cls,
        "bboxes": boxes,
        "batch_idx": bidx,
        "den": dens,
        "stems": [b["stem"] for b in batch],
        "ns": [b["n"] for b in batch],
    }


def det_loss_value(criterion, preds, batch):
    out = criterion(preds, batch)
    if isinstance(out, (list, tuple)):
        loss = out[0]
        items = out[1] if len(out) > 1 else None
    else:
        loss, items = out, None
    if torch.is_tensor(loss) and loss.numel() > 1:
        loss = loss.sum()
    return loss, items


def build_dual(ckpt, imgsz, device, head="p2p3"):
    YOLO = import_yolo()
    if not os.path.isfile(ckpt):
        raise SystemExit("missing yolo ckpt " + ckpt)
    yolo = YOLO(ckpt)
    det = yolo.model
    if hasattr(yolo, "overrides") and getattr(det, "args", None) is None:
        try:
            from ultralytics.utils import IterableSimpleNamespace

            det.args = IterableSimpleNamespace(**dict(yolo.overrides))
        except Exception:
            pass
    dual = DualYoloDen.from_detection_model(det, imgsz=imgsz, device="cpu", head=head)
    dual.to(device)
    dual.det.criterion = None
    ensure_criterion(dual.det)
    return dual, yolo


def make_optim(dual, lr_det, lr_head, wd, det_on):
    groups = [{"params": list(dual.head.parameters()), "lr": lr_head, "weight_decay": wd}]
    if det_on:
        groups.append({"params": [p for p in dual.det.parameters() if p.requires_grad], "lr": lr_det, "weight_decay": wd * 0.1})
    return torch.optim.AdamW(groups)


def den_terms(den, den_gt, args):
    loss_map, loss_sum, loss_tv, _, loss_sparse = joint_density_losses(
        den, den_gt, dense_tau=getattr(args, "dense_tau", 0.0)
    )
    loss = (
        args.w_map * loss_map
        + args.w_sum * loss_sum
        + args.w_tv * loss_tv
        + getattr(args, "w_sparse", 0.0) * loss_sparse
    )
    return loss, loss_map, loss_sum, loss_tv, loss_sparse


def run_epoch(dual, loader, device, optim, args, train, freeze_det):
    dual.train(train)
    dual.set_det_trainable((not freeze_det) and train)
    tot = {"loss": 0.0, "det": 0.0, "map": 0.0, "sum": 0.0, "tv": 0.0, "sparse": 0.0, "den_mae": 0.0, "n": 0}
    n_img = 0
    for step, batch in enumerate(loader):
        if args.max_batches and step >= args.max_batches:
            break
        imgs = batch["img"].to(device, non_blocking=True)
        den_gt = batch["den"].to(device, non_blocking=True).unsqueeze(1)
        yolo_batch = {
            "img": imgs,
            "cls": batch["cls"].to(device),
            "bboxes": batch["bboxes"].to(device),
            "batch_idx": batch["batch_idx"].to(device),
        }
        if train:
            preds, den = dual(imgs)
            loss_det, _ = det_loss_value(dual.det.criterion, preds, yolo_batch)
            den_loss, loss_map, loss_sum, loss_tv, loss_sparse = den_terms(den, den_gt, args)
            loss = args.w_det * loss_det + den_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual.parameters(), 5.0)
            optim.step()
        else:
            with torch.no_grad():
                preds, den = dual(imgs)
                loss_det, _ = det_loss_value(dual.det.criterion, preds, yolo_batch)
                den_loss, loss_map, loss_sum, loss_tv, loss_sparse = den_terms(den, den_gt, args)
                loss = args.w_det * loss_det + den_loss
        bsz = int(imgs.shape[0])
        pred_n = den.flatten(1).sum(1).detach()
        gt_n = torch.tensor(batch["ns"], device=device, dtype=pred_n.dtype)
        mae = (pred_n - gt_n).abs().sum().item()
        tot["loss"] += float(loss.detach().item())
        tot["det"] += float(loss_det.detach().item())
        tot["map"] += float(loss_map.detach().item())
        tot["sum"] += float(loss_sum.detach().item())
        tot["tv"] += float(loss_tv.detach().item())
        tot["sparse"] += float(loss_sparse.detach().item())
        tot["den_mae"] += mae
        tot["n"] += 1
        n_img += bsz
        if train and (step + 1) % 20 == 0:
            print(
                "step %s loss=%.3f det=%.3f map=%.4f sum=%.3f tv=%.4f sparse=%.3f den_mae=%.2f"
                % (
                    step + 1,
                    tot["loss"] / tot["n"],
                    tot["det"] / tot["n"],
                    tot["map"] / tot["n"],
                    tot["sum"] / tot["n"],
                    tot["tv"] / tot["n"],
                    tot["sparse"] / tot["n"],
                    tot["den_mae"] / max(n_img, 1),
                ),
                flush=True,
            )
    n = max(tot["n"], 1)
    out = {k: tot[k] / n for k in ("loss", "det", "map", "sum", "tv", "sparse")}
    out["den_mae"] = tot["den_mae"] / float(max(n_img, 1))
    return out


def save_ckpt(path, dual, epoch, extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {
        "epoch": epoch,
        "det": dual.det.state_dict(),
        "head": dual.head.state_dict(),
        "head_type": dual.head_kind(),
    }
    if hasattr(dual.head, "p2"):
        rec["head_c"] = (int(dual.head.p2.in_channels), int(dual.head.p3.in_channels))
    if hasattr(dual.head, "chs"):
        rec["head_chs"] = list(dual.head.chs)
    rec.update(extra)
    torch.save(rec, path)


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis in ("0", "1", "0,1"):
        raise SystemExit("refusing GPU 0/1, got %s" % vis)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=P2_BEST)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr-det", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--w-det", type=float, default=1.0)
    ap.add_argument("--w-map", type=float, default=10.0)
    ap.add_argument("--w-sum", type=float, default=0.1)
    ap.add_argument("--w-tv", type=float, default=0.05)
    ap.add_argument("--w-sparse", type=float, default=0.0)
    ap.add_argument("--dense-tau", type=float, default=0.0)
    ap.add_argument("--freeze-det", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--head", default="p2p3", choices=("p2p3", "fpn"))
    ap.add_argument("--out", default=os.path.join(RUNS, "YOLO-DenHead-SUM"))
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_batches = args.max_batches or 3
        args.out = args.out + "-smoke"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "visible", vis, "ckpt", args.ckpt, flush=True)
    name = os.environ.get("SWANLAB_RUN") or (
        "YOLO-DenFPN-SUM" if args.head == "fpn" else "YOLO-DenHead-SUM"
    )
    run = None if args.smoke else swan(name, vars(args))
    dual, _yolo = build_dual(args.ckpt, args.imgsz, device, head=args.head)
    train_ds = PestDetDen("train", args.imgsz, augment=True, sigma=args.sigma)
    val_ds = PestDetDen("val", args.imgsz, augment=False, sigma=args.sigma)
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=max(0, args.workers // 2),
        pin_memory=True,
        collate_fn=collate,
    )
    print("n train", len(train_ds), "val", len(val_ds), flush=True)
    best = 1e9
    hist = []
    optim = None
    prev_freeze = None
    for epoch in range(1, args.epochs + 1):
        freeze = epoch <= args.freeze_det
        if optim is None or freeze != prev_freeze:
            dual.set_det_trainable(not freeze)
            optim = make_optim(dual, args.lr_det, args.lr_head, args.wd, det_on=not freeze)
            prev_freeze = freeze
            print("optim freeze_det=%s n_param_groups=%s" % (freeze, len(optim.param_groups)), flush=True)
        t0 = time.time()
        tr = run_epoch(dual, train_loader, device, optim, args, True, freeze)
        va = run_epoch(dual, val_loader, device, None, args, False, freeze)
        rec = {
            "epoch": epoch,
            "freeze_det": freeze,
            "train": tr,
            "val": va,
            "sec": time.time() - t0,
        }
        hist.append(rec)
        print(
            "epoch %s freeze=%s train_loss=%.3f val_loss=%.3f val_den_mae=%.3f val_sparse=%.3f sec=%.0f"
            % (epoch, freeze, tr["loss"], va["loss"], va["den_mae"], va["sparse"], rec["sec"]),
            flush=True,
        )
        extra = {
            "val_den_mae": va["den_mae"],
            "val_loss": va["loss"],
            "yolo_src": args.ckpt,
            "head_type": args.head,
            "dense_tau": args.dense_tau,
            "w_sparse": args.w_sparse,
        }
        save_ckpt(os.path.join(args.out, "last.pt"), dual, epoch, extra)
        score = va["loss"] if args.dense_tau > 0 else va["den_mae"]
        if score < best:
            best = score
            save_ckpt(os.path.join(args.out, "best.pt"), dual, epoch, extra)
            print("best", "val_loss" if args.dense_tau > 0 else "val_den_mae", best, flush=True)
        if hasattr(dual.head, "scale_gates"):
            g = dual.head.scale_gates().detach().cpu().tolist()
            print("fpn gates", ["%.3f" % x for x in g], flush=True)
        json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=2)
        if run:
            try:
                import swanlab

                swanlab.log(
                    {
                        "train/loss": tr["loss"],
                        "train/det": tr["det"],
                        "train/map": tr["map"],
                        "train/sum": tr["sum"],
                        "train/den_mae": tr["den_mae"],
                        "val/loss": va["loss"],
                        "val/den_mae": va["den_mae"],
                        "epoch": epoch,
                    }
                )
            except Exception:
                pass
    print("done best_val_den_mae", best, "out", args.out, flush=True)
    if run:
        try:
            import swanlab

            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
