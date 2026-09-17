# -*- coding: utf-8 -*-
"""YOLOv8-P2 detection head + density head on the same P2/P3 neck.

One forward yields boxes and D. Density is fused from the two highest-resolution
Detect inputs (P2 then P3 on yolov8-p2; P3 then P4 on a 3-scale detector).
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_detect(model):
    last = None
    seq = model.model if hasattr(model, "model") else model
    for m in seq.modules():
        name = m.__class__.__name__
        if name in ("Detect", "v8Detect", "v10Detect", "Detect_v8"):
            last = m
    if last is not None:
        return last
    if hasattr(seq, "__getitem__"):
        return seq[-1]
    raise RuntimeError("no Detect module")


class ConvGNSiLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1):
        super(ConvGNSiLU, self).__init__()
        p = d * (k // 2)
        ng = 8 if c_out % 8 == 0 else (4 if c_out % 4 == 0 else 1)
        self.cv = nn.Conv2d(c_in, c_out, k, padding=p, dilation=d, bias=False)
        self.gn = nn.GroupNorm(ng, c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.cv(x)))


class NeckDensityHead(nn.Module):
    """1x1 project P2/P3, upsample P3, 3x3 fuse, ReLU density."""

    def __init__(self, c2, c3, mid=128):
        super(NeckDensityHead, self).__init__()
        self.p2 = nn.Conv2d(c2, mid, 1, bias=True)
        self.p3 = nn.Conv2d(c3, mid, 1, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, p2, p3):
        a = self.p2(p2)
        b = self.p3(p3)
        if b.shape[-2:] != a.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([a, b], 1))


class FPNDensityHead(nn.Module):
    """Use every YOLO Detect scale (P2..P5).

    Top-down FPN: P5 (crowd context) upsamples onto P4/P3/P2 (small-insect detail).
    Learnable sigmoid gates pick which scale to trust. A dilated tail then
    expands receptive field at P2 resolution, CSRNet-style, without pooling away
    the high-res map YOLO already paid for.
    """

    def __init__(self, chs, mid=128):
        super(FPNDensityHead, self).__init__()
        chs = [int(c) for c in chs]
        self.chs = chs
        self.n = len(chs)
        if self.n < 2:
            raise ValueError("FPNDensityHead needs >=2 FPN scales")
        self.lateral = nn.ModuleList([nn.Conv2d(c, mid, 1, bias=True) for c in chs])
        self.gate = nn.Parameter(torch.zeros(self.n))
        self.down = nn.ModuleList([ConvGNSiLU(mid, mid, 3, d=1) for _ in range(self.n - 1)])
        self.backend = nn.Sequential(
            ConvGNSiLU(mid, mid, 3, d=2),
            ConvGNSiLU(mid, mid, 3, d=2),
            ConvGNSiLU(mid, mid, 3, d=2),
            ConvGNSiLU(mid, mid // 2, 3, d=1),
            nn.Conv2d(mid // 2, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def scale_gates(self):
        return torch.sigmoid(self.gate)

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            raise TypeError("FPNDensityHead expects a list of FPN maps")
        if len(feats) != self.n:
            raise ValueError("got %s FPN maps, built for %s" % (len(feats), self.n))
        g = self.scale_gates()
        laterals = [g[i] * self.lateral[i](feats[i]) for i in range(self.n)]
        x = laterals[-1]
        for i in range(self.n - 2, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.down[i](x + laterals[i])
        return self.backend(x)


def downsample_density(gt_full, dh, dw):
    b, _, h, w = gt_full.shape
    pooled = F.adaptive_avg_pool2d(gt_full, (dh, dw))
    return pooled * float(h * w) / float(max(dh * dw, 1))


def upsample_density(den_low, h, w):
    if den_low.dim() == 2:
        den_low = den_low.unsqueeze(0).unsqueeze(0)
    elif den_low.dim() == 3:
        den_low = den_low.unsqueeze(0)
    den_low = torch.nan_to_num(den_low.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    h = max(int(h), 1)
    w = max(int(w), 1)
    up = F.interpolate(den_low, size=(h, w), mode="bilinear", align_corners=False)
    up = torch.nan_to_num(up, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    src = float(den_low.sum().item())
    dst = float(up.sum().item())
    if dst < 1e-12:
        return up
    return up * (src / dst)


def joint_density_losses(pred, gt_full, dense_tau=0.0):
    """Pixel MSE (sum/B) + count L1 + normalized-map L1.

    dense_tau > 0: supervise only GT pixels >= tau (crowds). Sparse mass is
    returned separately so the trainer can push D to 0 there (boxes count isolates).
    """
    gt = downsample_density(gt_full, pred.shape[-2], pred.shape[-1])
    b = pred.shape[0]
    if dense_tau and float(dense_tau) > 0:
        mask = (gt >= float(dense_tau)).to(pred.dtype)
        loss_map = ((pred - gt).pow(2) * mask).sum() / float(max(b, 1))
        loss_sum = (
            (pred * mask).flatten(1).sum(1) - (gt * mask).flatten(1).sum(1)
        ).abs().mean()
        ps = (pred * mask).flatten(1).sum(1).clamp_min(1e-6).view(b, 1, 1, 1)
        gs = (gt * mask).flatten(1).sum(1).clamp_min(1e-6).view(b, 1, 1, 1)
        denom = mask.sum().clamp_min(1.0)
        loss_tv = (((pred / ps) - (gt / gs)).abs() * mask).sum() / denom
        loss_sparse = (pred * (1.0 - mask)).flatten(1).sum(1).mean()
        return loss_map, loss_sum, loss_tv, gt, loss_sparse
    loss_map = F.mse_loss(pred, gt, reduction="sum") / float(max(b, 1))
    loss_sum = (pred.flatten(1).sum(1) - gt.flatten(1).sum(1)).abs().mean()
    ps = pred.flatten(1).sum(1).clamp_min(1e-6).view(b, 1, 1, 1)
    gs = gt.flatten(1).sum(1).clamp_min(1e-6).view(b, 1, 1, 1)
    loss_tv = ((pred / ps) - (gt / gs)).abs().mean()
    loss_sparse = pred.new_zeros(())
    return loss_map, loss_sum, loss_tv, gt, loss_sparse


def ensure_det_args(det):
    """YOLO26 loss reads hyp.box as attributes, not dict keys."""
    try:
        from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, IterableSimpleNamespace
    except Exception:
        DEFAULT_CFG = None
        DEFAULT_CFG_DICT = {"box": 7.5, "cls": 0.5, "dfl": 1.5}

        class IterableSimpleNamespace(object):
            def __init__(self, **kw):
                self.__dict__.update(kw)

    base = dict(DEFAULT_CFG_DICT)
    cur = getattr(det, "args", None)
    extra = {}
    if isinstance(cur, dict):
        extra = {k: v for k, v in cur.items() if v is not None}
    elif cur is not None and not hasattr(cur, "box"):
        extra = dict(cur) if hasattr(cur, "items") else dict(vars(cur))
    elif cur is not None and hasattr(cur, "box"):
        return det
    if extra:
        base.update(extra)
        det.args = IterableSimpleNamespace(**base)
    elif cur is None:
        det.args = DEFAULT_CFG if DEFAULT_CFG is not None else IterableSimpleNamespace(**base)
    return det


def ensure_criterion(det):
    ensure_det_args(det)
    crit = getattr(det, "criterion", None)
    if crit is None:
        det.criterion = det.init_criterion()
        crit = det.criterion
    if getattr(crit, "hyp", None) is not None and not hasattr(crit.hyp, "box"):
        crit.hyp = det.args
    else:
        crit.hyp = det.args
    return crit


class DualYoloDen(nn.Module):
    """DetectionModel + density head. One backbone, two heads."""

    def __init__(self, det, head):
        super(DualYoloDen, self).__init__()
        self.det = det
        self.head = head
        self._feats = None
        self._detect = _find_detect(det)
        self._hook = self._detect.register_forward_pre_hook(self._on_detect)

    def _on_detect(self, module, inputs):
        self._feats = inputs[0]

    def neck_feats(self):
        feats = self._feats
        if feats is None:
            raise RuntimeError("Detect hook did not fire")
        if not isinstance(feats, (list, tuple)):
            raise RuntimeError("Detect expected a feature list, got %s" % type(feats))
        if len(feats) < 2:
            raise RuntimeError("need >=2 Detect scales, got %s" % len(feats))
        return list(feats)

    def p2p3(self):
        feats = self.neck_feats()
        return feats[0], feats[1]

    def forward(self, x):
        preds = self.det(x)
        feats = self.neck_feats()
        if isinstance(self.head, FPNDensityHead):
            den = self.head(feats)
        else:
            den = self.head(feats[0], feats[1])
        return preds, den

    def set_det_trainable(self, flag):
        for p in self.det.parameters():
            p.requires_grad = bool(flag)

    def head_kind(self):
        return "fpn" if isinstance(self.head, FPNDensityHead) else "p2p3"

    @classmethod
    def from_detection_model(cls, det, imgsz=640, device=None, head="fpn"):
        device = device or next(det.parameters()).device
        detect = _find_detect(det)
        holder = {}

        def _cap(m, inp):
            holder["f"] = inp[0]

        h = detect.register_forward_pre_hook(_cap)
        was = det.training
        det.eval()
        with torch.no_grad():
            _ = det(torch.zeros(1, 3, int(imgsz), int(imgsz), device=device))
        h.remove()
        if was:
            det.train()
        f = holder.get("f")
        if f is None:
            raise RuntimeError("could not capture Detect inputs")
        chs = [int(t.shape[1]) for t in f]
        tags = []
        for t in f:
            h = int(t.shape[-2])
            stride = int(round(float(imgsz) / float(max(h, 1))))
            tags.append({4: "P2", 8: "P3", 16: "P4", 32: "P5"}.get(stride, "s%s" % stride))
        print(
            "neck %s chs=%s"
            % (" ".join(["%s%s" % (tags[i], tuple(t.shape)) for i, t in enumerate(f)]), chs),
            flush=True,
        )
        kind = str(head or "fpn").lower()
        if kind in ("fpn", "all", "p2p5"):
            hd = FPNDensityHead(chs)
        else:
            hd = NeckDensityHead(chs[0], chs[1])
        print("density head", kind, hd.__class__.__name__, flush=True)
        return cls(det, hd)
