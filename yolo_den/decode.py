# -*- coding: utf-8 -*-
"""Density-guided NMS and mutually exclusive patch routing.

Zhao-style DG-NMS (mild): T_i = Nt0 + 0.3 minmax(d_i) in [0.5, 0.8], then
decay remaining densities after a keep. Adaptive NMS (Liu CVPR 2019) is the
coarser ancestor that only raises Nt.

On this pest set, raising Nt in crowds mostly un-does duplicate suppression,
so the decoder that is meant to help also has:

  tight  — lower Nt where density is high (more suppression)
  bal    — fixed Nt, then two-sided patch routing (too few OR too many boxes)
  gate   — fixed Nt, dense patches (mass >= tau_m) always use the integral
  blob   — density connected components, not a patch grid: crowds integrate,
           isolated insects keep boxes, unsupported boxes are dropped

Detection and integral never share the same region.
"""
from __future__ import print_function

import numpy as np


class DecodeCfg(object):
    def __init__(
        self,
        box_side=20.0,
        nt0=0.5,
        nt_alpha=0.15,
        nt_max=0.95,
        patch=32,
        tau_r=0.6,
        tau_r_hi=1.4,
        tau_m=3.0,
        zhao_gain=0.3,
        zhao_sigma=0.5,
        tight_gain=0.15,
        tight_min=0.35,
        blob_tmean=2.5,
        blob_tfloor=1e-5,
        blob_close=11,
        blob_dilate=3,
        blob_tau_miss=0.5,
        blob_drop_out=True,
        mode="full",
    ):
        self.box_side = float(box_side)
        self.nt0 = float(nt0)
        self.nt_alpha = float(nt_alpha)
        self.nt_max = float(nt_max)
        self.patch = int(patch)
        self.tau_r = float(tau_r)
        self.tau_r_hi = float(tau_r_hi)
        self.tau_m = float(tau_m)
        self.zhao_gain = float(zhao_gain)
        self.zhao_sigma = float(zhao_sigma)
        self.tight_gain = float(tight_gain)
        self.tight_min = float(tight_min)
        self.blob_tmean = float(blob_tmean)
        self.blob_tfloor = float(blob_tfloor)
        self.blob_close = int(blob_close)
        self.blob_dilate = int(blob_dilate)
        self.blob_tau_miss = float(blob_tau_miss)
        self.blob_drop_out = bool(blob_drop_out)
        self.mode = str(mode)


def points_to_boxes(points_yx, box_side):
    """PET points are (row, col). Boxes are xyxy in pixel coordinates."""
    if points_yx is None or len(points_yx) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    pts = np.asarray(points_yx, dtype=np.float32)
    half = 0.5 * float(box_side)
    y = pts[:, 0]
    x = pts[:, 1]
    boxes = np.stack([x - half, y - half, x + half, y + half], axis=1)
    return boxes.astype(np.float32)


def box_iou_matrix(boxes):
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    xx1 = np.maximum(x1[:, None], x1[None, :])
    yy1 = np.maximum(y1[:, None], y1[None, :])
    xx2 = np.minimum(x2[:, None], x2[None, :])
    yy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    union = area[:, None] + area[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(iou, 0.0)
    return iou.astype(np.float32)


def box_diou_matrix(boxes):
    """DIoU = IoU - rho^2(centers) / c^2. Same layout as box_iou_matrix."""
    iou = box_iou_matrix(boxes)
    n = boxes.shape[0]
    if n == 0:
        return iou
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    dist2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
    xx1 = np.minimum(boxes[:, 0][:, None], boxes[:, 0][None, :])
    yy1 = np.minimum(boxes[:, 1][:, None], boxes[:, 1][None, :])
    xx2 = np.maximum(boxes[:, 2][:, None], boxes[:, 2][None, :])
    yy2 = np.maximum(boxes[:, 3][:, None], boxes[:, 3][None, :])
    c2 = (xx2 - xx1) ** 2 + (yy2 - yy1) ** 2
    diou = iou - dist2 / np.maximum(c2, 1e-9)
    np.fill_diagonal(diou, 0.0)
    return diou.astype(np.float32)


def box_masses(density, boxes):
    h, w = density.shape
    masses = np.zeros((boxes.shape[0],), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        xa = int(max(0, np.floor(x1)))
        ya = int(max(0, np.floor(y1)))
        xb = int(min(w, np.ceil(x2)))
        yb = int(min(h, np.ceil(y2)))
        if xb > xa and yb > ya:
            masses[i] = float(density[ya:yb, xa:xb].sum())
    return masses


def nms_thresholds(masses, cfg):
    extra = np.maximum(masses - 1.0, 0.0) * cfg.nt_alpha
    nt = np.clip(cfg.nt0 + extra, cfg.nt0, cfg.nt_max)
    return nt.astype(np.float32)


def greedy_nms(boxes, scores, thresholds, overlap=None):
    """Classic greedy NMS; each kept box uses its own overlap threshold.

    overlap defaults to IoU; pass box_diou_matrix(boxes) for DIoU-NMS.
    """
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    iou = box_iou_matrix(boxes) if overlap is None else overlap
    suppressed = np.zeros(n, dtype=np.bool_)
    keep = []
    for i in order:
        i = int(i)
        if suppressed[i]:
            continue
        keep.append(i)
        thr = float(thresholds[i])
        for j in order:
            j = int(j)
            if j == i or suppressed[j]:
                continue
            if float(iou[i, j]) >= thr:
                suppressed[j] = True
    return np.asarray(keep, dtype=np.int64)


def _patch_slices(h, w, patch):
    rows = []
    y = 0
    while y < h:
        x = 0
        y1 = min(h, y + patch)
        while x < w:
            x1 = min(w, x + patch)
            rows.append((y, y1, x, x1))
            x = x1
        y = y1
    return rows


def sample_density_centers(density, boxes):
    h, w = density.shape
    d = np.zeros((boxes.shape[0],), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx = int(np.clip(round(0.5 * (x1 + x2)), 0, w - 1))
        cy = int(np.clip(round(0.5 * (y1 + y2)), 0, h - 1))
        d[i] = float(density[cy, cx])
    return d


def minmax01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def zhao_dg_nms(boxes, scores, density, cfg):
    """Zhao et al. DG-NMS: T_i = 0.5 + 0.3 minmax(d_i), then decay remaining density.

    T stays in [0.5, 0.8]. After a box is kept, neighbors decay:
        d_i <- d_i * exp(-IoU^2 / sigma)
    so later thresholds fall and duplicates are less likely to survive.
    """
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    d = minmax01(sample_density_centers(density, boxes))
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    iou = box_iou_matrix(boxes)
    live = np.ones(n, dtype=np.bool_)
    keep = []
    sig = max(float(cfg.zhao_sigma), 1e-6)
    gain = float(cfg.zhao_gain)
    nt0 = float(cfg.nt0)
    for t in order:
        t = int(t)
        if not live[t]:
            continue
        keep.append(t)
        live[t] = False
        for j in order:
            j = int(j)
            if not live[j]:
                continue
            ov = float(iou[t, j])
            thr = nt0 + gain * float(np.clip(d[j], 0.0, 1.0))
            if ov > thr:
                live[j] = False
            else:
                d[j] *= np.exp(-(ov * ov) / sig)
    return np.asarray(keep, dtype=np.int64)


def tight_nms(boxes, scores, density, cfg):
    """Dense regions get a *lower* IoU threshold (more duplicate removal)."""
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    f = minmax01(sample_density_centers(density, boxes))
    thr = np.clip(cfg.nt0 - cfg.tight_gain * f, cfg.tight_min, cfg.nt0)
    return greedy_nms(boxes, scores, thr)


def route_patch_records(density, boxes, keep, cfg):
    """Same routing as route_patches, plus per-patch records for visualization."""
    h, w = density.shape
    n_keep = int(keep.shape[0])
    centers = np.zeros((n_keep, 2), dtype=np.float32)
    if n_keep:
        kboxes = boxes[keep]
        centers[:, 0] = np.clip(0.5 * (kboxes[:, 1] + kboxes[:, 3]), 0, h - 1e-4)
        centers[:, 1] = np.clip(0.5 * (kboxes[:, 0] + kboxes[:, 2]), 0, w - 1e-4)
    used = np.zeros(n_keep, dtype=np.bool_)
    det_n = 0.0
    den_n = 0.0
    n_switch = 0
    recs = []
    for y0, y1, x0, x1 in _patch_slices(h, w, cfg.patch):
        mass = float(density[y0:y1, x0:x1].sum())
        if n_keep:
            in_p = (
                (centers[:, 0] >= y0)
                & (centers[:, 0] < y1)
                & (centers[:, 1] >= x0)
                & (centers[:, 1] < x1)
            )
            n_box = int(in_p.sum())
        else:
            in_p = None
            n_box = 0
        ratio = n_box / max(mass, 1e-6)
        too_few = ratio < cfg.tau_r
        too_many = ratio > cfg.tau_r_hi
        mass_only = cfg.mode == "gate" or cfg.tau_r_hi <= 0
        switch = mass >= cfg.tau_m and (True if mass_only else (too_few or too_many))
        recs.append(
            {
                "y0": int(y0),
                "y1": int(y1),
                "x0": int(x0),
                "x1": int(x1),
                "mass": mass,
                "n_box": int(n_box),
                "switch": bool(switch),
            }
        )
        if switch:
            den_n += mass
            n_switch += 1
        else:
            det_n += float(n_box)
            if in_p is not None:
                used[in_p] = True
    keep_det = keep[used] if n_keep else keep
    return det_n, den_n, n_switch, keep_det, recs


def _window_max(mask, k):
    r = max(int(k) // 2, 0)
    if r <= 0:
        return mask.astype(np.bool_)
    m = mask.astype(np.uint8)
    h, w = m.shape
    padded = np.pad(m, r, mode="constant")
    acc = np.zeros((h, w), dtype=np.uint8)
    side = 2 * r + 1
    for dy in range(side):
        for dx in range(side):
            acc = np.maximum(acc, padded[dy : dy + h, dx : dx + w])
    return acc.astype(np.bool_)


def _window_min(mask, k):
    r = max(int(k) // 2, 0)
    if r <= 0:
        return mask.astype(np.bool_)
    m = mask.astype(np.uint8)
    h, w = m.shape
    padded = np.pad(m, r, mode="constant", constant_values=1)
    acc = np.ones((h, w), dtype=np.uint8)
    side = 2 * r + 1
    for dy in range(side):
        for dx in range(side):
            acc = np.minimum(acc, padded[dy : dy + h, dx : dx + w])
    return acc.astype(np.bool_)


def _binary_close(mask, k):
    k = int(k)
    if k <= 1:
        return mask.astype(np.bool_)
    try:
        from scipy.ndimage import binary_closing

        st = np.ones((k, k), dtype=bool)
        return binary_closing(mask, structure=st)
    except Exception:
        return _window_min(_window_max(mask, k), k)


def _binary_dilate(mask, k):
    k = int(k)
    if k <= 0:
        return mask.astype(np.bool_)
    try:
        from scipy.ndimage import binary_dilation

        st = np.ones((k, k), dtype=bool)
        return binary_dilation(mask, structure=st)
    except Exception:
        return _window_max(mask, k)


def _cc_label_np(mask):
    m = np.asarray(mask, dtype=np.bool_)
    h, w = m.shape
    labels = np.zeros((h, w), dtype=np.int32)
    nlab = 0
    for y in range(h):
        row = m[y]
        for x in range(w):
            if not row[x] or labels[y, x]:
                continue
            nlab += 1
            stack = [(y, x)]
            labels[y, x] = nlab
            while stack:
                cy, cx = stack.pop()
                y0 = cy - 1 if cy else 0
                y1 = cy + 2 if cy + 1 < h else h
                x0 = cx - 1 if cx else 0
                x1 = cx + 2 if cx + 1 < w else w
                for ny in range(y0, y1):
                    for nx in range(x0, x1):
                        if m[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = nlab
                            stack.append((ny, nx))
    return labels, nlab


def _cc_label(mask):
    try:
        from scipy.ndimage import label

        labels, nlab = label(mask)
        return labels.astype(np.int32), int(nlab)
    except Exception:
        return _cc_label_np(mask)


def density_blobs(density, cfg, close=None, dilate=None):
    """Threshold + close + dilate, then 8-connected components on D."""
    d = np.nan_to_num(np.asarray(density, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    t = max(float(cfg.blob_tfloor), float(cfg.blob_tmean) * float(max(d.mean(), 0.0)))
    mask = d >= t
    ck = cfg.blob_close if close is None else int(close)
    dk = cfg.blob_dilate if dilate is None else int(dilate)
    mask = _binary_close(mask, ck)
    mask = _binary_dilate(mask, dk)
    labels, nlab = _cc_label(mask)
    return d, labels, nlab, float(t)


def blob_route(density, boxes, keep, cfg, close=None, dilate=None, drop_out=None):
    """Connected-component exclusive routing.

    Density grows its own regions (not a 32x32 grid). A component with mass
    >= tau_m is a crowd: count = integral, drop boxes inside. Smaller
    components keep detection boxes. Boxes that sit on empty background
    (no component) are dropped as unsupported false positives. A small
    component with no box still contributes its mass if it looks like a miss.
    """
    d, labels, nlab, t = density_blobs(density, cfg, close=close, dilate=dilate)
    h, w = d.shape
    masses = np.bincount(labels.ravel(), weights=d.ravel(), minlength=nlab + 1).astype(np.float64)
    n_of = np.zeros(nlab + 1, dtype=np.int32)
    n_keep = int(keep.shape[0])
    box_lab = np.zeros(n_keep, dtype=np.int32)
    if n_keep:
        kb = boxes[keep]
        cy = np.clip(np.round(0.5 * (kb[:, 1] + kb[:, 3])).astype(np.int32), 0, h - 1)
        cx = np.clip(np.round(0.5 * (kb[:, 0] + kb[:, 2])).astype(np.int32), 0, w - 1)
        box_lab = labels[cy, cx]
        for lab in box_lab:
            n_of[int(lab)] += 1
    if drop_out is None:
        drop_out = bool(cfg.blob_drop_out)
    used = np.zeros(n_keep, dtype=np.bool_)
    det_n = 0.0
    den_n = 0.0
    n_dense = 0
    n_sparse = 0
    n_miss = 0
    kinds = {}
    for lab in range(1, nlab + 1):
        mass = float(masses[lab])
        n_box = int(n_of[lab])
        if mass >= cfg.tau_m:
            den_n += mass
            n_dense += 1
            kinds[lab] = "dense"
        else:
            n_sparse += 1
            if n_box > 0:
                det_n += float(n_box)
                kinds[lab] = "sparse"
                if n_keep:
                    used[box_lab == lab] = True
            elif mass >= cfg.blob_tau_miss:
                den_n += mass
                n_miss += 1
                kinds[lab] = "miss"
            else:
                kinds[lab] = "noise"
    if n_keep and not drop_out:
        out = box_lab == 0
        det_n += float(out.sum())
        used[out] = True
    keep_det = keep[used] if n_keep else keep
    stats = {
        "t": t,
        "n_cc": int(nlab),
        "n_dense": int(n_dense),
        "n_sparse": int(n_sparse),
        "n_miss": int(n_miss),
        "n_switch": int(n_dense + n_miss),
        "n_drop": int(n_keep - int(used.sum())) if n_keep else 0,
        "kinds": kinds,
        "labels": labels,
    }
    return det_n, den_n, stats, keep_det


def route_patches(density, boxes, keep, cfg):
    """Mutually exclusive patch routing.

    If mass M is large and the box/density ratio is too low (inseparable) or
    too high (duplicate boxes), count = M and drop those boxes. Otherwise
    count the kept boxes.
    """
    det_n, den_n, n_switch, keep_det, _recs = route_patch_records(density, boxes, keep, cfg)
    return det_n, den_n, n_switch, keep_det


def decode_count(points_yx, scores, density, cfg=None):
    """Return count plus branch diagnostics.

    modes:
      pet       — raw point count (after the detector's own score filter)
      den       — density integral only
      nms_fixed — boxes + constant Nt0, no routing
      nms_diou  — boxes + DIoU-NMS, constant Nt0, no routing
      nms_den   — density-guided NMS, no routing
      full      — density-guided NMS + exclusive integral on unexplained patches
    """
    if cfg is None:
        cfg = DecodeCfg()
    pts = np.zeros((0, 2), dtype=np.float32) if points_yx is None else np.asarray(points_yx, dtype=np.float32)
    sc = np.zeros((0,), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
    den = np.nan_to_num(np.asarray(density, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if den.ndim != 2:
        raise ValueError("density must be HxW")
    rec = {
        "count_pet": float(pts.shape[0]),
        "count_den": float(den.sum()),
        "n_switch": 0,
        "n_nms": 0,
        "det_n": 0.0,
        "den_n": 0.0,
    }
    mode = cfg.mode
    if mode == "pet":
        rec["count"] = rec["count_pet"]
        return rec
    if mode == "den":
        rec["count"] = rec["count_den"]
        return rec

    side = cfg.box_side
    if side <= 0:
        if pts.shape[0] >= 2:
            d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            side = float(np.clip(np.median(d.min(axis=1)) * 0.9, 8.0, 32.0))
        else:
            side = 20.0
    boxes = points_to_boxes(pts, side)
    out = decode_boxes(boxes, sc, den, cfg)
    out["count_pet"] = rec["count_pet"]
    return out


def decode_boxes(boxes_xyxy, scores, density, cfg=None):
    """Same decoder as decode_count, but boxes are detector xyxy (not expanded points)."""
    if cfg is None:
        cfg = DecodeCfg()
    boxes = np.zeros((0, 4), dtype=np.float32) if boxes_xyxy is None else np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    sc = np.zeros((0,), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] != sc.shape[0]:
        n = min(boxes.shape[0], sc.shape[0])
        boxes = boxes[:n]
        sc = sc[:n]
    den = np.nan_to_num(np.asarray(density, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if den.ndim != 2:
        raise ValueError("density must be HxW")
    rec = {
        "count_det": float(boxes.shape[0]),
        "count_pet": float(boxes.shape[0]),
        "count_den": float(den.sum()),
        "n_switch": 0,
        "n_nms": 0,
        "det_n": 0.0,
        "den_n": 0.0,
    }
    mode = cfg.mode
    if mode in ("pet", "det", "yolo"):
        rec["count"] = rec["count_det"]
        rec["det_n"] = rec["count_det"]
        return rec
    if mode == "den":
        rec["count"] = rec["count_den"]
        return rec
    masses = box_masses(den, boxes)
    if mode == "nms_fixed":
        keep = greedy_nms(boxes, sc, np.full((boxes.shape[0],), cfg.nt0, dtype=np.float32))
        rec["mean_nt"] = cfg.nt0
    elif mode == "nms_diou":
        keep = greedy_nms(
            boxes,
            sc,
            np.full((boxes.shape[0],), cfg.nt0, dtype=np.float32),
            overlap=box_diou_matrix(boxes),
        )
        rec["mean_nt"] = cfg.nt0
    elif mode == "zhao":
        keep = zhao_dg_nms(boxes, sc, den, cfg)
        rec["mean_nt"] = cfg.nt0 + 0.5 * cfg.zhao_gain
    elif mode == "tight":
        keep = tight_nms(boxes, sc, den, cfg)
        rec["mean_nt"] = cfg.nt0 - 0.5 * cfg.tight_gain
    elif mode == "nms_den":
        thr = nms_thresholds(masses, cfg)
        keep = greedy_nms(boxes, sc, thr)
        rec["mean_nt"] = float(thr.mean()) if thr.size else cfg.nt0
    else:
        # full / zhao_bal / tight_bal / bal: NMS then two-sided routing
        if mode in ("zhao_bal", "full_zhao"):
            keep = zhao_dg_nms(boxes, sc, den, cfg)
        elif mode in ("tight_bal",):
            keep = tight_nms(boxes, sc, den, cfg)
        elif mode in ("full", "nms_den_bal"):
            thr = nms_thresholds(masses, cfg)
            keep = greedy_nms(boxes, sc, thr)
            rec["mean_nt"] = float(thr.mean()) if thr.size else cfg.nt0
        else:
            # bal, fix_bal, full_fix, gate, blob*, blob_diou*: fixed Nt, then routing
            ov = box_diou_matrix(boxes) if str(mode).startswith("blob_diou") else None
            keep = greedy_nms(
                boxes,
                sc,
                np.full((boxes.shape[0],), cfg.nt0, dtype=np.float32),
                overlap=ov,
            )
            rec["mean_nt"] = cfg.nt0
    rec["n_nms"] = int(keep.shape[0])
    rec["mean_box_mass"] = float(masses.mean()) if masses.size else 0.0
    if mode in ("nms_fixed", "nms_den", "zhao", "tight", "nms_diou"):
        rec["count"] = float(keep.shape[0])
        rec["det_n"] = float(keep.shape[0])
        return rec
    if str(mode).startswith("blob"):
        close = 1 if mode == "blob_open" else None
        drop = False if mode == "blob_keep" else None
        det_n, den_n, stats, keep_det = blob_route(den, boxes, keep, cfg, close=close, drop_out=drop)
        rec["det_n"] = float(det_n)
        rec["den_n"] = float(den_n)
        rec["n_switch"] = int(stats["n_switch"])
        rec["n_cc"] = int(stats["n_cc"])
        rec["n_dense"] = int(stats["n_dense"])
        rec["n_sparse"] = int(stats["n_sparse"])
        rec["n_miss"] = int(stats["n_miss"])
        rec["n_drop"] = int(stats["n_drop"])
        rec["n_det_boxes"] = int(keep_det.shape[0])
        rec["count"] = float(det_n + den_n)
        rec["blob_t"] = float(stats["t"])
        return rec
    det_n, den_n, n_switch, keep_det = route_patches(den, boxes, keep, cfg)
    rec["det_n"] = float(det_n)
    rec["den_n"] = float(den_n)
    rec["n_switch"] = int(n_switch)
    rec["n_det_boxes"] = int(keep_det.shape[0])
    rec["count"] = float(det_n + den_n)
    return rec


def _self_test():
    h = w = 64
    den = np.zeros((h, w), dtype=np.float32)
    den[8:24, 8:24] = 8.0 / (16.0 * 16.0)
    den[40:48, 40:48] = 1.0 / (8.0 * 8.0)
    pts = np.array([[16.0, 16.0], [18.0, 18.0], [44.0, 44.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.95], dtype=np.float32)
    full = decode_count(pts, scores, den, DecodeCfg(box_side=16, patch=32, tau_r=0.6, tau_m=2.0, mode="full"))
    nms = decode_count(pts, scores, den, DecodeCfg(box_side=16, mode="nms_den"))
    boxes = points_to_boxes(pts, 16)
    boxed = decode_boxes(boxes, scores, den, DecodeCfg(box_side=16, patch=32, tau_r=0.6, tau_m=2.0, mode="full"))
    zhao = decode_boxes(boxes, scores, den, DecodeCfg(box_side=16, mode="zhao"))
    bal = decode_boxes(boxes, scores, den, DecodeCfg(box_side=16, patch=32, tau_r=0.6, tau_r_hi=1.4, tau_m=2.0, mode="bal"))
    gate = decode_boxes(boxes, scores, den, DecodeCfg(box_side=16, patch=32, tau_m=2.0, mode="gate"))
    fp = np.concatenate([boxes, np.array([[1.0, 1.0, 9.0, 9.0]], dtype=np.float32)], axis=0)
    sc_fp = np.concatenate([scores, np.array([0.4], dtype=np.float32)], axis=0)
    blob = decode_boxes(fp, sc_fp, den, DecodeCfg(box_side=16, tau_m=3.0, blob_close=5, blob_dilate=1, mode="blob"))
    assert nms["count"] >= 1
    assert full["count"] > 0
    assert abs(boxed["count"] - full["count"]) < 1e-4
    assert zhao["count"] >= 1
    assert bal["count"] > 0
    assert gate["count"] > 0
    assert blob["n_drop"] >= 1
    assert blob["count"] > 0
    print(
        "dennms.decode ok",
        full,
        "zhao",
        zhao["count"],
        "bal",
        bal["count"],
        "gate",
        gate["count"],
        "blob",
        blob["count"],
        "drop",
        blob["n_drop"],
        "cc",
        blob["n_cc"],
    )
    return full


if __name__ == "__main__":
    _self_test()
