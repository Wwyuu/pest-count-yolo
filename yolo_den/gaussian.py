# -*- coding: utf-8 -*-
"""Point GT -> density map (isotropic Gaussian, sigma=4, unit mass)."""
from __future__ import print_function

import numpy as np


def gaussian_density_yx(points_yx, h, w, sigma=4.0):
    den = np.zeros((h, w), dtype=np.float32)
    if points_yx is None:
        return den
    pts = np.asarray(points_yx, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        return den
    r = int(max(1, round(3 * sigma)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    g /= g.sum()
    for y, x in pts:
        cx, cy = int(round(x)), int(round(y))
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        gx0, gx1 = x0 - (cx - r), x1 - (cx - r)
        gy0, gy1 = y0 - (cy - r), y1 - (cy - r)
        den[y0:y1, x0:x1] += g[gy0:gy1, gx0:gx1]
    return den
