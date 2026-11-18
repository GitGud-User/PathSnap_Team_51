"""Per-observation labels derived from the reference route (train / student_test only).

The .route file is an ordered arc sequence R_0..R_{m-1}, not a label per GPS fix. We assign
each fix i an index j(i) into the route with a monotonic dynamic programme:

    D[i, j] = c(i, j) + min_{k <= j} D[i-1, k]          (j never moves backwards in time)
    c(i, j) = min(dist(fix_i, arc R_j), CAP_M)

The cap stops long off-route stretches (e.g. trips 33/54) from dragging the alignment.
Label(i) = R_{j(i)} if dist(fix_i, R_{j(i)}) <= MATCH_M, else -1 (UNMATCHED: excluded from
training targets and from accuracy denominators, and counted in reports).
"""
from __future__ import annotations

import numpy as np

from pathsnap.data.loaders import Trip

MATCH_M = 50.0
CAP_M = 60.0


def _point_seg_dist(p, a, b):
    ab = b - a
    L2 = (ab ** 2).sum(-1)
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / np.where(L2 > 0, L2, 1)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.hypot(*(p[:, None, :] - proj).transpose(2, 0, 1))


def align_labels(trip: Trip, route: np.ndarray) -> np.ndarray:
    route = route[(route >= 0) & (route < trip.n_arcs)]
    xy = trip.nodes_xy
    a, b = xy[trip.arcs[route, 0]], xy[trip.arcs[route, 1]]
    pts = trip.gps_xy
    n, m = len(pts), len(route)
    back = np.empty((n, m), dtype=np.int32)
    dist_rows = np.empty((n, m), dtype=np.float32)
    prev = None
    ar = np.arange(m)
    for s in range(0, n, 512):  # chunked distance computation
        dist_rows[s:s + 512] = _point_seg_dist(pts[s:s + 512], a, b)
    for i in range(n):
        c = np.minimum(dist_rows[i], CAP_M)
        if prev is None:
            cur = c.astype(np.float64)
            back[i] = ar
        else:
            cm = np.minimum.accumulate(prev)
            arg = np.maximum.accumulate(np.where(prev <= cm, ar, 0))
            cur = c + cm
            back[i] = arg
        prev = cur
    j = int(np.argmin(prev))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    labels = route[path].astype(np.int64)
    labels[dist_rows[np.arange(n), path] > MATCH_M] = -1
    return labels
