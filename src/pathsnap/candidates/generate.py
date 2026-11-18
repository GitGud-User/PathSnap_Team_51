"""Candidate road-segment generation with a spatial index (shapely STRtree, an R-tree).

For each GPS fix we return up to K directed arcs whose segment lies within RADIUS_M of the
fix (projected metres), ordered by perpendicular distance. Arc ids are preserved exactly
(candidate row -> arc_id), which the classifier and HMM rely on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from pathsnap.data.loaders import Trip

RADIUS_M = 50.0
TOP_K = 8


class ArcIndex:
    """Spatial index over one trip's valid arcs (self-loops / bad refs excluded)."""

    def __init__(self, trip: Trip):
        u, v = trip.arcs[:, 0], trip.arcs[:, 1]
        n = trip.n_nodes
        self.valid_arcs = np.flatnonzero((u >= 0) & (v >= 0) & (u < n) & (v < n) & (u != v))
        xy = trip.nodes_xy
        self.a = xy[u[self.valid_arcs]]
        self.b = xy[v[self.valid_arcs]]
        self.tree = STRtree(shapely.linestrings(np.stack([self.a, self.b], axis=1)))


def generate(trip: Trip, index: ArcIndex | None = None, radius_m: float = RADIUS_M,
             top_k: int = TOP_K) -> pd.DataFrame:
    """Rows: obs_idx, cand_rank, arc_id, perp_dist_m, proj_t, proj_x, proj_y."""
    idx = index or ArcIndex(trip)
    pts = trip.gps_xy
    obs, loc = idx.tree.query(shapely.points(pts), predicate="dwithin", distance=radius_m)
    if len(obs) == 0:
        return pd.DataFrame(columns=["obs_idx", "cand_rank", "arc_id", "perp_dist_m", "proj_t",
                                     "proj_x", "proj_y"])
    a, b, p = idx.a[loc], idx.b[loc], pts[obs]
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1), 0, 1)
    proj = a + t[:, None] * ab
    d = np.hypot(*(p - proj).T)
    df = pd.DataFrame({"obs_idx": obs, "arc_id": idx.valid_arcs[loc], "perp_dist_m": d,
                       "proj_t": t, "proj_x": proj[:, 0], "proj_y": proj[:, 1]})
    df = df.sort_values(["obs_idx", "perp_dist_m"], kind="stable")
    df["cand_rank"] = df.groupby("obs_idx").cumcount()
    df = df[df.cand_rank < top_k].reset_index(drop=True)
    return df[["obs_idx", "cand_rank", "arc_id", "perp_dist_m", "proj_t", "proj_x", "proj_y"]]
