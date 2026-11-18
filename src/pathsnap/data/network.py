"""Road-network structure analysis for one trip (directed graph: node -> node via arc)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.data.loaders import Trip


def arc_lengths_m(trip: Trip) -> np.ndarray:
    """Length of each arc in metres (projected). NaN for arcs with invalid node refs."""
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    ok = (u >= 0) & (v >= 0) & (u < trip.n_nodes) & (v < trip.n_nodes)
    out = np.full(len(u), np.nan)
    xy = trip.nodes_xy
    out[ok] = np.hypot(*(xy[v[ok]] - xy[u[ok]]).T)
    return out


def network_stats(trip: Trip) -> dict:
    n, a = trip.n_nodes, trip.n_arcs
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    valid = (u >= 0) & (v >= 0) & (u < n) & (v < n)
    uv, vv = u[valid], v[valid]

    # duplicates and reverse twins (two-way streets are stored as two arcs)
    key = uv.astype(np.int64) * (n + 1) + vv
    rkey = vv.astype(np.int64) * (n + 1) + uv
    uniq = np.unique(key)
    has_rev = np.isin(rkey, uniq)

    deg = np.bincount(np.concatenate([uv, vv]), minlength=n)
    g = coo_matrix((np.ones(len(uv)), (uv, vv)), shape=(n, n)).tocsr()
    n_wcc, wcc_lab = connected_components(g, directed=True, connection="weak")
    n_scc, scc_lab = connected_components(g, directed=True, connection="strong")
    wcc_sizes = np.bincount(wcc_lab)
    scc_sizes = np.bincount(scc_lab)

    L = arc_lengths_m(trip)
    Lv = L[~np.isnan(L)]
    return {
        "nodes": n,
        "arcs": a,
        "arc_bad_ref": int((~valid).sum()),
        "self_loops": int((uv == vv).sum()),
        "dup_arcs": int(len(key) - len(uniq)),
        "reverse_pair_pct": float(100 * has_rev.mean()) if len(uv) else np.nan,
        "isolated_nodes": int((deg == 0).sum()),
        "wcc_count": int(n_wcc),
        "largest_wcc_nodes_pct": float(100 * wcc_sizes.max() / n) if n else np.nan,
        "scc_count": int(n_scc),
        "largest_scc_nodes_pct": float(100 * scc_sizes.max() / n) if n else np.nan,
        "dead_end_nodes": int(((np.bincount(uv, minlength=n) == 0) & (deg > 0)).sum()),
        "arc_len_median_m": float(np.median(Lv)) if len(Lv) else np.nan,
        "arc_len_p95_m": float(np.percentile(Lv, 95)) if len(Lv) else np.nan,
        "arc_len_max_m": float(Lv.max()) if len(Lv) else np.nan,
        "zero_len_arcs": int((Lv < 0.01).sum()),
    }
