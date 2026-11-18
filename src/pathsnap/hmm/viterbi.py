"""HMM map-matching decoded with Viterbi.

Hidden state at time t : one candidate directed arc s of GPS fix t (from the candidate table).
Emission              : log e_t(s) = alpha * log P_clf(s) + (1 - alpha) * (-(d_t(s) / sigma)^2 / 2)
                         P_clf = selected classifier's probability, d = perpendicular distance.
Transition s -> s'    : log a = -| d_net(s,s') - ||g_t - g_{t-1}|| | / beta  + C(s, s')
                         d_net = directed NETWORK route distance between the two projections
                         (Newson-Krumm): remaining tail of s + Dijkstra shortest path from
                         to_node(s) to from_node(s') + head of s', arc lengths in projected
                         metres. Shortest paths are searched up to ||g_t - g_{t-1}|| + cutoff
                         and cached per node pair. No path within that limit is treated as the
                         limit itself -> a soft penalty of cutoff/beta (never -inf), so the
                         decoder can always recover from a bad stretch.
                         C = 0 (same arc, forward) | 0 (s' starts where s ends) |
                             -hop2 (one arc in between) | -backward (same arc, moving back) |
                             -gamma (not connected within 2 arcs) - a soft penalty, never -inf.
                         Legacy fallback: if nodes_xy is not passed, straight-line distance
                         between projections stands in for d_net (the old v0 behaviour).
Viterbi recurrence    : delta_t(s') = log e_t(s') + max_s [ delta_{t-1}(s) + log a(s, s') ]
                         psi_t(s')   = argmax_s [ ... ]; backtrack from argmax delta_T.
Gaps                  : a new, independent segment starts when dt > gap_s or a fix has no
                         candidates (that fix is returned as -1 = unmatched).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

@dataclass
class HMMParams:
    alpha: float = 0.8
    sigma: float = 10.0
    beta: float = 20.0
    gamma: float = 8.0
    hop2: float = 1.0
    backward: float = 3.0
    gap_s: float = 60.0
    cutoff: float = 200.0    # metres of route-distance slack beyond the GPS step (soft cap)

    def to_dict(self):
        return asdict(self)

def node_adjacency(arcs: np.ndarray, n_nodes: int) -> csr_matrix:
    u, v = arcs[:, 0], arcs[:, 1]
    ok = (u >= 0) & (v >= 0)
    return csr_matrix((np.ones(ok.sum(), dtype=np.int8), (u[ok], v[ok])), shape=(n_nodes, n_nodes))

def arc_lengths_m(arcs: np.ndarray, nodes_xy: np.ndarray) -> np.ndarray:
    """Length of every arc in projected metres (arcs are straight node->node segments)."""
    u, v = arcs[:, 0], arcs[:, 1]
    ok = (u >= 0) & (v >= 0) & (u < len(nodes_xy)) & (v < len(nodes_xy))
    L = np.zeros(len(arcs), dtype=float)
    L[ok] = np.hypot(*(nodes_xy[v[ok]] - nodes_xy[u[ok]]).T)
    return np.maximum(L, 1e-9)

def weighted_adjacency(arcs: np.ndarray, arc_len: np.ndarray, n_nodes: int) -> csr_matrix:
    """Directed graph weighted by arc length; parallel arcs keep the shortest one."""
    u, v = arcs[:, 0], arcs[:, 1]
    ok = (u >= 0) & (v >= 0) & (u < n_nodes) & (v < n_nodes) & (u != v)
    u, v, w = u[ok], v[ok], arc_len[ok]
    order = np.lexsort((w, v, u))
    u, v, w = u[order], v[order], w[order]
    keep = np.ones(len(u), dtype=bool)
    keep[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
    return csr_matrix((w[keep], (u[keep], v[keep])), shape=(n_nodes, n_nodes))

def _node_pair_distances(W: csr_matrix, cache: dict, src: np.ndarray, tgt: np.ndarray,
                         mask: np.ndarray, limit: float) -> np.ndarray:
    """Shortest-path lengths for the masked (src, tgt) node pairs; inf where no path exists
    within `limit`. Cached per node pair: a finite distance is exact; a cached inf is only
    reusable when its search limit was >= the current limit."""
    D = np.full(mask.shape, np.inf)
    ii, jj = np.nonzero(mask)
    if len(ii) == 0:
        return D
    stale: dict = {}
    for a, b in zip(ii, jj):
        key = (int(src[a]), int(tgt[b]))
        hit = cache.get(key)
        if hit is not None and (np.isfinite(hit[0]) or hit[1] >= limit):
            D[a, b] = hit[0] if hit[0] <= limit else np.inf
        else:
            stale.setdefault(key[0], []).append((a, b, key[1]))
    if stale:
        sources = np.fromiter(stale.keys(), dtype=np.int64, count=len(stale))
        rows = np.atleast_2d(dijkstra(W, directed=True, indices=sources, limit=limit))
        for s, row in zip(sources, rows):
            for a, b, tnode in stale[int(s)]:
                d = float(row[tnode])
                cache[(int(s), tnode)] = (d, limit)
                D[a, b] = d
    return D

def decode(table: pd.DataFrame, prob: np.ndarray, gps_xy: np.ndarray, t: np.ndarray,
           arcs: np.ndarray, n_nodes: int, params: HMMParams = HMMParams(),
           adj: csr_matrix | None = None,
           nodes_xy: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Returns (arc per fix, -1 where unmatched) and runtime in ms.

    Pass nodes_xy (projected node coordinates, metres) to use directed network route
    distance (Dijkstra, capped at gps step + params.cutoff) in the transition model.
    Without nodes_xy the legacy straight-line approximation is used (v0 behaviour).
    """
    t0 = time.perf_counter()
    P = params
    adj = adj if adj is not None else node_adjacency(arcs, n_nodes)
    n = len(gps_xy)
    out = np.full(n, -1, dtype=np.int64)
    if len(table) == 0:
        return out, 0.0
    if nodes_xy is not None:
        alen = arc_lengths_m(arcs, nodes_xy)
        W = weighted_adjacency(arcs, alen, n_nodes)
        sp_cache: dict = {}
    else:
        alen = W = sp_cache = None
    tb = table.assign(_p=np.clip(prob, 1e-6, 1.0)).sort_values(["obs_idx", "cand_rank"])
    obs = tb.obs_idx.to_numpy()
    starts = np.searchsorted(obs, np.arange(n))
    ends = np.searchsorted(obs, np.arange(n), side="right")
    arc_a, d_a = tb.arc_id.to_numpy(), tb.perp_dist_m.to_numpy()
    pt_a, p_a = tb.proj_t.to_numpy(), tb._p.to_numpy()
    q_a = tb[["proj_x", "proj_y"]].to_numpy()
    emis_a = P.alpha * np.log(p_a) + (1 - P.alpha) * (-0.5 * (d_a / P.sigma) ** 2)

    delta = None
    seg = []          # (fix index, slice) of current segment
    backs = []

    def flush():
        if not seg:
            return
        j = int(np.argmax(delta))
        for k in range(len(seg) - 1, -1, -1):
            i, sl = seg[k]
            out[i] = arc_a[sl][j]
            if k > 0:
                j = backs[k][j]

    for i in range(n):
        sl = slice(starts[i], ends[i])
        if sl.start == sl.stop:                       # no candidate -> break segment
            flush(); seg.clear(); backs.clear(); delta = None
            continue
        if delta is not None and (t[i] - t[seg[-1][0]] > P.gap_s):
            flush(); seg.clear(); backs.clear(); delta = None
        e = emis_a[sl]
        if delta is None:
            delta = e.copy(); seg.append((i, sl)); backs.append(None)
            continue
        pi, psl = seg[-1]
        A, B = arc_a[psl], arc_a[sl]
        gps_d = float(np.hypot(*(gps_xy[i] - gps_xy[pi])))
        same = A[:, None] == B[None, :]
        back = same & (pt_a[sl][None, :] < pt_a[psl][:, None] - 0.05)
        succ = arcs[A, 1][:, None] == arcs[B, 0][None, :]
        hop2 = np.asarray(adj[arcs[A, 1]][:, arcs[B, 0]].todense()) > 0
        C = np.where(same, 0.0, np.where(succ, 0.0, np.where(hop2, -P.hop2, -P.gamma)))
        C = np.where(back, -P.backward, C)
        if W is None:                                 # legacy: straight-line v0 fallback
            q_d = np.hypot(*(q_a[sl][None, :, :] - q_a[psl][:, None, :]).transpose(2, 0, 1))
            la = -np.abs(q_d - gps_d) / P.beta
        else:                                         # directed network route distance
            limit = gps_d + P.cutoff
            d_net = ((1.0 - pt_a[psl]) * alen[A])[:, None] + (pt_a[sl] * alen[B])[None, :]
            general = ~same & ~succ
            if general.any():
                D = _node_pair_distances(W, sp_cache, arcs[A, 1], arcs[B, 0], general, limit)
                d_net = d_net + np.where(general, D, 0.0)
            d_same = np.abs(pt_a[sl][None, :] - pt_a[psl][:, None]) * alen[A][:, None]
            d_net = np.where(same, d_same, d_net)
            d_net = np.minimum(d_net, limit)          # no path within limit -> soft cap
            la = -np.abs(d_net - gps_d) / P.beta
        score = delta[:, None] + la + C
        bp = np.argmax(score, axis=0)
        delta = score[bp, np.arange(len(B))] + e
        seg.append((i, sl)); backs.append(bp)
    flush()
    return out, (time.perf_counter() - t0) * 1000

def route_from_fixes(arc_per_fix: np.ndarray) -> np.ndarray:
    """Collapse per-fix arcs into the traversed arc sequence (consecutive duplicates removed)."""
    s = arc_per_fix[arc_per_fix >= 0]
    if len(s) == 0:
        return s
    keep = np.concatenate([[True], s[1:] != s[:-1]])
    return s[keep]
