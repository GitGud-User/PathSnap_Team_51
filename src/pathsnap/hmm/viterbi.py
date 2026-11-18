"""HMM map-matching decoded with Viterbi.

Hidden state at time t : one candidate directed arc s of GPS fix t (from the candidate table).
Emission              : log e_t(s) = alpha * log P_clf(s) + (1 - alpha) * (-(d_t(s) / sigma)^2 / 2)
                         P_clf = selected classifier's probability, d = perpendicular distance.
Transition s -> s'    : log a = -| ||q' - q|| - ||g_t - g_{t-1}|| | / beta  + C(s, s')
                         q, q' = projections of the fixes on the arcs, g = GPS positions.
                         C = 0 (same arc, forward) | 0 (s' starts where s ends) |
                             -hop2 (one arc in between) | -backward (same arc, moving back) |
                             -gamma (not connected within 2 arcs) – a soft penalty, never -inf,
                             so the decoder can always recover from a bad stretch.
                         v0 simplification: straight-line distance between projections stands in
                         for network route distance (documented limitation; Dijkstra is next).
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


@dataclass
class HMMParams:
    alpha: float = 0.8
    sigma: float = 10.0
    beta: float = 20.0
    gamma: float = 8.0
    hop2: float = 1.0
    backward: float = 3.0
    gap_s: float = 60.0

    def to_dict(self):
        return asdict(self)


def node_adjacency(arcs: np.ndarray, n_nodes: int) -> csr_matrix:
    u, v = arcs[:, 0], arcs[:, 1]
    ok = (u >= 0) & (v >= 0)
    return csr_matrix((np.ones(ok.sum(), dtype=np.int8), (u[ok], v[ok])), shape=(n_nodes, n_nodes))


def decode(table: pd.DataFrame, prob: np.ndarray, gps_xy: np.ndarray, t: np.ndarray,
           arcs: np.ndarray, n_nodes: int, params: HMMParams = HMMParams(),
           adj: csr_matrix | None = None) -> tuple[np.ndarray, float]:
    """Returns (arc per fix, -1 where unmatched) and runtime in ms."""
    t0 = time.perf_counter()
    P = params
    adj = adj if adj is not None else node_adjacency(arcs, n_nodes)
    n = len(gps_xy)
    out = np.full(n, -1, dtype=np.int64)
    if len(table) == 0:
        return out, 0.0
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
        gps_d = np.hypot(*(gps_xy[i] - gps_xy[pi]))
        q_d = np.hypot(*(q_a[sl][None, :, :] - q_a[psl][:, None, :]).transpose(2, 0, 1))
        la = -np.abs(q_d - gps_d) / P.beta
        same = A[:, None] == B[None, :]
        back = same & (pt_a[sl][None, :] < pt_a[psl][:, None] - 0.05)
        succ = arcs[A, 1][:, None] == arcs[B, 0][None, :]
        hop2 = np.asarray(adj[arcs[A, 1]][:, arcs[B, 0]].todense()) > 0
        C = np.where(same, 0.0, np.where(succ, 0.0, np.where(hop2, -P.hop2, -P.gamma)))
        C = np.where(back, -P.backward, C)
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
