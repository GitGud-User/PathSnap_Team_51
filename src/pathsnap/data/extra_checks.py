"""Extra audit checks added after cross-checking a teammate's independent analysis.

Run:  python -m pathsnap.data.extra_checks
Out:  results/eda/extra_checks.csv  (+ printed summary)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree

from pathsnap.config import DEV_PARTITIONS, RESULTS_DIR
from pathsnap.data.loaders import list_trips, load_reference_route, load_trip


def main() -> None:
    rows = []
    for part in DEV_PARTITIONS:
        for tid in list_trips(part):
            t = load_trip(tid, part)
            r = load_reference_route(tid, part)
            ts = t.gps.t.to_numpy()
            u, v = t.arcs[:, 0], t.arcs[:, 1]
            g = coo_matrix((np.ones(len(u)), (u, v)), shape=(t.n_nodes, t.n_nodes))
            _, lab = connected_components(g, directed=True, connection="weak")
            big = np.bincount(lab).argmax()
            nxy = t.nodes_xy
            L = np.hypot(*(nxy[v] - nxy[u]).T)
            ok = np.flatnonzero(u != v)
            tree = STRtree(shapely.linestrings(np.stack([nxy[u[ok]], nxy[v[ok]]], 1)))
            pts = shapely.points(t.gps_xy)
            c20 = np.bincount(tree.query(pts, predicate="dwithin", distance=20)[0], minlength=t.n_obs)
            c50 = np.bincount(tree.query(pts, predicate="dwithin", distance=50)[0], minlength=t.n_obs)
            _, dn = tree.query_nearest(pts, return_distance=True, all_matches=False)
            rows.append(dict(
                partition=part, trip=tid,
                fractional_timestamps=int((ts != np.round(ts)).sum()),
                route_in_largest_wcc=bool(np.all(lab[u[r]] == big)),
                dup_node_coords=int(len(t.nodes) - len(np.unique(t.nodes, axis=0))),
                zero_len_arcs_exact=int((L == 0).sum()),
                arcs_under_1m=int((L < 1).sum()),
                route_revisits=int(len(r) - len(np.unique(r))),
                arcs_within_20m_median=float(np.median(c20)),
                arcs_within_50m_median=float(np.median(c50)),
                obs_no_arc_within_20m_pct=float(100 * np.mean(c20 == 0)),
                nearest_arc_median_m=float(np.median(dn)),
                nearest_arc_p90_m=float(np.percentile(dn, 90)),
                gps_start_to_route_start_m=float(np.hypot(*(t.gps_xy[0] - nxy[u[r[0]]]))),
                gps_end_to_route_end_m=float(np.hypot(*(t.gps_xy[-1] - nxy[v[r[-1]]]))),
            ))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS_DIR / "eda" / "extra_checks.csv", index=False)
    print(d.describe().T.round(2).to_string())

    # Where in time do the off-route fixes occur? (deciles of each trip)
    g2r = pd.read_csv(RESULTS_DIR / "eda" / "gps_to_route.csv.gz", dtype={"trip": str})
    for tid in ("00000033", "00000054"):
        x = g2r[g2r.trip == tid].dist_m.to_numpy()
        n = len(x)
        dec = [round(100 * np.mean(x[i * n // 10:(i + 1) * n // 10] > 50)) for i in range(10)]
        print(f"{tid} % fixes >50 m from route, by time decile: {dec}")


if __name__ == "__main__":
    main()
