"""Per (GPS fix, candidate arc) feature table.

Every feature is computable from .track/.nodes/.arcs alone (never from .route), and none
encodes absolute location (no lon/lat, no ids), because each trip is a different city.

Run (build training/dev tables):  python -m pathsnap.features.build
Output: results/features/<partition>.parquet  (git-ignored, regenerate with this command)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pathsnap.candidates.generate import ArcIndex, generate
from pathsnap.config import DEV_PARTITIONS, RESULTS_DIR
from pathsnap.data.loaders import Trip, list_trips, load_reference_route, load_trip
from pathsnap.geo import angle_diff_deg, bearing_deg
from pathsnap.labels.align import align_labels

FEATURES = [
    "perp_dist_m", "cand_rank", "dist_minus_min", "dist_ratio_min", "n_candidates",
    "heading_diff_deg", "speed_mps", "heading_reliable", "arc_len_m", "proj_t",
    "to_out_degree", "has_reverse", "prev_same_arc_dist", "next_same_arc_dist",
]


def gps_kinematics(trip: Trip) -> pd.DataFrame:
    xy, t = trip.gps_xy, trip.gps.t.to_numpy()
    n = len(xy)
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    prv = np.maximum(np.arange(n) - 1, 0)
    d = xy[nxt] - xy[prv]
    dt = t[nxt] - t[prv]
    speed = np.hypot(*d.T) / np.where(dt > 0, dt, np.nan)
    return pd.DataFrame({"gps_heading": bearing_deg(d[:, 0], d[:, 1]),
                         "speed_mps": np.nan_to_num(speed, nan=0.0)})


def build_trip_table(trip: Trip, index: ArcIndex | None = None) -> pd.DataFrame:
    c = generate(trip, index)
    if c.empty:
        return c
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    xy = trip.nodes_xy
    arc = c.arc_id.to_numpy()
    vec = xy[v[arc]] - xy[u[arc]]
    kin = gps_kinematics(trip)
    c["arc_len_m"] = np.hypot(*vec.T)
    c["heading_diff_deg"] = angle_diff_deg(kin.gps_heading.to_numpy()[c.obs_idx],
                                           bearing_deg(vec[:, 0], vec[:, 1]))
    c["speed_mps"] = kin.speed_mps.to_numpy()[c.obs_idx]
    c["heading_reliable"] = (c.speed_mps > 2.0).astype(np.int8)
    g = c.groupby("obs_idx").perp_dist_m
    mn = g.transform("min")
    c["dist_minus_min"] = c.perp_dist_m - mn
    c["dist_ratio_min"] = (c.perp_dist_m + 1) / (mn + 1)
    c["n_candidates"] = g.transform("size")
    outdeg = np.bincount(u[u >= 0], minlength=trip.n_nodes)
    c["to_out_degree"] = outdeg[v[arc]]
    key = set(zip(u.tolist(), v.tolist()))
    c["has_reverse"] = np.fromiter(((int(v[a]), int(u[a])) in key for a in arc), np.int8, len(arc))
    # Is this same arc also a candidate for the previous / next fix, and how far was it?
    lookup = c.set_index(["obs_idx", "arc_id"]).perp_dist_m
    for name, shift in (("prev_same_arc_dist", -1), ("next_same_arc_dist", 1)):
        k = pd.MultiIndex.from_arrays([c.obs_idx + shift, c.arc_id])
        c[name] = lookup.reindex(k).fillna(99.0).to_numpy()
    c.insert(0, "trip_id", trip.trip_id)
    return c


def add_labels(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    table["true_arc"] = labels[table.obs_idx.to_numpy()]
    table["y"] = (table.arc_id == table.true_arc).astype(np.int8)
    return table


def main() -> None:
    out = RESULTS_DIR / "features"
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for part in DEV_PARTITIONS:
        tabs = []
        for tid in list_trips(part):
            t0 = time.perf_counter()
            trip = load_trip(tid, part)
            tab = build_trip_table(trip)
            labels = align_labels(trip, load_reference_route(tid, part))
            tab = add_labels(tab, labels)
            labelled = labels >= 0
            hit = tab[tab.y == 1].obs_idx.unique()
            stats.append({"partition": part, "trip": tid, "obs": trip.n_obs,
                          "labelled_obs": int(labelled.sum()),
                          "obs_with_candidates": int(tab.obs_idx.nunique()),
                          "candidate_miss": int(labelled.sum() - len(hit)),
                          "candidate_miss_rate": float(1 - len(hit) / max(1, labelled.sum())),
                          "rows": len(tab), "seconds": round(time.perf_counter() - t0, 2)})
            tabs.append(tab)
            print(f"[features] {part} {tid} rows={len(tab)} miss={stats[-1]['candidate_miss_rate']:.3f}")
        pd.concat(tabs, ignore_index=True).to_parquet(out / f"{part}.parquet", index=False)
    s = pd.DataFrame(stats)
    s.to_csv(RESULTS_DIR / "features" / "candidate_quality.csv", index=False)
    for part, g in s.groupby("partition"):
        print(part, "labelled", g.labelled_obs.sum(), "of", g.obs.sum(),
              "| candidate miss rate", round(g.candidate_miss.sum() / g.labelled_obs.sum(), 4))


if __name__ == "__main__":
    main()
