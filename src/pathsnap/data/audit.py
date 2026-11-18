"""Stage 2 – full dataset audit (train + student_test).

Run:
    python -m pathsnap.data.audit

Outputs (results/eda/):
    trip_audit.csv          one row per trip, ~50 checks
    audit_summary.json      partition-level totals
    data_quality_log.csv    one row per detected issue (trip, issue, value, decision)
    gps_steps.csv.gz        per-step dt / distance / speed (input for EDA plots)
    gps_to_route.csv.gz     per-observation distance to the nearest reference-route arc

Reference routes are read here ONLY to audit their validity and how well they line up with
the GPS. They never feed a feature or model. The evaluator partition is never touched.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from pathsnap.config import DEV_PARTITIONS, RESULTS_DIR
from pathsnap.data.loaders import list_trips, load_reference_route, load_trip, Trip
from pathsnap.data.network import arc_lengths_m, network_stats
from pathsnap.geo import haversine_m

OUT = RESULTS_DIR / "eda"

# Thresholds used only to FLAG issues (nothing is removed at this stage).
GAP_FLAG_S = 60.0          # time gap worth splitting an HMM segment on (decided in Stage 3/8)
JUMP_FLAG_MPS = 50.0       # 180 km/h implied speed = almost certainly a GPS jump
FAR_FROM_ROUTE_M = 50.0    # GPS point implausibly far from the reference route


def gps_stats(trip: Trip) -> tuple[dict, pd.DataFrame]:
    g = trip.gps
    lon, lat, t = g["lon"].to_numpy(), g["lat"].to_numpy(), g["t"].to_numpy()
    dt = np.diff(t)
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = np.where(dt > 0, step / dt, np.nan)
    s = {
        "track_rows": trip.parse.rows["track"],
        "track_bad_rows": trip.parse.bad_rows["track"],
        "obs": trip.n_obs,
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "dt_eq1_pct": float(100 * np.mean(np.isclose(dt, 1.0))),
        "dt_gt1_count": int((dt > 1 + 1e-6).sum()),
        "dt_gt_gap_flag": int((dt > GAP_FLAG_S).sum()),
        "dt_max": float(dt.max()),
        "dt_nonpos_count": int((dt <= 0).sum()),
        "dup_coord_steps": int((step == 0).sum()),
        "gps_path_m": float(step.sum()),
        "speed_median_mps": float(np.nanmedian(spd)),
        "speed_p99_mps": float(np.nanpercentile(spd, 99)),
        "speed_max_mps": float(np.nanmax(spd)),
        "jumps_gt_flag": int((spd > JUMP_FLAG_MPS).sum()),
        "lon_min": float(lon.min()), "lon_max": float(lon.max()),
        "lat_min": float(lat.min()), "lat_max": float(lat.max()),
        "utm_epsg": trip.projector.crs.to_epsg(),
    }
    steps = pd.DataFrame({"dt": dt, "step_m": step, "speed_mps": spd})
    return s, steps


def route_stats(trip: Trip, route: np.ndarray, arc_len: np.ndarray) -> tuple[dict, np.ndarray]:
    A = trip.n_arcs
    ok = (route >= 0) & (route < A)
    r = route[ok]
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    breaks = int(np.sum(v[r[:-1]] != u[r[1:]])) if len(r) > 1 else 0
    consec_dup = int(np.sum(r[:-1] == r[1:])) if len(r) > 1 else 0

    # Distance of each GPS fix to the nearest reference arc (projected metres).
    ua = np.unique(r)
    xy = trip.nodes_xy
    lines = shapely.linestrings(np.stack([xy[u[ua]], xy[v[ua]]], axis=1))
    tree = STRtree(lines)
    pts = shapely.points(trip.gps_xy)
    _, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)

    s = {
        "route_len": int(len(route)),
        "route_bad_ref": int((~ok).sum()),
        "route_consec_dup": consec_dup,
        "route_breaks": breaks,
        "route_unique_arcs": int(len(ua)),
        "route_revisits": int(len(r) - len(ua)),
        "route_len_m": float(np.nansum(arc_len[r])),
        "gps2route_median_m": float(np.median(dist)),
        "gps2route_p95_m": float(np.percentile(dist, 95)),
        "gps2route_max_m": float(dist.max()),
        "gps_far_from_route": int((dist > FAR_FROM_ROUTE_M).sum()),
        "gps_far_from_route_pct": float(100 * np.mean(dist > FAR_FROM_ROUTE_M)),
    }
    return s, dist


def quality_issues(row: dict) -> list[dict]:
    """Translate audit numbers into a readable issue log with the decision taken."""
    out = []

    def add(issue, value, decision):
        out.append({"partition": row["partition"], "trip": row["trip"], "issue": issue,
                    "value": value, "decision": decision})

    if row["track_bad_rows"]:
        add("malformed_gps_rows", row["track_bad_rows"], "excluded from GPS (cannot be placed)")
    if row["dt_gt_gap_flag"]:
        add(f"time_gaps_gt_{int(GAP_FLAG_S)}s", row["dt_gt_gap_flag"],
            f"kept; HMM will start a new segment (max gap {row['dt_max']:.0f}s)")
    if row["jumps_gt_flag"]:
        add(f"implied_speed_gt_{int(JUMP_FLAG_MPS)}mps", row["jumps_gt_flag"],
            "kept; treated as noisy fixes (down-weighted by emission/transition)")
    if row["dt_nonpos_count"]:
        add("non_increasing_timestamps", row["dt_nonpos_count"], "to be sorted / de-duplicated")
    if row["arc_bad_ref"]:
        add("arcs_with_invalid_node_ref", row["arc_bad_ref"], "arc kept (id preserved) but never a candidate")
    if row["self_loops"]:
        add("self_loop_arcs", row["self_loops"], "kept; zero-length, excluded from candidates")
    if row["route_breaks"]:
        add("reference_route_breaks", row["route_breaks"], "allowed as gaps during label alignment")
    if row["gps_far_from_route_pct"] > 5:
        add("gps_far_from_reference_route_pct", round(row["gps_far_from_route_pct"], 1),
            f"points >{int(FAR_FROM_ROUTE_M)}m from route get label UNMATCHED (Stage 3)")
    if row["largest_wcc_nodes_pct"] < 90:
        add("fragmented_network_largest_wcc_pct", round(row["largest_wcc_nodes_pct"], 1),
            "HMM must tolerate 'no path' transitions")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows, issues, steps_all, g2r_all = [], [], [], []

    for part in DEV_PARTITIONS:
        for tid in list_trips(part):
            trip = load_trip(tid, part)
            files = sorted(p.suffix for p in trip.path.iterdir())
            arc_len = arc_lengths_m(trip)
            gs, steps = gps_stats(trip)
            ns = network_stats(trip)
            route = load_reference_route(tid, part)
            rs, dist = route_stats(trip, route, arc_len)

            row = {"partition": part, "trip": tid, "files": "|".join(files),
                   "track_kb": round((trip.path / f"{tid}.track").stat().st_size / 1024, 1),
                   "arcs_kb": round((trip.path / f"{tid}.arcs").stat().st_size / 1024, 1),
                   **gs, **ns, "node_bad_rows": trip.parse.bad_rows["nodes"],
                   "arc_bad_rows": trip.parse.bad_rows["arcs"], **rs}
            rows.append(row)
            issues += quality_issues(row)
            steps.insert(0, "trip", tid)
            steps.insert(0, "partition", part)
            steps_all.append(steps)
            g2r_all.append(pd.DataFrame({"partition": part, "trip": tid,
                                         "obs_idx": np.arange(trip.n_obs), "dist_m": dist}))
            print(f"[audit] {part:12s} {tid}  obs={trip.n_obs:6d}  arcs={trip.n_arcs:7d}  "
                  f"gps2route_med={rs['gps2route_median_m']:.1f}m")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trip_audit.csv", index=False)
    pd.DataFrame(issues).to_csv(OUT / "data_quality_log.csv", index=False)
    pd.concat(steps_all).to_csv(OUT / "gps_steps.csv.gz", index=False, float_format="%.4f")
    pd.concat(g2r_all).to_csv(OUT / "gps_to_route.csv.gz", index=False, float_format="%.3f")

    summary = {}
    for part, g in df.groupby("partition"):
        summary[part] = {
            "trips": int(len(g)),
            "trips_with_all_4_files": int((g["files"] == ".arcs|.nodes|.route|.track").sum()),
            "observations": int(g["obs"].sum()),
            "malformed_gps_rows": int(g["track_bad_rows"].sum()),
            "gps_km": round(g["gps_path_m"].sum() / 1000, 1),
            "pct_steps_exactly_1s": round(
                100 * (1 - g["dt_gt1_count"].sum() / (g["obs"] - 1).sum()), 2),
            "trips_with_gap_gt_60s": int((g["dt_max"] > GAP_FLAG_S).sum()),
            "jump_steps_gt_50mps": int(g["jumps_gt_flag"].sum()),
            "nodes_total": int(g["nodes"].sum()),
            "arcs_total": int(g["arcs"].sum()),
            "arcs_per_trip_min_max": [int(g["arcs"].min()), int(g["arcs"].max())],
            "isolated_nodes": int(g["isolated_nodes"].sum()),
            "trips_multi_wcc": int((g["wcc_count"] > 1).sum()),
            "route_arcs_total": int(g["route_len"].sum()),
            "route_breaks_total": int(g["route_breaks"].sum()),
            "gps_far_from_route_total": int(g["gps_far_from_route"].sum()),
            "gps2route_median_of_medians_m": round(float(g["gps2route_median_m"].median()), 2),
        }
    summary["runtime_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[audit] wrote {OUT}")


if __name__ == "__main__":
    main()
