"""Static EDA figures for the report (all computed from results/eda/* produced by the audit,
plus raw trips for the trajectory panels).

Run (after `python -m pathsnap.data.audit`):
    python -m pathsnap.viz.eda_plots

Output: results/eda/figures/*.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from pathsnap.config import RESULTS_DIR
from pathsnap.data.loaders import load_reference_route, load_trip
from pathsnap.viz.folium_maps import nearby_arcs

EDA = RESULTS_DIR / "eda"
FIG = EDA / "figures"

# Fixed categorical order: partition colour follows the partition everywhere.
PART_COLOR = {"train": "#2a78d6", "student_test": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
NETWORK, GPS, REF = "#c3c2b7", "#2a78d6", "#008300"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 10, "legend.frameon": False,
})


def _save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"[plots] {name}")


def _hist_by_partition(ax, df, col, bins, log=False):
    for part, g in df.groupby("partition", sort=False):
        ax.hist(g[col].dropna(), bins=bins, color=PART_COLOR[part], alpha=0.75,
                label=f"{part} (n={len(g)})", edgecolor="white", linewidth=0.6)
    if log:
        ax.set_xscale("log")
    ax.legend()


def fig_inventory(a):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    _hist_by_partition(axs[0], a, "obs", np.logspace(2, 4.5, 25), log=True)
    axs[0].set(title="GPS observations per trip", xlabel="observations (log)", ylabel="trips")
    _hist_by_partition(axs[1], a.assign(km=a.gps_path_m / 1000), "km", np.logspace(0.5, 2.5, 25), log=True)
    axs[1].set(title="GPS path length per trip", xlabel="km (log)", ylabel="trips")
    _hist_by_partition(axs[2], a.assign(h=a.duration_s / 3600), "h", np.logspace(-2, 2, 25), log=True)
    axs[2].set(title="Trip duration (incl. gaps)", xlabel="hours (log)", ylabel="trips")
    _save(fig, "01_inventory.png")


def fig_missing(a):
    """Missing-value analysis: malformed rows per file type, and seconds with no GPS fix."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 2.2]})
    bad = pd.Series({"track": a.track_bad_rows.sum(), "nodes": a.node_bad_rows.sum(),
                     "arcs": a.arc_bad_rows.sum(), "route": a.route_bad_ref.sum()})
    axs[0].bar(bad.index, bad.values, color="#2a78d6", width=0.6)
    for i, v in enumerate(bad.values):
        axs[0].text(i, v, f"{int(v)}", ha="center", va="bottom", color=INK)
    axs[0].set(title="Malformed / invalid rows per file type", ylabel="rows (all 90 trips)",
               ylim=(0, max(1, bad.max()) * 1.3))
    # A 1 Hz track should have duration+1 fixes; the shortfall = seconds with no observation.
    miss = 100 * (1 - a.obs / (a.duration_s + 1))
    s = a.assign(miss=miss).sort_values("miss", ascending=False).reset_index(drop=True)
    axs[1].bar(s.index, s.miss, color=[PART_COLOR[p] for p in s.partition], width=0.85)
    axs[1].set_yscale("symlog", linthresh=1)
    axs[1].set(title="Share of seconds with no GPS fix (1 Hz expected)", xlabel="trips, sorted",
               ylabel="% missing (symlog)")
    for i in range(3):
        axs[1].text(i, s.miss[i], f" {s.trip[i][-2:]}", rotation=90, va="bottom", fontsize=7, color=INK)
    axs[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in PART_COLOR.values()],
                  labels=list(PART_COLOR))
    _save(fig, "02_missing_values.png")


def fig_sampling(steps):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    dt = steps.dt.clip(upper=600)
    bins = np.concatenate([[0.5, 1.5, 2.5, 5.5, 10.5, 30.5, 60.5, 120.5, 300.5, 600.5]])
    counts, _ = np.histogram(dt, bins=bins)
    labels = ["1", "2", "3–5", "6–10", "11–30", "31–60", "61–120", "121–300", "301+"]
    axs[0].bar(labels, counts, color="#2a78d6", width=0.7)
    axs[0].set_yscale("log")
    axs[0].set(title=f"Sampling interval Δt ({len(steps):,} steps, {100 * (steps.dt == 1).mean():.1f}% = 1 s)",
               xlabel="Δt (s)", ylabel="steps (log)")
    spd = steps.speed_mps.dropna().clip(upper=100)
    axs[1].hist(spd, bins=100, color="#2a78d6")
    axs[1].set_yscale("log")
    axs[1].axvline(50, color="#e34948", lw=1.5)
    axs[1].text(51, axs[1].get_ylim()[1] * 0.3, "50 m/s jump flag", color=INK)
    axs[1].set(title="Implied speed between consecutive fixes (clipped at 100)", xlabel="m/s", ylabel="steps (log)")
    _save(fig, "03_sampling_and_speed.png")


def fig_network(a):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.3))
    for part, g in a.groupby("partition", sort=False):
        axs[0].scatter(g.nodes, g.arcs, s=14, color=PART_COLOR[part], label=part)
        axs[1].scatter(g.arc_len_median_m, g.reverse_pair_pct, s=14, color=PART_COLOR[part])
        axs[2].scatter(g.wcc_count, g.largest_wcc_nodes_pct, s=14, color=PART_COLOR[part])
    axs[0].set(xscale="log", yscale="log", title="Road-network size per trip", xlabel="nodes", ylabel="arcs")
    axs[0].legend()
    axs[1].set(title="Arc length vs two-way share", xlabel="median arc length (m)",
               ylabel="% arcs with reverse twin")
    axs[2].set(xscale="log", title="Connectivity (weak components)", xlabel="# components (log)",
               ylabel="% nodes in largest")
    _save(fig, "04_network_structure.png")


def fig_gps_to_route(g2r, a):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    d = g2r.dist_m.clip(upper=100)
    axs[0].hist(d, bins=100, color="#2a78d6")
    axs[0].axvline(50, color="#e34948", lw=1.5)
    axs[0].set(title=f"GPS distance to nearest reference arc (median {g2r.dist_m.median():.1f} m)",
               xlabel="metres (clipped at 100)", ylabel="observations")
    s = a.sort_values("gps_far_from_route_pct", ascending=False).head(12)
    axs[1].barh(s.trip.str[-2:], s.gps_far_from_route_pct, color=[PART_COLOR[p] for p in s.partition])
    axs[1].invert_yaxis()
    axs[1].set(title="Trips with most fixes > 50 m from reference route", xlabel="% of fixes", ylabel="trip")
    _save(fig, "05_gps_to_reference_route.png")


def fig_world(a):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for part, g in a.groupby("partition", sort=False):
        ax.scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, s=22,
                   color=PART_COLOR[part], label=part, edgecolor="white", linewidth=0.5)
    ax.set(xlim=(-180, 180), ylim=(-60, 75), title="Trip locations – every trip has its own city network",
           xlabel="longitude", ylabel="latitude")
    ax.legend(loc="lower left")
    _save(fig, "06_trip_locations.png")


def fig_trajectories(specs):
    fig, axs = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 4.2))
    for ax, (part, tid, note) in zip(np.atleast_1d(axs), specs):
        trip = load_trip(tid, part)
        xy, nxy = trip.gps_xy, trip.nodes_xy
        x0, y0 = xy.min(0)
        arcs = nearby_arcs(trip, 300)
        segs = np.stack([nxy[trip.arcs[arcs, 0]], nxy[trip.arcs[arcs, 1]]], 1) - [x0, y0]
        ax.add_collection(LineCollection(segs / 1000, colors=NETWORK, linewidths=0.5))
        r = load_reference_route(tid, part)
        rseg = np.stack([nxy[trip.arcs[r, 0]], nxy[trip.arcs[r, 1]]], 1) - [x0, y0]
        ax.add_collection(LineCollection(rseg / 1000, colors=REF, linewidths=2))
        ax.plot((xy[:, 0] - x0) / 1000, (xy[:, 1] - y0) / 1000, ".", ms=1.2, color=GPS)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set(title=f"{part} {tid[-2:]}: {note}", xlabel="km east (UTM)", ylabel="km north")
    axs[0].legend(handles=[plt.Line2D([], [], color=c, lw=w, marker=m, ls=ls) for c, w, m, ls in
                           [(NETWORK, 1, None, "-"), (REF, 2, None, "-"), (GPS, 0, ".", "")]],
                  labels=["road network", "reference route", "raw GPS"], loc="best")
    _save(fig, "07_sample_trajectories.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(EDA / "trip_audit.csv", dtype={"trip": str})
    steps = pd.read_csv(EDA / "gps_steps.csv.gz")
    g2r = pd.read_csv(EDA / "gps_to_route.csv.gz", dtype={"trip": str})
    fig_inventory(a)
    fig_missing(a)
    fig_sampling(steps)
    fig_network(a)
    fig_gps_to_route(g2r, a)
    fig_world(a)
    fig_trajectories([("train", "00000001", "clean"), ("train", "00000033", "off-route cluster"),
                      ("student_test", "00000000", "student-test")])


if __name__ == "__main__":
    main()
