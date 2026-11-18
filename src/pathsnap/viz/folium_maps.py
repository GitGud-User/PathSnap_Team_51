"""Reusable Folium map builder.

Layers (each toggleable via LayerControl):
    Road network (clipped to the area around the GPS track)  – grey
    Raw GPS observations                                       – blue dots
    Flagged GPS (time gaps / speed jumps)                      – orange / red
    Reference route  (DEVELOPMENT PARTITIONS ONLY)             – green
    Extra routes (classifier / HMM, added in later stages)     – caller-supplied colours

The reference-route layer refuses the evaluator partition so ground truth can never be
drawn into anything the evaluator UI shows.
"""
from __future__ import annotations

import folium
import numpy as np
import shapely
from shapely import STRtree

from pathsnap.data.loaders import Trip

COLORS = {
    "network": "#9e9e9e",
    "gps": "#1f5fd6",
    "gap": "#ff8c00",
    "jump": "#111111",          # black rings, so they never look like the red classifier route
    "reference": "#2ca02c",
}


def _arc_coords(trip: Trip, arc_ids: np.ndarray) -> list:
    u, v = trip.arcs[arc_ids, 0], trip.arcs[arc_ids, 1]
    a, b = trip.nodes[u], trip.nodes[v]
    return [[[float(p[0]), float(p[1])], [float(q[0]), float(q[1])]] for p, q in zip(a, b)]


def nearby_arcs(trip: Trip, radius_m: float = 150.0) -> np.ndarray:
    """Arc ids within radius of any GPS fix (keeps maps of 300k-arc networks light)."""
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    ok = np.flatnonzero((u >= 0) & (v >= 0) & (u < trip.n_nodes) & (v < trip.n_nodes) & (u != v))
    xy = trip.nodes_xy
    lines = shapely.linestrings(np.stack([xy[u[ok]], xy[v[ok]]], axis=1))
    tree = STRtree(lines)
    _, hit = tree.query(shapely.points(trip.gps_xy), predicate="dwithin", distance=radius_m)
    return ok[np.unique(hit)]


def _geojson_lines(coords: list, color: str, weight: float, opacity: float, name: str):
    feat = {"type": "Feature", "properties": {},
            "geometry": {"type": "MultiLineString", "coordinates": coords}}
    return folium.GeoJson(feat, name=name,
                          style_function=lambda _: {"color": color, "weight": weight,
                                                    "opacity": opacity})


def _legend(items: list[tuple[str, str]]) -> folium.Element:
    """items: (label, colour); a label ending in '(points)' is drawn as a dot, others as a line."""
    def swatch(c, dot):
        if dot:
            return (f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                    f'border:2px solid {c};margin:0 9px 0 3px;vertical-align:middle"></span>')
        return (f'<span style="display:inline-block;width:16px;height:4px;background:{c};'
                f'margin-right:6px;vertical-align:middle"></span>')
    rows = "".join(f'<div>{swatch(c, label.endswith("(points)"))}{label.replace(" (points)", "")}</div>'
                   for label, c in items)
    html = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            'padding:8px 10px;border:1px solid #888;border-radius:6px;font:12px sans-serif">'
            f'<b>Legend</b>{rows}</div>')
    return folium.Element(html)


def trip_map(trip: Trip,
             reference_route: np.ndarray | None = None,
             extra_routes: dict[str, tuple[np.ndarray, str]] | None = None,
             gap_s: float = 60.0, jump_mps: float = 50.0,
             network_radius_m: float = 150.0, max_gps_markers: int = 4000,
             focus: tuple[int, int] | None = None, show_flags: bool = True) -> folium.Map:
    """focus=(first_fix, last_fix) zooms the initial view onto that stretch of the trip."""
    if reference_route is not None and trip.partition == "evaluator_test":
        raise PermissionError("Reference route must never be drawn for evaluator trips.")

    g = trip.gps
    lat, lon, t = g["lat"].to_numpy(), g["lon"].to_numpy(), g["t"].to_numpy()
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if focus is not None:
        a, b = focus
        m.fit_bounds([[lat[a:b + 1].min(), lon[a:b + 1].min()], [lat[a:b + 1].max(), lon[a:b + 1].max()]])
    else:
        m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    legend = []

    # Drawing order = stacking order: network, then GPS, then routes on top, so the
    # matched routes are never hidden under thousands of GPS dots.
    arcs = nearby_arcs(trip, network_radius_m)
    _geojson_lines(_arc_coords(trip, arcs), COLORS["network"], 2, 0.6,
                   f"Road network ({len(arcs)} arcs within {network_radius_m:.0f} m)").add_to(m)
    legend.append(("Road network", COLORS["network"]))

    # GPS points (subsampled only if very long, and the layer name says so)
    step = max(1, int(np.ceil(len(g) / max_gps_markers)))
    fg = folium.FeatureGroup(name=f"Raw GPS ({len(g)} fixes{'' if step == 1 else f', every {step}th shown'})")
    folium.PolyLine(list(zip(lat, lon)), color=COLORS["gps"], weight=1, opacity=0.3).add_to(fg)
    for i in range(0, len(g), step):
        folium.CircleMarker((lat[i], lon[i]), radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.6, weight=0,
                            tooltip=f"obs {i}  t={t[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    legend.append(("Raw GPS fixes (points)", COLORS["gps"]))

    # Reference route (development partitions only)
    if reference_route is not None:
        r = reference_route[(reference_route >= 0) & (reference_route < trip.n_arcs)]
        _geojson_lines(_arc_coords(trip, r), COLORS["reference"], 9, 0.45,
                       "Reference route (ground truth)").add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    for name, (arc_ids, color) in (extra_routes or {}).items():
        _geojson_lines(_arc_coords(trip, np.asarray(arc_ids)), color, 4, 0.95, name).add_to(m)
        legend.append((name, color))

    # Flagged fixes
    dt = np.diff(t)
    xy = trip.gps_xy
    spd = np.hypot(*np.diff(xy, axis=0).T) / np.where(dt > 0, dt, np.nan)
    n_gap, n_jump = int((dt > gap_s).sum()), int((spd > jump_mps).sum())
    flags = folium.FeatureGroup(name=f"Flagged GPS ({n_gap} gaps, {n_jump} jumps)", show=show_flags)
    for i in np.flatnonzero(dt > gap_s):
        folium.CircleMarker((lat[i + 1], lon[i + 1]), radius=7, color=COLORS["gap"], weight=3,
                            tooltip=f"gap of {dt[i]:.0f}s before obs {i + 1}").add_to(flags)
    for i in np.flatnonzero(spd > jump_mps):
        folium.CircleMarker((lat[i + 1], lon[i + 1]), radius=5, color=COLORS["jump"], weight=2,
                            tooltip=f"jump {spd[i]:.0f} m/s at obs {i + 1}").add_to(flags)
    flags.add_to(m)
    legend += [(f"Time gap > {gap_s:.0f}s (points)", COLORS["gap"]),
               (f"Speed > {jump_mps:.0f} m/s (points)", COLORS["jump"])]

    folium.Marker((lat[0], lon[0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    return m
