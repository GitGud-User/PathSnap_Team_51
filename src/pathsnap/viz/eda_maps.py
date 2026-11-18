"""Generate interactive Folium EDA maps for selected development trips.

Run:
    python -m pathsnap.viz.eda_maps                       # default showcase trips
    python -m pathsnap.viz.eda_maps --trips train:00000001 student_test:00000000

Output: results/eda/maps/<partition>_<trip>.html
"""
from __future__ import annotations

import argparse

from pathsnap.config import RESULTS_DIR
from pathsnap.data.loaders import load_reference_route, load_trip
from pathsnap.viz.folium_maps import trip_map

# Chosen from the audit to show one clean trip and each kind of anomaly.
DEFAULT_TRIPS = [
    "train:00000001",         # clean, small, Germany (also the trip in the official guide image)
    "train:00000032",         # most speed jumps (889)
    "train:00000033",         # 59 % of fixes form an off-route cluster (~0.35 x 1 km, long stop / car park drift)
    "train:00000090",         # 42-hour time gap
    "student_test:00000000",  # student-test example
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*", default=DEFAULT_TRIPS, help="partition:trip_id")
    args = ap.parse_args()
    out = RESULTS_DIR / "eda" / "maps"
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.trips:
        part, tid = spec.split(":")
        trip = load_trip(tid, part)
        m = trip_map(trip, reference_route=load_reference_route(tid, part))
        path = out / f"{part}_{tid}.html"
        m.save(str(path))
        print(f"[maps] {spec} -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
