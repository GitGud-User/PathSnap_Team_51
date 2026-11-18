"""Interactive Folium maps for report section 6 (classifier-only vs HMM).

Run (after tools/report_figures.py, which picks the close-up window):
    python tools/report_maps.py
Out: results/hmm/map_student_test_00000000.html          whole trip
     results/hmm/map_student_test_<trip>_zoom.html        opens zoomed on the window from
                                                            results/report/route_zoom_info.json
View them through http (see docs/DEPLOY.md) so the OpenStreetMap tiles load.
"""
from __future__ import annotations

import json

from pathsnap.config import RESULTS_DIR
from pathsnap.data.loaders import load_reference_route, load_trip
from pathsnap.hmm.viterbi import route_from_fixes
from pathsnap.pipeline import load_artifacts, predict_trip
from pathsnap.viz.folium_maps import trip_map

CLF, HMM = "#d62728", "#6250d6"


def build(tid: str, out_name: str, focus=None) -> None:
    trip = load_trip(tid, "student_test")
    out = predict_trip(trip, MODEL, PARAMS)
    m = trip_map(trip, reference_route=load_reference_route(tid, "student_test"),
                 extra_routes={"Classifier-only route": (route_from_fixes(out["clf_arcs"]), CLF),
                               "HMM/Viterbi route": (route_from_fixes(out["hmm_arcs"]), HMM)},
                 focus=focus, show_flags=False)
    path = RESULTS_DIR / "hmm" / out_name
    m.save(str(path))
    print("wrote", path)


if __name__ == "__main__":
    MODEL, PARAMS = load_artifacts()
    build("00000000", "map_student_test_00000000.html")
    z = json.loads((RESULTS_DIR / "report" / "route_zoom_info.json").read_text())
    a = z["window_start"]
    build(z["trip"], f"map_student_test_{z['trip']}_zoom.html", focus=(a, a + z["window_len"] - 1))
