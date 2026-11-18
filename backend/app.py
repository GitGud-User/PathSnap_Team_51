"""PathSnap API (Laptop 1).

Start:  uvicorn backend.app:app --host 0.0.0.0 --port 8000
Env:    PATHSNAP_EVAL_PARTITION  partition served to the UI (default: evaluator_test if it has
                                 trips, else student_test for rehearsals)
        PATHSNAP_DATA_DIR        data root (default ./data)

Every /api/predict call runs the full pipeline live (no cached or hard-coded outputs) and
logs each stage. Reference routes are read only inside `_evaluate`, server-side, and are
never included in a response.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

# joblib (used by scikit-learn) tries to count physical cores with a subprocess that fails on
# some Windows machines and prints a long, harmless traceback into the API log. joblib skips
# that check when LOKY_MAX_CPU_COUNT is below the logical core count, so set it to one less.
# Inference uses LightGBM's own threads, so this does not slow predictions.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathsnap.data.loaders import list_trips, load_reference_route, load_trip
from pathsnap.eval.metrics import pointwise_accuracy, route_continuity, route_f1
from pathsnap.hmm.viterbi import route_from_fixes
from pathsnap.labels.align import align_labels
from pathsnap.pipeline import load_artifacts, predict_trip

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(), logging.FileHandler("logs/api.log")])
log = logging.getLogger("pathsnap.api")

PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION") or (
    "evaluator_test" if list_trips("evaluator_test") else "student_test")

app = FastAPI(title="PathSnap API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL, PARAMS = load_artifacts()
log.info("model + HMM params loaded; serving partition '%s' (%d trips)", PARTITION,
         len(list_trips(PARTITION)))


def _arc_lines(trip, arc_ids):
    u, v = trip.arcs[arc_ids, 0], trip.arcs[arc_ids, 1]
    a, b = trip.nodes[u], trip.nodes[v]
    return [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(a, b)]


def _evaluate(trip, out) -> dict:
    """Server-side scoring against the hidden reference route (never returned)."""
    route = load_reference_route(trip.trip_id, PARTITION, allow_evaluator=True)
    labels = align_labels(trip, route)
    lab = labels >= 0
    tab = out["table"]
    hit = set(tab[tab.arc_id.to_numpy() == labels[tab.obs_idx.to_numpy()]].obs_idx.tolist())
    cmr = 1 - len(hit & set(np.flatnonzero(lab).tolist())) / max(1, lab.sum())
    hmm_c = route_continuity(out["hmm_arcs"], trip.arcs)
    clf_c = route_continuity(out["clf_arcs"], trip.arcs)
    return {
        "top1_accuracy_hmm": pointwise_accuracy(out["hmm_arcs"], labels),
        "top1_accuracy_classifier": pointwise_accuracy(out["clf_arcs"], labels),
        "route_continuity_hmm": hmm_c["continuity"],
        "route_continuity_classifier": clf_c["continuity"],
        "invalid_transitions_hmm": hmm_c["invalid_transitions"],
        "invalid_transitions_classifier": clf_c["invalid_transitions"],
        "candidate_miss_rate": cmr,
        **{k + "_hmm": v for k, v in route_f1(route_from_fixes(out["hmm_arcs"]), route).items()},
        "labelled_fixes": int(lab.sum()),
        "unmatched_fixes_hmm": int((out["hmm_arcs"] < 0).sum()),
    }


@app.get("/")
def root():
    """Landing response for the bare server/tunnel address."""
    return {"service": "PathSnap API", "status": "running", "partition": PARTITION,
            "endpoints": {"health": "/api/health", "trips": "/api/trips",
                          "predict": "POST /api/predict/{trip_id}", "interactive_docs": "/docs"}}


@app.get("/api/health")
def health():
    return {"status": "ok", "partition": PARTITION, "model": type(MODEL).__name__,
            "hmm_params": PARAMS.to_dict()}


@app.get("/api/trips")
def trips():
    ids = list_trips(PARTITION)
    log.info("GET /api/trips -> %d trips", len(ids))
    return {"partition": PARTITION, "trips": ids}


@app.post("/api/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in list_trips(PARTITION):
        log.warning("rejected unknown trip_id=%s", trip_id)
        raise HTTPException(404, f"trip {trip_id} not in partition {PARTITION}")
    t0 = time.perf_counter()
    log.info("predict %s | request received", trip_id)
    try:
        trip = load_trip(trip_id, PARTITION)
        log.info("predict %s | %d GPS fixes, %d road arcs -> candidates, features, LightGBM, HMM",
                 trip_id, trip.n_obs, trip.n_arcs)
        out = predict_trip(trip, MODEL, PARAMS)
        tm = out["timings"]
        log.info("predict %s | stage ms: candidates+features %.0f, classifier %.0f, viterbi %.0f", trip_id,
                 tm["candidates_features_ms"], tm["classifier_ms"], tm["hmm_ms"])
        metrics = _evaluate(trip, out)
    except Exception as e:  # noqa: BLE001
        log.exception("predict trip=%s failed", trip_id)
        raise HTTPException(500, f"pipeline failed: {e}") from e
    hmm_route = route_from_fixes(out["hmm_arcs"])
    clf_route = route_from_fixes(out["clf_arcs"])
    g = trip.gps
    step = max(1, len(g) // 3000)
    total = (time.perf_counter() - t0) * 1000
    log.info("predict %s | done %.0f ms | top1 hmm %.3f clf %.3f | continuity hmm %.3f clf %.3f",
             trip_id, total, metrics["top1_accuracy_hmm"], metrics["top1_accuracy_classifier"],
             metrics["route_continuity_hmm"], metrics["route_continuity_classifier"])
    return {
        "trip_id": trip_id,
        "n_fixes": trip.n_obs,
        "gps": g[["lat", "lon"]].to_numpy()[::step].round(6).tolist(),
        "hmm_route": _arc_lines(trip, hmm_route),
        "classifier_route": _arc_lines(trip, clf_route),
        "metrics": metrics,
        "timings_ms": {**{k: round(v, 1) for k, v in out["timings"].items()}, "total_ms": round(total, 1)},
    }
