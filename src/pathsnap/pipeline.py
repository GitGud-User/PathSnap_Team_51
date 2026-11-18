"""The single prediction pipeline used by experiments AND the API (keeps them consistent).

predict_trip(trip) : candidates -> features -> classifier probabilities -> HMM/Viterbi.
It never reads the reference route.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from pathsnap.config import PROJECT_ROOT
from pathsnap.data.loaders import Trip
from pathsnap.eval.metrics import classifier_choice
from pathsnap.features.build import FEATURES, build_trip_table
from pathsnap.hmm.viterbi import HMMParams, decode

FINAL_DIR = PROJECT_ROOT / "models" / "final"


def load_artifacts(model_dir: Path = FINAL_DIR):
    model = joblib.load(model_dir / "model.joblib")
    feats = json.loads((model_dir / "feature_list.json").read_text())
    assert feats == FEATURES, "feature list of saved model differs from code"
    hp = model_dir / "hmm_params.json"
    params = HMMParams(**json.loads(hp.read_text())) if hp.exists() else HMMParams()
    return model, params


def predict_trip(trip: Trip, model, params: HMMParams) -> dict:
    timings = {}
    t0 = time.perf_counter()
    table = build_trip_table(trip)
    timings["candidates_features_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    prob = model.predict_proba(table[FEATURES].to_numpy(float))[:, 1] if len(table) else np.array([])
    timings["classifier_ms"] = (time.perf_counter() - t0) * 1000
    clf = np.full(trip.n_obs, -1, dtype=np.int64)
    if len(table):
        ch = classifier_choice(table, prob)
        clf[ch.index.get_level_values(1).to_numpy()] = ch.to_numpy()
    hmm, ms = decode(table, prob, trip.gps_xy, trip.gps.t.to_numpy(), trip.arcs, trip.n_nodes, params)
    timings["hmm_ms"] = ms
    return {"table": table, "prob": prob, "clf_arcs": clf, "hmm_arcs": hmm, "timings": timings}
