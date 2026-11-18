"""Stage 8: tune decoder parameters, then compare classifier-only vs classifier + HMM/Viterbi.

Tuning (train data only, leakage-free): TUNE_TRIPS train trips are held out; an untuned
LightGBM is fitted on the remaining train trips; a small grid over (alpha, beta, gamma) is
scored by pointwise accuracy on the held-out trips. Comparison: final tuned model on the 10
student-test trips.

Run:  python -m pathsnap.hmm.compare        (needs models/final from pathsnap.models.optimise)
Out:  results/hmm/{decoder_grid.csv, clf_vs_hmm_per_trip.csv, clf_vs_hmm_summary.csv,
      clf_vs_hmm.png, map_<trip>.html}, models/final/hmm_params.json
"""
from __future__ import annotations

import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import RANDOM_SEED, RESULTS_DIR
from pathsnap.data.loaders import list_trips, load_reference_route, load_trip
from pathsnap.eval.metrics import pointwise_accuracy, route_continuity, route_f1
from pathsnap.features.build import FEATURES
from pathsnap.hmm.viterbi import HMMParams, decode, route_from_fixes
from pathsnap.labels.align import align_labels
from pathsnap.models.registry import MODELS
from pathsnap.models.train_all import load, training_rows
from pathsnap.pipeline import FINAL_DIR, load_artifacts, predict_trip
from pathsnap.viz.folium_maps import trip_map

OUT = RESULTS_DIR / "hmm"
TUNE_TRIPS = 12
GRID = {"alpha": [0.6, 0.8, 1.0], "beta": [10.0, 30.0], "gamma": [4.0, 8.0], "cutoff":[100.0, 300.0]}


def evaluate(trip, out, labels, route):
    return {"pointwise_acc_clf": pointwise_accuracy(out["clf_arcs"], labels),
            "pointwise_acc_hmm": pointwise_accuracy(out["hmm_arcs"], labels),
            **{f"{k}_clf": v for k, v in route_continuity(out["clf_arcs"], trip.arcs).items()},
            **{f"{k}_hmm": v for k, v in route_continuity(out["hmm_arcs"], trip.arcs).items()},
            **{f"{k}_clf": v for k, v in route_f1(route_from_fixes(out["clf_arcs"]), route).items()},
            **{f"{k}_hmm": v for k, v in route_f1(route_from_fixes(out["hmm_arcs"]), route).items()},
            "hmm_ms": out["timings"]["hmm_ms"]}


def tune() -> HMMParams:
    tr = load("train")
    rng = np.random.default_rng(RANDOM_SEED)
    small = [t for t, n in tr.groupby("trip_id").obs_idx.max().items() if n < 4000]
    hold = list(rng.choice(small, TUNE_TRIPS, replace=False))
    fit_rows = training_rows(tr[~tr.trip_id.isin(hold)])
    m = MODELS["lightgbm"][1]().fit(fit_rows[FEATURES].to_numpy(float), fit_rows.y.to_numpy())
    cache = []
    for tid in hold:
        trip = load_trip(tid, "train")
        tab = tr[tr.trip_id == tid].reset_index(drop=True)
        p = m.predict_proba(tab[FEATURES].to_numpy(float))[:, 1]
        lab = np.full(trip.n_obs, -1)
        lab[tab.obs_idx] = tab.true_arc
        cache.append((trip, tab, p, lab))
    rows = []
    for a, b, g, c in itertools.product(GRID["alpha"], GRID["beta"], GRID["gamma"], GRID["cutoff"]):
        prm = HMMParams(alpha=a, beta=b, gamma=g, cutoff=c)
        accs = []
        for trip, tab, p, lab in cache:
            hmm, _ = decode(tab, p, trip.gps_xy, trip.gps.t.to_numpy(), trip.arcs, trip.n_nodes, prm, nodes_xy=trip.nodes_xy)
            accs.append(pointwise_accuracy(hmm, lab))
        rows.append({"alpha": a, "beta": b, "gamma": g, "cutoff": c, "mean_pointwise_acc": float(np.mean(accs))})
        print(f"[hmm-grid] alpha={a} beta={b} gamma={g} cutoff={c} acc={rows[-1]['mean_pointwise_acc']:.4f}")    
      
    grid = pd.DataFrame(rows).sort_values("mean_pointwise_acc", ascending=False)
    grid.to_csv(OUT / "decoder_grid.csv", index=False)
    (OUT / "tuning_trips.json").write_text(json.dumps(sorted(hold)))
    best = grid.iloc[0]
    return HMMParams(alpha=float(best.alpha), beta=float(best.beta), gamma=float(best.gamma),cutoff=float(best.cutoff))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = tune()
    (FINAL_DIR / "hmm_params.json").write_text(json.dumps(params.to_dict(), indent=2))
    model, params = load_artifacts()
    rows = []
    for tid in list_trips("student_test"):
        trip = load_trip(tid, "student_test")
        route = load_reference_route(tid, "student_test")
        labels = align_labels(trip, route)
        out = predict_trip(trip, model, params)
        rows.append({"trip": tid, "obs": trip.n_obs, **evaluate(trip, out, labels, route)})
        print(f"[hmm] {tid} clf={rows[-1]['pointwise_acc_clf']:.4f} hmm={rows[-1]['pointwise_acc_hmm']:.4f}")
        if tid == "00000000":
            m = trip_map(trip, reference_route=route, extra_routes={
                "Classifier-only route": (route_from_fixes(out["clf_arcs"]), "#d62728"),
                "HMM/Viterbi route": (route_from_fixes(out["hmm_arcs"]), "#6250d6")})
            m.save(str(OUT / f"map_student_test_{tid}.html"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "clf_vs_hmm_per_trip.csv", index=False)
    summ = df.drop(columns=["trip"]).mean(numeric_only=True).to_frame("mean_over_10_trips")
    summ.to_csv(OUT / "clf_vs_hmm_summary.csv")
    print(summ.to_string())
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (m, title) in zip(axs, [("pointwise_acc", "Pointwise accuracy"),
                                    ("continuity", "Route continuity"), ("route_f1", "Route arc F1")]):
        x = np.arange(len(df))
        ax.bar(x - 0.2, df[f"{m}_clf"], 0.4, color="#eb6834", label="classifier only")
        ax.bar(x + 0.2, df[f"{m}_hmm"], 0.4, color="#2a78d6", label="classifier + HMM")
        ax.set(title=f"{title} (student-test)", xticks=x, xticklabels=df.trip.str[-2:], xlabel="trip")
    axs[0].legend()
    fig.savefig(OUT / "clf_vs_hmm.png", bbox_inches="tight", dpi=130)


if __name__ == "__main__":
    main()
