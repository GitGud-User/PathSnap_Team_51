"""Extra figures used only by the technical report.

Run after models/train_all, models/optimise and hmm/compare:
    python tools/report_figures.py
Out: results/report/{train_vs_cv.png, route_zoom.png, route_zoom_info.json}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from pathsnap.config import RESULTS_DIR
from pathsnap.data.loaders import load_reference_route, load_trip
from pathsnap.hmm.viterbi import route_from_fixes
from pathsnap.labels.align import align_labels
from pathsnap.pipeline import load_artifacts, predict_trip

OUT = RESULTS_DIR / "report"
NAMES = {"logreg": "Logistic reg.", "knn": "KNN", "random_forest": "Random forest",
         "hist_gb": "HistGB", "lightgbm": "LightGBM", "mlp": "MLP"}


def train_vs_cv():
    c = pd.read_csv(RESULTS_DIR / "models" / "comparison.csv")
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(x - 0.27, c.train_top1_mean, 0.27, color="#9ec5f4", label="training folds")
    ax.bar(x, c.cv_top1_mean, 0.27, color="#2a78d6", label="validation folds (CV)")
    ax.bar(x + 0.27, c.st_top1, 0.27, color="#eb6834", label="student-test")
    for i, r in c.iterrows():
        ax.text(i - 0.27, r.train_top1_mean + 0.003, f"{r.train_top1_mean:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set(xticks=x, xticklabels=[NAMES[m] for m in c.model], ylabel="Top-1 accuracy",
           ylim=(0.75, 1.02), title="Train vs validation vs student-test Top-1 (over/underfitting check)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", color="#e1e0d9")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "train_vs_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def route_zoom():
    per = pd.read_csv(RESULTS_DIR / "hmm" / "clf_vs_hmm_per_trip.csv", dtype={"trip": str})
    per["fixed"] = per.invalid_transitions_clf - per.invalid_transitions_hmm
    tid = per.sort_values("fixed", ascending=False).trip.iloc[0]
    trip = load_trip(tid, "student_test")
    route = load_reference_route(tid, "student_test")
    labels = align_labels(trip, route)
    model, params = load_artifacts()
    out = predict_trip(trip, model, params)
    clf, hmm = out["clf_arcs"], out["hmm_arcs"]
    gain = ((clf != labels) & (hmm == labels) & (labels >= 0)).astype(int)
    W = 90
    i0 = int(np.argmax(np.convolve(gain, np.ones(W), "valid")))
    sl = slice(i0, i0 + W)
    xy, nxy, arcs = trip.gps_xy, trip.nodes_xy, trip.arcs

    def segs(ids):
        ids = np.asarray(ids)
        ids = ids[ids >= 0]
        return np.stack([nxy[arcs[ids, 0]], nxy[arcs[ids, 1]]], 1)

    win = xy[sl]
    x0, y0 = win.min(0) - 60
    x1, y1 = win.max(0) + 60
    mid = (nxy[arcs[:, 0]] + nxy[arcs[:, 1]]) / 2
    near = np.flatnonzero((mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1))
    ref_ids = np.unique(labels[sl][labels[sl] >= 0])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, ids, col, title in [(axs[0], clf[sl], "#eb6834", "Classifier only"),
                                (axs[1], hmm[sl], "#1a5cff", "Classifier + HMM/Viterbi")]:
        ax.add_collection(LineCollection(segs(near), colors="#d0cfca", linewidths=1))
        ax.add_collection(LineCollection(segs(ref_ids), colors="#2ca02c", linewidths=7, alpha=0.35))
        ax.add_collection(LineCollection(segs(np.unique(ids)), colors=col, linewidths=2.2))
        ax.plot(win[:, 0], win[:, 1], ".", color="#111", ms=3)
        acc = float((ids[labels[sl] >= 0] == labels[sl][labels[sl] >= 0]).mean())
        ax.set(title=f"{title}  (accuracy in window {acc:.0%})", xlim=(x0, x1), ylim=(y0, y1))
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    axs[0].legend(handles=[plt.Line2D([], [], color="#d0cfca", lw=1),
                           plt.Line2D([], [], color="#2ca02c", lw=7, alpha=0.35),
                           plt.Line2D([], [], color="#eb6834", lw=2.2),
                           plt.Line2D([], [], color="#1a5cff", lw=2.2),
                           plt.Line2D([], [], color="#111", marker=".", ls="")],
                  labels=["road network", "reference route", "classifier arcs", "HMM arcs", "GPS fixes"],
                  loc="lower left", fontsize=7)
    fig.suptitle(f"Student-test trip {tid}, fixes {i0}–{i0 + W - 1}: where the HMM corrects the classifier")
    fig.savefig(OUT / "route_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (OUT / "route_zoom_info.json").write_text(json.dumps(
        {"trip": tid, "window_start": i0, "window_len": W,
         "fixes_corrected_in_window": int(gain[sl].sum())}, indent=2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    train_vs_cv()
    route_zoom()
    print("wrote", OUT)
