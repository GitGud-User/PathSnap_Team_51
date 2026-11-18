"""Stage 5-6: train and compare six classifiers.

Protocol
  * Training data: the 80 train trips only. GroupKFold(5) grouped by trip_id, so a trip's
    fixes never appear in both the fit and validation folds.
  * Rows: (fix, candidate) pairs. Only labelled fixes are used as training targets.
  * Fixes are subsampled every STRIDE-th observation (1 Hz fixes are near-duplicates; this
    keeps runtime laptop-friendly). KNN is additionally capped at KNN_MAX_ROWS.
  * Student-test (10 trips) is used only for the final comparison, on ALL its fixes.

Run:  python -m pathsnap.models.train_all
Out:  results/models/{comparison.csv, cv_folds.csv, configs.json, *.png}
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from pathsnap.config import RANDOM_SEED, RESULTS_DIR
from pathsnap.eval.metrics import classifier_choice, route_continuity, row_metrics, top1_accuracy
from pathsnap.features.build import FEATURES
from pathsnap.models.registry import MODELS, OBJECTIVES

STRIDE = 4
KNN_MAX_ROWS = 150_000
OUT = RESULTS_DIR / "models"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def load(part: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "features" / f"{part}.parquet")


def n_labelled(part: str) -> int:
    q = pd.read_csv(RESULTS_DIR / "features" / "candidate_quality.csv")
    return int(q[q.partition == part].labelled_obs.sum())


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.true_arc >= 0) & (df.obs_idx % STRIDE == 0)].reset_index(drop=True)


def fit(name, X, y):
    m = MODELS[name][1]()
    if name == "knn" and len(X) > KNN_MAX_ROWS:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(X), KNN_MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    t0 = time.perf_counter()
    m.fit(X, y)
    return m, time.perf_counter() - t0


def continuity_of(df: pd.DataFrame, score: np.ndarray, arcs_by_trip: dict) -> float:
    pred = classifier_choice(df, score)
    vals = []
    for tid, s in pred.groupby(level=0):
        vals.append(route_continuity(s.to_numpy(), arcs_by_trip[tid])["continuity"])
    return float(np.mean(vals))


def main() -> None:
    from pathsnap.data.loaders import load_trip

    OUT.mkdir(parents=True, exist_ok=True)
    tr_all, st = load("train"), load("student_test")
    tr = training_rows(tr_all)
    X, y, groups = tr[FEATURES].to_numpy(float), tr.y.to_numpy(), tr.trip_id.to_numpy()
    Xs = st[FEATURES].to_numpy(float)
    st_arcs = {t: load_trip(t, "student_test").arcs for t in st.trip_id.unique()}
    n_lab_st = n_labelled("student_test")
    print(f"train rows={len(tr)} (pos rate {y.mean():.3f}), student-test rows={len(st)}")

    folds, rows, cms = [], [], {}
    gkf = GroupKFold(n_splits=5)
    for name, (family, _) in MODELS.items():
        f1s, t1s, lls, tr_f1s, tr_t1s, tr_lls = [], [], [], [], [], []
        for k, (a, b) in enumerate(gkf.split(X, y, groups)):
            m, _ = fit(name, X[a], y[a])
            p = m.predict_proba(X[b])[:, 1]
            r = row_metrics(y[b], p)
            t1 = top1_accuracy(tr.iloc[b], p)
            # Same metrics on the data the model was fitted on (every 5th fix of the training
            # fold, whole candidate groups kept) -> train-vs-validation gap for over/underfitting.
            a_eval = a[tr.obs_idx.to_numpy()[a] % (STRIDE * 5) == 0]
            pa = m.predict_proba(X[a_eval])[:, 1]
            ra = row_metrics(y[a_eval], pa)
            t1a = top1_accuracy(tr.iloc[a_eval], pa)
            folds.append({"model": name, "fold": k, **r, "top1": t1,
                          "train_macro_f1": ra["macro_f1"], "train_top1": t1a,
                          "train_log_loss": ra["log_loss"]})
            f1s.append(r["macro_f1"]); t1s.append(t1); lls.append(r["log_loss"])
            tr_f1s.append(ra["macro_f1"]); tr_t1s.append(t1a); tr_lls.append(ra["log_loss"])
        m, t_train = fit(name, X, y)
        t0 = time.perf_counter()
        ps = m.predict_proba(Xs)[:, 1]
        t_inf = time.perf_counter() - t0
        rs = row_metrics(st.y.to_numpy(), ps)
        rows.append({
            "model": name, "family": family, "objective": OBJECTIVES[name],
            "cv_macro_f1_mean": np.mean(f1s), "cv_macro_f1_std": np.std(f1s),
            "cv_top1_mean": np.mean(t1s), "cv_top1_std": np.std(t1s), "cv_log_loss": np.mean(lls),
            "train_macro_f1_mean": np.mean(tr_f1s), "train_top1_mean": np.mean(tr_t1s),
            "train_log_loss_mean": np.mean(tr_lls),
            "st_accuracy": rs["accuracy"], "st_precision": rs["precision"],
            "st_recall": rs["recall"], "st_macro_f1": rs["macro_f1"], "st_log_loss": rs["log_loss"],
            "st_top1": top1_accuracy(st, ps, n_lab_st),
            "st_route_continuity": continuity_of(st, ps, st_arcs),
            "train_time_s": t_train, "infer_ms_per_1k_rows": 1000 * t_inf / len(Xs) * 1000,
        })
        cms[name] = confusion_matrix(st.y, (ps >= 0.5).astype(int))
        if name == "mlp":
            mlp = m[-1]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(mlp.loss_curve_, color=COLORS[0], lw=2, label="training loss (log loss)")
            ax2 = ax.twinx() if False else None
            ax.set(title="MLP training history (real, per epoch)", xlabel="epoch", ylabel="log loss")
            ax.legend()
            fig.savefig(OUT / "mlp_loss_curve.png", bbox_inches="tight", dpi=130)
            fig2, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(mlp.validation_scores_, color=COLORS[1], lw=2)
            ax.set(title="MLP internal validation accuracy per epoch", xlabel="epoch", ylabel="accuracy")
            fig2.savefig(OUT / "mlp_validation_curve.png", bbox_inches="tight", dpi=130)
            plt.close("all")
        print(f"[models] {name:14s} cvF1={np.mean(f1s):.4f} cvTop1={np.mean(t1s):.4f} "
              f"stTop1={rows[-1]['st_top1']:.4f} stF1={rs['macro_f1']:.4f} ll={rs['log_loss']:.4f} "
              f"train={t_train:.1f}s")

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "cv_folds.csv", index=False)
    (OUT / "configs.json").write_text(json.dumps(
        {n: str(MODELS[n][1]()) for n in MODELS} | {"stride": STRIDE, "knn_max_rows": KNN_MAX_ROWS,
                                                    "cv": "GroupKFold(5) by trip_id", "seed": RANDOM_SEED},
        indent=2))
    plots(comp, pd.DataFrame(folds), cms)


def plots(comp, folds, cms):
    names = list(comp.model)
    col = {n: COLORS[i] for i, n in enumerate(names)}

    def bar(metric, title, fname, err=None, lower_better=False):
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(names, comp[metric], color=[col[n] for n in names],
               yerr=comp[err] if err else None, capsize=3)
        for i, v in enumerate(comp[metric]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set(title=title + (" (lower is better)" if lower_better else ""))
        ax.grid(axis="y", color="#e1e0d9")
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(OUT / fname, bbox_inches="tight", dpi=130)
        plt.close(fig)

    bar("cv_macro_f1_mean", "Cross-validation macro-F1 (GroupKFold 5, train)", "cv_macro_f1.png", "cv_macro_f1_std")
    bar("cv_top1_mean", "Cross-validation Top-1 accuracy (train)", "cv_top1.png", "cv_top1_std")
    bar("st_top1", "Student-test Top-1 accuracy", "st_top1.png")
    bar("st_macro_f1", "Student-test macro-F1", "st_macro_f1.png")
    bar("st_log_loss", "Student-test log loss", "st_log_loss.png", lower_better=True)
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for ax, n in zip(axs.flat, names):
        cm = cms[n]
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8)
        ax.set(title=n, xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true")
    fig.suptitle("Student-test confusion matrices (candidate rows, threshold 0.5)")
    fig.savefig(OUT / "confusion_matrices.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
