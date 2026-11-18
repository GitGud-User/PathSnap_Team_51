"""Stage 7: Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (LightGBM).

Objective : mean observation-level Top-1 accuracy over GroupKFold(3) on the 80 train trips
            (Top-1 is what the HMM consumes; log loss is recorded per trial as well).
Never uses student-test or evaluator data for the search. Student-test is used once,
afterwards, to compare tuned vs untuned with identical metrics.

Run:  python -m pathsnap.models.optimise [--trials 25] [--timeout 600]
Out:  results/optuna/{trials.csv, best_params.json, search_space.json, tuned_vs_untuned.csv,
      optimisation_history.png}, models/final/{model.joblib, feature_list.json, model_card.json}
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.config import PROJECT_ROOT, RANDOM_SEED, RESULTS_DIR
from pathsnap.eval.metrics import row_metrics, top1_accuracy
from pathsnap.features.build import FEATURES
from pathsnap.models.registry import MODELS
from pathsnap.models.train_all import load, n_labelled, training_rows

OUT = RESULTS_DIR / "optuna"
FINAL = PROJECT_ROOT / "models" / "final"

SEARCH_SPACE = {
    "num_leaves": "int 15..255 (log)", "learning_rate": "float 0.01..0.3 (log)",
    "n_estimators": "int 100..800", "min_child_samples": "int 10..500 (log)",
    "colsample_bytree": "float 0.5..1.0", "subsample": "float 0.5..1.0",
    "reg_lambda": "float 1e-8..10 (log)", "reg_alpha": "float 1e-8..10 (log)",
}


def suggest(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 500, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
    )


def make(params):
    return LGBMClassifier(subsample_freq=1, random_state=RANDOM_SEED, verbose=-1, **params)


def cv_eval(model_fn, tr, X, y, groups, k=3):
    t1, f1, ll = [], [], []
    for a, b in GroupKFold(k).split(X, y, groups):
        m = model_fn().fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        r = row_metrics(y[b], p)
        t1.append(top1_accuracy(tr.iloc[b], p)); f1.append(r["macro_f1"]); ll.append(r["log_loss"])
    return float(np.mean(t1)), float(np.mean(f1)), float(np.mean(ll))


def student_eval(m, st, n_lab):
    t0 = time.perf_counter()
    p = m.predict_proba(st[FEATURES].to_numpy(float))[:, 1]
    inf = (time.perf_counter() - t0) * 1000 / len(st) * 1000
    r = row_metrics(st.y.to_numpy(), p)
    return {"st_top1": top1_accuracy(st, p, n_lab), "st_macro_f1": r["macro_f1"],
            "st_log_loss": r["log_loss"], "infer_ms_per_1k_rows": inf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)

    tr = training_rows(load("train"))
    st = load("student_test")
    X, y, g = tr[FEATURES].to_numpy(float), tr.y.to_numpy(), tr.trip_id.to_numpy()

    def objective(trial):
        params = suggest(trial)
        t1, f1, ll = cv_eval(lambda: make(params), tr, X, y, g)
        trial.set_user_attr("cv_macro_f1", f1)
        trial.set_user_attr("cv_log_loss", ll)
        return t1

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
                                study_name="lightgbm_top1")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    opt_time = time.perf_counter() - t0

    study.trials_dataframe().to_csv(OUT / "trials.csv", index=False)
    (OUT / "search_space.json").write_text(json.dumps(
        {"sampler": "TPE (seed 42)", "objective": "maximize mean GroupKFold(3) Top-1 on train",
         "n_trials_requested": args.trials, "timeout_s": args.timeout, "space": SEARCH_SPACE}, indent=2))
    (OUT / "best_params.json").write_text(json.dumps(
        {"best_value_cv_top1": study.best_value, "best_params": study.best_params,
         "n_trials_completed": len(study.trials), "optimisation_seconds": opt_time}, indent=2))

    # tuned vs untuned with identical protocol
    rows, fitted = [], {}
    n_lab = n_labelled("student_test")
    for label, fn in (("untuned", MODELS["lightgbm"][1]), ("tuned", lambda: make(study.best_params))):
        t1, f1, ll = cv_eval(fn, tr, X, y, g)
        t0 = time.perf_counter()
        m = fn().fit(X, y)
        fitted[label] = m
        rows.append({"model": label, "cv_top1": t1, "cv_macro_f1": f1, "cv_log_loss": ll,
                     "train_time_s": time.perf_counter() - t0, **student_eval(m, st, n_lab)})
    pd.DataFrame(rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)

    # Final model = higher CV Top-1 (train only). Student-test is reported, never used to choose.
    chosen = max(rows, key=lambda r: r["cv_top1"])["model"]
    joblib.dump(fitted[chosen], FINAL / "model.joblib")
    (FINAL / "feature_list.json").write_text(json.dumps(FEATURES, indent=2))
    (FINAL / "model_card.json").write_text(json.dumps(
        {"model": f"LightGBM ({chosen} configuration)",
         "params": fitted[chosen].get_params(),
         "selection_rule": "higher GroupKFold(3) CV Top-1 on train between untuned and Optuna-best",
         "cv_top1": {r["model"]: r["cv_top1"] for r in rows},
         "trained_on": "80 train trips, labelled fixes, every 4th fix",
         "evidence": ["results/models/comparison.csv", "results/optuna/tuned_vs_untuned.csv"]},
        indent=2, default=str))
    print(f"final model: {chosen} configuration")

    vals = [t.value for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(len(vals)), vals, "o", color="#2a78d6", ms=4, label="trial")
    ax.plot(range(len(vals)), np.maximum.accumulate(vals), color="#eb6834", lw=2, label="best so far")
    ax.set(title="Optuna TPE optimisation history (LightGBM)", xlabel="trial", ylabel="CV Top-1")
    ax.legend()
    fig.savefig(OUT / "optimisation_history.png", bbox_inches="tight", dpi=130)
    print(pd.DataFrame(rows).to_string())
    print("best", study.best_params, "time", round(opt_time, 1))


if __name__ == "__main__":
    main()
