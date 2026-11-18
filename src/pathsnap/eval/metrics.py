"""Shared metric definitions (used by model experiments, the HMM comparison and the API)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score)


def row_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    """Binary (candidate is / is not the true arc) metrics on candidate rows."""
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def classifier_choice(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """Classifier-only prediction: the highest-scoring candidate arc per (trip, obs)."""
    d = df[["trip_id", "obs_idx", "arc_id"]].assign(score=score)
    best = d.loc[d.groupby(["trip_id", "obs_idx"]).score.idxmax()]
    return best.set_index(["trip_id", "obs_idx"]).arc_id


def top1_accuracy(df: pd.DataFrame, score: np.ndarray, n_labelled: int | None = None) -> float:
    """Observation-level Top-1: share of LABELLED fixes whose top-scored candidate is the
    reference arc. Labelled fixes with no correct candidate count as errors."""
    pred = classifier_choice(df, score)
    truth = df.groupby(["trip_id", "obs_idx"]).true_arc.first()
    lab = truth[truth >= 0]
    correct = (pred.reindex(lab.index) == lab).sum()
    denom = n_labelled if n_labelled is not None else len(lab)
    return float(correct / max(1, denom))


def route_continuity(arc_seq: np.ndarray, arcs: np.ndarray) -> dict:
    """Over consecutive matched fixes whose arc changes: share where to(prev) == from(next)."""
    s = arc_seq[arc_seq >= 0]
    ch = np.flatnonzero(s[1:] != s[:-1])
    if len(ch) == 0:
        return {"continuity": 1.0, "invalid_transitions": 0, "arc_changes": 0}
    ok = arcs[s[ch], 1] == arcs[s[ch + 1], 0]
    return {"continuity": float(ok.mean()), "invalid_transitions": int((~ok).sum()),
            "arc_changes": int(len(ch))}


def pointwise_accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    lab = labels >= 0
    return float((pred[lab] == labels[lab]).mean()) if lab.any() else float("nan")


def route_f1(pred_arcs: np.ndarray, ref_route: np.ndarray) -> dict:
    """Sequence-level: arc-set precision/recall/F1 of the decoded route vs reference route."""
    P, R = set(pred_arcs[pred_arcs >= 0].tolist()), set(ref_route.tolist())
    tp = len(P & R)
    prec = tp / max(1, len(P))
    rec = tp / max(1, len(R))
    return {"route_precision": prec, "route_recall": rec,
            "route_f1": 2 * prec * rec / max(1e-9, prec + rec)}
