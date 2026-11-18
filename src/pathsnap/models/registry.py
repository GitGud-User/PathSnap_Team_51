"""The six candidate classifiers and their initial (untuned) configurations."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.config import RANDOM_SEED as S

MODELS = {
    "logreg": ("Linear", lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
    "knn": ("Instance-based", lambda: make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
    "random_forest": ("Bagging ensemble", lambda: RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=S)),
    "hist_gb": ("Boosting", lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=S)),
    "lightgbm": ("Boosting", lambda: LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=S, verbose=-1)),
    "mlp": ("Neural network", lambda: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                        batch_size=1024, max_iter=60, early_stopping=True,
                                        n_iter_no_change=5, random_state=S))),
}

# Loss / objective, for the report
OBJECTIVES = {
    "logreg": "L2-regularised binary cross-entropy (log loss)",
    "knn": "none (lazy learner); probability = distance-weighted neighbour vote",
    "random_forest": "Gini impurity per split; bagged trees, probability = mean leaf frequency",
    "hist_gb": "binary log loss, gradient boosting on histograms",
    "lightgbm": "binary log loss, leaf-wise gradient boosting",
    "mlp": "binary cross-entropy minimised with Adam, early stopping on 10% validation",
}
