"""Smoke tests for the six candidate classifiers in pathsnap.models.registry.

Each model must build from the registry, train on a small synthetic sample shaped like our
real feature table, and return one valid probability per row. This catches a broken model
configuration before a long training run.  Run: pytest -q tests/test_models.py
"""
import numpy as np
import pytest

from pathsnap.features.build import FEATURES
from pathsnap.models.registry import MODELS, OBJECTIVES


def _toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    # label depends on the first two features, like distance and heading in the real data
    y = ((X[:, 0] + 0.5 * X[:, 1]) < 0).astype(int)
    return X, y


def test_registry_has_six_models_with_objectives():
    assert len(MODELS) == 6
    assert set(MODELS) == set(OBJECTIVES)
    for name, (family, factory) in MODELS.items():
        assert isinstance(family, str) and family
        assert callable(factory)


@pytest.mark.parametrize("name", list(MODELS))
def test_model_trains_and_returns_probabilities(name):
    X, y = _toy_data()
    model = MODELS[name][1]()
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))
    # every model should learn this easy rule much better than chance
    acc = ((p >= 0.5).astype(int) == y).mean()
    assert acc > 0.7, f"{name} accuracy {acc:.2f}"


def test_feature_count_matches_saved_model():
    import json
    from pathsnap.config import PROJECT_ROOT
    saved = json.loads((PROJECT_ROOT / "models" / "final" / "feature_list.json").read_text())
    assert saved == FEATURES
