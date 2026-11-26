"""Pipeline fits for every model, and training and serving stay consistent."""
import numpy as np
import pytest

from src.config import load_config
from src.ingest import load_raw
from src.pipeline import build_pipeline


@pytest.fixture(scope="module")
def xy():
    cfg = load_config()
    df = load_raw(cfg).head(400)
    target = cfg["task"]["target"]
    return cfg, df.drop(columns=[target]), df[target].astype(int)


@pytest.mark.parametrize("name", ["baseline", "logistic_regression", "hist_gradient_boosting"])
def test_pipeline_fits_and_scores(xy, name):
    cfg, X, y = xy
    pipe = build_pipeline(cfg, name)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X.iloc[:10])[:, 1]
    assert len(proba) == 10
    assert ((proba >= 0) & (proba <= 1)).all()


def test_serving_applies_feature_engineering_internally(xy):
    cfg, X, y = xy
    pipe = build_pipeline(cfg, "logistic_regression")
    pipe.fit(X, y)
    one = X.iloc[[0]].copy()
    # The caller supplies only raw columns; the pipeline must add engineered ones itself.
    assert "mean_arterial_pressure" not in one.columns
    proba = pipe.predict_proba(one)[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0


def test_batch_and_single_predictions_match(xy):
    cfg, X, y = xy
    pipe = build_pipeline(cfg, "logistic_regression")
    pipe.fit(X, y)
    batch = pipe.predict_proba(X.iloc[:5])[:, 1]
    singles = np.array([pipe.predict_proba(X.iloc[[i]])[:, 1][0] for i in range(5)])
    assert np.allclose(batch, singles, atol=1e-9)
