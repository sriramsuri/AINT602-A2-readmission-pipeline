"""Stage 3 - feature engineering (leakage-safe).

Four engineered features, each computable from information available at discharge, so
none of them introduces future information:

* mean_arterial_pressure - average perfusion pressure, diastolic + (systolic - diastolic)/3.
* pulse_pressure - systolic minus diastolic, a marker of vascular stiffness.
* comorbidity_count - diabetes + hypertension, a simple chronic-burden score.
* bmi_category - clinical BMI band, which lets a linear model use BMI non-linearly.

They are wrapped in a scikit-learn transformer and placed inside the Pipeline. That is
deliberate: the prediction path then applies the identical logic used in training, so
training and serving cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_NUMERIC = ["mean_arterial_pressure", "pulse_pressure", "comorbidity_count"]
ENGINEERED_CATEGORICAL = ["bmi_category"]

BMI_BINS = [0.0, 18.5, 25.0, 30.0, np.inf]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered feature columns added.

    Pure and row-wise: no statistics are learned from the data, so the function is
    safe to apply to a single new record exactly as it is applied to the training set.
    """
    out = df.copy()
    sys_bp = pd.to_numeric(out["systolic_bp"], errors="coerce")
    dia_bp = pd.to_numeric(out["diastolic_bp"], errors="coerce")
    out["mean_arterial_pressure"] = dia_bp + (sys_bp - dia_bp) / 3.0
    out["pulse_pressure"] = sys_bp - dia_bp
    out["comorbidity_count"] = (
        pd.to_numeric(out["diabetes"], errors="coerce").fillna(0)
        + pd.to_numeric(out["hypertension"], errors="coerce").fillna(0)
    )
    bmi = pd.to_numeric(out["bmi"], errors="coerce")
    out["bmi_category"] = (
        pd.cut(bmi, bins=BMI_BINS, labels=BMI_LABELS, right=False).astype("object")
    )
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """scikit-learn transformer that adds the engineered features.

    Nothing is fitted (transform is a pure function of each row), so fit is a no-op.
    Keeping it as a pipeline step is what guarantees train-serve consistency.
    """

    def fit(self, X, y=None):  # noqa: N803 (scikit-learn naming convention)
        return self

    def transform(self, X):  # noqa: N803
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame as input.")
        return add_engineered_features(X)

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL, dtype=object)
