"""Stage 4 - the end-to-end model Pipeline.

One scikit-learn Pipeline holds every step: engineered features, preprocessing and the
estimator. Two design choices matter for the assessment.

* The ColumnTransformer selects inputs by name and uses remainder='drop', so the
  excluded columns (patient_id and mortality) and the target can never leak into the
  model even if they are present in the input frame.
* Imputation, scaling and one-hot encoding are fitted on training data only (they are
  inside the Pipeline, fitted in train.fit). The unknown-category setting on the encoder
  keeps serving robust to values never seen during training.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, FeatureEngineer

CANDIDATE_MODELS = ("logistic_regression", "hist_gradient_boosting")


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Column-wise preprocessing. Numeric columns are imputed and scaled; binary columns
    are imputed only; categorical columns are imputed and one-hot encoded (dense output,
    because the gradient-boosting model needs dense input)."""
    numeric = list(cfg["features"]["numeric"]) + ENGINEERED_NUMERIC
    binary = list(cfg["features"]["binary"])
    categorical = list(cfg["features"]["categorical"]) + ENGINEERED_CATEGORICAL

    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    binary_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("bin", binary_pipe, binary),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def make_estimator(name: str, cfg: dict):
    """Return an unfitted estimator by name."""
    seed = cfg.get("seed", 42)
    if name == "logistic_regression":
        # class_weight balances the minority (readmitted) class at training time.
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "baseline":
        strategy = cfg["model"].get("baseline", "most_frequent")
        return DummyClassifier(strategy=strategy, random_state=seed)
    raise ValueError(f"Unknown model name: {name!r}")


def build_pipeline(cfg: dict, model_name: str) -> Pipeline:
    """Assemble engineer -> preprocess -> model. Preprocessing is applied uniformly to
    every model (the baseline ignores its inputs, which is why it is a baseline)."""
    return Pipeline(
        [
            ("engineer", FeatureEngineer()),
            ("preprocess", build_preprocessor(cfg)),
            ("model", make_estimator(model_name, cfg)),
        ]
    )
