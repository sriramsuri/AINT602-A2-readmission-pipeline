"""Stage 7 - batch prediction on new data.

Loads the saved pipeline and applies it to a CSV of new records. Because the saved
object is the whole Pipeline (feature engineering, preprocessing and model), the
prediction path runs the identical transformations used in training, so training and
serving cannot drift apart. Input needs the feature columns only; no target is required.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config, resolve
from .validate import load_schema, validate

log = logging.getLogger("predict")


def load_model(cfg: dict):
    # The model artifact is produced by this project's own train.py (first-party and
    # trusted), so joblib loading here is not deserialising from an untrusted source.
    return joblib.load(resolve(cfg, cfg["paths"]["model_out"]))


def predict_frame(bundle: dict, df: pd.DataFrame):
    pipe, thr = bundle["pipeline"], bundle["threshold"]
    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= thr).astype(int)
    return proba, pred


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate readmission predictions for new data.")
    parser.add_argument("--input", required=True, help="CSV of new records (feature columns; no target needed).")
    parser.add_argument("--output", default=None, help="Where to write the predictions CSV.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    df = pd.read_csv(in_path)

    # Serve-time validation is lenient: quality issues are warnings, which the pipeline's
    # imputer and unknown-category handling absorb. Structural errors still stop the run.
    schema = load_schema(cfg)
    report = validate(df, schema, require_target=False, strict=False)
    for c in report.errors:
        log.error("validation error - %s: %s", c["check"], c["detail"])
    report.raise_for_errors()
    for c in report.warnings:
        log.warning("%s: %s", c["check"], c["detail"])

    bundle = load_model(cfg)
    proba, pred = predict_frame(bundle, df)

    id_col = cfg["task"]["id_column"]
    out = pd.DataFrame()
    if id_col in df.columns:
        out[id_col] = df[id_col]
    out["readmission_probability"] = proba.round(4)
    out["readmission_prediction"] = pred
    out_path = Path(args.output) if args.output else resolve(cfg, cfg["paths"]["predictions_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"records={len(out)} predicted_positive={int(pred.sum())} threshold={bundle['threshold']:.3f}")
    print(out.head(10).to_string(index=False))
    print(f"predictions -> {out_path}")


if __name__ == "__main__":
    main()
