"""Stage 5 - training, benchmarking and model selection.

Strategy: a stratified train / validation / test split. Models are fitted on train,
compared on validation, and the chosen candidate is then refitted on train + validation
and reported once on the held-out test set. The operating threshold is selected on
validation, never on test, so the test metrics are not tuned on their own data.

Every run is tracked with MLflow and summarised in artifacts/metrics.json and
artifacts/run_metadata.json, which together record what produced the saved model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .config import load_config, resolve
from .ingest import load_raw
from .pipeline import build_pipeline
from .validate import load_schema, validate

# MLflow 3.x gates the local file-store backend behind this flag. We keep the file store
# because its on-disk run tree (params, metrics, artifacts) is easy to inspect and commit
# as evidence; this opts back into it without needing a database server.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

log = logging.getLogger("train")


def data_hash(path: Path) -> str:
    """SHA-256 of the raw data file, recorded for provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_splits(df: pd.DataFrame, cfg: dict) -> dict:
    """Stratified train / validation / test split, driven by config."""
    target = cfg["task"]["target"]
    stratify_on = cfg["split"].get("stratify", True)
    seed = cfg["seed"]
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_trv, X_test, y_trv, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"],
        stratify=y if stratify_on else None, random_state=seed,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trv, y_trv, test_size=cfg["split"]["val_size"],
        stratify=y_trv if stratify_on else None, random_state=seed,
    )
    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_trainval": X_trv, "y_trainval": y_trv,
    }


def _proba(estimator, X) -> np.ndarray:
    return estimator.predict_proba(X)[:, 1]


def metrics_at(y_true, proba, threshold: float) -> dict:
    """Full metric set at one operating threshold."""
    y_pred = (proba >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        out["roc_auc"] = None
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


def choose_threshold(y_true, proba, objective: str = "f1") -> float:
    """Select an operating threshold on validation data by maximising F1."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    return float(thresholds[int(np.argmax(f1))])


def env_versions() -> dict:
    import sklearn

    import mlflow

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "mlflow": mlflow.__version__,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train, benchmark and select the model.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    import mlflow

    df = load_raw(cfg)
    schema = load_schema(cfg)
    report = validate(df, schema, require_target=True, strict=True)
    report.raise_for_errors()
    log.info("Validation passed: %d rows, %d warning(s)", report.n_rows, len(report.warnings))

    splits = make_splits(df, cfg)
    seed = cfg["seed"]
    target = cfg["task"]["target"]
    candidates = list(cfg["model"]["candidates"])

    mlflow.set_tracking_uri(str(resolve(cfg, "mlruns")))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Fit baseline + candidates on train, compare on validation.
    val_results: dict = {}
    fitted: dict = {}
    for name in ["baseline"] + candidates:
        pipe = build_pipeline(cfg, name)
        pipe.fit(splits["X_train"], splits["y_train"])
        fitted[name] = pipe
        proba_val = _proba(pipe, splits["X_val"])
        val_pr = float(average_precision_score(splits["y_val"], proba_val))
        try:
            val_roc = float(roc_auc_score(splits["y_val"], proba_val))
        except ValueError:
            val_roc = None
        val_results[name] = {"pr_auc": val_pr, "roc_auc": val_roc}
        with mlflow.start_run(run_name=f"{name}-val"):
            mlflow.log_params({"model": name, "seed": seed, "target": target,
                               "n_train": int(len(splits["y_train"])), "n_val": int(len(splits["y_val"]))})
            metrics_to_log = {"val_pr_auc": val_pr}
            if val_roc is not None:
                metrics_to_log["val_roc_auc"] = val_roc
            mlflow.log_metrics(metrics_to_log)
        log.info("%-22s val PR-AUC=%.4f ROC-AUC=%s", name, val_pr, val_roc)

    best_name = max(candidates, key=lambda n: val_results[n]["pr_auc"])
    log.info("Selected candidate on validation PR-AUC: %s", best_name)

    # Operating threshold selected on validation only.
    configured = cfg["model"].get("decision_threshold")
    if configured is None:
        proba_val_best = _proba(fitted[best_name], splits["X_val"])
        threshold = choose_threshold(splits["y_val"], proba_val_best,
                                     cfg["model"].get("threshold_objective", "f1"))
    else:
        threshold = float(configured)
    log.info("Operating threshold: %.3f", threshold)

    # Refit baseline + every candidate on train+val, report once on test.
    test_metrics: dict = {}
    final_pipe = None
    for name in ["baseline"] + candidates:
        pipe = build_pipeline(cfg, name)
        pipe.fit(splits["X_trainval"], splits["y_trainval"])
        proba_test = _proba(pipe, splits["X_test"])
        thr = 0.5 if name == "baseline" else threshold
        test_metrics[name] = metrics_at(splits["y_test"], proba_test, thr)
        if name == best_name:
            final_pipe = pipe

    # Persist the selected model and its metadata.
    model_path = resolve(cfg, cfg["paths"]["model_out"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": final_pipe, "threshold": threshold, "model_name": best_name,
         "features": cfg["features"], "target": target, "seed": seed},
        model_path,
    )

    metrics = {
        "target": target,
        "selected_model": best_name,
        "threshold": threshold,
        "validation": val_results,
        "test": test_metrics,
        "class_balance": {
            "train_positive_rate": float(np.mean(splits["y_train"])),
            "test_positive_rate": float(np.mean(splits["y_test"])),
        },
    }
    metrics_path = resolve(cfg, cfg["paths"]["metrics_out"])
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    raw_path = resolve(cfg, cfg["data"]["raw_path"])
    run_meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "data_file": cfg["data"]["raw_path"],
        "data_sha256": data_hash(raw_path),
        "n_rows": int(len(df)),
        "split_sizes": {k: int(len(splits[f"y_{k}"])) for k in ("train", "val", "test", "trainval")},
        "selected_model": best_name,
        "threshold": threshold,
        "candidates": candidates,
        "library_versions": env_versions(),
        "model_artifact": cfg["paths"]["model_out"],
    }
    meta_path = resolve(cfg, cfg["paths"]["run_meta_out"])
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2)

    # Record the selected run and its artefacts in MLflow.
    with mlflow.start_run(run_name=f"{best_name}-final"):
        sel, base = test_metrics[best_name], test_metrics["baseline"]
        mlflow.log_params({"selected_model": best_name, "threshold": threshold, "seed": seed,
                           "n_trainval": int(len(splits["y_trainval"])), "n_test": int(len(splits["y_test"]))})
        final_metrics = {"test_pr_auc": sel["pr_auc"], "test_f1": sel["f1"],
                         "test_recall": sel["recall"], "test_precision": sel["precision"],
                         "baseline_pr_auc": base["pr_auc"]}
        if sel["roc_auc"] is not None:
            final_metrics["test_roc_auc"] = sel["roc_auc"]
        mlflow.log_metrics(final_metrics)
        for p in (model_path, metrics_path, meta_path):
            mlflow.log_artifact(str(p))

    sel = test_metrics[best_name]
    print(f"selected={best_name} threshold={threshold:.3f}")
    print(f"test PR-AUC selected={sel['pr_auc']:.4f} baseline={test_metrics['baseline']['pr_auc']:.4f} "
          f"ROC-AUC={sel['roc_auc']}")
    print(f"model    -> {model_path}")
    print(f"metrics  -> {metrics_path}")
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()
