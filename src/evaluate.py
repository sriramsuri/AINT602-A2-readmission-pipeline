"""Stage 6 - evaluation of the saved model.

Loads the saved pipeline, rebuilds the identical held-out test split (same seed), and
produces the figures and the numeric report the technical report refers to: ROC and
precision-recall curves, a confusion matrix, a calibration curve and a fairness slice
that reports error rates per group of the sex column.
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend so figures render in CI without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import load_config, resolve
from .ingest import load_raw
from .train import make_splits

log = logging.getLogger("evaluate")


def _fig_path(cfg: dict, name: str):
    d = resolve(cfg, cfg["paths"]["figures_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def plot_roc(y, proba, path):
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(y, proba, path):
    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    base = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"Model (PR-AUC = {ap:.3f})")
    ax.axhline(base, ls="--", color="grey", label=f"Base rate = {base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, threshold):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Not readmitted", "Readmitted"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y, proba, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        true_p, pred_p = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    except ValueError:
        true_p, pred_p = calibration_curve(y, proba, n_bins=10, strategy="uniform")
    ax.plot(pred_p, true_p, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (test set)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fairness(group_metrics, path, slice_col):
    groups = list(group_metrics.keys())
    recalls = [group_metrics[g]["recall"] for g in groups]
    rates = [group_metrics[g]["selection_rate"] for g in groups]
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(x - width / 2, recalls, width, label="Recall")
    ax.bar(x + width / 2, rates, width, label="Flagged rate")
    ax.set_xticks(x, groups)
    ax.set_xlabel(slice_col)
    ax.set_ylabel("Rate")
    ax.set_title(f"Error rates by {slice_col} (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fairness_slice(X_test, y_test, proba, threshold, slice_col) -> dict:
    y_pred = (proba >= threshold).astype(int)
    y_true = y_test.to_numpy()
    res = {}
    for g in sorted(X_test[slice_col].dropna().unique().tolist()):
        mask = (X_test[slice_col] == g).to_numpy()
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        res[str(g)] = {
            "n": int(mask.sum()),
            "positive_rate": float(np.mean(yt)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "selection_rate": float(np.mean(yp)),
        }
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate the saved model on the test set.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # The model artifact is produced by this project's own train.py (first-party and
    # trusted), so loading it with joblib is not deserialising from an untrusted source.
    bundle = joblib.load(resolve(cfg, cfg["paths"]["model_out"]))
    pipe, threshold = bundle["pipeline"], bundle["threshold"]

    df = load_raw(cfg)
    splits = make_splits(df, cfg)
    X_test, y_test = splits["X_test"], splits["y_test"]
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    try:
        roc = float(roc_auc_score(y_test, proba))
    except ValueError:
        roc = None
    slice_col = cfg["fairness"]["slice_column"]
    groups = fairness_slice(X_test, y_test, proba, threshold, slice_col)

    report = {
        "model": bundle["model_name"],
        "threshold": float(threshold),
        "n_test": int(len(y_test)),
        "test_positive_rate": float(np.mean(y_test)),
        "roc_auc": roc,
        "pr_auc": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_test, proba)),
        "confusion_matrix": {"tn": int(cm[0][0]), "fp": int(cm[0][1]),
                             "fn": int(cm[1][0]), "tp": int(cm[1][1])},
        "fairness_by_" + slice_col: groups,
    }

    plot_roc(y_test, proba, _fig_path(cfg, "roc_curve.png"))
    plot_pr(y_test, proba, _fig_path(cfg, "pr_curve.png"))
    plot_confusion(cm, _fig_path(cfg, "confusion_matrix.png"), threshold)
    plot_calibration(y_test, proba, _fig_path(cfg, "calibration.png"))
    plot_fairness(groups, _fig_path(cfg, "fairness_by_sex.png"), slice_col)

    out = resolve(cfg, cfg["paths"]["eval_out"])
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"model={report['model']} ROC-AUC={roc} PR-AUC={report['pr_auc']:.4f} "
          f"recall={report['recall']:.3f} precision={report['precision']:.3f} F1={report['f1']:.3f}")
    print(f"report  -> {out}")
    print(f"figures -> {resolve(cfg, cfg['paths']['figures_dir'])}")


if __name__ == "__main__":
    main()
