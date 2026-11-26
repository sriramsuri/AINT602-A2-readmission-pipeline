"""Render the pipeline architecture diagram from code.

Generating the figure from the same stage names used in the implementation keeps the
diagram, the report and the code aligned: if a stage is renamed, the diagram changes
with it. Writes reports/figures/architecture.png and architecture.svg.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .config import load_config, resolve  # noqa: E402

STAGES = [
    "Raw data\n(data/raw/synthetic_clinical_dataset.csv)",
    "Ingestion\n(src/ingest.py)",
    "Validation\n(src/validate.py + data/schema.json)",
    "Feature engineering\n(src/features.py)",
    "Preprocessing\n(ColumnTransformer: impute, scale, encode)",
    "Train + benchmark\n(src/train.py: baseline vs candidates)",
    "Evaluation\n(src/evaluate.py: metrics, calibration, fairness)",
    "Model artifact\n(artifacts/model.joblib + threshold)",
    "Batch prediction\n(src/predict.py: new data to predictions)",
]


def _box(ax, cx, cy, w, h, text, fc, fs=9):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            fc=fc, ec="#33475b", lw=1.2,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p1, p2, style="-", color="#33475b"):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.2,
            color=color, linestyle=style, shrinkA=2, shrinkB=2,
        )
    )


def build(path_png, path_svg) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(STAGES)
    top, bottom = 0.93, 0.11
    ys = [top - i * (top - bottom) / (n - 1) for i in range(n)]
    cx, w, h = 0.33, 0.46, 0.066
    centers = []
    for y, label in zip(ys, STAGES):
        fc = "#dbe5f1"
        if label.startswith("Raw"):
            fc = "#e8e8e8"
        elif label.startswith("Model artifact"):
            fc = "#d9eed2"
        elif label.startswith("Batch prediction"):
            fc = "#fbe4cf"
        _box(ax, cx, y, w, h, label, fc)
        centers.append((cx, y))
    for i in range(n - 1):
        _arrow(ax, (cx, centers[i][1] - h / 2), (cx, centers[i + 1][1] + h / 2))

    # Experiment tracking, fed by train and evaluate.
    tx = 0.80
    track_y = (centers[5][1] + centers[6][1]) / 2
    _box(ax, tx, track_y, 0.34, 0.11,
         "Experiment tracking\n(MLflow runs,\nartifacts/metrics.json,\nartifacts/run_metadata.json)",
         "#ece2f4")
    _arrow(ax, (cx + w / 2, centers[5][1]), (tx - 0.17, track_y), style="--", color="#7a5ba6")
    _arrow(ax, (cx + w / 2, centers[6][1]), (tx - 0.17, track_y), style="--", color="#7a5ba6")

    # Monitoring, fed by prediction (a consideration beyond the project scope).
    my = 0.055
    _box(ax, tx, my, 0.34, 0.09,
         "Monitoring (beyond project)\ndata drift, validation failures,\nscore ranges; feeds revalidation\nand retraining",
         "#fbe1e1")
    _arrow(ax, (centers[-1][0] + w / 2, centers[-1][1]), (tx - 0.17, my), style="--", color="#b5534f")

    ax.text(0.5, 0.985, "30-Day Readmission Prediction - Pipeline Architecture",
            ha="center", va="top", fontsize=12, fontweight="bold")
    fig.savefig(path_png, dpi=160, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the architecture diagram.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    figures = resolve(cfg, cfg["paths"]["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    png, svg = figures / "architecture.png", figures / "architecture.svg"
    build(png, svg)
    print(f"architecture diagram -> {png}")


if __name__ == "__main__":
    main()
