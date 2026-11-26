# Clinical 30-Day Readmission MLOps Pipeline

A small but complete, reproducible machine learning pipeline that predicts whether a
patient will be readmitted to hospital within 30 days of discharge. It is built for
AINT602 Assessment 2 and is designed so that another team member can inspect it, run
it, reproduce the result and reuse it on new data.

The point of this project is not the highest possible accuracy. It is an ML system that
is reproducible, testable, maintainable and responsible across the lifecycle.

- **Task:** binary classification, target `readmission_30d`.
- **Unit of prediction:** one hospital discharge (one patient episode).
- **Prediction time:** the moment of discharge. Only information available by then is
  used as a feature. `mortality` and `patient_id` are excluded on purpose (see
  `config.yaml`).
- **Repository:** https://github.com/sriramsuri/AINT602-A2-readmission-pipeline

## Repository layout

```
.
├─ README.md                 This file.
├─ config.yaml               Single source of truth: seed, split, features, paths.
├─ requirements.txt          Top-level dependencies.
├─ requirements.lock.txt     Exact pinned versions from the build environment.
├─ Makefile                  One-command stages for Unix/macOS.
├─ run.ps1                   One-command stages for Windows PowerShell.
├─ data/
│  ├─ raw/synthetic_clinical_dataset.csv   Raw input data (synthetic).
│  └─ schema.json            Data contract enforced by the validator.
├─ src/
│  ├─ config.py   ingest.py   validate.py   features.py
│  ├─ pipeline.py train.py    evaluate.py   predict.py
│  └─ make_diagram.py         Generates the architecture figure from the code.
├─ tests/                    pytest: schema, features, leakage, train-serve parity.
├─ artifacts/                Saved model, metrics, run metadata, validation output.
├─ mlruns/                   MLflow experiment tracking store.
├─ reports/figures/          Architecture diagram and evaluation figures.
└─ .github/workflows/ci.yml  Lint, tests and validation on every push.
```

## Setup

Python 3.11 or newer (developed and tested on 3.14). From the project root:

```bash
# create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

For an exact match of the build environment, use `pip install -r requirements.lock.txt`.

## Run the pipeline

Each stage is a module you can run on its own, so the workflow is transparent and
repeatable.

```bash
python -m src.ingest       # load and summarise the raw data
python -m src.validate     # run schema and data-quality checks
python -m src.train        # split, train baseline + candidates, track runs, save model
python -m src.evaluate     # metrics, figures and a fairness slice for the saved model
python -m src.predict --input data/sample_new_patients.csv --output artifacts/predictions.csv
```

Or run the whole thing with one command:

```bash
make all          # Unix / macOS
./run.ps1 all     # Windows PowerShell
```

## Reproduce the selected model

The run is defined by three things: the code (pinned by Git), `config.yaml` and the
raw data in `data/raw/`. To reproduce the selected model and its metrics:

```bash
pip install -r requirements.lock.txt
python -m src.train       # writes artifacts/model.joblib and artifacts/metrics.json
python -m src.evaluate    # writes artifacts/evaluation_report.json and reports/figures/
```

`artifacts/run_metadata.json` records the data hash, library versions, parameters,
chosen model, operating threshold and the metrics, so any saved model can be traced
back to exactly what produced it. The same information is logged to MLflow under the
`readmission_30d` experiment (`mlflow ui` to browse it).

## Tests and quality checks

```bash
ruff check src tests      # static checks
pytest -q                 # unit and integration tests
```

The tests cover schema validation (including deliberately corrupted inputs), the
engineered features, a guard that leakage columns never reach the model, and
train-serve consistency (the prediction path applies the same preprocessing as
training). The same checks run automatically in GitHub Actions on every push
(`.github/workflows/ci.yml`).

## Predict on new data

`src/predict.py` loads the saved pipeline and applies the identical preprocessing and
feature logic used in training, so training and serving cannot drift. Input is a CSV
with the feature columns (no target needed); output is a CSV of probabilities and
labels. A small example input is generated at `data/sample_new_patients.csv`.

## Data note

The dataset is synthetic and contains no real patient records. It is included in the
repository because it is small and lets anyone reproduce the result without external
access.

## Results summary

Populated from the latest run; full numbers are in `artifacts/metrics.json` and the
technical report. See `reports/figures/` for the ROC, precision-recall, calibration,
confusion-matrix and fairness figures.

| Model | ROC AUC | PR AUC | Recall (readmit) | Precision (readmit) |
|-------|---------|--------|------------------|---------------------|
| Baseline (most frequent) | _see metrics.json_ | | | |
| Logistic regression | | | | |
| Hist gradient boosting | | | | |

## Academic context

Built for AINT602 Assessment 2. The technical report and evidence appendix are in
`reports/`. AI assistance was used in building this project and is declared in the
accompanying AI Usage Statement.
