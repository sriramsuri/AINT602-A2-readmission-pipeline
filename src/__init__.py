"""Reusable MLOps pipeline for 30-day hospital readmission prediction.

Stages (each a module in this package):
    ingest    -> load raw data from the configured source
    validate  -> schema and data-quality checks
    features  -> leakage-safe engineered features (a scikit-learn transformer)
    pipeline  -> the end-to-end preprocessing + model Pipeline
    train     -> split, fit baseline and candidates, track runs, save the model
    evaluate  -> metrics, figures and a fairness slice for the saved model
    predict   -> apply the saved pipeline to new data

AINT602 Assessment 2.
"""

__version__ = "1.0.0"
