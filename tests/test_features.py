"""Engineered features compute the intended values and the transformer adds them."""
import pandas as pd

from src.features import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    FeatureEngineer,
    add_engineered_features,
)


def _row(**overrides):
    base = dict(systolic_bp=120.0, diastolic_bp=80.0, diabetes=1, hypertension=0, bmi=27.0)
    base.update(overrides)
    return pd.DataFrame([base])


def test_pulse_pressure_and_map():
    out = add_engineered_features(_row(systolic_bp=120.0, diastolic_bp=90.0))
    assert out["pulse_pressure"].iloc[0] == 30.0
    assert abs(out["mean_arterial_pressure"].iloc[0] - (90 + 30 / 3)) < 1e-9


def test_comorbidity_count():
    out = add_engineered_features(_row(diabetes=1, hypertension=1))
    assert out["comorbidity_count"].iloc[0] == 2


def test_bmi_category_bands():
    expected = {15.0: "underweight", 22.0: "normal", 27.0: "overweight", 35.0: "obese"}
    for bmi, label in expected.items():
        out = add_engineered_features(_row(bmi=bmi))
        assert out["bmi_category"].iloc[0] == label


def test_transformer_adds_expected_columns():
    out = FeatureEngineer().fit_transform(_row())
    for col in ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL:
        assert col in out.columns
