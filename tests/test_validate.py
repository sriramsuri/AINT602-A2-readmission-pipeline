"""Validation behaviour, including deliberately corrupted inputs."""
import pytest

from src.config import load_config
from src.ingest import load_raw
from src.validate import load_schema, validate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def schema(cfg):
    return load_schema(cfg)


@pytest.fixture(scope="module")
def raw(cfg):
    return load_raw(cfg)


def test_clean_data_passes(raw, schema):
    report = validate(raw, schema, require_target=True, strict=True)
    assert report.ok, report.errors


def test_missing_column_is_error(raw, schema):
    bad = raw.drop(columns=["glucose"])
    report = validate(bad, schema, require_target=True, strict=True)
    assert not report.ok
    assert any("glucose" in c["check"] for c in report.errors)


def test_out_of_range_is_error_when_strict(raw, schema):
    bad = raw.copy()
    bad.loc[bad.index[:5], "age"] = 500
    report = validate(bad, schema, require_target=True, strict=True)
    assert not report.ok
    assert any(c["check"] == "range[age]" for c in report.errors)


def test_out_of_range_is_warning_when_lenient(raw, schema):
    bad = raw.copy()
    bad.loc[bad.index[:5], "age"] = 500
    report = validate(bad, schema, require_target=False, strict=False)
    assert report.ok
    assert any(c["check"] == "range[age]" for c in report.warnings)


def test_unexpected_category_flagged(raw, schema):
    bad = raw.copy()
    bad.loc[bad.index[:3], "sex"] = "Unknown"
    report = validate(bad, schema, require_target=True, strict=True)
    assert not report.ok
    assert any(c["check"] == "allowed[sex]" for c in report.errors)


def test_duplicate_ids_is_error(raw, schema):
    bad = raw.copy()
    bad.loc[bad.index[1], "patient_id"] = bad.loc[bad.index[0], "patient_id"]
    report = validate(bad, schema, require_target=True, strict=True)
    assert not report.ok
    assert any(c["check"] == "no_duplicate_ids" for c in report.errors)
