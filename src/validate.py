"""Stage 2 - validation.

Check an incoming batch of data against the contract in data/schema.json before it
is allowed near a model. The same module is used in two modes:

* strict mode (training): data-quality problems are errors and stop the run, so a
  corrupt batch can never quietly train a model.
* lenient mode (serving): the same problems are warnings. The pipeline's imputer and
  the one-hot encoder (handle_unknown='ignore') then absorb them, so a single odd
  value does not crash a live prediction.

Checks: required columns present, no unexpected columns, parseable data types, missing
values, numeric ranges, categorical value sets, duplicate rows and duplicate ids, and a
valid binary target when one is required.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from .config import load_config, resolve
from .ingest import load_raw

log = logging.getLogger("validate")


@dataclass
class ValidationReport:
    """Structured result of validating one batch of data."""

    n_rows: int = 0
    mode: str = "strict"
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, level: str, detail: str, count: int = 0) -> None:
        self.checks.append(
            {"check": name, "passed": passed, "level": level, "detail": detail, "count": count}
        )

    @property
    def errors(self) -> list[dict]:
        return [c for c in self.checks if not c["passed"] and c["level"] == "error"]

    @property
    def warnings(self) -> list[dict]:
        return [c for c in self.checks if not c["passed"] and c["level"] == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "mode": self.mode,
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "checks": self.checks,
        }

    def raise_for_errors(self) -> None:
        if not self.ok:
            lines = "; ".join(f"{c['check']}: {c['detail']}" for c in self.errors)
            raise ValueError(f"Validation failed with {len(self.errors)} error(s): {lines}")


def load_schema(cfg: dict) -> dict:
    schema_path = resolve(cfg, cfg["data"]["schema_path"])
    with open(schema_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_missing(series: pd.Series) -> pd.Series:
    """Treat NaN and blank strings as missing (works for object and string dtypes)."""
    missing = series.isna()
    if not pd.api.types.is_numeric_dtype(series):
        stripped = series.astype("string").str.strip()
        missing = missing | stripped.eq("")
    return missing


def validate(
    df: pd.DataFrame,
    schema: dict,
    require_target: bool = True,
    strict: bool = True,
) -> ValidationReport:
    """Validate df against schema and return a ValidationReport."""
    report = ValidationReport(n_rows=len(df), mode="strict" if strict else "lenient")
    soft = "error" if strict else "warning"  # level for data-quality (not structural) issues
    columns = schema["columns"]

    # 1. Required columns present. A missing required column is always an error.
    for name, spec in columns.items():
        needed = spec.get("required", False) or (
            require_target and spec.get("required_for_training", False)
        )
        if needed:
            present = name in df.columns
            report.add(
                f"column_present[{name}]",
                present,
                "error",
                "present" if present else "required column is missing",
            )

    # 2. Unexpected columns are a warning (extra data is tolerated but noted).
    unexpected = [c for c in df.columns if c not in columns]
    report.add(
        "no_unexpected_columns",
        not unexpected,
        "warning",
        "none" if not unexpected else f"unexpected columns: {unexpected}",
        count=len(unexpected),
    )

    # Per-column data-quality checks (only for columns that are present).
    for name, spec in columns.items():
        if name not in df.columns:
            continue
        col = df[name]
        dtype = spec.get("dtype")

        # 3. Data type is parseable.
        if dtype in ("int", "float"):
            coerced = pd.to_numeric(col, errors="coerce")
            unparseable = int((coerced.isna() & col.notna()).sum())
            report.add(
                f"dtype[{name}]",
                unparseable == 0,
                "error",
                "ok" if unparseable == 0 else f"{unparseable} value(s) not numeric",
                count=unparseable,
            )
            if dtype == "int":
                non_int = int((coerced.dropna() % 1 != 0).sum())
                if non_int:
                    report.add(
                        f"integer[{name}]",
                        False,
                        "warning",
                        f"{non_int} non-integer value(s) in an integer column",
                        count=non_int,
                    )
            numeric = coerced
        else:
            numeric = None

        # 4. Missing values.
        n_missing = int(_is_missing(col).sum())
        if name == schema_target(schema) and require_target:
            report.add(
                f"target_present[{name}]",
                n_missing == 0,
                "error",
                "ok" if n_missing == 0 else f"{n_missing} missing target value(s)",
                count=n_missing,
            )
        elif spec.get("required", False):
            report.add(
                f"no_missing[{name}]",
                n_missing == 0,
                soft,
                "ok" if n_missing == 0 else f"{n_missing} missing value(s)",
                count=n_missing,
            )

        # 5. Numeric ranges.
        if numeric is not None and ("min" in spec or "max" in spec):
            lo = spec.get("min", float("-inf"))
            hi = spec.get("max", float("inf"))
            out = int(((numeric < lo) | (numeric > hi)).sum())
            report.add(
                f"range[{name}]",
                out == 0,
                soft,
                "ok" if out == 0 else f"{out} value(s) outside [{lo}, {hi}]",
                count=out,
            )

        # 6. Categorical / allowed value set.
        if "allowed" in spec:
            allowed = set(spec["allowed"])
            present_vals = set(col.dropna().unique().tolist())
            bad = present_vals - allowed
            report.add(
                f"allowed[{name}]",
                not bad,
                soft,
                "ok" if not bad else f"unexpected values: {sorted(map(str, bad))}",
                count=len(bad),
            )

    # 7. Duplicate rows and duplicate ids.
    rules = schema.get("row_rules", {})
    if rules.get("no_duplicate_rows"):
        dups = int(df.duplicated().sum())
        report.add("no_duplicate_rows", dups == 0, soft, "ok" if dups == 0 else f"{dups} duplicate row(s)", count=dups)
    id_col = _id_column(schema)
    if rules.get("no_duplicate_ids") and id_col in df.columns:
        dup_ids = int(df[id_col].duplicated().sum())
        report.add("no_duplicate_ids", dup_ids == 0, "error", "ok" if dup_ids == 0 else f"{dup_ids} duplicate id(s)", count=dup_ids)

    return report


def schema_target(schema: dict) -> str | None:
    for name, spec in schema["columns"].items():
        if spec.get("role") == "target":
            return name
    return None


def _id_column(schema: dict) -> str | None:
    for name, spec in schema["columns"].items():
        if spec.get("role") == "id":
            return name
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validate the raw dataset against the schema.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--lenient", action="store_true", help="Treat data-quality issues as warnings.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    schema = load_schema(cfg)
    df = load_raw(cfg)
    report = validate(df, schema, require_target=True, strict=not args.lenient)

    out_path = resolve(cfg, cfg["paths"]["validation_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)

    print(f"rows={report.n_rows} mode={report.mode} ok={report.ok} "
          f"errors={len(report.errors)} warnings={len(report.warnings)}")
    for c in report.errors + report.warnings:
        print(f"  [{c['level']}] {c['check']}: {c['detail']}")
    print(f"report written to {out_path}")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
