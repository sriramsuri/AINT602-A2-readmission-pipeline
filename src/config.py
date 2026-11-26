"""Load and expose the project configuration.

Keeping configuration in one YAML file (not scattered through the code) is what
lets another person answer "which settings produced this model?" by reading a
single file. Every stage imports load_config() so they all agree on the seed,
the split and the feature lists.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Project root is the parent of the src/ package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read config.yaml and return it as a dict.

    Parameters
    ----------
    path:
        Optional explicit path. Defaults to config.yaml at the project root.
    """
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve(cfg: dict, rel: str) -> Path:
    """Resolve a repository-relative path against the project root."""
    return (PROJECT_ROOT / rel).resolve()


def all_feature_columns(cfg: dict) -> list[str]:
    """Return the raw input columns the model is allowed to use (pre-engineering)."""
    f = cfg["features"]
    return list(f["numeric"]) + list(f["binary"]) + list(f["categorical"])
