"""Stage 1 - ingestion.

Load the raw clinical dataset from the source named in config.yaml. The loader
is deliberately small and repeatable: one function, one source, a clear error
if the file is missing. A URL or database source could be swapped in here
without touching any downstream stage.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import load_config, resolve

log = logging.getLogger("ingest")


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load the raw dataset defined in config.yaml and return a DataFrame."""
    raw_path = resolve(cfg, cfg["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "See the README for how to place the dataset in data/raw/."
        )
    df = pd.read_csv(raw_path)
    log.info("Loaded %d rows and %d columns from %s", df.shape[0], df.shape[1], raw_path.name)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Load and summarise the raw dataset.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_raw(cfg)
    print(f"rows={df.shape[0]} cols={df.shape[1]}")
    print("columns:", list(df.columns))
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
