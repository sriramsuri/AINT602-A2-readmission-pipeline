"""Create a small example of new, unlabelled records for the prediction demo.

Takes a handful of rows from the dataset and removes the target and the excluded
mortality column, which mimics the information that would be available at prediction
time (everything known at discharge, but not the outcome itself).
"""
from __future__ import annotations

import argparse
import logging

from .config import load_config, resolve
from .ingest import load_raw

log = logging.getLogger("make_sample")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Write a small sample of new patients (no labels).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)

    df = load_raw(cfg)
    drop = [cfg["task"]["target"], "mortality"]
    sample = df.drop(columns=[c for c in drop if c in df.columns]).head(args.n)

    out = resolve(cfg, cfg["paths"]["sample_input"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"wrote {len(sample)} sample record(s) -> {out}")


if __name__ == "__main__":
    main()
