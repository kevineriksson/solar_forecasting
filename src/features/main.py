"""Stage 2 entrypoint: build features from interim parquet, write per-split tables.

Usage:
    python -m src.features.main --params params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.ingest.splits import compute_splits

from .io import read_interim, write_split_parquet
from .pipeline import FeatureConfig, build_features, split_features

LOG = logging.getLogger("features")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(args.params).resolve().parent
    params = yaml.safe_load(Path(args.params).read_text())

    interim_root = repo_root / params["paths"]["interim"]
    parquet_root = interim_root / "parquet"
    features_root = repo_root / params["paths"]["features"]
    splits_manifest_path = repo_root / params["paths"]["splits_manifest"]

    LOG.info("reading interim parquet: %s", parquet_root)
    df = read_interim(parquet_root)
    LOG.info("interim rows=%d cols=%d", len(df), df.shape[1])

    # Rebuild the SplitManifest from params + observed data range so we don't
    # rely on parsing the JSON manifest. The two are derived from the same
    # inputs (Stage 1 reads the same params), so they will agree.
    manifest = compute_splits(
        reference_now=pd.Timestamp(params["splits"]["reference_now"]),
        data_first_ts=df["period_end"].iloc[0],
        data_last_ts=df["period_end"].iloc[-1],
        splits_cfg=params["splits"],
    )

    cfg = FeatureConfig.from_params(params)
    LOG.info(
        "building features: lags=%s rolling=%s lagged_vars=%s",
        cfg.lags_steps,
        cfg.rolling_means_steps,
        cfg.lagged_variables,
    )
    full = build_features(df, cfg)
    LOG.info("feature rows=%d cols=%d", len(full), full.shape[1])

    by_split = split_features(full, manifest)
    counts = {name: len(sub) for name, sub in by_split.items()}
    LOG.info("split row counts: %s", counts)

    features_root.mkdir(parents=True, exist_ok=True)
    for name, sub in by_split.items():
        out_path = features_root / f"{name}.parquet"
        LOG.info("writing %s rows -> %s", len(sub), out_path)
        write_split_parquet(sub, out_path)

    # Sanity manifest for inspection (NOT a DVC output — splits.json from Stage 1
    # remains the source of truth for split boundaries).
    sanity_path = features_root / "features_manifest.json"
    sanity_path.write_text(
        json.dumps(
            {
                "n_columns": int(full.shape[1]),
                "columns": list(full.columns),
                "warmup_dropped_rows": int(len(df) - len(full)),
                "split_row_counts": counts,
                "reference_now": _iso(manifest.reference_now),
                "splits_manifest": str(splits_manifest_path.relative_to(repo_root)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    LOG.info("features complete")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 features")
    parser.add_argument("--params", default="params.yaml", help="path to params.yaml")
    return parser.parse_args(argv)


def _iso(ts: pd.Timestamp) -> str:
    return ts.isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
