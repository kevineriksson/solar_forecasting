"""Stage 1 entrypoint: ingest raw Solcast CSV, validate, partition, emit splits.

Usage:
    python -m src.ingest.main --params params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from .io import read_raw_csv, write_partitioned_parquet
from .schema import PERIOD_COL, TIMESTAMP_COL, validate
from .splits import assign_splits, compute_splits

LOG = logging.getLogger("ingest")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(args.params).resolve().parent
    params = yaml.safe_load(Path(args.params).read_text())

    raw_path = repo_root / params["data"]["raw_file_path"]
    interim_root = repo_root / params["paths"]["interim"]
    parquet_root = interim_root / "parquet"
    splits_manifest_path = repo_root / params["paths"]["splits_manifest"]

    LOG.info("reading raw file: %s", raw_path)
    df = read_raw_csv(raw_path, params["data"]["raw_columns"])
    LOG.info("read %d rows, %d columns", len(df), df.shape[1])

    LOG.info("validating schema")
    report = validate(df)
    LOG.info(
        "schema OK: rows=%d first=%s last=%s",
        report.n_rows,
        report.first_ts,
        report.last_ts,
    )

    LOG.info("computing splits from reference_now=%s", params["splits"]["reference_now"])
    manifest = compute_splits(
        reference_now=pd.Timestamp(params["splits"]["reference_now"]),
        data_first_ts=report.first_ts,
        data_last_ts=report.last_ts,
        splits_cfg=params["splits"],
    )

    # Assign + verify disjoint coverage before any write.
    split_series = assign_splits(df, manifest, TIMESTAMP_COL)
    counts = split_series.value_counts().to_dict()
    assert sum(counts.values()) == len(df), "split coverage check failed"
    LOG.info("split row counts: %s", counts)

    # Drop the now-validated period column before persisting.
    df_out = df.drop(columns=[PERIOD_COL])

    LOG.info("writing partitioned parquet -> %s", parquet_root)
    write_partitioned_parquet(df_out, parquet_root)

    LOG.info("writing splits manifest -> %s", splits_manifest_path)
    splits_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    splits_manifest_path.write_text(
        json.dumps(
            {
                "reference_now": _iso(manifest.reference_now),
                "data_first_ts": _iso(manifest.data_first_ts),
                "data_last_ts": _iso(manifest.data_last_ts),
                "splits": {
                    s.name: s.to_dict(int(counts.get(s.name, 0))) for s in manifest.as_tuple()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    LOG.info("ingest complete")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 ingest")
    parser.add_argument("--params", default="params.yaml", help="path to params.yaml")
    return parser.parse_args(argv)


def _iso(ts: pd.Timestamp) -> str:
    return ts.isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
