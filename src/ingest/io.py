"""I/O for Stage 1: read raw Solcast CSV, write partitioned Parquet."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import NUMERIC_COLS, PERIOD_COL, TIMESTAMP_COL, rename_to_canonical


def read_raw_csv(path: Path, raw_columns: Mapping[str, str]) -> pd.DataFrame:
    """Read the raw Solcast CSV and return a canonical, typed DataFrame.

    Output columns: TIMESTAMP_COL (tz-aware UTC), PERIOD_COL, plus all NUMERIC_COLS
    as float64. Extra columns from the source are dropped.
    """
    df = pd.read_csv(path)
    df = rename_to_canonical(df, raw_columns)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    for col in NUMERIC_COLS:
        df[col] = df[col].astype("float64")

    keep = [TIMESTAMP_COL, PERIOD_COL, *NUMERIC_COLS]
    return df[keep].reset_index(drop=True)


def write_partitioned_parquet(df: pd.DataFrame, root: Path) -> None:
    """Write df to <root>/year=YYYY/month=MM/part.parquet, partitioned by year/month.

    Idempotent: the root directory is fully replaced on each call so DVC sees a
    deterministic output tree (no leftover files from previous runs).
    """
    if df.empty:
        raise ValueError("refusing to write empty DataFrame to partitioned Parquet")

    out_df = df.copy()
    out_df["year"] = out_df[TIMESTAMP_COL].dt.year.astype("int32")
    out_df["month"] = out_df[TIMESTAMP_COL].dt.month.astype("int32")

    if root.exists():
        _rm_tree(root)
    root.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(out_df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=["year", "month"],
        existing_data_behavior="error",
        # Force a deterministic single-file-per-partition layout.
        basename_template="part-{i}.parquet",
    )


def _rm_tree(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    for child in path.iterdir():
        _rm_tree(child)
    path.rmdir()
