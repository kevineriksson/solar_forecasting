"""I/O for Stage 2: read partitioned interim Parquet, write per-split feature tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingest.schema import NUMERIC_COLS, TIMESTAMP_COL


def read_interim(parquet_root: Path) -> pd.DataFrame:
    """Load all partitioned interim Parquet files into one UTC-sorted DataFrame."""
    if not parquet_root.exists():
        raise FileNotFoundError(f"interim parquet root not found: {parquet_root}")

    dataset = pq.ParquetDataset(str(parquet_root))
    table = dataset.read()
    df = table.to_pandas()

    # Partition columns (year, month) are added by pyarrow as int categories; drop.
    for partition_col in ("year", "month"):
        if partition_col in df.columns:
            df = df.drop(columns=[partition_col])

    # Ensure UTC tz-aware and sorted.
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Re-assert dtypes for numeric columns (Parquet round-trip can introduce float32).
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    return df


def write_split_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write `df` as a deterministic single-file Parquet at `path`.

    Caller is responsible for stable column order and row order. We replace any
    existing file so DVC sees a clean output.
    """
    if df.empty:
        raise ValueError(f"refusing to write empty DataFrame to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        str(path),
        compression="snappy",
        # Fixed row group size for determinism across runs.
        row_group_size=64 * 1024,
    )
