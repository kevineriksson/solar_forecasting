"""Unit tests for src/ingest/io.py — round-trip and partitioning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from src.ingest.io import write_partitioned_parquet
from src.ingest.schema import PERIOD_COL, TIMESTAMP_COL
from tests.unit._fixtures import make_canonical_frame

CANONICAL_NUMERIC_COLS = [
    "ghi",
    "dni",
    "dhi",
    "gti",
    "air_temp",
    "wind_speed_100m",
    "zenith",
    "azimuth",
]


def _read_back(root: Path) -> pd.DataFrame:
    table = ds.dataset(str(root), format="parquet", partitioning="hive").to_table()
    return table.to_pandas()


def test_roundtrip_preserves_values(tmp_path: Path):
    df = make_canonical_frame(n_rows=96 * 70).drop(columns=[PERIOD_COL])  # ~70 days, spans months
    root = tmp_path / "parquet"
    write_partitioned_parquet(df, root)

    out = _read_back(root)
    # The partition columns (year, month) come back as extras; drop them before compare.
    out = out.drop(columns=["year", "month"])
    # Sort both by timestamp to compare regardless of partition ordering.
    df_sorted = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    out_sorted = out.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    for col in CANONICAL_NUMERIC_COLS:
        pd.testing.assert_series_equal(out_sorted[col], df_sorted[col], check_dtype=False)
    pd.testing.assert_series_equal(
        out_sorted[TIMESTAMP_COL].dt.tz_convert("UTC"),
        df_sorted[TIMESTAMP_COL],
        check_names=False,
    )


def test_partitioning_creates_year_month_dirs(tmp_path: Path):
    # Frame spanning Jan -> Feb 2025.
    df = make_canonical_frame(start="2025-01-30T00:00:00Z", n_rows=96 * 5).drop(
        columns=[PERIOD_COL]
    )
    root = tmp_path / "parquet"
    write_partitioned_parquet(df, root)

    dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert dirs == ["year=2025"]
    months = sorted(p.name for p in (root / "year=2025").iterdir() if p.is_dir())
    assert months == ["month=1", "month=2"]


def test_second_write_replaces_first(tmp_path: Path):
    root = tmp_path / "parquet"
    df_a = make_canonical_frame(start="2025-01-01T00:00:00Z", n_rows=96).drop(columns=[PERIOD_COL])
    write_partitioned_parquet(df_a, root)
    df_b = make_canonical_frame(start="2025-06-01T00:00:00Z", n_rows=96, seed=1).drop(
        columns=[PERIOD_COL]
    )
    write_partitioned_parquet(df_b, root)
    # Only the second write's partition (year=2025/month=6) should exist.
    months = sorted(p.name for p in (root / "year=2025").iterdir() if p.is_dir())
    assert months == ["month=6"]
