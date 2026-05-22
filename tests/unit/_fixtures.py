"""Synthetic fixtures for ingest unit tests.

Build small, well-formed DataFrames at 15-minute resolution that mirror the
canonical post-rename ingest schema. Tests mutate copies of these to exercise
each schema rule independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.schema import NUMERIC_COLS, PERIOD_COL, TIMESTAMP_COL


def make_canonical_frame(
    start: str = "2025-01-01T00:00:00Z",
    n_rows: int = 96 * 4,  # 4 days
    seed: int = 0,
) -> pd.DataFrame:
    """A clean, schema-valid frame with the canonical column names."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n_rows, freq="15min", tz="UTC")
    df = pd.DataFrame({TIMESTAMP_COL: ts, PERIOD_COL: "PT15M"})
    # Set numeric columns to plausible, schema-valid values.
    df["ghi"] = rng.uniform(0, 800, n_rows)
    df["dni"] = rng.uniform(0, 900, n_rows)
    df["dhi"] = rng.uniform(0, 400, n_rows)
    df["gti"] = rng.uniform(0, 850, n_rows)
    df["air_temp"] = rng.uniform(-30, 35, n_rows)
    df["wind_speed_100m"] = rng.uniform(0, 20, n_rows)
    df["zenith"] = rng.uniform(0, 180, n_rows)
    df["azimuth"] = rng.uniform(-180, 180, n_rows)
    cols = [TIMESTAMP_COL, PERIOD_COL, *NUMERIC_COLS]
    return df[cols].reset_index(drop=True)
