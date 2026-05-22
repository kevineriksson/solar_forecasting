"""Unit tests for src/ingest/schema.py (synthetic fixtures only)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.schema import (
    PERIOD_COL,
    TIMESTAMP_COL,
    SchemaError,
    rename_to_canonical,
    validate,
)
from tests.unit._fixtures import make_canonical_frame


def test_happy_path_passes():
    df = make_canonical_frame()
    report = validate(df)
    assert report.n_rows == len(df)
    assert report.first_ts == df[TIMESTAMP_COL].iloc[0]
    assert report.last_ts == df[TIMESTAMP_COL].iloc[-1]


def test_missing_required_column_raises():
    df = make_canonical_frame().drop(columns=["dni"])
    with pytest.raises(SchemaError, match="missing required columns"):
        validate(df)


@pytest.mark.parametrize("col", ["ghi", "dni", "dhi", "gti"])
def test_negative_irradiance_raises(col):
    df = make_canonical_frame()
    df.loc[0, col] = -1.0
    with pytest.raises(SchemaError, match=f"{col!r} has"):
        validate(df)


def test_zenith_out_of_range_raises():
    df = make_canonical_frame()
    df.loc[0, "zenith"] = 181.0
    with pytest.raises(SchemaError, match="zenith"):
        validate(df)


def test_duplicate_timestamp_raises():
    df = make_canonical_frame()
    df.loc[1, TIMESTAMP_COL] = df.loc[0, TIMESTAMP_COL]
    with pytest.raises(SchemaError, match="duplicate"):
        validate(df)


def test_non_monotonic_raises():
    df = make_canonical_frame()
    # Swap two non-adjacent rows so duplicates are not introduced but order breaks.
    df.loc[5, TIMESTAMP_COL], df.loc[10, TIMESTAMP_COL] = (
        df.loc[10, TIMESTAMP_COL],
        df.loc[5, TIMESTAMP_COL],
    )
    with pytest.raises(SchemaError, match="monotonic"):
        validate(df)


def test_gap_raises():
    df = make_canonical_frame()
    # Drop one row to create a 30-minute gap.
    df = df.drop(index=3).reset_index(drop=True)
    with pytest.raises(SchemaError, match="gaps"):
        validate(df)


def test_period_column_must_be_constant_pt15m():
    df = make_canonical_frame()
    df.loc[0, PERIOD_COL] = "PT30M"
    with pytest.raises(SchemaError, match="constant"):
        validate(df)


def test_nan_in_numeric_column_raises():
    df = make_canonical_frame()
    df.loc[0, "ghi"] = pd.NA
    df["ghi"] = df["ghi"].astype("float64")
    with pytest.raises(SchemaError, match="NaN"):
        validate(df)


def test_rename_inverts_raw_columns():
    raw = pd.DataFrame(
        {
            "period_end": pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC"),
            "period": ["PT15M", "PT15M"],
            "ghi": [0, 1],
            "dni": [0, 1],
            "dhi": [0, 1],
            "gti": [0, 1],
            "air_temp": [0, 1],
            "wind_speed_100m": [0, 1],
            "zenith": [90, 80],
            "azimuth": [0, 0],
        }
    )
    raw_columns = {
        "timestamp": "period_end",
        "ghi": "ghi",
        "dni": "dni",
        "dhi": "dhi",
        "gti": "gti",
        "air_temp": "air_temp",
        "wind_speed_100m": "wind_speed_100m",
        "zenith": "zenith",
        "azimuth": "azimuth",
    }
    renamed = rename_to_canonical(raw, raw_columns)
    assert TIMESTAMP_COL in renamed.columns
    assert PERIOD_COL in renamed.columns
    assert "wind_speed_100m" in renamed.columns


def test_rename_missing_source_column_raises():
    raw = pd.DataFrame({"period_end": [pd.Timestamp("2025-01-01", tz="UTC")]})
    raw_columns = {"timestamp": "period_end", "ghi": "ghi"}
    with pytest.raises(SchemaError, match="missing expected source columns"):
        rename_to_canonical(raw, raw_columns)
