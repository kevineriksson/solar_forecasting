"""Schema validation for raw Solcast data.

Pure functions: take a DataFrame, raise SchemaError on the first violation.
No I/O, no globals. Validation happens AFTER renaming raw columns to canonical names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

TIMESTAMP_COL = "period_end"
PERIOD_COL = "period"
EXPECTED_PERIOD_VALUE = "PT15M"
RESOLUTION = pd.Timedelta(minutes=15)

NUMERIC_COLS = (
    "ghi",
    "dni",
    "dhi",
    "gti",
    "air_temp",
    "wind_speed_100m",
    "zenith",
    "azimuth",
)

NON_NEGATIVE_COLS = ("ghi", "dni", "dhi", "gti")


class SchemaError(ValueError):
    """Raised when raw data fails a schema validation rule."""


@dataclass(frozen=True)
class SchemaReport:
    n_rows: int
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp


def rename_to_canonical(df: pd.DataFrame, raw_columns: Mapping[str, str]) -> pd.DataFrame:
    """Rename columns from file-specific names to canonical names from params.yaml.

    raw_columns maps canonical_name -> file_column_name. We invert that to do the rename.
    """
    file_to_canonical = {file_name: canonical for canonical, file_name in raw_columns.items()}
    # Canonical 'timestamp' is just an alias for the canonical period_end column name.
    if "timestamp" in raw_columns:
        file_to_canonical[raw_columns["timestamp"]] = TIMESTAMP_COL

    missing = [src for src in file_to_canonical if src not in df.columns]
    if missing:
        raise SchemaError(f"raw file missing expected source columns: {missing}")

    return df.rename(columns=file_to_canonical)


def validate(df: pd.DataFrame) -> SchemaReport:
    """Run all schema checks. Returns a report on success; raises SchemaError on failure."""
    _check_required_columns(df)
    _check_period_constant(df)
    _check_dtypes(df)
    _check_ranges(df)
    _check_timestamp_monotonic_unique(df)
    _check_no_gaps(df)
    return SchemaReport(
        n_rows=len(df),
        first_ts=df[TIMESTAMP_COL].iloc[0],
        last_ts=df[TIMESTAMP_COL].iloc[-1],
    )


def _check_required_columns(df: pd.DataFrame) -> None:
    required = (TIMESTAMP_COL, PERIOD_COL, *NUMERIC_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns after rename: {missing}")


def _check_period_constant(df: pd.DataFrame) -> None:
    unique_periods = df[PERIOD_COL].unique()
    if len(unique_periods) != 1 or unique_periods[0] != EXPECTED_PERIOD_VALUE:
        raise SchemaError(
            f"{PERIOD_COL!r} must be constant {EXPECTED_PERIOD_VALUE!r}; "
            f"found unique values: {list(unique_periods)}"
        )


def _check_dtypes(df: pd.DataFrame) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL]):
        raise SchemaError(f"{TIMESTAMP_COL!r} must be a datetime dtype")
    tz = getattr(df[TIMESTAMP_COL].dtype, "tz", None)
    if tz is None:
        raise SchemaError(f"{TIMESTAMP_COL!r} must be timezone-aware (UTC)")
    if str(tz) != "UTC":
        raise SchemaError(f"{TIMESTAMP_COL!r} timezone must be UTC, got {tz}")

    for col in NUMERIC_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"{col!r} must be numeric, got {df[col].dtype}")
        if df[col].isna().any():
            raise SchemaError(f"{col!r} contains NaN values")


def _check_ranges(df: pd.DataFrame) -> None:
    for col in NON_NEGATIVE_COLS:
        if (df[col] < 0).any():
            bad = int((df[col] < 0).sum())
            raise SchemaError(f"{col!r} has {bad} negative value(s); must be >= 0")
    zenith = df["zenith"]
    if (zenith < 0).any() or (zenith > 180).any():
        raise SchemaError("'zenith' out of range; must be in [0, 180]")


def _check_timestamp_monotonic_unique(df: pd.DataFrame) -> None:
    ts = df[TIMESTAMP_COL]
    if ts.duplicated().any():
        n = int(ts.duplicated().sum())
        raise SchemaError(f"{TIMESTAMP_COL!r} has {n} duplicate value(s)")
    if not ts.is_monotonic_increasing:
        raise SchemaError(f"{TIMESTAMP_COL!r} is not strictly monotonic increasing")


def _check_no_gaps(df: pd.DataFrame) -> None:
    diffs = df[TIMESTAMP_COL].diff().dropna()
    bad = (diffs != RESOLUTION).to_numpy()
    if bad.any():
        first_bad_pos = int(bad.argmax()) + 1  # +1 because diff drops first row
        raise SchemaError(
            f"timestamp gaps detected: {int(bad.sum())} non-15-minute step(s); "
            f"first at index {first_bad_pos} (ts={df[TIMESTAMP_COL].iloc[first_bad_pos]})"
        )
