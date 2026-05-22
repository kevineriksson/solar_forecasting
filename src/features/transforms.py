"""Pure feature transforms: k_t, lags, rolling means, calendar, night mask.

All transforms operate on a single UTC-indexed DataFrame and return new columns.
Lags and rolling means strictly look BACKWARD only (no leakage of t into features at t).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

CS_EPSILON = 1.0e-3  # W/m² — anything below this is treated as night/no-sun


def compute_kt(ghi: pd.Series, cs_ghi: pd.Series, clip_min: float, clip_max: float) -> pd.Series:
    """k_t = ghi / cs_ghi, with night handling and clipping.

    Where cs_ghi <= CS_EPSILON we set k_t = 0.0 (night; k_t is undefined and we
    refuse to propagate NaN into a feature column).
    """
    daylight = cs_ghi > CS_EPSILON
    kt = pd.Series(np.zeros(len(ghi), dtype="float64"), index=ghi.index, name="k_t")
    kt.loc[daylight] = (ghi.loc[daylight] / cs_ghi.loc[daylight]).astype("float64")
    return kt.clip(lower=clip_min, upper=clip_max)


def night_mask(zenith: pd.Series, threshold_deg: float) -> pd.Series:
    """1 where zenith >= threshold (sun below/at horizon), else 0."""
    return (zenith >= threshold_deg).astype("int8").rename("is_night")


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """sin/cos encodings of hour-of-day and day-of-year. Inputs are UTC."""
    if index.tz is None:
        raise ValueError("calendar features require a timezone-aware index")
    hour = index.hour + index.minute / 60.0
    doy = index.dayofyear.to_numpy(dtype="float64")
    two_pi = 2.0 * np.pi
    return pd.DataFrame(
        {
            "hour_sin": np.sin(two_pi * hour / 24.0),
            "hour_cos": np.cos(two_pi * hour / 24.0),
            "doy_sin": np.sin(two_pi * doy / 365.25),
            "doy_cos": np.cos(two_pi * doy / 365.25),
        },
        index=index,
    ).astype("float64")


def add_lags(df: pd.DataFrame, columns: Iterable[str], lags_steps: Sequence[int]) -> pd.DataFrame:
    """Append `{col}_lag{k}` columns. `lag k` at time t is the value at t-k steps."""
    out = {}
    for col in columns:
        s = df[col]
        for k in lags_steps:
            if k <= 0:
                raise ValueError(f"lag must be positive (look-back); got {k}")
            out[f"{col}_lag{k}"] = s.shift(k).astype("float64")
    return pd.DataFrame(out, index=df.index)


def add_rolling_means(
    df: pd.DataFrame, columns: Iterable[str], windows_steps: Sequence[int]
) -> pd.DataFrame:
    """Append `{col}_roll{w}` columns.

    The window at time t covers strictly past values [t-w, t-1] (exclusive of t),
    enforced by `.shift(1)` before `.rolling(w)`. `min_periods=w` so early rows
    where the window isn't yet full are NaN until the warmup elapses.
    """
    out = {}
    for col in columns:
        s = df[col]
        for w in windows_steps:
            if w <= 0:
                raise ValueError(f"rolling window must be positive; got {w}")
            out[f"{col}_roll{w}"] = (
                s.shift(1).rolling(window=w, min_periods=w).mean().astype("float64")
            )
    return pd.DataFrame(out, index=df.index)
