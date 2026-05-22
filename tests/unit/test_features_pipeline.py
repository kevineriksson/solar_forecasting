"""Integration-ish test for build_features + split_features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.pipeline import FeatureConfig, build_features, split_features
from src.ingest.schema import TIMESTAMP_COL
from src.ingest.splits import compute_splits

PARAMS = {
    "site": {
        "name": "montana_wolfpoint",
        "latitude": 48.30783,
        "longitude": -105.1017,
        "altitude_m": 640,
        "timezone": "UTC",
    },
    "features": {
        "lags_steps": [1, 4, 12],
        "rolling_means_steps": [4, 12],
        "kt_clip_min": 0.0,
        "kt_clip_max": 1.5,
        "night_zenith_threshold_deg": 90.0,
        "lagged_variables": ["ghi", "dni", "dhi", "k_t", "air_temp", "wind_speed_100m"],
    },
}


def _toy_frame(n: int = 96 * 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-06-01T00:00:00Z", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            TIMESTAMP_COL: ts,
            "ghi": rng.uniform(0, 800, n),
            "dni": rng.uniform(0, 900, n),
            "dhi": rng.uniform(0, 400, n),
            "gti": rng.uniform(0, 850, n),
            "air_temp": rng.uniform(-30, 35, n),
            "wind_speed_100m": rng.uniform(0, 20, n),
            "zenith": rng.uniform(0, 180, n),
            "azimuth": rng.uniform(-180, 180, n),
        }
    )


def test_build_features_no_nans_and_warmup_dropped():
    df = _toy_frame()
    cfg = FeatureConfig.from_params(PARAMS)
    out = build_features(df, cfg)
    # Longest lag/window is 12 -> drop 12 leading rows.
    assert len(out) == len(df) - 12
    assert not out.isna().any().any()
    # k_t must be in the configured range.
    assert (out["k_t"] >= 0.0).all() and (out["k_t"] <= 1.5).all()
    # Calendar columns are present.
    for col in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
        assert col in out.columns
    # Night mask is int8 0/1.
    assert out["is_night"].dtype == np.dtype("int8")
    assert set(out["is_night"].unique()).issubset({0, 1})


def test_build_features_deterministic():
    df = _toy_frame()
    cfg = FeatureConfig.from_params(PARAMS)
    a = build_features(df, cfg)
    b = build_features(df, cfg)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_split_features_coverage():
    # 30 days of data, reference_now at the end -> all three splits non-empty.
    n = 96 * 365  # 1 year
    df = _toy_frame(n)
    cfg = FeatureConfig.from_params(PARAMS)
    full = build_features(df, cfg)

    splits_cfg = {
        "promo_months_back_start": 8,
        "promo_months_back_end": 6,
        "replay_months": 6,
    }
    # Reference_now AT data_last_ts is allowed (splits.py accepts <=).
    manifest = compute_splits(
        reference_now=df[TIMESTAMP_COL].iloc[-1],
        data_first_ts=df[TIMESTAMP_COL].iloc[0],
        data_last_ts=df[TIMESTAMP_COL].iloc[-1],
        splits_cfg=splits_cfg,
    )
    by_split = split_features(full, manifest)
    assert set(by_split.keys()) == {"train", "promo", "replay"}
    assert sum(len(v) for v in by_split.values()) == len(full)
    # Splits in time-order.
    assert by_split["train"][TIMESTAMP_COL].iloc[-1] < by_split["promo"][TIMESTAMP_COL].iloc[0]
    assert by_split["promo"][TIMESTAMP_COL].iloc[-1] < by_split["replay"][TIMESTAMP_COL].iloc[0]
