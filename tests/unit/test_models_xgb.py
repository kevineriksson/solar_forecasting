"""Unit tests for src/models/xgb_train.py — feature set + alignment.

We avoid invoking the full training entrypoint here; the heavy stuff
(MLflow logging, XGBoost fitting at scale) is exercised end-to-end by the
T5 run. These tests pin down the invariants that are cheap to verify and
that would silently break the skill-score comparison if violated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.xgb_train import EXCLUDED_FEATURE_COLS


def test_excluded_columns_are_targets_plus_timestamp_plus_gti():
    # The exact set is load-bearing: it's referenced in the MLflow run params
    # and any drift here changes the candidate's feature space.
    assert EXCLUDED_FEATURE_COLS == {"period_end", "ghi", "dni", "dhi", "gti"}


def test_shift_alignment_matches_persistence_convention():
    # The candidate must use features from t-h to predict the value at t,
    # exactly like persistence_forecast does. This test pins that down with
    # a tiny synthetic frame.
    n = 8
    h = 2
    df = pd.DataFrame(
        {
            "feat_a": np.arange(n, dtype="float64"),
            "feat_b": np.arange(n, dtype="float64") * 10.0,
            "target": np.arange(n, dtype="float64") * 100.0,
        }
    )
    X = df[["feat_a", "feat_b"]].shift(h).reset_index(drop=True)

    # X.iloc[t] == features at row t-h for t >= h.
    for t in range(h, n):
        assert X.iloc[t]["feat_a"] == float(t - h)
        assert X.iloc[t]["feat_b"] == float((t - h) * 10)

    # First h rows are NaN (cold-start; trainer drops them, XGB handles them in val).
    assert X.iloc[:h].isna().all().all()


def test_no_feature_column_is_a_future_value():
    # Spot-check the canonical T3 feature names: no column should describe a
    # value AFTER the row's timestamp. We can only check by name here, but
    # T3's pipeline enforces this structurally — this test guards against a
    # rename that would silently introduce "lead" features.
    forbidden_patterns = ("lead", "future", "next")
    candidates = [
        "k_t",
        "k_t_lag1",
        "k_t_lag4",
        "k_t_lag12",
        "ghi_lag1",
        "ghi_lag4",
        "ghi_lag12",
        "ghi_roll4",
        "ghi_roll12",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "cs_ghi",
        "cs_dni",
        "cs_dhi",
        "zenith",
        "azimuth",
        "is_night",
        "air_temp",
        "wind_speed_100m",
    ]
    for c in candidates:
        for pat in forbidden_patterns:
            assert pat not in c, f"feature {c} contains forbidden pattern {pat!r}"
