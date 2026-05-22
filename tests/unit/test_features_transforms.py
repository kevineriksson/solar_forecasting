"""Unit tests for src/features/transforms.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.transforms import (
    add_lags,
    add_rolling_means,
    calendar_features,
    compute_kt,
    night_mask,
)


def _index(n: int = 16) -> pd.DatetimeIndex:
    return pd.date_range("2025-06-21T00:00:00Z", periods=n, freq="15min", tz="UTC")


def test_kt_zero_when_clearsky_is_zero():
    idx = _index(8)
    ghi = pd.Series([0.0] * 4 + [500.0] * 4, index=idx)
    cs_ghi = pd.Series([0.0] * 4 + [1000.0] * 4, index=idx)
    kt = compute_kt(ghi, cs_ghi, clip_min=0.0, clip_max=1.5)
    assert (kt.iloc[:4] == 0.0).all()
    assert (kt.iloc[4:] == 0.5).all()


def test_kt_clips_above_one_point_five():
    idx = _index(4)
    ghi = pd.Series([2000.0, 1000.0, 100.0, 0.0], index=idx)
    cs_ghi = pd.Series([1000.0, 1000.0, 1000.0, 1000.0], index=idx)
    kt = compute_kt(ghi, cs_ghi, clip_min=0.0, clip_max=1.5)
    # 2.0 -> clipped to 1.5; 1.0 stays; 0.1 stays; 0/positive cs is 0.
    assert kt.iloc[0] == pytest.approx(1.5)
    assert kt.iloc[1] == pytest.approx(1.0)
    assert kt.iloc[2] == pytest.approx(0.1)
    assert kt.iloc[3] == pytest.approx(0.0)
    assert (kt >= 0.0).all() and (kt <= 1.5).all()


def test_night_mask_threshold():
    idx = _index(5)
    zenith = pd.Series([10.0, 85.0, 90.0, 95.0, 180.0], index=idx)
    nm = night_mask(zenith, threshold_deg=90.0)
    assert nm.tolist() == [0, 0, 1, 1, 1]
    assert nm.dtype == np.dtype("int8")


def test_calendar_features_unit_circle():
    idx = _index(96 * 2)  # 2 days
    cal = calendar_features(idx)
    assert list(cal.columns) == ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    # sin^2 + cos^2 == 1 to machine precision.
    np.testing.assert_allclose(cal["hour_sin"] ** 2 + cal["hour_cos"] ** 2, 1.0, atol=1e-12)
    np.testing.assert_allclose(cal["doy_sin"] ** 2 + cal["doy_cos"] ** 2, 1.0, atol=1e-12)


def test_calendar_features_require_tz():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")  # naive
    with pytest.raises(ValueError):
        calendar_features(idx)


def test_lags_look_backward_only():
    idx = _index(8)
    df = pd.DataFrame({"x": np.arange(8, dtype="float64")}, index=idx)
    lags = add_lags(df, ["x"], [1, 4])
    assert list(lags.columns) == ["x_lag1", "x_lag4"]
    # lag1 at position i should equal x[i-1]; first row is NaN.
    assert np.isnan(lags["x_lag1"].iloc[0])
    assert lags["x_lag1"].iloc[1] == 0.0
    assert lags["x_lag1"].iloc[7] == 6.0
    # lag4 at position 4 should equal x[0].
    assert lags["x_lag4"].iloc[4] == 0.0
    assert np.isnan(lags["x_lag4"].iloc[3])


def test_rolling_means_exclude_current_step():
    idx = _index(8)
    df = pd.DataFrame({"x": np.arange(8, dtype="float64")}, index=idx)
    rolls = add_rolling_means(df, ["x"], [4])
    # At index 4, window covers x[0..3] = mean(0,1,2,3) = 1.5.
    assert rolls["x_roll4"].iloc[4] == pytest.approx(1.5)
    # First 4 rows: not enough history -> NaN.
    assert rolls["x_roll4"].iloc[:4].isna().all()
    # Verify the window does NOT include the current step (else mean at i=4 would be 2.0).


def test_lags_reject_nonpositive():
    idx = _index(4)
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    with pytest.raises(ValueError):
        add_lags(df, ["x"], [0])
    with pytest.raises(ValueError):
        add_lags(df, ["x"], [-1])
