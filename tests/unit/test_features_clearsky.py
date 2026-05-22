"""Unit tests for src/features/clearsky.py — pvlib wrapper for the fixed site."""

from __future__ import annotations

import pandas as pd

from src.features.clearsky import CLEARSKY_COLUMNS, compute_clearsky

SITE = {
    "name": "montana_wolfpoint",
    "latitude": 48.30783,
    "longitude": -105.1017,
    "altitude_m": 640,
    "timezone": "UTC",
}


def test_clearsky_zero_at_solar_midnight():
    # ~07:30 UTC ≈ 00:30 local at lon -105 (the deep night). Sun should be down.
    idx = pd.date_range("2025-06-21T07:00:00Z", periods=4, freq="15min", tz="UTC")
    cs = compute_clearsky(idx, SITE)
    assert list(cs.columns) == list(CLEARSKY_COLUMNS)
    assert (cs["cs_ghi"] < 1.0).all()
    assert (cs["cs_dni"] < 1.0).all()
    assert (cs["cs_dhi"] < 1.0).all()


def test_clearsky_positive_at_solar_noon():
    # ~19:00 UTC ≈ 12:00 local at lon -105 on summer solstice; sun is high.
    idx = pd.date_range("2025-06-21T19:00:00Z", periods=4, freq="15min", tz="UTC")
    cs = compute_clearsky(idx, SITE)
    assert (cs["cs_ghi"] > 500.0).all()
    assert (cs["cs_dni"] > 0).all()


def test_clearsky_naive_index_rejected():
    idx = pd.date_range("2025-06-21T19:00:00", periods=4, freq="15min")
    try:
        compute_clearsky(idx, SITE)
    except ValueError:
        return
    raise AssertionError("compute_clearsky should reject a naive index")


def test_clearsky_deterministic():
    idx = pd.date_range("2025-06-21T12:00:00Z", periods=96, freq="15min", tz="UTC")
    a = compute_clearsky(idx, SITE)
    b = compute_clearsky(idx, SITE)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
