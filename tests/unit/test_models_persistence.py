"""Unit tests for src/models/persistence.py — smart persistence forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.persistence import (
    PersistenceConfig,
    persistence_forecast,
    score_predictions,
)


def _cfg() -> PersistenceConfig:
    return PersistenceConfig(kt_clip_min=0.0, kt_clip_max=1.5)


def _frame(ghi: list[float], cs_ghi: list[float]) -> pd.DataFrame:
    n = len(ghi)
    assert len(cs_ghi) == n
    return pd.DataFrame(
        {
            "ghi": ghi,
            "dni": [0.0] * n,
            "dhi": [0.0] * n,
            "cs_ghi": cs_ghi,
            "cs_dni": [0.0] * n,
            "cs_dhi": [0.0] * n,
        }
    )


def test_clear_sky_is_exact_for_persistence():
    # Under perfect clear sky (ghi == cs_ghi), k_t == 1, so prediction == cs_ghi(t+h)
    # which equals ghi(t+h). Persistence should be a *perfect* forecaster here.
    df = _frame(
        ghi=[100, 200, 300, 400, 500, 600],
        cs_ghi=[100, 200, 300, 400, 500, 600],
    )
    pred = persistence_forecast(df, "ghi", horizon_steps=1, cfg=_cfg())
    # Row 0 cold-start NaN; rows 1..5 should match ghi exactly.
    assert np.isnan(pred.iloc[0])
    np.testing.assert_allclose(pred.iloc[1:].to_numpy(), df["ghi"].iloc[1:].to_numpy())


def test_night_at_valid_time_forces_zero():
    df = _frame(
        ghi=[500, 500, 0, 0],
        cs_ghi=[1000, 1000, 0.0, 0.0],  # nights at rows 2, 3
    )
    pred = persistence_forecast(df, "ghi", horizon_steps=1, cfg=_cfg())
    # Row 1 (cs_ghi > 0): kt(0)=0.5 → 0.5*1000=500
    # Rows 2, 3 (cs_ghi == 0 at valid time): forced to 0.
    assert pred.iloc[1] == 500.0
    assert pred.iloc[2] == 0.0
    assert pred.iloc[3] == 0.0


def test_cold_start_rows_are_nan():
    df = _frame(ghi=[100, 200, 300, 400], cs_ghi=[100, 200, 300, 400])
    pred = persistence_forecast(df, "ghi", horizon_steps=2, cfg=_cfg())
    assert pred.iloc[:2].isna().all()
    assert pred.iloc[2:].notna().all()


def test_horizon_must_be_positive():
    df = _frame(ghi=[100, 200], cs_ghi=[100, 200])
    with pytest.raises(ValueError, match="horizon_steps must be > 0"):
        persistence_forecast(df, "ghi", horizon_steps=0, cfg=_cfg())


def test_unknown_target_raises():
    df = _frame(ghi=[100, 200], cs_ghi=[100, 200])
    with pytest.raises(ValueError, match="unknown target"):
        persistence_forecast(df, "gti", horizon_steps=1, cfg=_cfg())


def test_score_predictions_basic():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pd.Series([1.0, 2.0, 3.0, 5.0])  # last diff is 1.0
    scores = score_predictions(y_true, y_pred)
    assert scores["n"] == 4
    assert scores["mae"] == pytest.approx(0.25)
    assert scores["rmse"] == pytest.approx(0.5)


def test_score_predictions_ignores_nans():
    y_true = pd.Series([1.0, 2.0, np.nan, 4.0])
    y_pred = pd.Series([np.nan, 2.0, 3.0, 4.0])
    scores = score_predictions(y_true, y_pred)
    # Only rows 1 and 3 overlap. Diffs: 0, 0. MAE=RMSE=0.
    assert scores["n"] == 2
    assert scores["mae"] == 0.0
    assert scores["rmse"] == 0.0
