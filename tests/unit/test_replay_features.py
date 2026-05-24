"""Unit tests for src.replay.features.

Covers the two invariants T11 cares about:

  1. ``build_features_as_of`` raises :class:`LeakageError` if asked to build the
     feature row at ``t`` from a history that contains rows after ``t``. This
     is the explicit "no future data" guard the replay loop is built around.

  2. ``build_features_as_of`` produces the same feature row at ``t`` that
     :func:`src.features.pipeline.build_features` produces when the input
     history is exactly truncated at ``t``. Together with (1), this proves the
     replay-time feature builder is a thin, correctness-preserving wrapper of
     Stage 2 — which is what T11 requires.

The :class:`ReplaySource` tests cover the runtime accessors used by the loop:
``feature_payload``, ``feature_sequence``, ``truths_at``, and the trailing
horizon-clip in :meth:`ReplaySource.timestamps`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeatureConfig, build_features
from src.ingest.schema import TIMESTAMP_COL
from src.replay.features import (
    GroundTruth,
    LeakageError,
    ReplaySource,
    build_features_as_of,
    enrich_for_persistence,
)

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


def _raw_history(n: int = 96 * 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="15min", tz="UTC")
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


# ---------------------------------------------------------------------------
# (1) leakage guard
# ---------------------------------------------------------------------------


def test_build_features_as_of_raises_when_history_contains_future_rows() -> None:
    df = _raw_history()
    cfg = FeatureConfig.from_params(PARAMS)
    # Pick a t in the middle of the history. The caller passes the *full*
    # history (which includes rows > t) and expects the function to refuse.
    t = pd.Timestamp(df[TIMESTAMP_COL].iloc[len(df) // 2])

    with pytest.raises(LeakageError) as excinfo:
        build_features_as_of(df, t, cfg)
    msg = str(excinfo.value)
    assert "timestamp >" in msg
    assert t.isoformat() in msg


def test_build_features_as_of_rejects_naive_timestamp() -> None:
    df = _raw_history()
    cfg = FeatureConfig.from_params(PARAMS)
    naive_t = pd.Timestamp("2025-01-02T00:00:00")  # tz-naive

    with pytest.raises(ValueError):
        build_features_as_of(df, naive_t, cfg)


# ---------------------------------------------------------------------------
# (2) equivalence with full Stage 2 pipeline
# ---------------------------------------------------------------------------


def test_build_features_as_of_matches_full_pipeline_row() -> None:
    """The wrapper must produce the exact row Stage 2 produced for the same t."""
    df = _raw_history()
    cfg = FeatureConfig.from_params(PARAMS)

    # Pre-compute features for the full series (Stage 2's contract).
    full = build_features(df, cfg)
    # Pick a t that is comfortably past the warmup window (longest lag = 12).
    t = pd.Timestamp(full[TIMESTAMP_COL].iloc[100])

    # Wrapper sees only history <= t. Pass it pre-truncated to satisfy the guard.
    truncated = df.loc[df[TIMESTAMP_COL] <= t].reset_index(drop=True)
    as_of = build_features_as_of(truncated, t, cfg)

    expected_row = full.loc[full[TIMESTAMP_COL] == t].iloc[0]
    for col in full.columns:
        if col == TIMESTAMP_COL:
            continue
        # Float columns may differ by float-rounding noise; require near-equality.
        np.testing.assert_allclose(
            float(as_of[col]),
            float(expected_row[col]),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"column {col!r} diverges between as-of and full pipeline",
        )


def test_build_features_as_of_returns_row_at_requested_timestamp() -> None:
    df = _raw_history()
    cfg = FeatureConfig.from_params(PARAMS)
    t = pd.Timestamp(df[TIMESTAMP_COL].iloc[200])

    truncated = df.loc[df[TIMESTAMP_COL] <= t].reset_index(drop=True)
    out = build_features_as_of(truncated, t, cfg)
    # The wrapper guarantees the last row is the requested t — implicit in the
    # AssertionError it raises otherwise. We assert observable behaviour: the
    # presence of the "current" target value at t.
    assert out["ghi"] == pytest.approx(float(df.loc[df[TIMESTAMP_COL] == t, "ghi"].iloc[0]))


# ---------------------------------------------------------------------------
# ReplaySource
# ---------------------------------------------------------------------------


def _features_frame(n: int = 96 * 8) -> pd.DataFrame:
    """A pre-computed Stage 2 frame stand-in: feature columns + targets."""
    df = _raw_history(n=n)
    cfg = FeatureConfig.from_params(PARAMS)
    return build_features(df, cfg)


def test_replay_source_feature_payload_strips_timestamp_only() -> None:
    feats = _features_frame()
    src = ReplaySource(
        feats, targets=("ghi", "dni", "dhi"), horizons_steps=(1, 4), horizon_labels=("15min", "1h")
    )
    t = pd.Timestamp(feats[TIMESTAMP_COL].iloc[50])
    payload = src.feature_payload(t)
    assert TIMESTAMP_COL not in payload
    # Sample numeric columns are present and float-valued.
    for col in ("ghi", "k_t", "ghi_lag1", "hour_sin"):
        assert col in payload
        assert isinstance(payload[col], int | float)


def test_replay_source_truths_at_returns_value_at_t_plus_h() -> None:
    feats = _features_frame()
    src = ReplaySource(
        feats, targets=("ghi", "dni", "dhi"), horizons_steps=(1, 4), horizon_labels=("15min", "1h")
    )
    idx = 30
    t = pd.Timestamp(feats[TIMESTAMP_COL].iloc[idx])

    truths_15 = src.truths_at(t, horizon_steps=1)
    assert truths_15 is not None
    expected_ghi = float(feats["ghi"].iloc[idx + 1])
    ghi_truth = next(gt for gt in truths_15 if gt.target == "ghi")
    assert isinstance(ghi_truth, GroundTruth)
    assert ghi_truth.value == pytest.approx(expected_ghi)
    assert ghi_truth.horizon_steps == 1
    assert ghi_truth.horizon_label == "15min"

    truths_1h = src.truths_at(t, horizon_steps=4)
    assert truths_1h is not None
    dni_truth = next(gt for gt in truths_1h if gt.target == "dni")
    assert dni_truth.value == pytest.approx(float(feats["dni"].iloc[idx + 4]))


def test_replay_source_truths_at_returns_none_past_window_end() -> None:
    feats = _features_frame()
    src = ReplaySource(
        feats, targets=("ghi",), horizons_steps=(1, 4), horizon_labels=("15min", "1h")
    )
    last_t = pd.Timestamp(feats[TIMESTAMP_COL].iloc[-1])
    # No row exists at last + 1 step.
    assert src.truths_at(last_t, horizon_steps=1) is None


def test_replay_source_timestamps_excludes_trailing_horizon() -> None:
    feats = _features_frame()
    src = ReplaySource(
        feats, targets=("ghi",), horizons_steps=(1, 4), horizon_labels=("15min", "1h")
    )
    iter_ts = src.timestamps()
    last_iter = iter_ts[-1]
    # The last scoreable timestamp is len-1 - max_horizon = len-1-4.
    expected_last = pd.Timestamp(feats[TIMESTAMP_COL].iloc[-1 - 4])
    assert last_iter == expected_last
    # Every iter_ts has both an in-window 15-min and 1-h truth.
    for t in iter_ts[-3:]:
        assert src.truths_at(t, 1) is not None
        assert src.truths_at(t, 4) is not None


def test_replay_source_feature_sequence_returns_oldest_first() -> None:
    feats = _features_frame()
    src = ReplaySource(
        feats, targets=("ghi",), horizons_steps=(1, 4), horizon_labels=("15min", "1h")
    )
    t = pd.Timestamp(feats[TIMESTAMP_COL].iloc[60])
    seq = src.feature_sequence(t, sequence_length=4)
    assert len(seq) == 4
    ts_seq = src.feature_timestamps(t, sequence_length=4)
    assert ts_seq[-1] == t
    assert ts_seq[0] == pd.Timestamp(feats[TIMESTAMP_COL].iloc[57])
    # Oldest-first ordering: the final element's "ghi" must equal feats[ghi][t]
    assert seq[-1]["ghi"] == pytest.approx(
        float(feats.loc[feats[TIMESTAMP_COL] == t, "ghi"].iloc[0])
    )


def test_replay_source_feature_sequence_rejects_when_not_enough_history() -> None:
    feats = _features_frame()
    src = ReplaySource(
        feats, targets=("ghi",), horizons_steps=(1, 4), horizon_labels=("15min", "1h")
    )
    t = pd.Timestamp(feats[TIMESTAMP_COL].iloc[1])
    with pytest.raises(ValueError):
        src.feature_sequence(t, sequence_length=24)


# ---------------------------------------------------------------------------
# Persistence enrichment
# ---------------------------------------------------------------------------


def test_enrich_for_persistence_adds_horizon_clearsky_columns() -> None:
    base = {
        "ghi": 500.0,
        "dni": 700.0,
        "dhi": 100.0,
        "cs_ghi": 600.0,
        "cs_dni": 800.0,
        "cs_dhi": 120.0,
    }
    t = pd.Timestamp("2025-06-15T12:00:00Z")  # daylight in Montana
    out = enrich_for_persistence(
        base,
        t,
        PARAMS["site"],
        horizons_steps=(1, 4),
        horizon_labels=("15min", "1h"),
    )
    for tgt in ("ghi", "dni", "dhi"):
        for lbl in ("15min", "1h"):
            key = f"cs_{tgt}_h{lbl}"
            assert key in out, f"missing {key}"
            assert out[key] >= 0.0
    # The original keys are still present (we don't mutate the input dict).
    for k, v in base.items():
        assert out[k] == v


def test_enrich_for_persistence_does_not_mutate_input() -> None:
    base = {"ghi": 500.0}
    out = enrich_for_persistence(
        base,
        pd.Timestamp("2025-06-15T12:00:00Z"),
        PARAMS["site"],
        horizons_steps=(1,),
        horizon_labels=("15min",),
    )
    assert "cs_ghi_h15min" in out
    assert "cs_ghi_h15min" not in base
