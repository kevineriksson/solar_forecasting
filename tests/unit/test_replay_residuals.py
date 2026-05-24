"""Sanity tests for the replay metrics layer.

The replay client computes residual = prediction - truth per (target, horizon)
and emits four series (residual, |residual|, prediction, truth) plus a counter.
These tests verify the metric module observes the right values when called and
that pre-declaring series populates every (target, horizon) label combination.
"""

from __future__ import annotations

import pytest

from src.replay import metrics as M


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Clear all replay-metric series before each test for isolation."""
    for collector in (
        M.predictions_total,
        M.request_failures_total,
        M.residual,
        M.residual_abs,
        M.prediction,
        M.truth,
        M.info,
    ):
        collector._metrics.clear()  # noqa: SLF001
    M.progress_ratio.set(0)
    M.simulated_clock_seconds.set(0)
    yield


def test_observe_residual_emits_signed_and_abs_residual() -> None:
    M.observe_residual(target="ghi", horizon="15min", pred_value=520.0, truth_value=500.0)
    r_child = M.residual.labels(target="ghi", horizon="15min")
    abs_child = M.residual_abs.labels(target="ghi", horizon="15min")
    pred_child = M.prediction.labels(target="ghi", horizon="15min")
    truth_child = M.truth.labels(target="ghi", horizon="15min")

    # Sum should reflect the signed and absolute residuals after one observation.
    assert r_child._sum.get() == pytest.approx(20.0)  # noqa: SLF001
    assert abs_child._sum.get() == pytest.approx(20.0)  # noqa: SLF001
    assert pred_child._sum.get() == pytest.approx(520.0)  # noqa: SLF001
    assert truth_child._sum.get() == pytest.approx(500.0)  # noqa: SLF001


def test_observe_residual_handles_negative_residual() -> None:
    M.observe_residual(target="dni", horizon="1h", pred_value=300.0, truth_value=450.0)
    r_child = M.residual.labels(target="dni", horizon="1h")
    abs_child = M.residual_abs.labels(target="dni", horizon="1h")
    assert r_child._sum.get() == pytest.approx(-150.0)  # noqa: SLF001
    assert abs_child._sum.get() == pytest.approx(150.0)  # noqa: SLF001


def test_observe_residual_increments_predictions_counter() -> None:
    counter = M.predictions_total.labels(target="dhi", horizon="15min")
    before = counter._value.get()  # noqa: SLF001
    M.observe_residual(target="dhi", horizon="15min", pred_value=50.0, truth_value=40.0)
    M.observe_residual(target="dhi", horizon="15min", pred_value=60.0, truth_value=70.0)
    assert counter._value.get() - before == pytest.approx(2.0)  # noqa: SLF001


def test_declare_series_pre_creates_every_target_horizon_combination() -> None:
    cols = tuple((t, lbl) for t in ("ghi", "dni", "dhi") for lbl in ("15min", "1h"))
    M.declare_series(cols)
    for tgt, lbl in cols:
        # Accessing .labels(...) without raising means the labelset exists.
        for series in (M.predictions_total, M.residual, M.residual_abs, M.prediction, M.truth):
            child = series.labels(target=tgt, horizon=lbl)
            assert child is not None


def test_set_info_clears_previous_labelset() -> None:
    M.set_info("xgboost", "1", "deadbeef", "cafebabe")
    M.set_info("lstm", "2", "1234abcd", "5678ef01")
    # Only the most recent set should remain — single Production model at a time.
    labels = list(M.info._metrics.keys())  # noqa: SLF001
    assert len(labels) == 1
    assert labels[0] == ("lstm", "2", "1234abcd", "5678ef01")
