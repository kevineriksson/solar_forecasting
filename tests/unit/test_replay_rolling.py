"""Unit tests for the T12 rolling-window helpers.

These are the only pieces of T12 logic that have non-trivial maths in them:
the PromQL alerts read these gauges directly, so a regression here would
silently break the dashboard and the drift_high / skill_score_low alerts.

We test:
  * Rolling MAE / RMSE on a hand-computed window.
  * Skill score sign: model better than persistence -> skill > 0; worse -> < 0.
  * Skill score returns NaN while the window is below ``min_samples``.
  * PSI is ~0 on identical distributions and increases monotonically as a
    feature distribution is shifted further from the reference.
  * PSI ignores non-finite samples instead of raising.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.replay.rolling import FeaturePSI, RollingResidualWindow

# ---------------------------------------------------------------------------
# RollingResidualWindow
# ---------------------------------------------------------------------------


def _seed_window(
    window: RollingResidualWindow,
    target: str,
    horizon: str,
    model_residuals: list[float],
    persistence_residuals: list[float],
) -> None:
    for m, p in zip(model_residuals, persistence_residuals, strict=True):
        window.observe(target, horizon, m, p)


def test_rolling_mae_rmse_match_handcomputed_values() -> None:
    win = RollingResidualWindow(
        window_steps=4, targets=("ghi",), horizon_labels=("15min",), min_samples=4
    )
    _seed_window(win, "ghi", "15min", [10.0, -20.0, 30.0, -40.0], [0.0, 0.0, 0.0, 0.0])

    assert win.mae("ghi", "15min") == pytest.approx(25.0)
    expected_rmse = math.sqrt((100 + 400 + 900 + 1600) / 4)
    assert win.rmse("ghi", "15min") == pytest.approx(expected_rmse)


def test_rolling_window_drops_oldest_samples_past_capacity() -> None:
    win = RollingResidualWindow(
        window_steps=2, targets=("ghi",), horizon_labels=("15min",), min_samples=2
    )
    # The first two should be evicted; only the final two survive.
    _seed_window(win, "ghi", "15min", [100.0, -100.0, 5.0, -5.0], [1.0, 1.0, 1.0, 1.0])
    assert win.mae("ghi", "15min") == pytest.approx(5.0)


def test_skill_score_positive_when_model_beats_persistence() -> None:
    win = RollingResidualWindow(
        window_steps=4, targets=("ghi",), horizon_labels=("15min",), min_samples=4
    )
    # Model residuals tight around 0; persistence is much worse.
    _seed_window(win, "ghi", "15min", [1.0, -1.0, 1.0, -1.0], [10.0, -10.0, 10.0, -10.0])
    skill = win.skill("ghi", "15min")
    assert skill > 0.5  # model is ~10x better than persistence by RMSE


def test_skill_score_negative_when_model_worse_than_persistence() -> None:
    win = RollingResidualWindow(
        window_steps=4, targets=("ghi",), horizon_labels=("15min",), min_samples=4
    )
    _seed_window(win, "ghi", "15min", [10.0, -10.0, 10.0, -10.0], [1.0, -1.0, 1.0, -1.0])
    assert win.skill("ghi", "15min") < -0.5


def test_rolling_gauges_nan_until_min_samples_reached() -> None:
    win = RollingResidualWindow(
        window_steps=10, targets=("ghi",), horizon_labels=("15min",), min_samples=5
    )
    _seed_window(win, "ghi", "15min", [1.0, 2.0], [1.0, 2.0])
    assert math.isnan(win.mae("ghi", "15min"))
    assert math.isnan(win.rmse("ghi", "15min"))
    assert math.isnan(win.skill("ghi", "15min"))


def test_skill_score_nan_when_persistence_has_zero_rmse() -> None:
    # If persistence is perfect (RMSE_p == 0) the skill score is undefined; we
    # publish NaN rather than +inf/-inf so the dashboard doesn't spike.
    win = RollingResidualWindow(
        window_steps=4, targets=("ghi",), horizon_labels=("15min",), min_samples=4
    )
    _seed_window(win, "ghi", "15min", [5.0, -5.0, 5.0, -5.0], [0.0, 0.0, 0.0, 0.0])
    assert math.isnan(win.skill("ghi", "15min"))


def test_fill_ratio_progresses_monotonically() -> None:
    win = RollingResidualWindow(
        window_steps=4, targets=("ghi",), horizon_labels=("15min",), min_samples=1
    )
    assert win.fill_ratio() == 0.0
    win.observe("ghi", "15min", 1.0, 1.0)
    win.observe("ghi", "15min", 1.0, 1.0)
    assert win.fill_ratio() == pytest.approx(0.5)
    for _ in range(2):
        win.observe("ghi", "15min", 1.0, 1.0)
    assert win.fill_ratio() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FeaturePSI
# ---------------------------------------------------------------------------


def _normal_reference(seed: int = 0, size: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=size).tolist()


def test_psi_near_zero_when_current_matches_reference() -> None:
    reference = {"k_t": _normal_reference(seed=1)}
    psi = FeaturePSI(reference_values=reference, window_steps=2000, min_samples=100)
    # Feed in samples from the same distribution.
    rng = np.random.default_rng(2)
    for v in rng.normal(loc=0.0, scale=1.0, size=2000):
        psi.observe("k_t", float(v))
    value = psi.psi("k_t")
    assert value < 0.1, f"PSI on matching distributions should be ~0, got {value}"


def test_psi_increases_with_distribution_shift() -> None:
    reference = {"k_t": _normal_reference(seed=3)}
    rng = np.random.default_rng(4)
    psi_values = []
    for shift in (0.0, 0.5, 1.5):
        psi = FeaturePSI(reference_values=reference, window_steps=2000, min_samples=100)
        for v in rng.normal(loc=shift, scale=1.0, size=2000):
            psi.observe("k_t", float(v))
        psi_values.append(psi.psi("k_t"))
    assert psi_values[0] < psi_values[1] < psi_values[2]
    # The large 1.5-sigma shift should comfortably exceed the 0.2 alert
    # threshold used by the PrometheusRule.
    assert psi_values[2] > 0.2


def test_psi_nan_until_min_samples_reached() -> None:
    reference = {"k_t": _normal_reference(seed=5, size=1000)}
    psi = FeaturePSI(reference_values=reference, window_steps=200, min_samples=50)
    for v in (0.1, 0.2, 0.3):  # only 3 samples
        psi.observe("k_t", v)
    assert math.isnan(psi.psi("k_t"))


def test_psi_ignores_unknown_feature() -> None:
    psi = FeaturePSI(
        reference_values={"k_t": _normal_reference(seed=6, size=500)}, window_steps=100
    )
    psi.observe("not_a_feature", 1.0)  # must not raise
    assert math.isnan(psi.psi("not_a_feature"))


def test_psi_ignores_non_finite_samples() -> None:
    reference = {"k_t": _normal_reference(seed=7, size=1000)}
    psi = FeaturePSI(reference_values=reference, window_steps=200, min_samples=50)
    rng = np.random.default_rng(8)
    for v in rng.normal(loc=0.0, scale=1.0, size=100):
        psi.observe("k_t", float(v))
    psi.observe("k_t", float("nan"))
    psi.observe("k_t", float("inf"))
    # Should still return a finite PSI value despite the bad inputs.
    value = psi.psi("k_t")
    assert math.isfinite(value)


def test_psi_handles_constant_reference_without_raising() -> None:
    # Degenerate reference distribution (single value); binning needs to
    # degrade gracefully rather than crash, so an alert never silently dies
    # on a misconfigured feature.
    psi = FeaturePSI(reference_values={"flat": [3.0] * 100}, window_steps=10, min_samples=5)
    for _ in range(10):
        psi.observe("flat", 3.0)
    assert math.isfinite(psi.psi("flat"))
