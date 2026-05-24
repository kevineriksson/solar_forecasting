"""Prometheus instruments for the replay client (Stage 6 / T11).

The replay client exposes ``/metrics`` on its own HTTP server (default :9090).
Prometheus scrapes it via a ServiceMonitor — push-style metrics are an
anti-pattern, so the loop writes locally and waits to be polled.

Series exposed:

  solar_replay_predictions_total            Counter   {target,horizon}
  solar_replay_request_failures_total       Counter   {reason}
  solar_replay_residual                     Histogram {target,horizon}
  solar_replay_residual_abs                 Histogram {target,horizon}
  solar_replay_prediction                   Histogram {target,horizon}
  solar_replay_truth                        Histogram {target,horizon}
  solar_replay_request_latency_seconds      Histogram
  solar_replay_progress_ratio               Gauge
  solar_replay_simulated_clock_seconds      Gauge
  solar_replay_info                         Gauge=1   {model_type,model_version,git_commit,dvc_hash}
  solar_replay_rolling_mae                  Gauge     {target,horizon}
  solar_replay_rolling_rmse                 Gauge     {target,horizon}
  solar_replay_rolling_skill                Gauge     {target,horizon}
  solar_replay_feature_psi                  Gauge     {feature}
  solar_replay_rolling_window_filled        Gauge

The histograms share bucketing with the serving app where it makes sense, so
T12's PromQL queries can use the same boundaries.

The four rolling gauges are computed in the replay client over a fixed-size
trailing window (``monitoring.drift.window_steps`` in params.yaml) and updated
after every scored prediction. PSI is computed against a reference histogram
snapshotted from the training split at client startup. PromQL cannot express
PSI directly, so the gauge approach is the simplest viable design.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Irradiance in W/m^2 — matches src.serving.metrics so panels can stack.
_IRRADIANCE_BUCKETS = (0, 25, 50, 100, 200, 400, 600, 800, 1000, 1200)

# Residuals can be negative (pred < truth) or positive. The buckets are signed.
_RESIDUAL_BUCKETS = (
    -1000.0,
    -500.0,
    -250.0,
    -100.0,
    -50.0,
    -25.0,
    -10.0,
    -1.0,
    0.0,
    1.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
)

# |residual| bucketing — useful for rate(...) MAE-style queries in PromQL.
_ABS_RESIDUAL_BUCKETS = (0, 1, 5, 10, 25, 50, 100, 250, 500, 1000)

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


predictions_total = Counter(
    "solar_replay_predictions_total",
    "Predictions emitted by the replay client by target and horizon.",
    labelnames=("target", "horizon"),
)

request_failures_total = Counter(
    "solar_replay_request_failures_total",
    "Predict requests that failed, labeled by reason.",
    labelnames=("reason",),
)

residual = Histogram(
    "solar_replay_residual",
    "Replay residual (prediction - truth) in W/m^2.",
    labelnames=("target", "horizon"),
    buckets=_RESIDUAL_BUCKETS,
)

residual_abs = Histogram(
    "solar_replay_residual_abs",
    "Absolute replay residual |prediction - truth| in W/m^2 — for MAE in PromQL.",
    labelnames=("target", "horizon"),
    buckets=_ABS_RESIDUAL_BUCKETS,
)

prediction = Histogram(
    "solar_replay_prediction",
    "Replayed prediction values in W/m^2.",
    labelnames=("target", "horizon"),
    buckets=_IRRADIANCE_BUCKETS,
)

truth = Histogram(
    "solar_replay_truth",
    "Replay ground-truth values in W/m^2.",
    labelnames=("target", "horizon"),
    buckets=_IRRADIANCE_BUCKETS,
)

request_latency = Histogram(
    "solar_replay_request_latency_seconds",
    "End-to-end POST /predict latency from the replay client (seconds).",
    buckets=_LATENCY_BUCKETS,
)

progress_ratio = Gauge(
    "solar_replay_progress_ratio",
    "Fraction of the replay window already simulated, in [0, 1].",
)

simulated_clock_seconds = Gauge(
    "solar_replay_simulated_clock_seconds",
    "Unix timestamp (UTC) of the current simulated replay step.",
)

info = Gauge(
    "solar_replay_info",
    "Provenance for the replay run; value always 1.",
    labelnames=("model_type", "model_version", "git_commit", "dvc_hash"),
)

rolling_mae = Gauge(
    "solar_replay_rolling_mae",
    "Rolling MAE over the trailing window (W/m^2).",
    labelnames=("target", "horizon"),
)

rolling_rmse = Gauge(
    "solar_replay_rolling_rmse",
    "Rolling RMSE over the trailing window (W/m^2).",
    labelnames=("target", "horizon"),
)

rolling_skill = Gauge(
    "solar_replay_rolling_skill",
    "Rolling skill score vs naive persistence: 1 - RMSE_model / RMSE_persistence.",
    labelnames=("target", "horizon"),
)

feature_psi = Gauge(
    "solar_replay_feature_psi",
    "Population Stability Index of recent feature values vs training reference.",
    labelnames=("feature",),
)

rolling_window_filled = Gauge(
    "solar_replay_rolling_window_filled",
    "Fraction of the rolling window currently populated, in [0, 1]. "
    "Skill/PSI gauges are NaN until this reaches 1.0.",
)


def declare_series(
    output_columns: tuple[tuple[str, str], ...],
    failure_reasons: tuple[str, ...] = (
        "connect",
        "timeout",
        "http_5xx",
        "http_4xx",
        "parse",
        "no_truth",
    ),
) -> None:
    """Pre-touch label combinations so the first scrape sees every series."""
    for tgt, lbl in output_columns:
        predictions_total.labels(target=tgt, horizon=lbl)
        residual.labels(target=tgt, horizon=lbl)
        residual_abs.labels(target=tgt, horizon=lbl)
        prediction.labels(target=tgt, horizon=lbl)
        truth.labels(target=tgt, horizon=lbl)
        rolling_mae.labels(target=tgt, horizon=lbl)
        rolling_rmse.labels(target=tgt, horizon=lbl)
        rolling_skill.labels(target=tgt, horizon=lbl)
    for reason in failure_reasons:
        request_failures_total.labels(reason=reason)


def declare_feature_psi_series(features: tuple[str, ...]) -> None:
    """Pre-touch PSI gauges so the dashboard sees every tracked feature."""
    for feat in features:
        feature_psi.labels(feature=feat)


def set_info(model_type: str, model_version: str, git_commit: str, dvc_hash: str) -> None:
    info._metrics.clear()  # noqa: SLF001
    info.labels(
        model_type=model_type,
        model_version=model_version,
        git_commit=git_commit,
        dvc_hash=dvc_hash,
    ).set(1)


def observe_residual(target: str, horizon: str, pred_value: float, truth_value: float) -> None:
    diff = pred_value - truth_value
    residual.labels(target=target, horizon=horizon).observe(diff)
    residual_abs.labels(target=target, horizon=horizon).observe(abs(diff))
    prediction.labels(target=target, horizon=horizon).observe(pred_value)
    truth.labels(target=target, horizon=horizon).observe(truth_value)
    predictions_total.labels(target=target, horizon=horizon).inc()
