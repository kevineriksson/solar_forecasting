"""Prometheus instruments for the serving app.

Series exposed (label sets pre-declared so each cardinality stays bounded):

  solar_predict_latency_seconds        Histogram     {endpoint}
  solar_predict_requests_total         Counter       {endpoint,status_code}
  solar_prediction_value               Histogram     {target,horizon}
  solar_input_feature_value            Histogram     {feature}
  solar_model_info                     Gauge=1       {model_type,version,git_commit,dvc_hash}

The /metrics endpoint is mounted as an ASGI sub-app in app.py via
`prometheus_client.make_asgi_app()`.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Endpoint names we instrument. Pre-declared so Prometheus sees the series
# from the first scrape even before traffic arrives.
ENDPOINTS = ("predict", "healthz")

# Input features to track distribution on. A small allowlist keeps cardinality
# in check; we deliberately don't expose every feature column.
TRACKED_FEATURES = ("k_t", "ghi_lag1", "zenith")

# Buckets: latency in seconds, generous range for a CPU forecast.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

# Buckets: irradiance in W/m^2. Covers night through bright midday GHI.
_IRRADIANCE_BUCKETS = (0, 25, 50, 100, 200, 400, 600, 800, 1000, 1200)

# Buckets: feature value — kept generic since the tracked features differ in scale.
# k_t in [0, 1.5]; zenith in [0, 180]; ghi_lag1 ~ irradiance.
_FEATURE_BUCKETS = (-50, 0, 1, 5, 25, 100, 250, 500, 750, 1000, 1500)


predict_latency = Histogram(
    "solar_predict_latency_seconds",
    "Latency of serving HTTP endpoints in seconds.",
    labelnames=("endpoint",),
    buckets=_LATENCY_BUCKETS,
)

predict_requests = Counter(
    "solar_predict_requests_total",
    "Count of HTTP requests by endpoint and status code.",
    labelnames=("endpoint", "status_code"),
)

prediction_value = Histogram(
    "solar_prediction_value",
    "Predicted irradiance (W/m^2) per target and horizon.",
    labelnames=("target", "horizon"),
    buckets=_IRRADIANCE_BUCKETS,
)

input_feature_value = Histogram(
    "solar_input_feature_value",
    "Observed input feature values (allowlisted).",
    labelnames=("feature",),
    buckets=_FEATURE_BUCKETS,
)

# Single-valued gauge per (model_type, version, ...). Re-emitted on (re)load.
model_info = Gauge(
    "solar_model_info",
    "Current Production model provenance; value is always 1.",
    labelnames=("model_type", "version", "git_commit", "dvc_hash"),
)


def declare_endpoint_series() -> None:
    """Pre-declare label combinations for endpoint-scoped metrics.

    Prometheus only sees a series after it's been incremented or observed at
    least once. Pre-touching common (endpoint, status_code) pairs makes the
    /metrics output stable from the first scrape, which T12's alert rules and
    dashboards depend on.
    """
    for ep in ENDPOINTS:
        # Trigger the Histogram so the series shows up even before traffic.
        predict_latency.labels(endpoint=ep)
        for code in ("200", "422", "500", "503"):
            predict_requests.labels(endpoint=ep, status_code=code)


def declare_output_series(output_columns: tuple[tuple[str, str], ...]) -> None:
    """Pre-declare prediction_value series for every (target, horizon) cell."""
    for t, lbl in output_columns:
        prediction_value.labels(target=t, horizon=lbl)


def declare_feature_series() -> None:
    """Pre-declare input_feature_value series for the allowlist."""
    for feat in TRACKED_FEATURES:
        input_feature_value.labels(feature=feat)


def set_model_info(model_type: str, version: str, git_commit: str, dvc_hash: str) -> None:
    """Publish current Production model provenance as a label-only gauge."""
    # Clear any prior labelset first — model_type/version change on reload.
    model_info._metrics.clear()  # noqa: SLF001 — prometheus_client has no public reset
    model_info.labels(
        model_type=model_type, version=version, git_commit=git_commit, dvc_hash=dvc_hash
    ).set(1)


def observe_prediction(
    predictions: dict[str, float], output_columns: tuple[tuple[str, str], ...]
) -> None:
    """Emit one observation per (target, horizon) for a single /predict call."""
    for t, lbl in output_columns:
        key = f"{t}_{lbl}"
        if key in predictions:
            prediction_value.labels(target=t, horizon=lbl).observe(predictions[key])


def observe_input_features(features: dict[str, float]) -> None:
    """Emit one observation per tracked feature (only those present in payload)."""
    for feat in TRACKED_FEATURES:
        if feat in features:
            input_feature_value.labels(feature=feat).observe(float(features[feat]))
