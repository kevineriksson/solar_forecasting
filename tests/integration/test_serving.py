"""T10 integration tests for the FastAPI serving app.

These tests do NOT touch MLflow or the real cluster. We monkey-patch
`load_production_model` to return a deterministic stub handle, so the tests
exercise the request schema, response shape, /healthz transitions, /metrics
exposition, error cases, and the "refuses to start" lifespan contract.

The cluster-side verification (real Production model, Prometheus target Up,
archive-the-model crash-loop) happens manually after deployment — see
k8s/serving/README.md.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.serving import app as app_module
from src.serving import loader as loader_module
from src.serving.loader import LoaderError, ModelHandle, ModelInfo

# ---------- stub handles ----------


def _make_info(
    *,
    model_type: str = "xgboost",
    feature_columns: tuple[str, ...] = (
        "k_t",
        "ghi_lag1",
        "zenith",
        "cs_ghi",
        "cs_dni",
        "cs_dhi",
    ),
    sequence_length: int = 1,
) -> ModelInfo:
    return ModelInfo(
        model_type=model_type,
        version="42",
        run_id="run_abc",
        git_commit="cafef00d" + "0" * 32,
        dvc_hash="deadbeef" + "0" * 56,
        feature_columns=feature_columns,
        output_columns=(
            ("ghi", "15min"),
            ("ghi", "1h"),
            ("dni", "15min"),
            ("dni", "1h"),
            ("dhi", "15min"),
            ("dhi", "1h"),
        ),
        targets=("ghi", "dni", "dhi"),
        horizon_labels=("15min", "1h"),
        sequence_length=sequence_length,
    )


class _StubHandle(ModelHandle):
    """Returns a deterministic mapping from feature row -> 6 numbers."""

    def __init__(self, info: ModelInfo) -> None:
        self.info = info

    def predict(self, features: list[dict[str, float]]) -> dict[str, float]:
        L = self.info.sequence_length
        if len(features) != L:
            raise ValueError(f"expected {L} timesteps, got {len(features)}")
        last = features[-1]
        missing = [c for c in self.info.feature_columns if c not in last]
        if missing:
            raise ValueError(f"missing features: {missing}")
        # Stable predictable values keyed off k_t so test assertions are easy.
        kt = float(last.get("k_t", 0.5))
        return {
            "ghi_15min": kt * 1000,
            "ghi_1h": kt * 900,
            "dni_15min": kt * 800,
            "dni_1h": kt * 700,
            "dhi_15min": kt * 200,
            "dhi_1h": kt * 180,
        }


# ---------- fixtures ----------


@pytest.fixture(autouse=True)
def _reset_prom_registry():
    """Reset the metric singletons between tests so label-counter state is fresh.

    prometheus_client's default REGISTRY is process-global; we touch it via
    our singleton metrics. Clearing the metric `_metrics` dicts is enough for
    Counter/Histogram label-set hygiene without unregistering the metrics
    themselves (which would break their module-level references).
    """
    from src.serving import metrics as M

    for m in (
        M.predict_latency,
        M.predict_requests,
        M.prediction_value,
        M.input_feature_value,
        M.model_info,
    ):
        m._metrics.clear()  # noqa: SLF001 — see metrics.set_model_info comment
    yield


def _build_client(
    monkeypatch: pytest.MonkeyPatch, *, info: ModelInfo, fail: bool = False
) -> TestClient:
    """Build a TestClient with `load_production_model` monkey-patched.

    `fail=True` causes the loader to raise LoaderError on startup, which is
    the "no Production model" case.
    """

    def fake_loader(*_args: Any, **_kwargs: Any) -> ModelHandle:
        if fail:
            raise LoaderError("no Production version registered")
        return _StubHandle(info)

    monkeypatch.setattr(loader_module, "load_production_model", fake_loader)
    monkeypatch.setattr(app_module, "load_production_model", fake_loader)

    fresh_app = app_module.create_app()
    return TestClient(fresh_app)


# ---------- tests ----------


def test_predict_tabular_returns_six_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _make_info(model_type="xgboost", sequence_length=1)
    with _build_client(monkeypatch, info=info) as client:
        payload = {
            "timestamp_utc": "2026-05-10T12:00:00Z",
            "features": {
                "k_t": 0.8,
                "ghi_lag1": 700.0,
                "zenith": 30.0,
                "cs_ghi": 900.0,
                "cs_dni": 850.0,
                "cs_dhi": 200.0,
            },
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()

    assert body["model_version"] == "42"
    assert body["model_type"] == "xgboost"
    # Six named irradiance floats — schema contract.
    for key in ("ghi_15min", "ghi_1h", "dni_15min", "dni_1h", "dhi_15min", "dhi_1h"):
        assert isinstance(body[key], float), f"{key} missing/not float"
    assert body["ghi_15min"] == pytest.approx(800.0)
    assert body["ghi_1h"] == pytest.approx(720.0)


def test_predict_missing_feature_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _make_info(model_type="xgboost", sequence_length=1)
    with _build_client(monkeypatch, info=info) as client:
        payload = {
            "timestamp_utc": "2026-05-10T12:00:00Z",
            "features": {"k_t": 0.8},  # missing the rest
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422, r.text
        assert "missing features" in r.text


def test_predict_sequence_for_lstm(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _make_info(
        model_type="lstm",
        feature_columns=("k_t", "ghi_lag1", "zenith"),
        sequence_length=3,
    )
    rows = [
        {"k_t": 0.1, "ghi_lag1": 100.0, "zenith": 30.0},
        {"k_t": 0.5, "ghi_lag1": 500.0, "zenith": 28.0},
        {"k_t": 0.9, "ghi_lag1": 900.0, "zenith": 25.0},
    ]
    ts = ["2026-05-10T11:30:00Z", "2026-05-10T11:45:00Z", "2026-05-10T12:00:00Z"]
    with _build_client(monkeypatch, info=info) as client:
        r = client.post("/predict", json={"timestamps_utc": ts, "features": rows})
        assert r.status_code == 200, r.text
        body = r.json()
    # Stub keys off the last row's k_t.
    assert body["ghi_15min"] == pytest.approx(900.0)
    assert body["model_type"] == "lstm"


def test_predict_sequence_length_mismatch_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _make_info(
        model_type="lstm",
        feature_columns=("k_t", "ghi_lag1", "zenith"),
        sequence_length=3,
    )
    short_rows = [{"k_t": 0.5, "ghi_lag1": 500.0, "zenith": 28.0}] * 2  # only 2, need 3
    ts = ["2026-05-10T11:45:00Z", "2026-05-10T12:00:00Z"]
    with _build_client(monkeypatch, info=info) as client:
        r = client.post("/predict", json={"timestamps_utc": ts, "features": short_rows})
        assert r.status_code == 422, r.text
        assert "3 timesteps" in r.text or "expected 3" in r.text


def test_predict_request_shape_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both timestamp_utc and timestamps_utc set, or both unset, must 422."""
    info = _make_info()
    with _build_client(monkeypatch, info=info) as client:
        # Neither set.
        r = client.post("/predict", json={"features": {"k_t": 0.5}})
        assert r.status_code == 422
        # Both set.
        r = client.post(
            "/predict",
            json={
                "timestamp_utc": "2026-05-10T12:00:00Z",
                "timestamps_utc": ["2026-05-10T12:00:00Z"],
                "features": {"k_t": 0.5},
            },
        )
        assert r.status_code == 422


def test_healthz_ok_after_load(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _make_info()
    with _build_client(monkeypatch, info=info) as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "xgboost"
    assert body["model_version"] == "42"
    assert body["git_commit"].startswith("cafef00d")


def test_app_refuses_to_start_when_no_production_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """LoaderError during lifespan must propagate and abort startup."""
    info = _make_info()
    with pytest.raises(LoaderError):
        # `with TestClient(...)` runs lifespan startup eagerly on __enter__.
        with _build_client(monkeypatch, info=info, fail=True) as client:
            client.get("/healthz")


def test_metrics_endpoint_exposes_expected_series(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _make_info(model_type="xgboost", sequence_length=1)
    with _build_client(monkeypatch, info=info) as client:
        # Hit /predict once so series get observed beyond the pre-declared zero state.
        client.post(
            "/predict",
            json={
                "timestamp_utc": "2026-05-10T12:00:00Z",
                "features": {
                    "k_t": 0.7,
                    "ghi_lag1": 600.0,
                    "zenith": 35.0,
                    "cs_ghi": 800.0,
                    "cs_dni": 750.0,
                    "cs_dhi": 180.0,
                },
            },
        )
        r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    # Each named metric must be present.
    for name in (
        "solar_predict_latency_seconds",
        "solar_predict_requests_total",
        "solar_prediction_value",
        "solar_input_feature_value",
        "solar_model_info",
    ):
        assert name in text, f"metric {name} missing from /metrics"
    # Provenance labels must be on solar_model_info.
    assert 'model_type="xgboost"' in text
    assert 'version="42"' in text
    # Tracked features pre-declared (k_t observed).
    assert 'feature="k_t"' in text
    # Pre-declared output cells (prometheus_client sorts label names alphabetically,
    # so the on-the-wire order is horizon,target — assert presence of each piece).
    assert 'horizon="15min"' in text and 'target="ghi"' in text
