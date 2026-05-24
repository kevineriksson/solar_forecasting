"""FastAPI serving app (Stage 5 / T10).

Endpoints:
  POST /predict   — request: timestamp + features (single row for tabular
                    models; list of L rows for LSTM). response: 6 named floats.
  GET  /healthz   — readiness/liveness. 503 until the Production model loads.
  GET  /metrics   — Prometheus text-format, mounted via ASGI sub-app.

Startup contract: if the Production model cannot be loaded, the app refuses
to start. uvicorn will exit non-zero; the Kubernetes Deployment crash-loops.
This is what makes the "refuses to start when MLflow has no Production model"
done-when check work.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field, model_validator

from src.serving import metrics as M
from src.serving.loader import LoaderError, ModelHandle, ModelInfo, load_production_model

LOG = logging.getLogger("serving.app")


# --------------------------------------------------------------------- schemas


class PredictRequest(BaseModel):
    """Polymorphic predict request.

    Tabular (xgboost / persistence) — single timestep:
        {"timestamp_utc": "2026-05-10T12:00:00Z", "features": {"k_t": 0.9, ...}}

    Sequence (lstm) — list of timesteps, oldest first, length = sequence_length:
        {"timestamps_utc": ["...", ...], "features": [{"k_t": 0.9, ...}, ...]}

    Exactly one of (timestamp_utc, timestamps_utc) must be present, and the
    `features` shape must match.
    """

    timestamp_utc: datetime | None = Field(default=None, description="As-of UTC time (tabular).")
    timestamps_utc: list[datetime] | None = Field(
        default=None, description="As-of UTC times, oldest first (sequence)."
    )
    features: dict[str, float] | list[dict[str, float]] = Field(
        description="Feature row (tabular) or list of rows (sequence)."
    )

    @model_validator(mode="after")
    def _check_shape(self) -> PredictRequest:
        single = self.timestamp_utc is not None
        multi = self.timestamps_utc is not None
        if single == multi:  # both or neither
            raise ValueError("request must set exactly one of timestamp_utc / timestamps_utc")
        if single and not isinstance(self.features, dict):
            raise ValueError("with timestamp_utc, features must be a single dict")
        if multi and not isinstance(self.features, list):
            raise ValueError("with timestamps_utc, features must be a list of dicts")
        if multi:
            assert isinstance(self.features, list)
            assert isinstance(self.timestamps_utc, list)
            if len(self.features) != len(self.timestamps_utc):
                raise ValueError(
                    f"features ({len(self.features)}) and timestamps_utc "
                    f"({len(self.timestamps_utc)}) length mismatch"
                )
        return self

    def as_timestep_list(self) -> tuple[datetime, list[dict[str, float]]]:
        """Return (as-of timestamp, list of feature dicts)."""
        if self.timestamp_utc is not None:
            assert isinstance(self.features, dict)
            return self.timestamp_utc, [self.features]
        assert self.timestamps_utc is not None
        assert isinstance(self.features, list)
        return self.timestamps_utc[-1], self.features


class PredictResponse(BaseModel):
    """Six named irradiance forecasts plus the as-of timestamp.

    All values in W/m^2. Field naming matches the loader's output keys:
    `{target}_{horizon_label}`.
    """

    forecast_origin_utc: datetime
    model_version: str
    model_type: str
    ghi_15min: float
    ghi_1h: float
    dni_15min: float
    dni_1h: float
    dhi_15min: float
    dhi_1h: float


class HealthzResponse(BaseModel):
    status: str
    model_type: str | None = None
    model_version: str | None = None
    git_commit: str | None = None
    dvc_hash: str | None = None


# --------------------------------------------------------------------- lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    params_path = Path(os.environ.get("SOLAR_PARAMS_PATH", "params.yaml"))
    params = yaml.safe_load(params_path.read_text())
    serving_cfg = params["serving"]
    model_name = str(serving_cfg["model_name"])
    model_stage = str(serving_cfg["model_stage"])

    LOG.info("loading model %s @ %s from MLflow", model_name, model_stage)
    try:
        handle = load_production_model(model_name, stage=model_stage)
    except LoaderError as exc:
        # Surfacing the failure here means lifespan raises -> uvicorn exits.
        LOG.error("model load failed: %s", exc)
        raise

    app.state.handle = handle
    app.state.ready = True

    # Pre-declare Prometheus series so the first scrape sees them.
    M.declare_endpoint_series()
    M.declare_output_series(handle.info.output_columns)
    M.declare_feature_series()
    M.set_model_info(
        handle.info.model_type,
        handle.info.version,
        handle.info.git_commit,
        handle.info.dvc_hash,
    )
    LOG.info(
        "model ready: type=%s version=%s git=%s",
        handle.info.model_type,
        handle.info.version,
        handle.info.git_commit[:8],
    )

    yield

    # Shutdown — nothing to clean up; let the process exit.
    app.state.ready = False


# --------------------------------------------------------------------- app


def create_app() -> FastAPI:
    app = FastAPI(title="solar-forecaster", lifespan=lifespan)
    app.state.ready = False
    app.state.handle = None

    # /metrics as an ASGI sub-app. Mount BEFORE routes so it has its own scope.
    app.mount("/metrics", make_asgi_app())

    @app.middleware("http")
    async def _instrument(request: Request, call_next):
        # /metrics is a sub-app — don't double-count it.
        if request.url.path.startswith("/metrics"):
            return await call_next(request)
        endpoint = (
            "predict"
            if request.url.path == "/predict"
            else ("healthz" if request.url.path == "/healthz" else "other")
        )
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            M.predict_requests.labels(endpoint=endpoint, status_code="500").inc()
            M.predict_latency.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
            raise
        M.predict_requests.labels(endpoint=endpoint, status_code=status).inc()
        M.predict_latency.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
        return response

    @app.get("/healthz", response_model=HealthzResponse)
    async def healthz(request: Request) -> Any:
        if not getattr(request.app.state, "ready", False) or request.app.state.handle is None:
            return JSONResponse(
                status_code=503,
                content=HealthzResponse(status="loading").model_dump(),
            )
        info: ModelInfo = request.app.state.handle.info
        return HealthzResponse(
            status="ok",
            model_type=info.model_type,
            model_version=info.version,
            git_commit=info.git_commit,
            dvc_hash=info.dvc_hash,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, body: PredictRequest) -> PredictResponse:
        handle: ModelHandle | None = getattr(request.app.state, "handle", None)
        if handle is None or not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="model not ready")

        as_of, feature_steps = body.as_timestep_list()

        try:
            preds = handle.predict(feature_steps)
        except ValueError as exc:
            # Per-feature validation, length mismatch, etc.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Instrument: prediction histograms + tracked input features.
        M.observe_prediction(preds, handle.info.output_columns)
        # Track inputs from the most recent timestep — the one closest to "now."
        M.observe_input_features(feature_steps[-1])

        # Coerce to the explicit response shape. Missing keys would be a bug
        # in the loader; KeyError here surfaces as 500 (correct).
        return PredictResponse(
            forecast_origin_utc=as_of,
            model_version=handle.info.version,
            model_type=handle.info.model_type,
            ghi_15min=preds["ghi_15min"],
            ghi_1h=preds["ghi_1h"],
            dni_15min=preds["dni_15min"],
            dni_1h=preds["dni_1h"],
            dhi_15min=preds["dhi_15min"],
            dhi_1h=preds["dhi_1h"],
        )

    return app


app = create_app()
