"""Alertmanager → KFP retrain webhook receiver (T13).

Endpoints:
  POST /alert    — Alertmanager v4 webhook target. Bearer-token auth.
                   Submits a fresh KFP run for the *first* matching firing
                   alert in the payload. Re-fires within ``retrain.cooldown_minutes``
                   are debounced (200, no submit).
  GET  /healthz  — readiness/liveness probe.
  GET  /metrics  — Prometheus text exposition (counters + last-run gauge).

The receiver lives in the ``monitoring`` namespace and submits to
``ml-pipeline.kubeflow.svc.cluster.local:8888``. It uses the SAME pipeline
factory as ``pipelines/kubeflow/submit.py`` and the SAME ``solar-train:<sha>``
image — at the SHA the receiver itself was built from. Rationale: that's the
only training image guaranteed to be ``minikube image load``-ed into the
cluster when the alert fires. CI (or the human dev) is responsible for
rebuilding the receiver on each main push so the trigger always targets
the current ``main`` HEAD.

Build-time inputs (Dockerfile ARGs):
  GIT_SHA   — long sha of HEAD when the image was built
  DVC_HASH  — output of ``get_dvc_features_hash`` at build time

Both are also overridable at runtime via env vars (mainly for tests).

In-memory debounce: a single-replica receiver is fine; restarts are rare
and the alert ``for: 10m`` already filters flapping.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel

LOG = logging.getLogger("retrain.webhook")

DEFAULT_PARAMS_PATH = Path(os.environ.get("SOLAR_PARAMS_PATH", "/app/params.yaml"))


# --------------------------------------------------------------------- config


class RetrainConfig(BaseModel):
    """Resolved retrain config from params.yaml + env."""

    cooldown_seconds: float
    allowed_alerts: set[str]
    kfp_endpoint: str
    experiment: str
    image_name: str
    port: int
    git_sha: str
    dvc_hash: str
    webhook_token: str

    @classmethod
    def load(cls, params_path: Path | None = None) -> RetrainConfig:
        path = params_path or DEFAULT_PARAMS_PATH
        params = yaml.safe_load(path.read_text())
        rt = params["retrain"]

        token = os.environ.get("WEBHOOK_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "WEBHOOK_TOKEN env var is empty — refuse to start. "
                "Mount from the solar-retrain-webhook Secret."
            )
        git_sha = os.environ.get("RETRAIN_GIT_SHA", "").strip()
        if not git_sha:
            raise RuntimeError(
                "RETRAIN_GIT_SHA env var is empty — refuse to start. "
                "Set it from the Dockerfile ARG at build time."
            )
        dvc_hash = os.environ.get("RETRAIN_DVC_HASH", "").strip()
        if not dvc_hash:
            raise RuntimeError(
                "RETRAIN_DVC_HASH env var is empty — refuse to start. "
                "Set it from the Dockerfile ARG at build time."
            )

        return cls(
            cooldown_seconds=float(rt["cooldown_minutes"]) * 60.0,
            allowed_alerts=set(rt["alerts"]),
            kfp_endpoint=os.environ.get("KFP_ENDPOINT", str(rt["kfp_endpoint"])),
            experiment=str(rt["experiment"]),
            image_name=str(rt["image_name"]),
            port=int(rt.get("port", 8000)),
            git_sha=git_sha,
            dvc_hash=dvc_hash,
            webhook_token=token,
        )


# --------------------------------------------------------------------- metrics


def _build_metrics() -> dict[str, Any]:
    """One private registry per app — keeps unit tests independent."""
    registry = CollectorRegistry()
    return {
        "registry": registry,
        "runs_submitted": Counter(
            "solar_retrain_runs_submitted_total",
            "Total KFP runs submitted by the retrain webhook.",
            ["alertname"],
            registry=registry,
        ),
        "alerts_received": Counter(
            "solar_retrain_alerts_received_total",
            "Total Alertmanager webhook payloads received (after auth).",
            ["alertname", "status"],
            registry=registry,
        ),
        "alerts_debounced": Counter(
            "solar_retrain_alerts_debounced_total",
            "Alerts that landed inside the cooldown window and were skipped.",
            ["alertname"],
            registry=registry,
        ),
        "alerts_rejected": Counter(
            "solar_retrain_alerts_rejected_total",
            "Alerts dropped because alertname is not in retrain.alerts.",
            ["alertname"],
            registry=registry,
        ),
        "submit_failures": Counter(
            "solar_retrain_submit_failures_total",
            "Failures while talking to the KFP API.",
            registry=registry,
        ),
        "last_run_ts": Gauge(
            "solar_retrain_last_run_timestamp_seconds",
            "Unix timestamp of the last successfully submitted KFP run.",
            registry=registry,
        ),
    }


# --------------------------------------------------------------------- submitter


class KFPSubmitter:
    """Thin wrapper around kfp.Client so unit tests can swap it out."""

    def __init__(self, endpoint: str, experiment: str, image_name: str):
        self.endpoint = endpoint
        self.experiment = experiment
        self.image_name = image_name

    def submit(self, git_sha: str, dvc_hash: str) -> str:
        from kfp import compiler
        from kfp.client import Client

        from pipelines.kubeflow.pipeline import build_pipeline

        short_sha = git_sha[:7]
        image = f"{self.image_name}:{short_sha}"
        pipeline_fn = build_pipeline(image=image, git_sha=git_sha, dvc_hash=dvc_hash)

        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "solar-mlops-retrain.yaml"
            compiler.Compiler().compile(pipeline_fn, str(package))

            client = Client(host=self.endpoint)
            experiment = client.create_experiment(name=self.experiment)
            run_name = f"solar-mlops-retrain-{short_sha}-{int(time.time())}"
            run = client.run_pipeline(
                experiment_id=experiment.experiment_id,
                job_name=run_name,
                pipeline_package_path=str(package),
            )
            return str(run.run_id)


# --------------------------------------------------------------------- state


class _DebounceState:
    """Last-submit timestamp guard. Thread-safe — FastAPI may call us
    concurrently from threadpool workers when the handler is sync."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_submit_at: float | None = None

    def try_acquire(self, now: float, cooldown_seconds: float) -> bool:
        """Atomically check + claim a submit slot.

        Returns True iff the call should proceed; False if inside cooldown.
        """
        with self._lock:
            if self._last_submit_at is not None and now - self._last_submit_at < cooldown_seconds:
                return False
            self._last_submit_at = now
            return True

    def release(self) -> None:
        """Undo a claim — call when the downstream submit raised."""
        with self._lock:
            self._last_submit_at = None

    @property
    def last_submit_at(self) -> float | None:
        with self._lock:
            return self._last_submit_at


# --------------------------------------------------------------------- handler


def _extract_first_matching_alert(
    payload: dict[str, Any], allowed: set[str]
) -> tuple[str, str] | None:
    """Return (alertname, status) of the first firing alert we should act on.

    Alertmanager payload schema (v4):
      { "status": "firing"|"resolved", "alerts": [ {"status": ..., "labels": {"alertname": ...}}, ... ] }

    We only act on individual alerts whose status == "firing" AND whose
    alertname is in `allowed`. Resolved or unrelated alerts return None.
    """
    alerts = payload.get("alerts", [])
    if not isinstance(alerts, list):
        return None
    for a in alerts:
        if not isinstance(a, dict):
            continue
        status = a.get("status", "")
        labels = a.get("labels", {}) or {}
        alertname = labels.get("alertname", "") if isinstance(labels, dict) else ""
        if status == "firing" and alertname in allowed:
            return alertname, status
    return None


def _peek_alertname(payload: dict[str, Any]) -> str:
    """Cheap label-only extraction for telemetry on rejected/resolved payloads."""
    alerts = payload.get("alerts", []) or []
    if alerts and isinstance(alerts[0], dict):
        labels = alerts[0].get("labels", {}) or {}
        if isinstance(labels, dict):
            return str(labels.get("alertname", "")) or "unknown"
    # Group labels fallback (Alertmanager always includes these on grouped routes).
    gl = payload.get("groupLabels", {}) or {}
    if isinstance(gl, dict) and gl.get("alertname"):
        return str(gl["alertname"])
    return "unknown"


# --------------------------------------------------------------------- app


def create_app(
    config: RetrainConfig | None = None,
    submitter: KFPSubmitter | None = None,
) -> FastAPI:
    """Build the FastAPI app. Both deps are injectable for unit tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = config or RetrainConfig.load()
        sub = submitter or KFPSubmitter(
            endpoint=cfg.kfp_endpoint,
            experiment=cfg.experiment,
            image_name=cfg.image_name,
        )
        app.state.config = cfg
        app.state.submitter = sub
        app.state.debounce = _DebounceState()
        app.state.metrics = _build_metrics()
        LOG.info(
            "retrain webhook ready: git_sha=%s endpoint=%s cooldown=%.0fs alerts=%s",
            cfg.git_sha[:8],
            cfg.kfp_endpoint,
            cfg.cooldown_seconds,
            sorted(cfg.allowed_alerts),
        )
        yield

    app = FastAPI(title="solar-retrain-webhook", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz(request: Request) -> Any:
        cfg: RetrainConfig | None = getattr(request.app.state, "config", None)
        if cfg is None:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return {
            "status": "ok",
            "git_sha": cfg.git_sha,
            "dvc_hash": cfg.dvc_hash,
            "cooldown_seconds": cfg.cooldown_seconds,
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> Any:
        m = getattr(request.app.state, "metrics", None)
        if m is None:
            return JSONResponse(status_code=503, content={"status": "loading"})
        body = generate_latest(m["registry"])
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)

    @app.post("/alert")
    async def alert(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Any:
        cfg: RetrainConfig = request.app.state.config
        sub: KFPSubmitter = request.app.state.submitter
        debounce: _DebounceState = request.app.state.debounce
        m = request.app.state.metrics

        # --- auth ---
        expected = f"Bearer {cfg.webhook_token}"
        if authorization != expected:
            # Don't leak whether the header was missing vs wrong.
            raise HTTPException(status_code=401, detail="unauthorized")

        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="payload must be a json object")

        # --- filter ---
        match = _extract_first_matching_alert(payload, cfg.allowed_alerts)
        if match is None:
            # Resolved alerts, unrelated alerts, or empty payloads land here.
            alertname = _peek_alertname(payload)
            payload_status = str(payload.get("status", "unknown"))
            if alertname not in cfg.allowed_alerts:
                m["alerts_rejected"].labels(alertname=alertname).inc()
                return {
                    "status": "ignored",
                    "reason": "alertname_not_allowed",
                    "alertname": alertname,
                }
            # Allowed alertname but no firing alert inside (e.g. all resolved).
            m["alerts_received"].labels(alertname=alertname, status=payload_status).inc()
            return {"status": "ignored", "reason": "no_firing_alerts", "alertname": alertname}

        alertname, alert_status = match
        m["alerts_received"].labels(alertname=alertname, status=alert_status).inc()

        # --- debounce ---
        now = time.time()
        if not debounce.try_acquire(now, cfg.cooldown_seconds):
            m["alerts_debounced"].labels(alertname=alertname).inc()
            last = debounce.last_submit_at or now
            remaining = max(0.0, cfg.cooldown_seconds - (now - last))
            LOG.info(
                "debounced alert=%s remaining=%.0fs",
                alertname,
                remaining,
            )
            return {
                "status": "debounced",
                "reason": "cooldown",
                "alertname": alertname,
                "cooldown_remaining_seconds": remaining,
            }

        # --- submit ---
        try:
            run_id = sub.submit(git_sha=cfg.git_sha, dvc_hash=cfg.dvc_hash)
        except Exception as exc:
            # Roll back debounce so a transient KFP outage doesn't lock us out.
            debounce.release()
            m["submit_failures"].inc()
            LOG.exception("KFP submission failed")
            raise HTTPException(status_code=502, detail=f"kfp submit failed: {exc}") from exc

        m["runs_submitted"].labels(alertname=alertname).inc()
        m["last_run_ts"].set(now)
        LOG.info(
            "submitted run_id=%s alertname=%s git_sha=%s",
            run_id,
            alertname,
            cfg.git_sha[:8],
        )
        return {
            "status": "submitted",
            "alertname": alertname,
            "run_id": run_id,
            "git_sha": cfg.git_sha,
            "dvc_hash": cfg.dvc_hash,
        }

    return app


# Module-level app for uvicorn. Tests build their own via create_app(...).
def _eager_app() -> FastAPI:
    return create_app()


# uvicorn entrypoint: ``uvicorn src.retrain.webhook:app``
# The lifespan hook runs on first request — config loads at startup, not import.
app = _eager_app()
