"""Replay client — Stage 6 / T11.

Walks forward through the 6-month replay window, POSTs each step to the
serving ``/predict`` endpoint, looks up the in-window ground truth at
``t + horizon``, and emits residual metrics that Prometheus scrapes from the
replay pod's own ``/metrics`` endpoint.

CLI flags
=========

  --endpoint    : serving base URL (e.g. http://solar-serve.solar.svc.cluster.local:80)
  --speedup     : target request rate in req/s
  --start       : ISO-8601 UTC, optional; defaults to replay window start
  --end         : ISO-8601 UTC, optional; defaults to replay window end
  --params      : path to params.yaml (default: /app/params.yaml)
  --features    : path to replay.parquet (default: /app/data/features/replay.parquet)
  --metrics-port: port for the /metrics server (default: 9090)
  --max-requests: stop after this many requests (useful for short demos)

Operational notes
=================

  * Pull-style metrics: we expose ``/metrics`` on ``--metrics-port`` via
    ``prometheus_client.start_http_server``. Prometheus scrapes us. We never
    push.
  * Per-step throttling is wall-clock, derived from ``--speedup``. If we fall
    behind, we don't try to "catch up" with bursts — we just run as fast as
    we can without overshooting the steady-state rate.
  * The loop is single-threaded and uses a persistent ``requests.Session`` so
    connection setup doesn't bound throughput. At 500 rps over loopback /
    cluster DNS this comfortably saturates a single uvicorn worker.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from prometheus_client import start_http_server

from src.replay import metrics as M
from src.replay.features import ReplaySource, enrich_for_persistence

LOG = logging.getLogger("replay.client")

DEFAULT_PARAMS_PATH = "/app/params.yaml"
DEFAULT_FEATURES_PATH = "/app/data/features/replay.parquet"


# ---------------------------------------------------------------------------
# config + CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    endpoint: str
    speedup_rps: float
    params_path: Path
    features_path: Path
    metrics_port: int
    max_requests: int | None
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    request_timeout_s: float


def _parse_args(argv: list[str] | None) -> RunConfig:
    p = argparse.ArgumentParser(description="Replay client for Stage 6")
    p.add_argument(
        "--endpoint",
        default=os.environ.get("SERVING_ENDPOINT"),
        help="Serving base URL (e.g. http://solar-serve.solar.svc.cluster.local:80)",
    )
    p.add_argument(
        "--speedup",
        type=float,
        default=None,
        help="Target request rate in req/s; default from params.yaml",
    )
    p.add_argument("--params", default=os.environ.get("SOLAR_PARAMS_PATH", DEFAULT_PARAMS_PATH))
    p.add_argument(
        "--features", default=os.environ.get("SOLAR_REPLAY_FEATURES", DEFAULT_FEATURES_PATH)
    )
    p.add_argument("--metrics-port", type=int, default=int(os.environ.get("METRICS_PORT", "9090")))
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop after this many predictions (default: walk full window)",
    )
    p.add_argument("--start", type=str, default=None, help="ISO-8601 UTC start (inclusive)")
    p.add_argument("--end", type=str, default=None, help="ISO-8601 UTC end (inclusive)")
    p.add_argument(
        "--request-timeout", type=float, default=5.0, help="Per-request HTTP timeout in seconds"
    )
    args = p.parse_args(argv)

    if not args.endpoint:
        p.error("--endpoint (or $SERVING_ENDPOINT) is required")

    return RunConfig(
        endpoint=args.endpoint.rstrip("/"),
        speedup_rps=float(args.speedup) if args.speedup is not None else float("nan"),
        params_path=Path(args.params),
        features_path=Path(args.features),
        metrics_port=int(args.metrics_port),
        max_requests=int(args.max_requests) if args.max_requests is not None else None,
        start=pd.Timestamp(args.start) if args.start else None,
        end=pd.Timestamp(args.end) if args.end else None,
        request_timeout_s=float(args.request_timeout),
    )


# ---------------------------------------------------------------------------
# /healthz probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMetadata:
    model_type: str
    model_version: str
    git_commit: str
    dvc_hash: str


def probe_model(session: requests.Session, endpoint: str, timeout_s: float = 30.0) -> ModelMetadata:
    """Poll ``/healthz`` until the serving pod reports ``status == 'ok'``.

    Done-when criteria require zero errors in either pod's logs, so we don't
    fire a single /predict before the model is loaded.
    """
    deadline = time.monotonic() + timeout_s
    last_body: dict[str, Any] | None = None
    last_status: int | None = None
    while time.monotonic() < deadline:
        try:
            r = session.get(f"{endpoint}/healthz", timeout=5.0)
            last_status = r.status_code
            if r.status_code == 200:
                body: dict[str, Any] = r.json()
                last_body = body
                if str(body.get("status")) == "ok":
                    return ModelMetadata(
                        model_type=str(body.get("model_type", "")),
                        model_version=str(body.get("model_version", "")),
                        git_commit=str(body.get("git_commit", "")),
                        dvc_hash=str(body.get("dvc_hash", "")),
                    )
        except requests.RequestException as exc:
            LOG.info("healthz not reachable yet: %s", exc)
        time.sleep(1.0)
    raise RuntimeError(
        f"serving did not become ready within {timeout_s:.0f}s "
        f"(last status={last_status}, body={last_body})"
    )


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------


def _build_payload(
    source: ReplaySource,
    t: pd.Timestamp,
    model_type: str,
    sequence_length: int,
    site: dict[str, object],
    horizons_steps: tuple[int, ...],
    horizon_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Construct the JSON body for a /predict request at simulated time ``t``."""
    if model_type == "lstm":
        ts_list = source.feature_timestamps(t, sequence_length)
        feature_seq = source.feature_sequence(t, sequence_length)
        return {
            "timestamps_utc": [_iso(ts) for ts in ts_list],
            "features": feature_seq,
        }

    feature_row = source.feature_payload(t)
    if model_type == "persistence":
        feature_row = enrich_for_persistence(
            feature_row,
            t,
            site,
            horizons_steps,
            horizon_labels,
        )
    return {
        "timestamp_utc": _iso(t),
        "features": feature_row,
    }


def _iso(ts: pd.Timestamp) -> str:
    """ISO-8601 UTC string with 'Z' suffix (matches the FastAPI input parser)."""
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def _classify_failure(exc: Exception) -> str:
    if isinstance(exc, requests.ConnectionError):
        return "connect"
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.HTTPError):
        code = exc.response.status_code if exc.response is not None else 0
        return "http_5xx" if 500 <= code < 600 else "http_4xx"
    if isinstance(exc, ValueError | KeyError):
        return "parse"
    return "parse"


class _GracefulExit(Exception):
    """Raised when SIGTERM/SIGINT arrives so the loop unwinds cleanly."""


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):  # noqa: ANN001 — signal callback
        raise _GracefulExit(f"received signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def run(cfg: RunConfig) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    _install_signal_handlers()

    params = yaml.safe_load(cfg.params_path.read_text())
    site = dict(params["site"])
    forecast = params["forecast"]
    targets: tuple[str, ...] = tuple(forecast["targets"])
    horizons_steps: tuple[int, ...] = tuple(int(s) for s in forecast["horizons_steps"])
    horizon_labels: tuple[str, ...] = tuple(str(s) for s in forecast["horizon_labels"])
    output_columns = tuple((t, lbl) for t in targets for lbl in horizon_labels)

    replay_cfg = params.get("replay") or {}
    speedup_rps = (
        cfg.speedup_rps
        if (cfg.speedup_rps == cfg.speedup_rps)  # not NaN
        else float(replay_cfg.get("speedup_rps", 100.0))
    )
    if speedup_rps <= 0:
        raise ValueError("speedup_rps must be > 0")
    interval_s = 1.0 / speedup_rps

    sequence_length = int(
        params.get("training", {}).get("lstm", {}).get("sequence_length_steps", 1)
    )

    LOG.info("loading replay features from %s", cfg.features_path)
    features_df = pd.read_parquet(cfg.features_path)
    LOG.info("replay table: %d rows, %d columns", len(features_df), features_df.shape[1])

    source = ReplaySource(
        features_df=features_df,
        targets=targets,
        horizons_steps=horizons_steps,
        horizon_labels=horizon_labels,
    )

    # Probe serving first so the first metric we emit isn't a connect-failure.
    session = requests.Session()
    LOG.info("probing %s/healthz ...", cfg.endpoint)
    metadata = probe_model(session, cfg.endpoint)
    LOG.info(
        "model ready: type=%s version=%s git=%s",
        metadata.model_type,
        metadata.model_version,
        metadata.git_commit[:8],
    )
    if metadata.model_type == "lstm" and sequence_length < 1:
        raise RuntimeError("model_type=lstm requires training.lstm.sequence_length_steps >= 1")

    # Pre-declare series + provenance gauge, then expose /metrics.
    M.declare_series(output_columns)
    M.set_info(
        metadata.model_type,
        metadata.model_version,
        metadata.git_commit,
        metadata.dvc_hash,
    )
    LOG.info("starting metrics server on :%d", cfg.metrics_port)
    start_http_server(cfg.metrics_port)

    timestamps = source.timestamps(start=cfg.start, end=cfg.end)
    if metadata.model_type == "lstm":
        # Skip leading rows that don't have a full sequence of history yet.
        timestamps = [t for t in timestamps if _has_sequence(source, t, sequence_length)]
    total = len(timestamps)
    if cfg.max_requests is not None:
        total = min(total, cfg.max_requests)
        timestamps = timestamps[:total]
    LOG.info(
        "replay window: %s -> %s  (%d scoreable timestamps, target=%.0f rps)",
        _iso(timestamps[0]) if timestamps else "—",
        _iso(timestamps[-1]) if timestamps else "—",
        total,
        speedup_rps,
    )
    if total == 0:
        LOG.warning("no timestamps to replay; exiting")
        return 0

    return _drive_loop(
        cfg=cfg,
        session=session,
        source=source,
        timestamps=timestamps,
        total=total,
        interval_s=interval_s,
        model_type=metadata.model_type,
        sequence_length=sequence_length,
        site=site,
        targets=targets,
        horizons_steps=horizons_steps,
        horizon_labels=horizon_labels,
    )


def _has_sequence(source: ReplaySource, t: pd.Timestamp, sequence_length: int) -> bool:
    try:
        source.feature_timestamps(t, sequence_length)
        return True
    except ValueError:
        return False


def _drive_loop(
    *,
    cfg: RunConfig,
    session: requests.Session,
    source: ReplaySource,
    timestamps: Iterable[pd.Timestamp],
    total: int,
    interval_s: float,
    model_type: str,
    sequence_length: int,
    site: dict[str, object],
    targets: tuple[str, ...],
    horizons_steps: tuple[int, ...],
    horizon_labels: tuple[str, ...],
) -> int:
    predict_url = f"{cfg.endpoint}/predict"
    timeout = cfg.request_timeout_s

    sent = 0
    failed = 0
    no_truth = 0
    t_started = time.monotonic()
    next_due = time.monotonic()

    try:
        for t in timestamps:
            # Throttle: wait until our scheduled slot, but don't burst-catch-up.
            now = time.monotonic()
            if now < next_due:
                time.sleep(next_due - now)
            next_due = max(next_due + interval_s, time.monotonic())

            body = _build_payload(
                source,
                t,
                model_type,
                sequence_length,
                site,
                horizons_steps,
                horizon_labels,
            )

            req_t0 = time.perf_counter()
            try:
                resp = session.post(predict_url, json=body, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001 — we classify below
                failed += 1
                M.request_failures_total.labels(reason=_classify_failure(exc)).inc()
                if failed <= 5 or failed % 100 == 0:
                    LOG.warning("predict failed at %s: %s", _iso(t), exc)
                continue
            finally:
                M.request_latency.observe(time.perf_counter() - req_t0)

            # Score + emit residuals per (target, horizon).
            scored_any = False
            for steps, label in zip(horizons_steps, horizon_labels, strict=True):
                truths = source.truths_at(t, steps)
                if truths is None:
                    continue
                scored_any = True
                for gt in truths:
                    key = f"{gt.target}_{label}"
                    if key not in data:
                        M.request_failures_total.labels(reason="parse").inc()
                        continue
                    M.observe_residual(gt.target, label, float(data[key]), gt.value)
            if not scored_any:
                no_truth += 1
                M.request_failures_total.labels(reason="no_truth").inc()

            sent += 1
            M.simulated_clock_seconds.set(pd.Timestamp(t).timestamp())
            if total > 0:
                M.progress_ratio.set(sent / total)
            if sent % 1000 == 0:
                elapsed = time.monotonic() - t_started
                LOG.info(
                    "sent=%d/%d failed=%d  rate=%.1f rps  sim_t=%s",
                    sent,
                    total,
                    failed,
                    sent / max(elapsed, 1e-9),
                    _iso(t),
                )
    except _GracefulExit as exc:
        LOG.info("graceful exit: %s", exc)

    elapsed = time.monotonic() - t_started
    LOG.info(
        "done: sent=%d/%d failed=%d no_truth=%d elapsed=%.1fs avg_rps=%.1f",
        sent,
        total,
        failed,
        no_truth,
        elapsed,
        sent / max(elapsed, 1e-9),
    )
    # Idle a bit so Prometheus can scrape the final state before the pod exits.
    _idle_for_scrape(seconds=int(os.environ.get("SOLAR_REPLAY_LINGER_S", "30")))
    return 0


def _idle_for_scrape(seconds: int) -> None:
    if seconds <= 0:
        return
    LOG.info("lingering %ds so Prometheus can scrape final state", seconds)
    try:
        time.sleep(seconds)
    except _GracefulExit:
        pass


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
