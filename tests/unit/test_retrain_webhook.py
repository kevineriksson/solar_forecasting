"""Unit tests for the T13 retrain webhook receiver.

Covers the four invariants from the T13 prompt:

  1. Auth: missing / wrong bearer token → 401.
  2. Alert filtering: alerts outside `retrain.alerts` are ignored; resolved
     alerts don't trigger a submit.
  3. Debounce: a second firing inside the cooldown returns "debounced" and
     does NOT call the submitter.
  4. Submit path: a fresh firing alert calls the submitter exactly once,
     passes the configured git_sha + dvc_hash, and returns 200 with the
     run_id.

The submitter is swapped for a recording stub so no KFP / network is
touched. Cooldown is set to large/small values per test to exercise both
sides of the debounce.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.retrain.webhook import RetrainConfig, _DebounceState, create_app


class RecordingSubmitter:
    """Stand-in for KFPSubmitter — records calls, returns deterministic ids.

    `fail` toggles an exception on the next call to verify the route maps
    KFP failures to HTTP 502 and rolls back the debounce slot.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self.fail = False
        self._lock = threading.Lock()
        self._n = 0

    def submit(self, git_sha: str, dvc_hash: str) -> str:
        with self._lock:
            self._n += 1
            self.calls.append({"git_sha": git_sha, "dvc_hash": dvc_hash})
            if self.fail:
                raise RuntimeError("kfp blew up")
            return f"run-{self._n}"


TOKEN = "test-token-abc"


def _config(cooldown_seconds: float = 1800.0) -> RetrainConfig:
    return RetrainConfig(
        cooldown_seconds=cooldown_seconds,
        allowed_alerts={"SolarDriftHigh", "SolarSkillScoreLow"},
        kfp_endpoint="http://kfp.invalid",
        experiment="solar-mlops-retrain",
        image_name="solar-train",
        port=8000,
        git_sha="deadbeefcafef00d" * 2 + "00000000",  # 40 chars
        dvc_hash="abc123",
        webhook_token=TOKEN,
    )


@pytest.fixture()
def submitter() -> RecordingSubmitter:
    return RecordingSubmitter()


@pytest.fixture()
def client(submitter: RecordingSubmitter) -> Iterator[TestClient]:
    app = create_app(config=_config(), submitter=submitter)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------- shared payloads


def _firing(alertname: str = "SolarDriftHigh") -> dict:
    return {
        "status": "firing",
        "groupLabels": {"alertname": alertname},
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": alertname, "severity": "warning"},
            }
        ],
    }


def _resolved(alertname: str = "SolarDriftHigh") -> dict:
    return {
        "status": "resolved",
        "groupLabels": {"alertname": alertname},
        "alerts": [
            {
                "status": "resolved",
                "labels": {"alertname": alertname, "severity": "warning"},
            }
        ],
    }


# ---------------------------------------------------------- auth


def test_alert_rejects_missing_token(client: TestClient, submitter: RecordingSubmitter) -> None:
    r = client.post("/alert", json=_firing())
    assert r.status_code == 401
    assert submitter.calls == []


def test_alert_rejects_wrong_token(client: TestClient, submitter: RecordingSubmitter) -> None:
    r = client.post(
        "/alert",
        json=_firing(),
        headers={"Authorization": "Bearer not-the-real-token"},
    )
    assert r.status_code == 401
    assert submitter.calls == []


def test_alert_accepts_correct_token(client: TestClient, submitter: RecordingSubmitter) -> None:
    r = client.post(
        "/alert",
        json=_firing(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "submitted"
    assert len(submitter.calls) == 1


# ---------------------------------------------------------- alert filtering


def test_alert_ignores_unknown_alertname(client: TestClient, submitter: RecordingSubmitter) -> None:
    r = client.post(
        "/alert",
        json=_firing(alertname="KubePodCrashLooping"),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ignored"
    assert body["reason"] == "alertname_not_allowed"
    assert submitter.calls == []


def test_alert_ignores_resolved_status(client: TestClient, submitter: RecordingSubmitter) -> None:
    r = client.post(
        "/alert",
        json=_resolved(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ignored"
    assert body["reason"] == "no_firing_alerts"
    assert submitter.calls == []


def test_alert_skill_score_low_also_fires(
    client: TestClient, submitter: RecordingSubmitter
) -> None:
    r = client.post(
        "/alert",
        json=_firing(alertname="SolarSkillScoreLow"),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["alertname"] == "SolarSkillScoreLow"
    assert len(submitter.calls) == 1


def test_alert_rejects_non_json_body(client: TestClient, submitter: RecordingSubmitter) -> None:
    r = client.post(
        "/alert",
        content=b"not json",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 400
    assert submitter.calls == []


# ---------------------------------------------------------- debounce


def test_alert_debounces_second_firing_inside_cooldown(
    client: TestClient, submitter: RecordingSubmitter
) -> None:
    # Long cooldown — second call must always be debounced.
    headers = {"Authorization": f"Bearer {TOKEN}"}
    first = client.post("/alert", json=_firing(), headers=headers)
    second = client.post("/alert", json=_firing(), headers=headers)

    assert first.status_code == 200
    assert first.json()["status"] == "submitted"

    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "debounced"
    assert body["reason"] == "cooldown"
    assert body["alertname"] == "SolarDriftHigh"
    assert body["cooldown_remaining_seconds"] > 0

    # Only one submission happened despite two firings.
    assert len(submitter.calls) == 1


def test_alert_allows_resubmit_after_cooldown_expires(
    submitter: RecordingSubmitter,
) -> None:
    # Negative cooldown → every call is outside it.
    app = create_app(config=_config(cooldown_seconds=-1.0), submitter=submitter)
    with TestClient(app) as c:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        a = c.post("/alert", json=_firing(), headers=headers)
        b = c.post("/alert", json=_firing(), headers=headers)
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["status"] == "submitted"
    assert b.json()["status"] == "submitted"
    assert len(submitter.calls) == 2


def test_alert_submit_failure_rolls_back_debounce(
    submitter: RecordingSubmitter,
) -> None:
    # Cooldown is long; but the first (failing) call must NOT eat the slot.
    submitter.fail = True
    app = create_app(config=_config(cooldown_seconds=3600.0), submitter=submitter)
    with TestClient(app) as c:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        bad = c.post("/alert", json=_firing(), headers=headers)
        # KFP recovers — second call should succeed, not be debounced.
        submitter.fail = False
        good = c.post("/alert", json=_firing(), headers=headers)

    assert bad.status_code == 502
    assert good.status_code == 200
    assert good.json()["status"] == "submitted"
    assert len(submitter.calls) == 2  # 1 failed + 1 success


# ---------------------------------------------------------- submit path


def test_alert_passes_configured_sha_and_dvc_hash(
    client: TestClient, submitter: RecordingSubmitter
) -> None:
    cfg = _config()
    r = client.post(
        "/alert",
        json=_firing(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["git_sha"] == cfg.git_sha
    assert body["dvc_hash"] == cfg.dvc_hash
    assert body["run_id"] == "run-1"
    assert submitter.calls == [{"git_sha": cfg.git_sha, "dvc_hash": cfg.dvc_hash}]


# ---------------------------------------------------------- healthz / metrics


def test_healthz_returns_config_summary(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["git_sha"]
    assert body["dvc_hash"]
    assert body["cooldown_seconds"] == 1800.0


def test_metrics_exposes_submission_counters(
    client: TestClient, submitter: RecordingSubmitter
) -> None:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    # Fire one allowed + one disallowed + one debounce.
    client.post("/alert", json=_firing(), headers=headers)
    client.post("/alert", json=_firing(alertname="ThirdPartyAlert"), headers=headers)
    client.post("/alert", json=_firing(), headers=headers)  # debounced

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert 'solar_retrain_runs_submitted_total{alertname="SolarDriftHigh"} 1.0' in body
    assert 'solar_retrain_alerts_debounced_total{alertname="SolarDriftHigh"} 1.0' in body
    assert 'solar_retrain_alerts_rejected_total{alertname="ThirdPartyAlert"} 1.0' in body
    # last-run gauge set, non-zero.
    assert "solar_retrain_last_run_timestamp_seconds" in body


# ---------------------------------------------------------- _DebounceState unit


def test_debounce_state_atomic_under_concurrency() -> None:
    """Two concurrent try_acquire calls must yield exactly one True."""
    state = _DebounceState()
    cooldown = 60.0
    now = 1000.0

    results: list[bool] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        results.append(state.try_acquire(now, cooldown))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [False, True]
