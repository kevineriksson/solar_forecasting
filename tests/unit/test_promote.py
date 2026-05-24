"""Unit tests for src/promotion/promote.py.

Covers the three cases the T9 prompt calls out:

  1. A "worse" candidate (constant-prediction → strongly negative skill on
     promo) is rejected when a competent Production model is already in
     place.

  2. A "better" candidate (positive promo skill) is promoted when the
     current Production model is the constant-prediction one.

  3. The training-window leakage guardrail rejects a candidate whose
     `train_window_end` falls after the promo split's start timestamp.

Each test stands up a temporary file-based MLflow tracking URI and registry,
fabricates the three candidate runs (persistence + xgboost + lstm) with the
required tags/params/metrics, and calls `promote()` directly. This isolates
the promotion logic from the real MLflow server and from actual model
training — we are testing the decision, not the math behind `promo.mean_skill`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.promotion.promote import (
    EXPERIMENT_NAME,
    MODEL_NAME,
    GuardrailError,
    promote,
)

GIT_SHA = "a" * 40
DVC_HASH = "b" * 40
MARGIN = 0.02

# Time anchors mirror data/interim/splits.json shape: train ends before promo,
# promo ends before replay. Strings are ISO UTC with the Z suffix the trainers
# write.
TRAIN_END_OK = "2025-09-09T23:45:00Z"  # < promo.start
PROMO_START = "2025-09-10T00:00:00Z"
REPLAY_START = "2025-11-10T00:00:00Z"
DATA_END = "2026-05-10T00:00:00Z"


@pytest.fixture
def mlflow_local(tmp_path: Path):
    """Spin up MLflow with file backends rooted at tmp_path.

    Both tracking and registry use sqlite (registry can't use file:// URIs).
    """
    tracking_db = tmp_path / "mlflow.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    uri = f"sqlite:///{tracking_db}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    # Pre-create the experiment with an explicit artifact root inside tmp_path,
    # and select it as active so subsequent start_run() calls log there
    # (otherwise they hit the Default experiment and promote() can't find them).
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=str(artifacts))
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    yield client


@pytest.fixture
def splits_manifest(tmp_path: Path) -> Path:
    """Write a minimal splits.json with realistic boundaries."""
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "reference_now": DATA_END,
                "data_first_ts": "2007-01-01T00:15:00Z",
                "data_last_ts": DATA_END,
                "splits": {
                    "train": {
                        "start": "2007-01-01T00:15:00Z",
                        "end": PROMO_START,
                        "inclusive": "[)",
                        "n_rows": 600000,
                    },
                    "promo": {
                        "start": PROMO_START,
                        "end": REPLAY_START,
                        "inclusive": "[)",
                        "n_rows": 5856,
                    },
                    "replay": {
                        "start": REPLAY_START,
                        "end": DATA_END,
                        "inclusive": "[]",
                        "n_rows": 17377,
                    },
                },
            }
        )
    )
    return path


def _make_candidate_run(
    *,
    model_type: str,
    promo_mean_skill: float,
    git_sha: str = GIT_SHA,
    dvc_hash: str = DVC_HASH,
    train_window_end: str = TRAIN_END_OK,
) -> str:
    """Create one fake training run with the tags/params/metrics promote() expects."""
    run_name = {
        "persistence": "persistence_baseline",
        "xgboost": "xgboost_candidate",
        "lstm": "lstm_candidate",
    }[model_type]
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "git_commit": git_sha,
                "dvc_hash": dvc_hash,
                "model_type": model_type,
            }
        )
        mlflow.log_params(
            {
                "train_window_start": "2007-01-01T00:15:00Z",
                "train_window_end": train_window_end,
                "promo_window_start": PROMO_START,
                "promo_window_end": REPLAY_START,
            }
        )
        mlflow.log_metric("promo.mean_skill", promo_mean_skill)
        return run.info.run_id


def _make_three_candidates(
    *,
    persistence_skill: float = 0.0,
    xgb_skill: float,
    lstm_skill: float,
    train_window_end: str = TRAIN_END_OK,
) -> dict[str, str]:
    return {
        "persistence": _make_candidate_run(
            model_type="persistence",
            promo_mean_skill=persistence_skill,
            train_window_end=train_window_end,
        ),
        "xgboost": _make_candidate_run(
            model_type="xgboost",
            promo_mean_skill=xgb_skill,
            train_window_end=train_window_end,
        ),
        "lstm": _make_candidate_run(
            model_type="lstm",
            promo_mean_skill=lstm_skill,
            train_window_end=train_window_end,
        ),
    }


def _seed_production(client: MlflowClient, run_id: str) -> str:
    """Register a run as a model version and transition it to Production.

    Mirrors promote._register_and_transition exactly (artifact_uri as source)
    so the test's seeding path uses the same registry semantics as the code
    under test.
    """
    try:
        client.create_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException as exc:
        if "RESOURCE_ALREADY_EXISTS" not in str(exc) and "already exists" not in str(exc).lower():
            raise
    run = client.get_run(run_id)
    mv = client.create_model_version(name=MODEL_NAME, source=run.info.artifact_uri, run_id=run_id)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=False,
    )
    return mv.version


# --------------------------------------------------------------------- tests


def test_worse_candidate_rejected(mlflow_local: MlflowClient, splits_manifest: Path):
    """Constant-prediction-style candidate (skill << 0) loses to a competent Production."""
    # Seed an existing Production with promo.mean_skill = 0.30 (decent XGBoost).
    # Its tags don't need git_sha matching — promote() only looks at the
    # candidate runs for that, and reads Production via the model version.
    prod_run_id = _make_candidate_run(
        model_type="xgboost",
        promo_mean_skill=0.30,
        git_sha="9" * 40,  # different git_sha so it's not picked as a candidate
    )
    _seed_production(mlflow_local, prod_run_id)

    # Candidate set: persistence (0.0 by construction) + two losers far below margin.
    # The "winner" among these three by promo.mean_skill is persistence at 0.0,
    # but 0.0 vs 0.30 fails the margin check.
    _make_three_candidates(xgb_skill=-1.5, lstm_skill=-2.0)

    decision = promote(
        client=mlflow_local,
        git_sha=GIT_SHA,
        margin=MARGIN,
        splits_manifest_path=splits_manifest,
    )

    assert decision.decision == "reject_skill"
    assert decision.candidate_promo_skill == pytest.approx(
        0.0
    )  # persistence won the candidate pick
    assert decision.production_promo_skill == pytest.approx(0.30)
    # Production version unchanged.
    prod_after = mlflow_local.get_latest_versions(MODEL_NAME, stages=["Production"])
    assert len(prod_after) == 1
    assert prod_after[0].run_id == prod_run_id


def test_better_candidate_promoted(mlflow_local: MlflowClient, splits_manifest: Path):
    """Real-XGBoost-style candidate (positive skill) beats constant-prediction Production."""
    # Seed Production as a constant-prediction model: promo.mean_skill = -0.80.
    bad_prod_run = _make_candidate_run(
        model_type="xgboost",
        promo_mean_skill=-0.80,
        git_sha="9" * 40,
    )
    bad_prod_version = _seed_production(mlflow_local, bad_prod_run)

    # Candidate set: real XGBoost wins at 0.25 (matches what xgb_train logs on
    # the promo window after T9 changes). Persistence at 0.0; LSTM at 0.10.
    candidates = _make_three_candidates(xgb_skill=0.25, lstm_skill=0.10)

    decision = promote(
        client=mlflow_local,
        git_sha=GIT_SHA,
        margin=MARGIN,
        splits_manifest_path=splits_manifest,
    )

    assert decision.decision == "promote"
    assert decision.candidate_model_type == "xgboost"
    assert decision.candidate_run_id == candidates["xgboost"]
    assert decision.candidate_promo_skill == pytest.approx(0.25)
    assert decision.production_promo_skill == pytest.approx(-0.80)
    assert decision.new_version is not None

    # Registry state: new version is Production; old version was Archived.
    prod_now = mlflow_local.get_latest_versions(MODEL_NAME, stages=["Production"])
    assert len(prod_now) == 1
    assert prod_now[0].version == decision.new_version
    assert prod_now[0].run_id == candidates["xgboost"]

    archived = mlflow_local.get_latest_versions(MODEL_NAME, stages=["Archived"])
    assert any(v.version == bad_prod_version for v in archived), (
        f"old Production {bad_prod_version} was not archived; "
        f"current archived versions: {[v.version for v in archived]}"
    )


def test_guardrail_rejects_overlap(mlflow_local: MlflowClient, splits_manifest: Path):
    """Candidate whose train_window_end falls inside the promo window must be rejected."""
    # No Production needed — guardrail fires before the comparison.
    # train_window_end > promo.start triggers the leakage guardrail.
    bad_end = "2025-10-01T00:00:00Z"  # well inside the promo window
    _make_three_candidates(
        xgb_skill=0.50,  # would otherwise win, but guardrail must reject
        lstm_skill=0.40,
        train_window_end=bad_end,
    )

    with pytest.raises(GuardrailError) as exc_info:
        promote(
            client=mlflow_local,
            git_sha=GIT_SHA,
            margin=MARGIN,
            splits_manifest_path=splits_manifest,
        )

    assert "promo.start" in str(exc_info.value)
    # Nothing should have been registered. get_latest_versions raises
    # MlflowException("not found") when the registered model itself was
    # never created — exactly what we want post-guardrail.
    try:
        prod = mlflow_local.get_latest_versions(MODEL_NAME, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        prod = []
    assert prod == [], f"guardrail violation should not register a model: {prod}"
