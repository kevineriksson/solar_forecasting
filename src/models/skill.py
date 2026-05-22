"""Skill-score helpers.

Skill score on RMSE (the project-wide definition, CLAUDE.md §6 Stage 4):

    skill = 1 - RMSE_candidate / RMSE_baseline

T5 (XGBoost) and T6 (LSTM) both need the **same** per-fold persistence RMSE
to compute apples-to-apples skills against T4. Centralising the MLflow lookup
here means both trainers parse the metric keys identically.

The metric keys logged by `train_persistence.py` are:

    fold{i}.rmse.{target}.{horizon_label}     e.g.  fold0.rmse.ghi.15min
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

_FOLD_RMSE_RE = re.compile(r"^fold(?P<fold>\d+)\.rmse\.(?P<target>[a-z_]+)\.(?P<label>[^.]+)$")


@dataclass(frozen=True)
class PersistenceRunRef:
    run_id: str
    experiment_id: str
    git_commit: str
    dvc_hash: str


def find_persistence_run(
    experiment_name: str, run_name: str = "persistence_baseline"
) -> PersistenceRunRef:
    """Return the most recent successful persistence_baseline run for an experiment.

    Raises if no matching run exists or required tags are missing.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} not found")

    filter_str = (
        f"tags.mlflow.runName = '{run_name}' "
        "and tags.model_type = 'persistence' "
        "and attributes.status = 'FINISHED'"
    )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"no FINISHED MLflow run with runName={run_name!r} and model_type=persistence "
            f"in experiment {experiment_name!r}. Run T4 first."
        )

    run = runs[0]
    tags = run.data.tags
    git_commit = tags.get("git_commit", "")
    dvc_hash = tags.get("dvc_hash", "")
    if not git_commit or not dvc_hash:
        raise RuntimeError(
            f"persistence run {run.info.run_id} is missing required tags "
            f"(git_commit={git_commit!r} dvc_hash={dvc_hash!r})"
        )

    return PersistenceRunRef(
        run_id=run.info.run_id,
        experiment_id=experiment.experiment_id,
        git_commit=git_commit,
        dvc_hash=dvc_hash,
    )


def load_per_fold_rmse(run_id: str) -> dict[tuple[int, str, str], float]:
    """Return {(fold_idx, target, horizon_label): rmse} parsed from a run's metrics."""
    run = mlflow.get_run(run_id)
    out: dict[tuple[int, str, str], float] = {}
    for key, value in run.data.metrics.items():
        m = _FOLD_RMSE_RE.match(key)
        if not m:
            continue
        out[(int(m["fold"]), m["target"], m["label"])] = float(value)
    if not out:
        raise RuntimeError(
            f"persistence run {run_id} exposes no `fold*.rmse.*` metrics — "
            "did T4 actually log per-fold RMSE?"
        )
    return out


def skill_score(rmse_candidate: float, rmse_baseline: float) -> float:
    """1 - RMSE_candidate / RMSE_baseline.

    Returns -inf if baseline is exactly zero (degenerate; should not happen in
    real data since persistence has non-zero residuals at any horizon > 0).
    """
    if rmse_baseline <= 0.0:
        return float("-inf")
    return 1.0 - (rmse_candidate / rmse_baseline)
