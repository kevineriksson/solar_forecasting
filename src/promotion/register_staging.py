"""T8-minimum promotion step: pick best candidate, register as Staging.

This is the *placeholder* promotion used by the T8 Kubeflow pipeline so the
DAG terminates with a model version in the registry. T9 replaces it with the
full candidate-vs-Production-on-promotion-window logic.

Behaviour:
    1. Find the three latest FINISHED training runs in the `solar_forecaster`
       MLflow experiment that share the supplied `--git-sha` (model_type in
       {persistence, xgboost, lstm}).
    2. Pick the one with the highest `mean.skill` metric. Persistence has
       `mean.skill = 0` by construction, so it only wins if both ML models
       failed.
    3. `mlflow.register_model("runs:/<winner>", name="solar_forecaster")` and
       transition that new version to `Staging`.
    4. Print the winner + new version to stdout.

Usage (inside KFP container):
    python -m src.promotion.register_staging --git-sha <sha>
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.models.mlflow_utils import resolve_tracking_uri

LOG = logging.getLogger("promotion")

EXPERIMENT_NAME = "solar_forecaster"
MODEL_NAME = "solar_forecaster"
CANDIDATE_MODEL_TYPES = ("persistence", "xgboost", "lstm")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tracking_uri = resolve_tracking_uri()
    LOG.info("MLflow tracking URI: %s", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT_NAME!r} not found")

    # Pull all three candidate runs for this git_sha.
    filter_str = f"tags.git_commit = '{args.git_sha}' " "and attributes.status = 'FINISHED'"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    if not runs:
        raise RuntimeError(
            f"no FINISHED runs in {EXPERIMENT_NAME!r} with git_commit={args.git_sha}"
        )

    # Keep the most recent run per model_type — guards against re-submits.
    latest_by_type: dict[str, mlflow.entities.Run] = {}
    for run in runs:
        mt = run.data.tags.get("model_type", "")
        if mt in CANDIDATE_MODEL_TYPES and mt not in latest_by_type:
            latest_by_type[mt] = run

    missing = [mt for mt in CANDIDATE_MODEL_TYPES if mt not in latest_by_type]
    if missing:
        raise RuntimeError(
            f"missing required model_type(s) for git_commit={args.git_sha}: {missing}. "
            f"Found types: {sorted(latest_by_type)}"
        )

    LOG.info("candidate runs for git=%s:", args.git_sha[:8])
    scored: list[tuple[str, float, mlflow.entities.Run]] = []
    for mt, run in latest_by_type.items():
        skill = float(run.data.metrics.get("mean.skill", 0.0))
        LOG.info("  %-11s run_id=%s  mean.skill=%+.4f", mt, run.info.run_id, skill)
        scored.append((mt, skill, run))

    winner_type, winner_skill, winner_run = max(scored, key=lambda x: x[1])
    LOG.info(
        "winner: %s  run_id=%s  mean.skill=%+.4f", winner_type, winner_run.info.run_id, winner_skill
    )

    # Register and transition. We register the entire run as the model artifact;
    # T9 swaps this for a proper logged-model URI once trainers log models.
    model_uri = f"runs:/{winner_run.info.run_id}"
    LOG.info("registering %s as %s", model_uri, MODEL_NAME)
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    LOG.info("registered %s version %s", MODEL_NAME, mv.version)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    # Tag the version so T9 / dashboards can trace it back to the source run.
    client.set_model_version_tag(MODEL_NAME, mv.version, "git_commit", args.git_sha)
    client.set_model_version_tag(MODEL_NAME, mv.version, "model_type", winner_type)
    client.set_model_version_tag(MODEL_NAME, mv.version, "source_run_id", winner_run.info.run_id)
    client.set_model_version_tag(MODEL_NAME, mv.version, "mean_skill", f"{winner_skill:.6f}")

    print()
    print("=" * 78)
    print(f"promoted to Staging: {MODEL_NAME} v{mv.version}")
    print(f"  model_type   : {winner_type}")
    print(f"  source_run_id: {winner_run.info.run_id}")
    print(f"  mean.skill   : {winner_skill:+.4f}")
    print(f"  git_commit   : {args.git_sha}")
    print("=" * 78)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T8-minimum promotion: register Staging")
    parser.add_argument(
        "--git-sha",
        required=True,
        help="git commit shared by all three candidate runs (selector + provenance tag)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
