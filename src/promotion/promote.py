"""T9: candidate-vs-Production promotion on the promotion validation window.

Replaces the T8 `register_staging.py` placeholder with the real Stage 4 logic.

Decision flow:
    1. Identify the candidate run: among the three FINISHED training runs
       (persistence, xgboost, lstm) tagged with --git-sha, pick the one with
       the highest `promo.mean_skill` metric. That's the per-pipeline-run
       winner, scored on the *promotion validation window*, not on CV.

    2. Guardrail: refuse to promote if the candidate's training data overlaps
       the promo or replay windows. Spelled out below in `_check_no_leakage`.

    3. Score candidate vs. current `Production`:
         skill_score = 1 - RMSE_model / RMSE_persistence    per (target, horizon)
         mean_skill  = mean across the 6 outputs
       Both are read from `promo.mean_skill` already logged by the trainers
       (see T9 changes in src/models/{train_persistence,xgb_train,lstm_train}.py).

    4. Promote iff:
         - no current Production exists (first-run case), OR
         - candidate.promo.mean_skill - production.promo.mean_skill >= margin
       (margin from params.yaml -> promotion.margin).

    5. On promotion: register the candidate's run as a new version of the
       `solar_forecaster` MLflow model, transition that version to
       `Production`, and archive the previous Production version (if any).

Usage (inside KFP container):
    python -m src.promotion.promote --git-sha <sha> --params /app/params.yaml

Exit codes:
    0 — clean decision (promote OR reject for skill reasons).
    1 — guardrail violation, missing prerequisite, or unrecoverable error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src.models.mlflow_utils import resolve_tracking_uri

LOG = logging.getLogger("promotion")

EXPERIMENT_NAME = "solar_forecaster"
MODEL_NAME = "solar_forecaster"
CANDIDATE_MODEL_TYPES = ("persistence", "xgboost", "lstm")


class GuardrailError(RuntimeError):
    """Candidate training data overlaps the promo or replay window."""


@dataclass(frozen=True)
class PromotionDecision:
    decision: str  # "promote" | "reject_skill" | "reject_no_candidate"
    candidate_run_id: str
    candidate_model_type: str
    candidate_promo_skill: float
    production_run_id: str | None
    production_promo_skill: float | None
    margin: float
    new_version: str | None = None
    archived_versions: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(args.params).resolve().parent
    params = yaml.safe_load(Path(args.params).read_text())
    margin = float(params["promotion"]["margin"])
    splits_path = repo_root / params["paths"]["splits_manifest"]

    tracking_uri = resolve_tracking_uri()
    LOG.info("MLflow tracking URI: %s", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        decision = promote(
            client=client,
            git_sha=args.git_sha,
            margin=margin,
            splits_manifest_path=splits_path,
            experiment_name=EXPERIMENT_NAME,
            model_name=MODEL_NAME,
        )
    except GuardrailError as exc:
        LOG.error("guardrail violation: %s", exc)
        return 1

    _print_summary(decision)
    return 0


def promote(
    *,
    client: MlflowClient,
    git_sha: str,
    margin: float,
    splits_manifest_path: Path,
    experiment_name: str = EXPERIMENT_NAME,
    model_name: str = MODEL_NAME,
) -> PromotionDecision:
    """Run the promotion decision and return a structured result.

    Tests call this directly; `main()` wraps it for CLI use. Raises
    `GuardrailError` only on leakage; all other "reject" outcomes are
    returned as a `PromotionDecision`.
    """
    candidate_run, candidate_type = _select_candidate(client, experiment_name, git_sha)
    LOG.info(
        "candidate: type=%s run_id=%s promo.mean_skill=%+.4f",
        candidate_type,
        candidate_run.info.run_id,
        float(candidate_run.data.metrics["promo.mean_skill"]),
    )

    splits = _load_splits(splits_manifest_path)
    _check_no_leakage(candidate_run, splits)

    candidate_skill = float(candidate_run.data.metrics["promo.mean_skill"])

    prod_version, prod_skill, prod_run_id = _current_production(client, model_name)
    if prod_version is None:
        LOG.info("no current Production model — promoting candidate by default")
        new_version, archived = _register_and_transition(
            client, candidate_run.info.run_id, model_name
        )
        _tag_version(
            client, model_name, new_version, candidate_run, candidate_type, candidate_skill
        )
        return PromotionDecision(
            decision="promote",
            candidate_run_id=candidate_run.info.run_id,
            candidate_model_type=candidate_type,
            candidate_promo_skill=candidate_skill,
            production_run_id=None,
            production_promo_skill=None,
            margin=margin,
            new_version=new_version,
            archived_versions=archived,
            notes="no prior Production version existed",
        )

    # _current_production guarantees prod_skill is not None whenever prod_version
    # is not None; assert it here so mypy can narrow the type for the arithmetic.
    assert prod_skill is not None
    LOG.info(
        "current Production: version=%s run_id=%s promo.mean_skill=%+.4f",
        prod_version,
        prod_run_id,
        prod_skill,
    )
    delta = candidate_skill - prod_skill
    LOG.info("delta = candidate - production = %+.4f  (margin=%.4f)", delta, margin)

    if delta >= margin:
        new_version, archived = _register_and_transition(
            client, candidate_run.info.run_id, model_name
        )
        _tag_version(
            client, model_name, new_version, candidate_run, candidate_type, candidate_skill
        )
        return PromotionDecision(
            decision="promote",
            candidate_run_id=candidate_run.info.run_id,
            candidate_model_type=candidate_type,
            candidate_promo_skill=candidate_skill,
            production_run_id=prod_run_id,
            production_promo_skill=prod_skill,
            margin=margin,
            new_version=new_version,
            archived_versions=archived,
            notes=f"delta {delta:+.4f} >= margin {margin}",
        )

    return PromotionDecision(
        decision="reject_skill",
        candidate_run_id=candidate_run.info.run_id,
        candidate_model_type=candidate_type,
        candidate_promo_skill=candidate_skill,
        production_run_id=prod_run_id,
        production_promo_skill=prod_skill,
        margin=margin,
        notes=f"delta {delta:+.4f} < margin {margin}; Production retained",
    )


# --------------------------------------------------------------------- internals


def _select_candidate(
    client: MlflowClient, experiment_name: str, git_sha: str
) -> tuple[mlflow.entities.Run, str]:
    """Return (winning_run, model_type) — highest promo.mean_skill for this git_sha."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} not found")

    filter_str = f"tags.git_commit = '{git_sha}' and attributes.status = 'FINISHED'"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    if not runs:
        raise RuntimeError(f"no FINISHED runs in {experiment_name!r} with git_commit={git_sha}")

    latest_by_type: dict[str, mlflow.entities.Run] = {}
    for run in runs:
        mt = run.data.tags.get("model_type", "")
        if mt in CANDIDATE_MODEL_TYPES and mt not in latest_by_type:
            latest_by_type[mt] = run

    missing = [mt for mt in CANDIDATE_MODEL_TYPES if mt not in latest_by_type]
    if missing:
        raise RuntimeError(
            f"missing required model_type(s) for git_commit={git_sha}: {missing}. "
            f"Found types: {sorted(latest_by_type)}"
        )

    scored: list[tuple[str, float, mlflow.entities.Run]] = []
    for mt, run in latest_by_type.items():
        if "promo.mean_skill" not in run.data.metrics:
            raise RuntimeError(
                f"{mt} run {run.info.run_id} is missing 'promo.mean_skill' — "
                "re-run T4/T5/T6 with T9 changes applied"
            )
        skill = float(run.data.metrics["promo.mean_skill"])
        LOG.info(
            "  candidate %-11s run_id=%s promo.mean_skill=%+.4f",
            mt,
            run.info.run_id,
            skill,
        )
        scored.append((mt, skill, run))

    winner_type, _winner_skill, winner_run = max(scored, key=lambda x: x[1])
    return winner_run, winner_type


def _load_splits(path: Path) -> dict:
    """Parse splits.json; raise if the manifest is missing or malformed."""
    if not path.exists():
        raise RuntimeError(f"splits manifest not found at {path}")
    data = json.loads(path.read_text())
    for key in ("promo", "replay"):
        if key not in data.get("splits", {}):
            raise RuntimeError(f"splits manifest missing 'splits.{key}'")
    return data


def _check_no_leakage(run: mlflow.entities.Run, splits: dict) -> None:
    """Refuse promotion if candidate's training data overlaps promo or replay.

    Mechanics:
      - The candidate's training window is logged at training time as two
        params: `train_window_start` and `train_window_end` (ISO UTC strings,
        equal to the first/last timestamps in `data/features/train.parquet`).
      - splits.json (Stage 1 output) gives the inclusive-exclusive timestamps
        of the promo and replay windows for *this* pipeline run.
      - Leakage = candidate.train_window_end > splits.promo.start  OR
                  candidate.train_window_end > splits.replay.start.
        (Either condition means the trainer saw rows it shouldn't have.)
      - Additionally: candidate.dvc_hash must equal the dvc_hash recorded on
        the candidate run (trivially true) — we surface it so cross-pipeline
        candidates can be matched against the splits.json that was current
        when they were trained. We do NOT require equality with any global
        dvc_hash here because the KFP pipeline pins all candidates and the
        promote step to the same image, and therefore the same hashes.

    Raises `GuardrailError` on violation; otherwise returns None.
    """
    params = run.data.params
    tags = run.data.tags

    start = params.get("train_window_start")
    end = params.get("train_window_end")
    if not start or not end:
        raise GuardrailError(
            f"candidate run {run.info.run_id} is missing train_window_start/end params"
        )
    if "dvc_hash" not in tags:
        raise GuardrailError(f"candidate run {run.info.run_id} is missing the dvc_hash tag")

    promo_start = splits["splits"]["promo"]["start"]
    replay_start = splits["splits"]["replay"]["start"]

    # Lexicographic comparison on ISO-8601 UTC ('Z') strings is correct
    # because all timestamps share format YYYY-MM-DDTHH:MM:SSZ.
    if end > promo_start:
        raise GuardrailError(
            f"candidate train_window_end ({end}) > promo.start ({promo_start}) — "
            "training data overlaps the promotion validation window"
        )
    if end > replay_start:
        raise GuardrailError(
            f"candidate train_window_end ({end}) > replay.start ({replay_start}) — "
            "training data overlaps the replay window"
        )
    LOG.info(
        "guardrail OK: train_window=[%s, %s]  promo.start=%s  replay.start=%s",
        start,
        end,
        promo_start,
        replay_start,
    )


def _current_production(
    client: MlflowClient, model_name: str
) -> tuple[str | None, float | None, str | None]:
    """Return (version, promo_mean_skill, source_run_id) for the current Production.

    Returns (None, None, None) if there is no Production version yet, OR if
    the existing Production version's source run lacks `promo.mean_skill`
    (which means it was registered by T8 placeholder logic and predates the
    promo metric — treat that as "no comparable Production"). The latter
    case is logged and falls back to auto-promotion.
    """
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except mlflow.exceptions.MlflowException as exc:
        # Model not yet registered at all.
        LOG.info("model %r not yet registered (%s) — treating as no Production", model_name, exc)
        return (None, None, None)

    if not versions:
        return (None, None, None)
    mv = versions[0]
    run_id = mv.run_id
    if not run_id:
        LOG.warning(
            "Production version %s has no source run_id; cannot score — auto-promoting",
            mv.version,
        )
        return (None, None, None)
    try:
        run = client.get_run(run_id)
    except mlflow.exceptions.MlflowException:
        LOG.warning(
            "Production version %s source run %s missing; auto-promoting", mv.version, run_id
        )
        return (None, None, None)
    if "promo.mean_skill" not in run.data.metrics:
        LOG.warning(
            "Production version %s (run %s) has no promo.mean_skill — pre-T9 model; "
            "auto-promoting",
            mv.version,
            run_id,
        )
        return (None, None, None)
    return (mv.version, float(run.data.metrics["promo.mean_skill"]), run_id)


def _register_and_transition(
    client: MlflowClient, run_id: str, model_name: str
) -> tuple[str, tuple[str, ...]]:
    """Register run as a new version and transition it to Production.

    Returns (new_version, archived_versions). archived_versions is the
    versions that moved Production -> Archived as a side effect of the
    `archive_existing_versions=True` transition.
    """
    # Snapshot existing Production versions BEFORE transition so we can
    # report which got archived. Empty if first promotion.
    try:
        before = [v.version for v in client.get_latest_versions(model_name, stages=["Production"])]
    except mlflow.exceptions.MlflowException:
        before = []

    # Use create_model_version directly with the run's artifact URI instead of
    # mlflow.register_model("runs:/..."). The latter requires an MLmodel file
    # at the URI on MLflow >= 3.x; trainers don't log a model artifact in T9
    # (T10 wires that), so we register the run's artifact root as the source
    # and rely on the version tags + source_run_id for traceability.
    try:
        client.create_registered_model(model_name)
        LOG.info("created registered model %s", model_name)
    except mlflow.exceptions.MlflowException as exc:
        if "RESOURCE_ALREADY_EXISTS" not in str(exc) and "already exists" not in str(exc).lower():
            raise

    run = client.get_run(run_id)
    LOG.info("registering run %s as new version of %s", run_id, model_name)
    mv = client.create_model_version(
        name=model_name,
        source=run.info.artifact_uri,
        run_id=run_id,
    )
    LOG.info("registered %s version %s", model_name, mv.version)

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return mv.version, tuple(before)


def _tag_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    run: mlflow.entities.Run,
    model_type: str,
    promo_skill: float,
) -> None:
    """Stamp the new Production version with traceability tags."""
    tags = {
        "git_commit": run.data.tags.get("git_commit", ""),
        "dvc_hash": run.data.tags.get("dvc_hash", ""),
        "model_type": model_type,
        "source_run_id": run.info.run_id,
        "promo_mean_skill": f"{promo_skill:.6f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)


def _print_summary(d: PromotionDecision) -> None:
    print()
    print("=" * 78)
    print(f"promotion decision: {d.decision.upper()}")
    print("=" * 78)
    print(f"  candidate    : {d.candidate_model_type}  run_id={d.candidate_run_id}")
    print(f"    promo.mean_skill : {d.candidate_promo_skill:+.4f}")
    if d.production_run_id is not None:
        print(f"  prior Production:                 run_id={d.production_run_id}")
        print(f"    promo.mean_skill : {d.production_promo_skill:+.4f}")
        delta = d.candidate_promo_skill - (d.production_promo_skill or 0.0)
        print(f"  delta (cand - prod): {delta:+.4f}   (margin={d.margin})")
    else:
        print("  prior Production: (none)")
    if d.new_version:
        print(f"  new Production version: {d.new_version}")
        if d.archived_versions:
            print(f"  archived: {', '.join(d.archived_versions)}")
    if d.notes:
        print(f"  notes: {d.notes}")
    print("=" * 78)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T9: candidate-vs-Production promotion")
    parser.add_argument(
        "--git-sha",
        required=True,
        help="git commit shared by the three candidate training runs",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="path to params.yaml (for margin + splits manifest path)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
