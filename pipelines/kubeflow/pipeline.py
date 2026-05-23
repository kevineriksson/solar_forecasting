"""T8 — Kubeflow Pipelines DAG wiring Stages 1–4.

DAG:
                                ┌─► train_persistence ─┐
    ingest ─► features ────────►├─► train_xgb         ─┼─► promotion
                                └─► train_lstm        ─┘

All six pods run the same `solar-train:<git_sha>` image. Each pod:
  • uses the image-baked `params.yaml` at `/app/params.yaml` — kept in sync
    with the running git_sha by virtue of being rebuilt at the same sha
  • reads MinIO + MLflow creds from the `solar-mlops-creds` Secret
  • exports the pipeline-level git_sha / dvc_hash via GIT_COMMIT_OVERRIDE
    and DVC_HASH_OVERRIDE so src.models.mlflow_utils tags runs correctly

DVC stays outside KFP per design (see PROMPTS.md T8 prompt): a local
`dvc repro` + `bash scripts/02_publish_features.sh <sha>` populates
`s3://solar-features/<sha>/` before the pipeline runs.
"""

from __future__ import annotations

from collections.abc import Callable

from kfp import dsl, kubernetes

from .components import build_ops

SECRET_NAME = "solar-mlops-creds"

# Keys we copy from the Secret straight into the pod environment.
_SECRET_ENV_KEYS = {
    "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
    "MLFLOW_S3_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
}


def _attach_runtime(task: dsl.PipelineTask, git_sha: str, dvc_hash: str) -> dsl.PipelineTask:
    """Inject Secret + override env vars onto a task.

    `git_sha` and `dvc_hash` are plain Python strings (compile-time constants),
    not pipeline parameters: KFP's `set_env_variable` rejects parameter
    channels and only accepts static values. Re-binding the pipeline factory
    per submission keeps the run reproducible while satisfying that limit.
    """
    kubernetes.use_secret_as_env(task, secret_name=SECRET_NAME, secret_key_to_env=_SECRET_ENV_KEYS)
    task.set_env_variable("GIT_COMMIT_OVERRIDE", git_sha)
    task.set_env_variable("DVC_HASH_OVERRIDE", dvc_hash)
    # Image is built locally and `minikube image load`-ed; never tries a pull.
    kubernetes.set_image_pull_policy(task, "Never")
    # Disable KFP-level result caching: training runs must produce fresh MLflow
    # entries per submission even if the compiled YAML is byte-identical.
    task.set_caching_options(False)
    return task


def build_pipeline(image: str, git_sha: str, dvc_hash: str) -> Callable:
    """Return a compiled-ready @dsl.pipeline function bound to submission state.

    All three of (image, git_sha, dvc_hash) are baked at compile time. The
    resulting pipeline takes no runtime parameters — submitters get
    reproducibility for free by recompiling each run.
    """
    ops = build_ops(image)

    @dsl.pipeline(
        name="solar-mlops-t8",
        description="T8: ingest -> features -> {persistence, xgb, lstm} -> promotion",
    )
    def solar_pipeline():
        ingest = ops["ingest"]().set_display_name("ingest")
        _attach_runtime(ingest, git_sha, dvc_hash)

        features = ops["features"]().set_display_name("features")
        _attach_runtime(features, git_sha, dvc_hash)
        features.after(ingest)

        train_persistence = ops["train_persistence"]().set_display_name("train_persistence")
        _attach_runtime(train_persistence, git_sha, dvc_hash)
        train_persistence.after(features)

        train_xgb = ops["train_xgb"]().set_display_name("train_xgb")
        _attach_runtime(train_xgb, git_sha, dvc_hash)
        # XGB needs the persistence run to exist (it computes skill vs. it).
        train_xgb.after(train_persistence)

        train_lstm = ops["train_lstm"]().set_display_name("train_lstm")
        _attach_runtime(train_lstm, git_sha, dvc_hash)
        # LSTM also needs persistence (same skill-score lookup).
        train_lstm.after(train_persistence)

        promotion = ops["promotion"]().set_display_name("promotion")
        _attach_runtime(promotion, git_sha, dvc_hash)
        promotion.after(train_xgb, train_lstm)

    return solar_pipeline


__all__ = ["build_pipeline", "SECRET_NAME"]
