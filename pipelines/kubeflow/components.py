"""KFP v2 container components for the T8 pipeline.

Six components, all running the same `solar-train:<git-sha>` image. The image
already contains the source under `/app/src/` (see docker/train.Dockerfile), so
each component just needs:

  1. Feature parquets fetched from MinIO into `/app/data/` (the training scripts
     read `data/features/<split>.parquet` relative to params.yaml's repo root).
  2. The git_commit + dvc_hash exported as env vars so
     `src.models.mlflow_utils` tags every MLflow run correctly without needing
     a working git / dvc CLI inside the pod.
  3. The image-baked `params.yaml` at `/app/params.yaml` (kept in sync because
     we rebuild the image at each git_sha).

The component functions themselves take no inputs: pipeline-level
`git_sha` / `dvc_hash` are injected into each pod as env vars by
`pipeline._attach_runtime`. Passing them as KFP parameters would require
typed inputs whose only effect would be `set_env_variable`, which we already
do explicitly — KFP container_component plus unused scalar inputs trips its
artifact-type validation, so keeping signatures empty is also the path of
least resistance.
"""

from __future__ import annotations

from collections.abc import Callable

from kfp import dsl

# params.yaml is baked into the image at /app/params.yaml (see
# docker/train.Dockerfile). The training scripts use Path(args.params).resolve()
# to derive repo_root, so the file MUST sit at /app/ — a ConfigMap volume mount
# resolves through a `..data` symlink and breaks that assumption. Since we
# rebuild the image at every git_sha anyway, the baked params.yaml is the
# source of truth.
PARAMS_PATH = "/app/params.yaml"

# Wrap module invocations with a feature-fetch step. The training scripts
# expect `data/features/*.parquet` relative to repo root (`/app`).
_TRAIN_WRAPPER = (
    "set -euo pipefail; "
    'echo "[init] git_commit=$GIT_COMMIT_OVERRIDE dvc_hash=$DVC_HASH_OVERRIDE"; '
    "python -m src.common.fetch_features --git-sha $GIT_COMMIT_OVERRIDE; "
    "exec python -m {module} --params {params}"
)

_PROMOTION_CMD = (
    "set -euo pipefail; "
    'echo "[init] git_commit=$GIT_COMMIT_OVERRIDE"; '
    "exec python -m src.promotion.register_staging --git-sha $GIT_COMMIT_OVERRIDE"
)

_VERIFY_CMD = (
    "set -euo pipefail; "
    'echo "[{name}] verifying feature artifacts for git=$GIT_COMMIT_OVERRIDE"; '
    "exec python -m src.common.fetch_features "
    "--git-sha $GIT_COMMIT_OVERRIDE --target-root /tmp/verify"
)


def _make_training_op(name: str, module: str, image: str) -> Callable:
    """Container op running `python -m <module>` after a feature fetch."""

    @dsl.container_component
    def op():
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "-c"],
            args=[_TRAIN_WRAPPER.format(module=module, params=PARAMS_PATH)],
        )

    op.__name__ = name
    return op


def _make_promotion_op(image: str) -> Callable:
    """Promotion step: queries MLflow, registers winner as Staging."""

    @dsl.container_component
    def op():
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "-c"],
            args=[_PROMOTION_CMD],
        )

    op.__name__ = "promotion"
    return op


def _make_passthrough_op(name: str, image: str) -> Callable:
    """Hash-verification placeholder for ingest / features.

    The pipeline does NOT re-run DVC inside KFP (see PROMPTS.md T8 design
    note). These ops exist to (a) document where Stages 1–2 fit in the DAG
    and (b) fail fast if the publish-to-MinIO step was skipped, by attempting
    the same fetch the trainers will rely on.
    """

    @dsl.container_component
    def op():
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "-c"],
            args=[_VERIFY_CMD.format(name=name)],
        )

    op.__name__ = name
    return op


def build_ops(image: str) -> dict[str, Callable]:
    """Return the 6 component factories bound to the given image tag.

    Keys: ingest, features, train_persistence, train_xgb, train_lstm, promotion.
    """
    return {
        "ingest": _make_passthrough_op("ingest", image),
        "features": _make_passthrough_op("features", image),
        "train_persistence": _make_training_op(
            "train_persistence", "src.models.train_persistence", image
        ),
        "train_xgb": _make_training_op("train_xgb", "src.models.xgb_train", image),
        "train_lstm": _make_training_op("train_lstm", "src.models.lstm_train", image),
        "promotion": _make_promotion_op(image),
    }
