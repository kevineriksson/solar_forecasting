"""Shared MLflow helpers: tracking URI resolution + reproducibility tags.

Project invariant: every MLflow run MUST carry `git_commit` and `dvc_hash`
tags. We fail loudly here rather than silently logging an empty tag.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import yaml


def resolve_tracking_uri() -> str:
    """Return the MLflow tracking URI from MLFLOW_TRACKING_URI; raise if unset."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. "
            "Port-forward MLflow and export it, e.g.:\n"
            "  kubectl port-forward -n mlflow svc/mlflow 5000:5000\n"
            "  export MLFLOW_TRACKING_URI=http://localhost:5000"
        )
    return uri


def get_git_commit(repo_root: Path) -> str:
    """Return `git rev-parse HEAD` for the repo. Raise if not a git repo or empty.

    If `GIT_COMMIT_OVERRIDE` is set in the environment, use it verbatim. This is
    how KFP components (which don't have a working .git directory) supply the
    commit they were submitted with.
    """
    override = os.environ.get("GIT_COMMIT_OVERRIDE", "").strip()
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"failed to read git_commit from {repo_root}: {exc}") from exc
    commit = out.stdout.strip()
    if not commit:
        raise RuntimeError(f"git rev-parse HEAD returned empty in {repo_root}")
    return commit


def get_dvc_features_hash(repo_root: Path) -> str:
    """Hash identifying the exact T3 feature artifacts used for training.

    Pulls the `features` stage's `outs[].md5` values from `dvc.lock`,
    concatenates them in sorted-path order, and returns their sha256 hex digest.
    This binds an MLflow run to the precise feature Parquet files it consumed.

    If `DVC_HASH_OVERRIDE` is set in the environment, use it verbatim. KFP
    components ship without `dvc.lock` and rely on the submit-time hash being
    passed in.
    """
    override = os.environ.get("DVC_HASH_OVERRIDE", "").strip()
    if override:
        return override
    lock_path = repo_root / "dvc.lock"
    if not lock_path.exists():
        raise RuntimeError(f"dvc.lock not found at {lock_path}")

    lock = yaml.safe_load(lock_path.read_text())
    try:
        outs = lock["stages"]["features"]["outs"]
    except KeyError as exc:
        raise RuntimeError(f"features stage not in dvc.lock: {exc}") from exc

    pieces: list[str] = []
    for out in sorted(outs, key=lambda o: o["path"]):
        md5 = out.get("md5")
        if not md5:
            raise RuntimeError(f"features stage out missing md5: {out}")
        pieces.append(f"{out['path']}:{md5}")

    if not pieces:
        raise RuntimeError("features stage has no outs in dvc.lock")

    digest = hashlib.sha256("|".join(pieces).encode("utf-8")).hexdigest()
    return digest


def reproducibility_tags(repo_root: Path) -> dict[str, str]:
    """Build the mandatory {git_commit, dvc_hash} tag dict; raise on missing inputs."""
    return {
        "git_commit": get_git_commit(repo_root),
        "dvc_hash": get_dvc_features_hash(repo_root),
    }
