"""Submit the T8 pipeline to a Kubeflow Pipelines endpoint.

Usage:
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80   # in another terminal
    python -m pipelines.kubeflow.submit                            # uses git HEAD

Flags:
    --git-sha        full git commit (default: `git rev-parse HEAD`)
    --params-path    path to params.yaml (default: ./params.yaml)
    --dvc-hash       T3-features dvc hash (default: derived from dvc.lock)
    --image          image name (default: solar-train)
    --image-tag      image tag (default: short sha of --git-sha)
    --kfp-endpoint   KFP API URL (default: http://localhost:8080)
    --experiment     KFP experiment name (default: solar-mlops)
    --no-wait        return immediately after submit instead of waiting

The image is NOT pulled — submit.py assumes `minikube image load
solar-train:<tag>` was already run (see docker/README.md).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from kfp import compiler
from kfp.client import Client

from src.models.mlflow_utils import get_dvc_features_hash, get_git_commit

from .pipeline import build_pipeline

LOG = logging.getLogger("submit")

DEFAULT_ENDPOINT = "http://localhost:8080"
DEFAULT_EXPERIMENT = "solar-mlops"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(args.params_path).resolve().parent

    git_sha = args.git_sha or get_git_commit(repo_root)
    dvc_hash = args.dvc_hash or get_dvc_features_hash(repo_root)
    image_tag = args.image_tag or _short_sha(git_sha, repo_root)
    image = f"{args.image}:{image_tag}"

    LOG.info("git_sha   = %s", git_sha)
    LOG.info("dvc_hash  = %s", dvc_hash)
    LOG.info("image     = %s", image)
    LOG.info("endpoint  = %s", args.kfp_endpoint)

    pipeline_fn = build_pipeline(image=image, git_sha=git_sha, dvc_hash=dvc_hash)

    with tempfile.TemporaryDirectory() as tmp:
        package_path = Path(tmp) / "solar-mlops-t8.yaml"
        LOG.info("compiling pipeline -> %s", package_path)
        compiler.Compiler().compile(pipeline_fn, str(package_path))

        client = Client(host=args.kfp_endpoint)
        experiment = client.create_experiment(name=args.experiment)
        LOG.info("experiment %s id=%s", args.experiment, experiment.experiment_id)

        run_name = f"solar-mlops-{image_tag}"
        LOG.info("submitting run: %s", run_name)
        run = client.run_pipeline(
            experiment_id=experiment.experiment_id,
            job_name=run_name,
            pipeline_package_path=str(package_path),
        )
        LOG.info("run submitted: id=%s", run.run_id)
        print(f"\nKFP run: {args.kfp_endpoint}/#/runs/details/{run.run_id}")

        if args.no_wait:
            return 0

        LOG.info("waiting for run to finish (--no-wait to skip)…")
        final = client.wait_for_run_completion(run.run_id, timeout=args.timeout)
        status = final.state
        LOG.info("run finished: state=%s", status)
        return 0 if status == "SUCCEEDED" else 1


def _short_sha(full_sha: str, repo_root: Path) -> str:
    """Best-effort `git rev-parse --short`; falls back to first 7 chars."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", full_sha],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return full_sha[:7]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the T8 Kubeflow pipeline")
    parser.add_argument("--git-sha", help="full git commit (default: HEAD)")
    parser.add_argument("--params-path", default="params.yaml")
    parser.add_argument("--dvc-hash", help="T3 features dvc hash (default: from dvc.lock)")
    parser.add_argument("--image", default="solar-train")
    parser.add_argument("--image-tag", help="image tag (default: short sha)")
    parser.add_argument("--kfp-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds to wait for run")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
