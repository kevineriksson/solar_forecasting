"""Fetch feature parquets + splits manifest from MinIO into /app/data/.

Used by the T8 KFP components as a pod-startup step so the training scripts
find the same files they expect locally (under `data/features/`).

Contract:
    Reads from   s3://<bucket>/<git_sha>/{train,promo,replay}.parquet
                 s3://<bucket>/<git_sha>/splits.json
                 s3://<bucket>/<git_sha>/features_manifest.json
    Writes to    /app/data/features/{train,promo,replay}.parquet
                 /app/data/features/features_manifest.json
                 /app/data/interim/splits.json

The bucket defaults to `solar-features`. Endpoint + creds come from the
standard `MLFLOW_S3_ENDPOINT_URL` / `AWS_*` env vars MLflow already uses, so
one Secret feeds both clients.

Why a separate `solar-features/<git_sha>/` prefix instead of `dvc pull`:
    DVC stores blobs at MD5-derived paths that are hard to address from a
    pod without shipping `dvc.yaml` + `dvc.lock`. A flat per-sha prefix keeps
    the pod startup logic trivial and aligns the data with the immutable
    git_commit the pipeline already passes around. `scripts/02_publish_features.sh`
    populates this prefix from a local `dvc pull`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config

LOG = logging.getLogger("fetch_features")

DEFAULT_BUCKET = "solar-features"
DEFAULT_TARGET_ROOT = Path("/app")

KEYS = [
    ("train.parquet", "data/features/train.parquet"),
    ("promo.parquet", "data/features/promo.parquet"),
    ("replay.parquet", "data/features/replay.parquet"),
    ("features_manifest.json", "data/features/features_manifest.json"),
    ("splits.json", "data/interim/splits.json"),
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL")
    if not endpoint:
        raise RuntimeError("MLFLOW_S3_ENDPOINT_URL (or S3_ENDPOINT_URL) is required to reach MinIO")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY required (MinIO creds)")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    target_root = Path(args.target_root).resolve()
    LOG.info("fetching s3://%s/%s/  ->  %s", args.bucket, args.git_sha, target_root)

    for key_suffix, rel_path in KEYS:
        src_key = f"{args.git_sha}/{key_suffix}"
        dst = target_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("  GET s3://%s/%s -> %s", args.bucket, src_key, dst)
        s3.download_file(args.bucket, src_key, str(dst))

    LOG.info("fetch complete")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch T8 feature parquets from MinIO")
    parser.add_argument(
        "--git-sha",
        default=os.environ.get("GIT_COMMIT_OVERRIDE"),
        help="git commit (S3 prefix); defaults to $GIT_COMMIT_OVERRIDE",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="MinIO bucket")
    parser.add_argument(
        "--target-root",
        default=str(DEFAULT_TARGET_ROOT),
        help="local root that mirrors the repo (data/features will be under here)",
    )
    args = parser.parse_args(argv)
    if not args.git_sha:
        parser.error("--git-sha (or $GIT_COMMIT_OVERRIDE) is required")
    return args


if __name__ == "__main__":
    sys.exit(main())
