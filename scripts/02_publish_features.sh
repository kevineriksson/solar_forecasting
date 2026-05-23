#!/usr/bin/env bash
# Publish the local feature parquets + splits manifest to MinIO under a
# git-sha prefix so the T8 KFP pods can fetch them via src.common.fetch_features.
#
# Usage:
#   bash scripts/02_publish_features.sh                    # uses git HEAD
#   bash scripts/02_publish_features.sh <git_sha>          # explicit sha
#
# Requires the MinIO port-forward to be running:
#   kubectl port-forward -n minio svc/minio 9000:9000
set -euo pipefail

GIT_SHA="${1:-$(git rev-parse HEAD)}"
BUCKET="${BUCKET:-solar-features}"
ENDPOINT="${MLFLOW_S3_ENDPOINT_URL:-http://localhost:9000}"
ACCESS_KEY="${AWS_ACCESS_KEY_ID:-minioadmin}"
SECRET_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin123}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "publishing features to s3://${BUCKET}/${GIT_SHA}/  (endpoint=${ENDPOINT})"

python - "$REPO_ROOT" "$GIT_SHA" "$BUCKET" "$ENDPOINT" "$ACCESS_KEY" "$SECRET_KEY" <<'PY'
import sys
from pathlib import Path
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

repo_root, git_sha, bucket, endpoint, ak, sk = sys.argv[1:]
repo = Path(repo_root)

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=ak,
    aws_secret_access_key=sk,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

try:
    s3.head_bucket(Bucket=bucket)
except ClientError:
    print(f"creating bucket {bucket}")
    s3.create_bucket(Bucket=bucket)

uploads = [
    (repo / "data/features/train.parquet",            f"{git_sha}/train.parquet"),
    (repo / "data/features/promo.parquet",            f"{git_sha}/promo.parquet"),
    (repo / "data/features/replay.parquet",           f"{git_sha}/replay.parquet"),
    (repo / "data/features/features_manifest.json",   f"{git_sha}/features_manifest.json"),
    (repo / "data/interim/splits.json",               f"{git_sha}/splits.json"),
]
for src, key in uploads:
    if not src.exists():
        sys.exit(f"missing local file: {src} -- run `dvc pull` first")
    print(f"  PUT {src.name} -> s3://{bucket}/{key}")
    s3.upload_file(str(src), bucket, key)
print("publish complete")
PY
