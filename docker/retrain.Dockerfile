# syntax=docker/dockerfile:1.7
# Retrain webhook image (T13). Tiny FastAPI receiver that converts an
# Alertmanager webhook into a KFP run submission. Image bakes in the
# git_sha + dvc_hash it was built from — those are the SHAs it submits
# training with, since `solar-train:<sha>` must already be loaded into
# minikube and they share that SHA.
#
# Build:
#   GIT_SHA=$(git rev-parse HEAD)
#   DVC_HASH=$(python -c "from pathlib import Path; \
#     from src.models.mlflow_utils import get_dvc_features_hash; \
#     print(get_dvc_features_hash(Path('.')))")
#   docker build -f docker/retrain.Dockerfile \
#     --build-arg GIT_SHA=$GIT_SHA --build-arg DVC_HASH=$DVC_HASH \
#     -t solar-retrain:$(git rev-parse --short HEAD) .
#   minikube image load solar-retrain:$(git rev-parse --short HEAD)

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY docker/requirements-retrain.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

WORKDIR /app

# Source modules: the webhook itself + the pipeline factory it submits.
COPY src/ /app/src/
COPY pipelines/ /app/pipelines/
COPY params.yaml /app/params.yaml

# Build-time reproducibility tags. ARG keeps Dockerfile reusable across
# rebuilds; ENV makes them visible at runtime to RetrainConfig.load().
ARG GIT_SHA
ARG DVC_HASH
ENV RETRAIN_GIT_SHA=${GIT_SHA} \
    RETRAIN_DVC_HASH=${DVC_HASH} \
    PYTHONPATH=/app \
    SOLAR_PARAMS_PATH=/app/params.yaml

EXPOSE 8000

CMD ["uvicorn", "src.retrain.webhook:app", "--host", "0.0.0.0", "--port", "8000"]
