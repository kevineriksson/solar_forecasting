# syntax=docker/dockerfile:1.7
# Replay image: minimal — pandas, requests, prometheus_client, pvlib.
# Walks the 6-month replay window, POSTs to the serving endpoint, emits
# residual metrics. T11 implements src.replay.client.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY docker/requirements-replay.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

WORKDIR /app
COPY src/ /app/src/
COPY params.yaml /app/params.yaml

ENV PYTHONPATH=/app

# T11 implements src.replay.client. Smoke test overrides CMD to verify imports.
CMD ["python", "-m", "src.replay.client", "--help"]
