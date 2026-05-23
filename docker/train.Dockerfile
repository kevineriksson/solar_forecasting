# syntax=docker/dockerfile:1.7
# Training image: persistence baseline, XGBoost, LSTM all run from here.
# Multi-stage build: deps compile in a heavier builder, runtime ships only the venv.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY docker/requirements-train.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ /app/src/
COPY params.yaml /app/params.yaml

ENV PYTHONPATH=/app

# Default entrypoint is `python -m`; callers pass the module + flags as CMD.
# Examples:
#   docker run solar-train ... src.models.train_persistence --config params.yaml
#   docker run solar-train ... src.models.xgb_train         --config params.yaml
#   docker run solar-train ... src.models.lstm_train        --config params.yaml
ENTRYPOINT ["python", "-m"]
CMD ["src.models.xgb_train", "--help"]
