# docker/

One Dockerfile per concern: train, serve, replay. Built locally and loaded into
minikube via `minikube image load` — no remote registry for local dev.

## Images

| Image | Purpose | Base | Size (uncompressed) | Key deps |
|---|---|---|---|---|
| `solar-train` | Persistence baseline, XGBoost, LSTM training jobs. Entrypoint is `python -m`; pass the module as CMD. | `python:3.11-slim-bookworm`, multi-stage | ~2.2 GB | xgboost, torch (CPU), scikit-learn, pvlib, mlflow |
| `solar-serve` | FastAPI + uvicorn loading Production model from MLflow. CPU-only; torch deliberately excluded until an LSTM wins promotion (see `requirements-serve.txt`). | `python:3.11-slim-bookworm`, multi-stage | ~1.6 GB | fastapi, uvicorn, mlflow, xgboost, prometheus-client |
| `solar-replay` | Walks the 6-month replay window, POSTs to serving, emits residual metrics. | `python:3.11-slim-bookworm`, single-stage | ~960 MB | pandas, requests, pvlib, prometheus-client |

All three are built with BuildKit (`# syntax=docker/dockerfile:1.7`) and use a
pip cache mount so rebuilds reuse the wheel cache.

## Build + load

```bash
SHA=$(git rev-parse --short HEAD)

docker build -f docker/train.Dockerfile  -t solar-train:$SHA  -t solar-train:latest  .
docker build -f docker/serve.Dockerfile  -t solar-serve:$SHA  -t solar-serve:latest  .
docker build -f docker/replay.Dockerfile -t solar-replay:$SHA -t solar-replay:latest .

minikube image load solar-train:$SHA  -p solar-mlops
minikube image load solar-serve:$SHA  -p solar-mlops
minikube image load solar-replay:$SHA -p solar-mlops
```

## Smoke tests (T7 "Done when")

Each image is verified in-cluster with `kubectl run --rm`. The image entrypoints
for T10 (`src.serving.app`) and T11 (`src.replay.client`) don't exist yet, so the
smoke tests override CMD with an import check.

```bash
# train
kubectl run smoke-train --rm -i --restart=Never \
  --image=solar-train:$SHA --image-pull-policy=Never \
  --overrides='{"spec":{"containers":[{"name":"smoke-train","image":"solar-train:'$SHA'","imagePullPolicy":"Never","command":["python","-c","import xgboost, torch, sklearn, pvlib, mlflow; print(\"train OK\")"]}]}}'

# serve
kubectl run smoke-serve --rm -i --restart=Never \
  --image=solar-serve:$SHA --image-pull-policy=Never \
  --overrides='{"spec":{"containers":[{"name":"smoke-serve","image":"solar-serve:'$SHA'","imagePullPolicy":"Never","command":["python","-c","import fastapi, uvicorn, mlflow, prometheus_client, xgboost; print(\"serve OK\")"]}]}}'

# replay
kubectl run smoke-replay --rm -i --restart=Never \
  --image=solar-replay:$SHA --image-pull-policy=Never \
  --overrides='{"spec":{"containers":[{"name":"smoke-replay","image":"solar-replay:'$SHA'","imagePullPolicy":"Never","command":["python","-c","import pandas, requests, prometheus_client, pvlib; print(\"replay OK\")"]}]}}'
```

`--image-pull-policy=Never` forces minikube to use the locally-loaded image
instead of trying to pull from Docker Hub.

## Layout

```
docker/
├── README.md                 # this file
├── train.Dockerfile          # multi-stage, ships /opt/venv only
├── serve.Dockerfile          # multi-stage, FastAPI runtime
├── replay.Dockerfile         # single-stage, minimal client
├── requirements-train.txt
├── requirements-serve.txt    # torch intentionally absent; see header comment
└── requirements-replay.txt
```

The repo-root `.dockerignore` keeps `data/`, `.dvc/cache/`, `mlruns/`, notebooks,
tests, and Git/IDE noise out of the build context.

## Next

- T10 adds `src.serving.app:app`; the serve image's default CMD already points at it.
- T11 adds `src.replay.client`; the replay image's default CMD already points at it.
- If an LSTM is ever the Production model, add `torch==2.4.1+cpu` (from
  `https://download.pytorch.org/whl/cpu`) to `requirements-serve.txt` and rebuild.
