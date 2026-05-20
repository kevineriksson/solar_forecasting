# CLAUDE.md

> Guidance for Claude Code working on this repository.
> Project: **MLOps Pipeline for Short-Term Solar Irradiance Forecasting** (Montana, Solcast data, 2007–present).

---

## 1. Project Overview

This repo is an end-to-end MLOps pipeline that ingests Solcast historical irradiance data for a single Montana site, trains three forecasting models, deploys the winner behind a FastAPI service on Kubernetes, and **simulates production traffic by replaying the most recent 6 months of the dataset** against the live endpoint. Prometheus + Grafana monitor predictions and residuals in real time and trigger retraining on drift or skill degradation.

**The deliverable is the pipeline, not just the model.** Models exist to prove the pipeline carries a real signal end-to-end.

### Site (fixed)
- **Latitude:** 48.30783° N
- **Longitude:** -105.1017° E (northeastern Montana, near Wolf Point)
- **Panel tilt:** 46.59° (≈ latitude tilt)
- **Panel azimuth:** 180° (due south)
- **Timezone for storage:** UTC everywhere. Local time only at presentation layer (dashboards).

### Forecast targets
- Variables: GHI, DNI, DHI (each as a **direct model output** — DNI/DHI are not post-hoc derived from GHI)
- Horizons: **15-minute** (1 step) and **1-hour** (4 steps) ahead
- Data resolution: **15-minute** Solcast historical

### Models compared
| Model | Role | Input |
|---|---|---|
| Smart persistence on clear-sky index `k_t` | Baseline | Last observed `k_t` |
| XGBoost | Main tabular model | Engineered lag + calendar features |
| LSTM | Sequence model | Multivariate sequence windows |

### The three invariants
Every deployed model must be reproducible from this triple. **Do not break these.**
1. **Code in Git** — every change committed before training.
2. **Data in DVC** — every snapshot and feature table versioned.
3. **MLflow models tagged with `git_commit` + `dvc_hash`** — no exceptions.

---

## 2. Tech Stack (Fixed — Do Not Substitute)

| Layer | Tool | Where it runs |
|---|---|---|
| Source control | Git | Local + remote (GitHub) |
| Data versioning | DVC | Local CLI |
| Object store (DVC + MLflow artifacts) | **MinIO** | In-cluster (`minio` ns) |
| Containers | Docker | Local builds, loaded into minikube |
| Compute | **Kubernetes via minikube** | Local laptop |
| Orchestration | Kubeflow Pipelines (standalone) | In-cluster (`kubeflow` ns) |
| Experiment + model registry | MLflow | In-cluster (`mlflow` ns) |
| Serving | FastAPI | In-cluster (`solar` ns) |
| Metrics | Prometheus (via kube-prometheus-stack) | In-cluster (`monitoring` ns) |
| Dashboards + alerts | Grafana (via kube-prometheus-stack) | In-cluster (`monitoring` ns) |
| ML libs | scikit-learn, XGBoost, PyTorch (LSTM), pandas, pvlib | In training containers |

When in doubt, **prefer the stack above** over alternatives, even if an alternative seems simpler.

---

## 3. Cluster Target

**minikube on local laptop.** Required start config:

```bash
minikube start \
  --cpus=6 --memory=12288 --disk-size=60g \
  --kubernetes-version=v1.29.0 \
  --driver=docker
minikube addons enable storage-provisioner
minikube addons enable metrics-server
```

Notes for working with minikube:
- **Image loading:** after `docker build`, run `minikube image load <image>:<tag>` to make images available in-cluster. Do *not* push to a remote registry for local dev.
- **Service access:** prefer `kubectl port-forward` for accessing UIs (MLflow, KFP, Grafana) over `NodePort`.
- **Storage:** rely on minikube's default `standard` StorageClass.
- **DNS inside cluster:** services reach each other via `<svc>.<ns>.svc.cluster.local`. Use this in config, not `localhost`.

---

## 4. Repository Layout (target)

```
.
├── CLAUDE.md                  # this file
├── README.md
├── .env.example               # template; real .env is gitignored
├── dvc.yaml                   # DVC pipeline stages
├── params.yaml                # all hyperparameters + paths
├── data/                      # DVC-tracked, NOT in Git
│   ├── raw/                   # Solcast file(s)
│   ├── interim/               # partitioned + validated
│   └── features/              # versioned feature table
├── src/
│   ├── ingest/                # Stage 1
│   ├── features/              # Stage 2
│   ├── models/                # Stage 3: persistence, xgb, lstm
│   ├── promotion/             # Stage 4
│   ├── serving/               # Stage 5: FastAPI app
│   ├── replay/                # Stage 6: replay client
│   └── common/                # shared utils, schemas, metrics
├── pipelines/
│   └── kubeflow/              # KFP component + pipeline definitions
├── docker/
│   ├── train.Dockerfile
│   ├── serve.Dockerfile
│   └── replay.Dockerfile
├── k8s/
│   ├── README.md              # how to deploy the platform (T0)
│   ├── platform/              # MinIO, MLflow, KFP, monitoring (T0)
│   ├── serving/               # Deployment, Service, HPA (T10)
│   ├── replay/                # CronJob (T11)
│   └── monitoring/            # dashboards, alert rules (T12)
├── scripts/
│   ├── 00_cluster_up.sh
│   ├── 01_install_platform.sh
│   └── 99_teardown.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   └── data/
└── notebooks/                 # exploratory only, not in pipeline
```

---

## 5. Data Splits (Critical — Get This Right)

Three disjoint time windows. Leakage between them is the single most likely silent bug in this project.

| Split | Window | Used in |
|---|---|---|
| **Train** | everything older than 8 months from "now" | Stage 3 training + rolling-origin CV |
| **Promotion validation** | months 7 and 8 back (2 months) | Stage 4 candidate-vs-Production scoring |
| **Replay** | most recent 6 months | Stage 6 simulated production traffic |

Rules:
- Splits are defined by timestamp, not random sampling.
- "Now" is a fixed reference timestamp stored in `params.yaml` (`reference_now`), **not** `datetime.now()`. This keeps the pipeline reproducible.
- The replay window is **never** seen during training or promotion scoring.

At 15-minute resolution that gives roughly:
- Train: ~17 years × 35,040 rows/year ≈ **595,000 rows**
- Promotion: 2 months × ~5,840 rows/month ≈ **11,700 rows**
- Replay: 6 months × ~5,840 rows/month ≈ **35,000 rows** (the replay stream)

---

## 6. Pipeline Stages

The Kubeflow DAG has six stages. Each is a separate Docker image, takes versioned inputs, produces versioned outputs, and is independently re-runnable.

### Stage 1 — Ingestion & Versioning (`src/ingest/`)
Read raw Solcast file (15-min, UTC), partition by year-month, write Parquet to MinIO via DVC. Validate schema (columns, dtypes, ranges, monotonic UTC timestamp, no duplicates, no gaps > 1 step). Emit three split manifests for train / promotion / replay using `reference_now` from `params.yaml`.

### Stage 2 — Preprocessing & Features (`src/features/`)
Clear-sky GHI via `pvlib` for the fixed site (lat 48.30783, lon -105.1017). Compute `k_t = GHI / GHI_clearsky` (clip to `[0, 1.5]`, mask night where zenith ≥ 90°). Lags at `[1, 4, 12]` steps = `[15min, 1h, 3h]`. Rolling means over `[4, 12]` steps. Calendar features: `sin/cos(hour)`, `sin/cos(day_of_year)`. Night mask column. Write versioned feature tables per split.

### Stage 3 — Training & Experimentation (`src/models/`)
Three training jobs in parallel as separate KFP components. **Chronological rolling-origin CV** on the training split — no shuffled k-fold.

Each model produces **6 outputs**: {GHI, DNI, DHI} × {15min, 1h horizons}. DNI and DHI are direct outputs, not derived from GHI. Choose between multi-output regressors or per-output models — document the choice in the run params.

Each run logs to MLflow: params, metrics (MAE, RMSE, skill score vs persistence — per target, per horizon), artifacts, plus **`git_commit` + `dvc_hash` as tags**.

### Stage 4 — Registry & Promotion (`src/promotion/`)
Score the new candidate and the current Production model on the **promotion validation window**. Skill score: `1 - RMSE_model / RMSE_persistence`. Aggregate across the 6 outputs as the mean skill score (give all targets/horizons equal weight unless `params.yaml` says otherwise). If candidate beats current Production by ≥ `promotion_margin`, transition candidate to `Production` and archive previous.

### Stage 5 — Deployment (`src/serving/`)
FastAPI exposes `POST /predict` returning GHI/DNI/DHI for both horizons (6 numbers per call). Loads `Production` model from MLflow at startup. Health endpoint `/healthz`, metrics endpoint `/metrics` (Prometheus format). Build → `minikube image load` → roll out Kubernetes `Deployment` behind a `Service`.

### Stage 6 — Replay Monitoring & Retraining (`src/replay/`)
Kubernetes `CronJob` walks forward through the 6-month replay window at configurable speedup (~500 req/s clears the window in ~2 min). For each prediction, ground truth is already on disk → emit residual metric per target × horizon. Prometheus scrapes latency, request rate, input feature distributions, predictions, rolling skill score. Grafana alerts on drift or sustained skill degradation → triggers a fresh KFP run.

---

## 7. How Claude Code Should Work in This Repo

### Always
- Read `CLAUDE.md`, `params.yaml`, and the target stage's existing code before writing anything new.
- Keep changes scoped to **one task at a time** unless explicitly told otherwise.
- Update `dvc.yaml` and the relevant KFP component when adding/changing a stage's inputs or outputs.
- Add or update a test for any non-trivial change in `src/`.
- Use `params.yaml` for all tunables — never hardcode paths, horizons, lags, thresholds, or window sizes.
- Use `.env` (or Kubernetes Secrets in-cluster) for endpoints/credentials — never hardcode.
- Tag every MLflow run with `git_commit` and `dvc_hash`. If either is missing, fail loudly.
- Keep all timestamps in UTC internally. Convert only at presentation.

### Never
- Use `datetime.now()` to define data splits.
- Mix random k-fold CV with time series — always rolling-origin / expanding-window.
- Train, validate-for-promotion, and replay on overlapping time ranges.
- Substitute a different tool from the stack in Section 2 without flagging it first.
- Commit data, model binaries, or `.env` files to Git. They belong in DVC, MLflow, or secrets.
- Push Docker images to a remote registry for local dev — use `minikube image load`.
- Derive DNI or DHI from forecasted GHI. They are direct outputs.
- Log secrets, API keys, or full request payloads to Prometheus or stdout.

### When uncertain
Stop and ask, or propose two options with tradeoffs. Silent guesses on splits, horizons, or the promotion rule are expensive to undo later.

---

## 8. Task Breakdown for Claude Code

Each task is sized for one focused session. Each has a verifiable **Done when** check.

### Week 1 — Platform, data, features, first models

#### T0. Cluster + platform services bootstrap
Stand up minikube and deploy MinIO, MLflow, Kubeflow Pipelines, and kube-prometheus-stack.

- Run `scripts/00_cluster_up.sh` (minikube start + addons + namespaces).
- Run `scripts/01_install_platform.sh` (Helm + kustomize installs).
- Create `k8s/platform/values/` with Helm values files if defaults need overrides.
- Create `.env` from `.env.example`.
- Document port-forward commands in `k8s/README.md` for each UI.
- **Done when:** `kubectl get pods -A` shows all four stacks `Running` and you can reach MLflow UI, KFP UI, MinIO console, and Grafana via port-forward.

#### T1. Repo + tooling bootstrap
- Initialize Git, `.gitignore` (Python, Docker, DVC, IDE, `.env`).
- Pre-commit: `ruff`, `black`, `mypy`.
- Initialize DVC with MinIO as the remote (endpoint from `.env`).
- Create `params.yaml` (see Section 11 for the full content to use).
- Skeleton folder layout from Section 4.
- **Done when:** `dvc remote list` shows the MinIO remote, pre-commit runs clean on an empty commit, `params.yaml` validates.

#### T2. Solcast ingestion (Stage 1)
- Read raw Solcast file from `data/raw/`. Confirm columns match expected schema; handle Solcast's `period_end` UTC timestamp convention.
- Partition by year-month, write Parquet to MinIO via DVC.
- Schema validation: required columns present, dtypes correct, GHI/DNI/DHI/GTI ≥ 0, zenith in `[0, 180]`, no duplicate timestamps, monotonic UTC index, gaps flagged (15-min steps only).
- Emit three split manifests (timestamp ranges) using `reference_now` and the split rules.
- DVC stage in `dvc.yaml`.
- **Done when:** `dvc repro ingest` produces three non-overlapping splits, `dvc push` succeeds, unit tests pass on a synthetic fixture.

#### T3. Feature engineering (Stage 2)
- Clear-sky GHI via `pvlib` (`Location(48.30783, -105.1017, 'UTC', altitude=...)`; pick altitude from Solcast metadata or ~640 m for Wolf Point).
- Compute `k_t`, lags `[1, 4, 12]` steps, rolling means `[4, 12]` steps, calendar `sin/cos`, night mask.
- Write versioned feature tables per split to `data/features/` (DVC → MinIO).
- Unit tests: night mask correct at known sunrise/sunset for the site; `k_t` ∈ `[0, 1.5]` after clipping; no NaNs after the longest lag window.
- **Done when:** `dvc repro features` runs deterministically and produces identical hashes across two runs on the same input.

#### T4. Persistence baseline (Stage 3a)
- Smart persistence on `k_t`: at forecast time `t+h`, predict `k_t(t+h) = k_t(t)`, then forecast `GHI(t+h) = k_t(t) × GHI_clearsky(t+h)`.
- For DNI and DHI: also persistence on their own clear-sky-indexed values (treat each as a direct output, per project rules).
- Rolling-origin CV on training split, log MAE/RMSE per target per horizon to MLflow.
- This baseline defines the skill-score denominator for every other model. Get it right first.
- **Done when:** MLflow shows a `persistence` run tagged with `git_commit` and `dvc_hash`; metrics degrade with horizon as expected; sanity-check: GHI persistence on a clear winter day should look near-perfect.

#### T5. XGBoost model (Stage 3b)
- Per-target, per-horizon XGBoost regressors (6 models total), or one multi-output wrapper — document choice.
- Trained on engineered features from T3.
- Rolling-origin CV, log to MLflow with the same metric schema as T4.
- Compute skill score vs T4's persistence run.
- **Done when:** mean skill score > 0 on validation folds; otherwise debug features/leakage before T6.

#### T6. LSTM model (Stage 3c)
- PyTorch LSTM on multivariate sequence windows (e.g. 24 steps = 6 hours of history).
- Multi-output head: predict 6 values (3 targets × 2 horizons).
- Same CV protocol, same MLflow logging conventions as T4/T5.
- **Done when:** model trains to convergence, logs to MLflow, produces a skill score on the same validation folds as T5.

### Week 2 — Orchestration, serving, replay, monitoring

#### T7. Dockerize all stages
- One Dockerfile per concern: train, serve, replay.
- Pin Python, OS packages, lib versions. BuildKit caching.
- Build → `minikube image load <image>:<tag>` for each.
- **Done when:** all three images build, load into minikube, and run a smoke command via `kubectl run --rm`.

#### T8. Kubeflow pipeline (Stages 1–4 wired together)
- KFP components for ingest, features, train (×3 in parallel), promotion.
- Pass Git commit + DVC hashes through as pipeline parameters.
- One-click run via `pipelines/kubeflow/submit.py`.
- **Done when:** the pipeline succeeds end-to-end and a candidate model lands in MLflow `Staging`.

#### T9. Promotion logic (Stage 4)
- Candidate-vs-Production scoring on the promotion validation window.
- Promote if candidate beats Production by ≥ `promotion_margin`.
- Guardrail: refuse promotion if candidate's training data overlaps the promotion or replay windows (verify via `dvc_hash` + split manifest).
- **Done when:** deliberate "worse" candidate is rejected, deliberate "better" candidate is promoted, both verified by automated test.

#### T10. FastAPI serving (Stage 5)
- `POST /predict` → 6 numbers (GHI/DNI/DHI × 15min/1h). `GET /healthz`. `GET /metrics`.
- Loads `Production` from MLflow on startup; refuses to start if none exists.
- Prometheus instrumentation: latency histogram, request counter, prediction histograms per target/horizon, input feature histograms.
- `Deployment` + `Service` + readiness/liveness probes + `ServiceMonitor`.
- **Done when:** `curl /predict` returns a valid forecast, `/metrics` exposes expected series, Prometheus shows the target Up.

#### T11. Replay client (Stage 6)
- Walks forward through the 6-month replay window at configurable speedup.
- For each timestep: build the feature vector exactly as Stage 2 would (no future leakage), `POST /predict`, emit `residual = prediction - truth` per target × horizon to Prometheus.
- Runs as `CronJob` (or one-shot `Job` for demos).
- **Done when:** 2-minute demo run produces tens of thousands of predictions + residuals visible in Prometheus.

#### T12. Prometheus + Grafana dashboards & alerts
- `ServiceMonitor`s for serving and replay pods.
- Grafana dashboards (JSON in `k8s/monitoring/dashboards/`): latency, RPS, rolling MAE/RMSE/skill per target × horizon, input feature drift (PSI or KS vs training distribution), prediction distributions.
- `PrometheusRule` alerts: skill below threshold for N minutes; drift above threshold for N minutes.
- **Done when:** dashboards populate during a replay run and a deliberately injected drift fires the alert.

#### T13. Retraining trigger
- Alertmanager → webhook → KFP pipeline run.
- New run uses latest snapshot (splits roll forward), trains, promotes (or rejects).
- **Done when:** a fired alert kicks off a new KFP run automatically and the new candidate appears in MLflow.

#### T14. End-to-end test + final report
- `make e2e`: ingest → features → train all three → promote → deploy → replay → dashboard populated → trigger retrain.
- Final report: architecture, results table (MAE/RMSE/skill at both horizons for all three models), Grafana screenshots, "rebuild from commit + hash" demo.
- **Done when:** fresh clone + `make cluster-up && make platform && make e2e` reproduces the demo.

---

## 9. Definition of Done (for the whole project)

- `git clone` + minikube + `make` targets reproduce the full demo.
- Every model in MLflow `Production` history is traceable to an exact Git commit and DVC hash.
- Grafana shows live predictions and residuals from the replay stream.
- A simulated drift triggers an alert → KFP run → new candidate → promote-or-reject under the same rules as the initial run.
- README explains how to run it; CLAUDE.md (this file) explains how to extend it.

---

## 10. Quick Reference — Common Commands

```bash
# Cluster
minikube start --cpus=6 --memory=12288 --disk-size=60g --driver=docker
minikube stop
minikube delete

# Platform UIs (port-forward each in its own terminal)
kubectl port-forward -n mlflow      svc/mlflow            5000:5000
kubectl port-forward -n kubeflow    svc/ml-pipeline-ui    8080:80
kubectl port-forward -n minio       svc/minio-console     9001:9001
kubectl port-forward -n monitoring  svc/kps-grafana       3000:80

# DVC / data
dvc repro                         # run pipeline locally
dvc push                          # push artifacts to MinIO

# Train a single model locally against in-cluster MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.models.xgb_train --config params.yaml

# Build + load image into minikube
docker build -f docker/serve.Dockerfile -t solar-serve:$(git rev-parse --short HEAD) .
minikube image load solar-serve:$(git rev-parse --short HEAD)

# Submit KFP pipeline
python pipelines/kubeflow/submit.py --git-sha $(git rev-parse HEAD)

# Replay (dev mode)
kubectl port-forward -n solar svc/solar-serve 8000:80
python -m src.replay.client --speedup 500 --endpoint http://localhost:8000/predict
```

---

## 11. Resolved Parameters (final)

All decisions baked in. See `params.yaml` for the machine-readable version.

| Item | Value |
|---|---|
| Site latitude | 48.30783 |
| Site longitude | -105.1017 |
| Panel tilt | 46.59° |
| Panel azimuth | 180° |
| Timezone (storage) | UTC |
| Raw data resolution | 15 minutes |
| Forecast horizons | 15 min (1 step), 1 hour (4 steps) |
| Forecast targets | GHI, DNI, DHI — direct outputs each |
| Number of model outputs | 6 (3 targets × 2 horizons) |
| Lags (feature engineering) | 1, 4, 12 steps (15min, 1h, 3h) |
| Rolling-mean windows | 4, 12 steps (1h, 3h) |
| `k_t` clipping range | `[0, 1.5]` |
| Night mask | zenith ≥ 90° |
| Train window | older than 8 months from `reference_now` |
| Promotion window | months 7–8 back |
| Replay window | most recent 6 months |
| Promotion margin | 0.02 (skill score) |
| Cluster | minikube, 6 CPU / 12 GB / 60 GB |
| Object store | MinIO in-cluster |
| Image registry | None (`minikube image load`) |
