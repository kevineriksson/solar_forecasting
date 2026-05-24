# Solar Forecasting MLOps Pipeline

End-to-end MLOps pipeline for short-term solar irradiance forecasting at a
fixed site (Wolf Point, MT), running on a local minikube cluster.

- **Inputs:** Solcast historicals (2007–present, 15-min resolution, UTC)
- **Outputs:** GHI/DNI/DHI at 15 min and 1 h ahead — 6 forecasts per call
- **Models:** smart persistence (baseline), XGBoost, LSTM — promoted by skill score
- **Deployment:** FastAPI behind Kubernetes, scraped by Prometheus
- **Closed loop:** drift / skill alerts → webhook → fresh KFP retrain run

The deliverable is the pipeline, not just the model. See [docs/report.md](docs/report.md)
for the architecture diagram, results, reproducibility demo, and lessons learned.

## Prerequisites

| Tool | Version |
|---|---|
| `minikube` | ≥ 1.32 (driver: docker) |
| `kubectl` | ≥ 1.29 |
| `helm` | ≥ 3.14 |
| `docker` | ≥ 24 |
| `python` | 3.11 |
| `dvc` | ≥ 3.50 |

Plus ~12 GB free RAM and 60 GB free disk for minikube.

## Quick start

```bash
# 1. Clone + install Python deps
git clone <repo-url>
cd solar_forecasting
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env       # MinIO + MLflow + KFP endpoints (defaults are fine)

# 2. Bring up the cluster + platform services (~10 min)
make cluster-up
make platform

# 3. Run the full demo (~45-60 min on a fresh cluster, ~15 min on a warm one)
make e2e
```

`make e2e` runs the full chain:
build → features → pipeline → serve → monitoring → retrain → replay → trigger-drift → verify.

When it finishes, open the dashboards:

```bash
make ports   # MLflow:5000, KFP:8080, Grafana:3000, Prometheus:9090, MinIO:9001
```

## What lives where

| Path | What's there |
|---|---|
| [src/](src/) | Source for every stage: ingest, features, models, promotion, serving, replay, retrain |
| [pipelines/kubeflow/](pipelines/kubeflow/) | KFP DAG (`ingest → features → train ×3 → promotion`) |
| [k8s/](k8s/) | Manifests by namespace: `platform/`, `serving/`, `replay/`, `retrain/`, `monitoring/` |
| [docker/](docker/) | One Dockerfile per concern: train, serve, replay, retrain |
| [scripts/](scripts/) | `00_cluster_up.sh`, `01_install_platform.sh`, `02_publish_features.sh`, `99_teardown.sh`, `build_report.py`, `verify_e2e.sh` |
| [params.yaml](params.yaml) | Single source of truth for all tunables — paths, lags, thresholds, model hyperparameters |
| [dvc.yaml](dvc.yaml) | Data versioning stages |
| [docs/report.md](docs/report.md) | Final report: architecture, results, reproducibility, lessons learned |
| [docs/demo_script.md](docs/demo_script.md) | Live-presentation walkthrough |
| [tests/](tests/) | Unit + integration suites (`pytest tests/unit`) |

## Common commands

```bash
# Show every target the Makefile knows about.
make help

# Build all four Docker images at $(git rev-parse --short HEAD) and load them
# into minikube. Targets are individually runnable: `make build-train`, etc.
make build

# Run the pipeline only (skips the deploy / drift / verify steps).
make features pipeline

# Trigger a drift alert and watch the retrain webhook fire.
make trigger-drift

# Run end-to-end smoke checks (does NOT modify anything).
make verify

# Regenerate the MLflow-driven results table in docs/report.md.
make report

# Wipe the cluster.
make teardown
```

## Three invariants

Every deployed model must satisfy these:

1. **Code in Git** — every change committed before training.
2. **Data in DVC** — every snapshot and feature table versioned.
3. **MLflow models tagged with `git_commit` + `dvc_hash`** — no exceptions.

The reproducibility section of the report ([docs/report.md §4](docs/report.md))
demonstrates that any promoted model can be rebuilt from this triple.

## Tests

```bash
pip install -e .[dev]
pytest tests/unit          # ~5 s, no cluster needed
pytest tests/integration   # needs MLflow + MinIO port-forwards
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Alerts fire in Prometheus but receiver gets nothing | Alert missing a `namespace` label (kube-prometheus-stack `OnNamespace` matcher strategy) | Already fixed in [k8s/monitoring/alerts.yaml](k8s/monitoring/alerts.yaml). If a new alert is added, copy the `namespace: monitoring` label. |
| `make pipeline` fails with S3 404 on `train.parquet` | Features haven't been published for this SHA | `make features` first, or `bash scripts/02_publish_features.sh $(git rev-parse HEAD)` |
| Training pods OOMKilled mid-pipeline | Cluster too small for parallel xgb + lstm with serve + replay also running | `make teardown && MINIKUBE_MEM_MB=16384 make cluster-up` |
| `kubectl port-forward` complains about ports already in use | Stale port-forwards from a previous session | `pkill -f 'kubectl port-forward'` |
| MLflow port-forward to `:5000` fails on macOS | AirPlay Receiver binds 5000 | Use `MLFLOW_PORT=5001 make ports` (or System Settings → AirDrop & Handoff → off) |

## Contributing

Hooks: pre-commit runs `ruff`, `black`, and `mypy` on every commit. Install with
`pre-commit install`. Tests must pass: `pytest tests/unit`.

For architecture decisions and "how to extend this", read [docs/report.md](docs/report.md).

## License

MIT — see [LICENSE](LICENSE) if present.
