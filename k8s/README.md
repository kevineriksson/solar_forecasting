# Platform — local minikube

This directory holds the Kubernetes manifests and notes for the platform layer
(MinIO, MLflow, Kubeflow Pipelines, kube-prometheus-stack). The actual install
is driven by `scripts/01_install_platform.sh`. This file documents how to
reach each UI and the workarounds baked into the install for upstream
breakage.

## Bring it up

```bash
bash scripts/00_cluster_up.sh        # minikube + namespaces
bash scripts/01_install_platform.sh  # MinIO, MLflow, KFP, kube-prometheus-stack
```

Cluster spec: 6 CPU, 12 GB, 60 GB on minikube via docker driver. Anything below
~12 GB is known to be unstable — the initial install pulls dozens of images
in parallel and a 7 GB cluster melts under the load (Docker container becomes
unresponsive, `kubectl` returns `Unable to connect to the server: EOF`).

## Reaching each UI

Run each `port-forward` in its own terminal. They must stay running while
you use the UI.

| UI       | Command                                                                | Default creds                              |
|----------|------------------------------------------------------------------------|--------------------------------------------|
| MLflow   | `kubectl port-forward -n mlflow      svc/mlflow            5000:80`    | none                                        |
| KFP      | `kubectl port-forward -n kubeflow    svc/ml-pipeline-ui    8080:80`    | none                                        |
| MinIO    | `kubectl port-forward -n minio       svc/minio-console     9001:9001`  | `minioadmin` / `minioadmin123`              |
| Grafana  | `kubectl port-forward -n monitoring  svc/kps-grafana       3000:80`    | `admin` / `admin`                           |

Then open:

- MLflow   — http://localhost:5000
- KFP      — http://localhost:8080
- MinIO    — http://localhost:9001
- Grafana  — http://localhost:3000

### macOS port-5000 caveat

On modern macOS, AirPlay Receiver listens on port 5000 and will answer the
MLflow port-forward with `HTTP 403 Server: AirTunes/…`. If you hit that:

- Either disable AirPlay Receiver (System Settings → General → AirDrop &
  Handoff → AirPlay Receiver → Off), or
- Map a different local port: `kubectl port-forward -n mlflow svc/mlflow 5001:80`
  and use http://localhost:5001.

## In-cluster service DNS

For pods that need to reach these services (e.g., training jobs, FastAPI
serving), use the in-cluster DNS — never `localhost`:

| Service           | In-cluster URL                                            |
|-------------------|-----------------------------------------------------------|
| MinIO S3 API      | `http://minio.minio.svc.cluster.local:9000`               |
| MinIO console     | `http://minio.minio.svc.cluster.local:9001`               |
| MLflow            | `http://mlflow.mlflow.svc.cluster.local`  (port 80)       |
| MLflow Postgres   | `mlflow-postgresql.mlflow.svc.cluster.local:5432`         |
| KFP API           | `http://ml-pipeline.kubeflow.svc.cluster.local:8888`      |
| KFP UI            | `http://ml-pipeline-ui.kubeflow.svc.cluster.local`        |
| Prometheus        | `http://kps-kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090` |
| Grafana           | `http://kps-grafana.monitoring.svc.cluster.local`         |

## Workarounds baked into the install

These are encoded in `scripts/01_install_platform.sh`; documented here so the
next person knows _why_:

1. **MLflow chart requires bundled Postgres, not an external pointer.** The
   community-charts `mlflow` chart's JSON schema rejects setting
   `backendStore.postgres.enabled=true` together with `postgresql.enabled=true`.
   The bundled Bitnami subchart is the only supported in-chart Postgres path.
   We pass `postgresql.enabled=true` plus `postgresql.auth.{username,password,
   database}`; the chart auto-wires MLflow to it.

2. **KFP 2.2.0 images are gone from gcr.io.** The whole
   `gcr.io/ml-pipeline/*` tag list is empty as of this writing
   (`gcr.io/ml-pipeline/frontend`, `gcr.io/ml-pipeline/mysql`, etc.). KFP 2.4.0
   migrated to `ghcr.io/kubeflow/kfp-*` for first-party images; the install
   uses `KFP_VERSION=2.4.0` for that reason.

3. **KFP 2.4.0 still references the dead `gcr.io/ml-pipeline/minio` for its
   bundled artifact-storage MinIO.** The script patches the `minio`
   Deployment in `kubeflow` ns to use upstream `minio/minio:RELEASE.2024-02-09T21-25-16Z`
   and renames the legacy `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` env vars to
   `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` (required by the upstream image).
   The Deployment's strategy is `Recreate`; if the broken-image pod was
   created first it sits Pending forever and blocks the rollout — the script
   force-deletes it right after the patch so the new template can take effect.

## Common operations

```bash
# Status across all platform namespaces at a glance
kubectl get pods -n minio -n mlflow -n kubeflow -n monitoring

# Tail logs for a specific component
kubectl logs -n kubeflow deploy/ml-pipeline -f
kubectl logs -n mlflow   deploy/mlflow      -f

# Reset just one component (Helm-managed)
helm uninstall -n mlflow mlflow
bash scripts/01_install_platform.sh   # re-runs everything idempotently

# Nuke the whole platform and start over
minikube delete -p solar-mlops
bash scripts/00_cluster_up.sh
bash scripts/01_install_platform.sh
```

## Subdirectories

- `platform/values/` — Helm values files (currently empty; if a chart needs
  more overrides than two or three `--set` flags, drop a values file here and
  reference it from the script with `-f k8s/platform/values/<name>.yaml`).
- (later) `serving/` — Deployment/Service/HPA for the FastAPI predictor (T10).
- (later) `replay/` — CronJob for the replay client (T11).
- (later) `monitoring/` — ServiceMonitors, dashboards, alert rules (T12).
