#!/usr/bin/env bash
# scripts/01_install_platform.sh
# Install MinIO, MLflow, Kubeflow Pipelines, and kube-prometheus-stack.
# Assumes 00_cluster_up.sh has run successfully.
# Idempotent: safe to re-run; uses `helm upgrade --install`.

set -euo pipefail

# ---------- credentials ----------
# Override these via env vars for non-dev use. Defaults are class-demo-only.
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin123}"
MLFLOW_DB_PASSWORD="${MLFLOW_DB_PASSWORD:-mlflowpass}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin}"

KFP_VERSION="${KFP_VERSION:-2.4.0}"

echo "==> Adding Helm repos"
helm repo add minio              https://charts.min.io/                              2>/dev/null || true
helm repo add community-charts   https://community-charts.github.io/helm-charts      2>/dev/null || true
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update

# ---------- 1. MinIO ----------
echo "==> Installing MinIO"
helm upgrade --install minio minio/minio -n minio \
  --set rootUser="$MINIO_ROOT_USER" \
  --set rootPassword="$MINIO_ROOT_PASSWORD" \
  --set mode=standalone \
  --set replicas=1 \
  --set resources.requests.memory=512Mi \
  --set persistence.size=20Gi \
  --set buckets[0].name=dvc \
  --set buckets[0].policy=none \
  --set buckets[1].name=mlflow-artifacts \
  --set buckets[1].policy=none

echo "    Waiting for MinIO to be ready..."
kubectl rollout status -n minio statefulset/minio --timeout=600s || \
  kubectl rollout status -n minio deployment/minio --timeout=600s

# ---------- 2. MLflow ----------
echo "==> Installing MLflow"
# Deploy the chart's bundled Bitnami Postgres subchart. The chart auto-wires
# MLflow to it when postgresql.enabled=true (schema disallows also setting
# backendStore.postgres.enabled — it's for external DBs only).
# Artifacts go to the in-cluster MinIO via S3-compatible API.
helm upgrade --install mlflow community-charts/mlflow -n mlflow \
  --set postgresql.enabled=true \
  --set postgresql.auth.username=mlflow \
  --set postgresql.auth.password="$MLFLOW_DB_PASSWORD" \
  --set postgresql.auth.database=mlflow \
  --set backendStore.databaseMigration=true \
  --set backendStore.databaseConnectionCheck=true \
  --set artifactRoot.s3.enabled=true \
  --set artifactRoot.s3.bucket=mlflow-artifacts \
  --set artifactRoot.s3.path="" \
  --set artifactRoot.s3.awsAccessKeyId="$MINIO_ROOT_USER" \
  --set artifactRoot.s3.awsSecretAccessKey="$MINIO_ROOT_PASSWORD" \
  --set extraEnvVars.MLFLOW_S3_ENDPOINT_URL=http://minio.minio.svc.cluster.local:9000 \
  --set extraEnvVars.AWS_DEFAULT_REGION=us-east-1 \
  --set service.type=ClusterIP

echo "    Waiting for MLflow to be ready..."
kubectl rollout status -n mlflow deployment/mlflow --timeout=600s

# ---------- 3. Kubeflow Pipelines (standalone) ----------
echo "==> Installing Kubeflow Pipelines standalone (this takes a few minutes)"
# KFP is installed via kustomize, not Helm. The 'env/platform-agnostic' overlay
# is the lightest install suitable for local clusters.
export PIPELINE_VERSION="$KFP_VERSION"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
kubectl wait --for=condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"

# Workaround: KFP 2.2.0 manifests pin a gcr.io/ml-pipeline/minio image tag that
# no longer exists upstream. Patch the bundled MinIO deployment to a current
# upstream image, and rename the legacy env vars accepted by older MinIO
# releases (MINIO_ACCESS_KEY/MINIO_SECRET_KEY -> MINIO_ROOT_USER/MINIO_ROOT_PASSWORD).
echo "    Patching KFP bundled MinIO (legacy image tag removed upstream)"
kubectl patch deployment -n kubeflow minio --type='json' -p='[
  {"op":"replace","path":"/spec/template/spec/containers/0/image","value":"minio/minio:RELEASE.2024-02-09T21-25-16Z"},
  {"op":"replace","path":"/spec/template/spec/containers/0/env/0/name","value":"MINIO_ROOT_USER"},
  {"op":"replace","path":"/spec/template/spec/containers/0/env/1/name","value":"MINIO_ROOT_PASSWORD"}
]'
# The deployment uses Recreate strategy; if the original (broken-image) pod
# was already created it stays in Pending forever blocking rollout. Force-
# delete any existing minio pod so the new template takes effect.
kubectl delete pod -n kubeflow -l app=minio --force --grace-period=0 --ignore-not-found 2>/dev/null || true

echo "    Waiting for KFP to be ready (this is the slowest install)..."
kubectl wait --for=condition=Ready pods --all -n kubeflow --timeout=600s || \
  echo "    Some KFP pods still starting — check 'kubectl get pods -n kubeflow' in a minute."

# ---------- 4. kube-prometheus-stack ----------
echo "==> Installing kube-prometheus-stack"
helm upgrade --install kps prometheus-community/kube-prometheus-stack -n monitoring \
  --set grafana.adminPassword="$GRAFANA_ADMIN_PASSWORD" \
  --set grafana.service.type=ClusterIP \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.retention=7d \
  --set prometheus.prometheusSpec.resources.requests.memory=512Mi \
  --set alertmanager.alertmanagerSpec.resources.requests.memory=64Mi

echo "    Waiting for Grafana to be ready..."
kubectl rollout status -n monitoring deployment/kps-grafana --timeout=600s

# ---------- summary ----------
cat <<EOF

================================================================
Platform installed.

Port-forward each UI in its own terminal:

  kubectl port-forward -n mlflow      svc/mlflow            5000:80
  kubectl port-forward -n kubeflow    svc/ml-pipeline-ui    8080:80
  kubectl port-forward -n minio       svc/minio-console     9001:9001
  kubectl port-forward -n monitoring  svc/kps-grafana       3000:80

Then visit:
  MLflow   : http://localhost:5000
  KFP      : http://localhost:8080
  MinIO    : http://localhost:9001  (user: $MINIO_ROOT_USER)
  Grafana  : http://localhost:3000  (user: admin, pass: $GRAFANA_ADMIN_PASSWORD)

Save credentials to .env (gitignored). See .env.example as a template.
================================================================
EOF
