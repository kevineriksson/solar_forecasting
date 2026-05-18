#!/usr/bin/env bash
# scripts/00_cluster_up.sh
# Bring up the minikube cluster for the solar MLOps project.
# Idempotent: safe to re-run.

set -euo pipefail

PROFILE="${MINIKUBE_PROFILE:-solar-mlops}"
CPUS="${MINIKUBE_CPUS:-6}"
MEM_MB="${MINIKUBE_MEM_MB:-12288}"
DISK="${MINIKUBE_DISK:-60g}"
K8S_VERSION="${K8S_VERSION:-v1.29.0}"

echo "==> Checking prerequisites"
for cmd in minikube kubectl helm docker; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: '$cmd' is required but not installed." >&2
    exit 1
  fi
done

echo "==> Starting minikube profile '$PROFILE'"
if minikube status -p "$PROFILE" >/dev/null 2>&1; then
  echo "    Cluster already running, skipping start."
else
  minikube start \
    -p "$PROFILE" \
    --cpus="$CPUS" \
    --memory="$MEM_MB" \
    --disk-size="$DISK" \
    --kubernetes-version="$K8S_VERSION" \
    --driver=docker
fi

echo "==> Enabling addons"
minikube -p "$PROFILE" addons enable storage-provisioner
minikube -p "$PROFILE" addons enable metrics-server

echo "==> Setting kubectl context to '$PROFILE'"
kubectl config use-context "$PROFILE"

echo "==> Creating namespaces"
for ns in minio mlflow kubeflow monitoring solar; do
  kubectl get ns "$ns" >/dev/null 2>&1 || kubectl create ns "$ns"
done

echo
echo "Cluster up. Next: scripts/01_install_platform.sh"
kubectl get nodes