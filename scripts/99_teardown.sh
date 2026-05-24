#!/usr/bin/env bash
# scripts/99_teardown.sh
# Wipe the minikube cluster for the solar MLOps project.
# Idempotent: safe to re-run when nothing exists.

set -euo pipefail

PROFILE="${MINIKUBE_PROFILE:-solar-mlops}"

echo "==> Tearing down minikube profile '$PROFILE'"
if minikube status -p "$PROFILE" >/dev/null 2>&1; then
  minikube delete -p "$PROFILE"
else
  echo "    Cluster '$PROFILE' is not running."
  # Even when not running, the profile config can linger — purge if present.
  if minikube profile list 2>/dev/null | grep -q "^| *$PROFILE *|"; then
    minikube delete -p "$PROFILE"
  fi
fi

echo "==> Killing any leftover kubectl port-forward processes"
# These tend to accumulate during development; harmless if none exist.
pkill -f "kubectl port-forward" 2>/dev/null || true

echo
echo "Teardown complete. To rebuild from scratch:"
echo "  make cluster-up && make platform && make e2e"
