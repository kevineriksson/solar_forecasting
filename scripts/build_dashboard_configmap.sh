#!/usr/bin/env bash
# Build the Grafana dashboard ConfigMap from the JSON source-of-truth.
#
# The kube-prometheus-stack Grafana sidecar auto-imports ConfigMaps in any
# namespace that carry the label `grafana_dashboard=1`, so we ship the
# dashboard as a labelled ConfigMap rather than editing Grafana through its
# API. The JSON file in k8s/monitoring/dashboards/ is the source of truth;
# this script regenerates the ConfigMap whenever the JSON changes.
#
# Usage:
#   scripts/build_dashboard_configmap.sh
#
# Output:
#   k8s/monitoring/dashboard-configmap.yaml  (overwritten in place)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/k8s/monitoring/dashboards/solar_overview.json"
OUT="$ROOT/k8s/monitoring/dashboard-configmap.yaml"

if [[ ! -f "$SRC" ]]; then
  echo "missing dashboard JSON: $SRC" >&2
  exit 1
fi

# `kubectl create configmap --dry-run=client` is the canonical way to generate
# a ConfigMap manifest without contacting the cluster. We then post-process
# with a single yq-like substitution to add the Grafana sidecar label.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

kubectl create configmap solar-overview-dashboard \
  --namespace=monitoring \
  --from-file=solar_overview.json="$SRC" \
  --dry-run=client -o yaml > "$TMP"

# Inject the label the sidecar looks for. We use a portable awk insert rather
# than yq so this works on a vanilla CI box.
awk '
  /^metadata:/ { print; print "  labels:"; print "    grafana_dashboard: \"1\""; next }
  { print }
' "$TMP" > "$OUT"

echo "wrote $OUT"
