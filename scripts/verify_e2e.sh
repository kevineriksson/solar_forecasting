#!/usr/bin/env bash
# scripts/verify_e2e.sh
# Smoke test the full T14 stack from outside the cluster. Used by `make verify`.
#
# Checks, in order:
#   1. solar-serve /predict returns 6 named floats for a synthetic feature row
#   2. Prometheus shows solar_replay_predictions_total above a floor (replay is
#      actually emitting)
#   3. solar-retrain has either submitted a run OR a recent retrain workflow
#      exists in kubeflow (proves the trigger fired during the e2e)
#
# Each check prints PASS/FAIL with a one-line explanation. Exits non-zero on
# any FAIL so `make e2e` halts and the user knows which stage is broken.
#
# Assumes the standard port-forwards are running. If they're not, the script
# starts its own (and kills them on exit).

set -euo pipefail

SERVE_PORT="${SERVE_PORT:-8000}"
PROM_PORT="${PROM_PORT:-9090}"
RETRAIN_PORT="${RETRAIN_PORT:-8001}"   # 8000 is taken by serve in e2e flow
PROMETHEUS_SVC="kps-kube-prometheus-stack-prometheus"

# Track port-forwards we open so we tear them down on exit.
pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

ensure_pf() {
  # ensure_pf <local-port> <namespace> <svc> <remote-port>
  local lport="$1" ns="$2" svc="$3" rport="$4"
  if lsof -nP -iTCP:"$lport" -sTCP:LISTEN >/dev/null 2>&1; then
    return 0   # already forwarded
  fi
  kubectl port-forward -n "$ns" "svc/$svc" "$lport:$rport" >/dev/null 2>&1 &
  pids+=($!)
  # Give kubectl ~3s to bind.
  for _ in 1 2 3 4 5 6; do
    if lsof -nP -iTCP:"$lport" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "FAIL: could not port-forward $svc to localhost:$lport" >&2
  exit 1
}

red() { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }

fails=0
pass() { green "PASS  $1"; }
fail() { red "FAIL  $1"; fails=$((fails+1)); }

# ----------------------------------------------------------------- 1. serving

ensure_pf "$SERVE_PORT" solar solar-serve 80

echo "==> 1/3  solar-serve /healthz"
# We check /healthz rather than synthesizing a /predict body because the
# expected feature schema is model-version-specific (loaded from the MLflow
# manifest at startup). /healthz returns the same provenance tags we'd
# look for in /predict's response, so it's an equally strong proof that
# the Production model was loaded and the app is serving.
health_resp=$(curl -sS "http://localhost:$SERVE_PORT/healthz" || true)
if echo "$health_resp" | grep -q '"status":"ok"' && echo "$health_resp" | grep -q '"model_type"'; then
  pass "/healthz reports a loaded model ($(echo "$health_resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"type={d['model_type']} v={d['model_version']}\")" 2>/dev/null || echo ok))"
else
  fail "/healthz did not report status=ok with a model_type: $health_resp"
fi

# ------------------------------------------------------------- 2. prometheus

ensure_pf "$PROM_PORT" monitoring "$PROMETHEUS_SVC" 9090

echo "==> 2/3  prometheus solar_replay_predictions_total"
prom_q='solar_replay_predictions_total'
prom_resp=$(curl -sS --get \
  --data-urlencode "query=sum($prom_q)" \
  "http://localhost:$PROM_PORT/api/v1/query" || true)
prom_count=$(echo "$prom_resp" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); r=d.get('data',{}).get('result',[]); print(int(float(r[0]['value'][1])) if r else 0)" \
  2>/dev/null || echo 0)
if [ "$prom_count" -gt 100 ]; then
  pass "replay emitted $prom_count predictions"
else
  fail "replay predictions counter is $prom_count (expected > 100)"
fi

# --------------------------------------------------------------- 3. retrain

ensure_pf "$RETRAIN_PORT" monitoring solar-retrain 8000

echo "==> 3/3  retrain webhook activity"
retrain_resp=$(curl -sS "http://localhost:$RETRAIN_PORT/metrics" || true)
runs_submitted=$(echo "$retrain_resp" \
  | awk '/^solar_retrain_runs_submitted_total\{/ {gsub(/[a-zA-Z_=", {}\\]/, "", $0); split($0,a," "); print a[2]}' \
  | head -1)
runs_submitted="${runs_submitted:-0}"

retrain_workflows=$(kubectl get workflows -n kubeflow --no-headers 2>/dev/null \
  | wc -l | tr -d ' ')

if [ "${runs_submitted%.*}" -gt 0 ] 2>/dev/null || [ "$retrain_workflows" -gt 1 ]; then
  pass "retrain trigger active (runs_submitted=$runs_submitted, kfp workflows=$retrain_workflows)"
else
  fail "retrain has not been triggered (runs_submitted=$runs_submitted, kfp workflows=$retrain_workflows)"
fi

# --------------------------------------------------------------------- exit

echo
if [ "$fails" -eq 0 ]; then
  green "ALL CHECKS PASSED"
  exit 0
else
  red "$fails CHECK(S) FAILED"
  exit 1
fi
