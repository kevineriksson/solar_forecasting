# Retrain webhook (T13)

Alertmanager → KFP retrain bridge. When `SolarDriftHigh` or `SolarSkillScoreLow`
fires, Alertmanager POSTs to a tiny FastAPI service that submits a fresh
Kubeflow Pipelines run against the `solar-train:<sha>` image already loaded
into the cluster.

## What lives here

| File | Purpose |
|---|---|
| `secret.yaml` | Placeholder `solar-retrain-webhook` Secret (`WEBHOOK_TOKEN`). Rotate before applying. |
| `deployment.yaml` | Single-replica Deployment of `solar-retrain:<sha>`. |
| `service.yaml` | ClusterIP that Alertmanager + Prometheus dial. |
| `servicemonitor.yaml` | Scrapes `/metrics` (`solar_retrain_*` counters + last-run gauge). |
| `alertmanager-config.yaml` | `AlertmanagerConfig` CR routing `component=solar` alerts to the webhook. |

The receiver code lives at [`src/retrain/webhook.py`](../../src/retrain/webhook.py)
and reuses the pipeline factory from [`pipelines/kubeflow/pipeline.py`](../../pipelines/kubeflow/pipeline.py).

## Build + load + deploy

```bash
# 1. Pre-flight: pick the sha you want retrain to submit for, and confirm a
#    matching solar-train image is already loaded into minikube (the
#    receiver submits the same sha it was built from).
GIT_SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
minikube image ls | grep "solar-train:${SHORT}" \
  || (echo "build solar-train:${SHORT} first (see k8s/serving/README.md)" && exit 1)

# 2. Compute the same dvc features hash the trainer would log.
DVC_HASH=$(python -c "from pathlib import Path; \
  from src.models.mlflow_utils import get_dvc_features_hash; \
  print(get_dvc_features_hash(Path('.')))")

# 3. Build + load.
docker build \
  -f docker/retrain.Dockerfile \
  --build-arg GIT_SHA=${GIT_SHA} \
  --build-arg DVC_HASH=${DVC_HASH} \
  -t solar-retrain:${SHORT} .
minikube image load solar-retrain:${SHORT}

# 4. Rotate the shared secret (DO NOT commit the result).
TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")
kubectl create secret generic solar-retrain-webhook \
  -n monitoring \
  --from-literal=WEBHOOK_TOKEN=${TOKEN} \
  --dry-run=client -o yaml | kubectl apply -f -

# 5. Deploy (skip secret.yaml — step 4 already created it).
kubectl apply -f k8s/retrain/service.yaml
kubectl apply -f k8s/retrain/servicemonitor.yaml
kubectl apply -f k8s/retrain/alertmanager-config.yaml
# Pin the image tag in deployment.yaml to ${SHORT} before applying:
sed -i.bak "s|solar-retrain:.*|solar-retrain:${SHORT}|" k8s/retrain/deployment.yaml
kubectl apply -f k8s/retrain/deployment.yaml
```

## Verify

```bash
# Receiver healthy?
kubectl port-forward -n monitoring svc/solar-retrain 8000:8000 &
curl -s localhost:8000/healthz | jq

# Smoke a fake firing alert through the auth path:
curl -s -X POST localhost:8000/alert \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "status":"firing",
    "alerts":[
      {"status":"firing","labels":{"alertname":"SolarDriftHigh"}}
    ]
  }' | jq
# Expect: {"status":"submitted","run_id":"…","alertname":"SolarDriftHigh",…}

# Repeat within retrain.cooldown_minutes — debounce kicks in:
curl -s -X POST localhost:8000/alert \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"status":"firing","alerts":[{"status":"firing","labels":{"alertname":"SolarDriftHigh"}}]}' \
  | jq
# Expect: {"status":"debounced","reason":"cooldown",…}
```

End-to-end via Alertmanager: lower the PSI threshold in
`k8s/monitoring/alerts.yaml` to ~0.05, re-apply the PrometheusRule, kick off a
replay run, wait for `SolarDriftHigh` to fire, then watch
`kubectl logs -n monitoring deploy/solar-retrain` for `submitted run_id=…`.
Restore the real threshold (`3.0`) and re-apply when done.

## Debounce

`retrain.cooldown_minutes` (default 30) in [`params.yaml`](../../params.yaml).
State is in-memory — single replica, `strategy: Recreate`. A pod restart
clears the cooldown; that's a conscious tradeoff to keep the deploy simple
(no Redis, no CR). The alert's `for: 10m` already filters short-lived spikes.
