# k8s/monitoring

Grafana dashboards and PrometheusRule alerts for the solar forecasting
service (T12).

## Layout

| Path | Purpose |
|---|---|
| `dashboards/solar_overview.json` | Source-of-truth Grafana dashboard. Edit this file. |
| `dashboard-configmap.yaml` | Generated ConfigMap that ships the JSON to Grafana. **Do not hand-edit** — regenerate via the script below. |
| `alerts.yaml` | `PrometheusRule` with `SolarDriftHigh` and `SolarSkillScoreLow`. |

## How the dashboard reaches Grafana

The platform's kube-prometheus-stack ships Grafana with a sidecar that
watches every namespace for ConfigMaps labelled `grafana_dashboard=1`. We
ship the dashboard as one such ConfigMap so the GitOps loop is just
`kubectl apply -f`. No Grafana-API calls, no UI clicks.

## Apply

```bash
# Regenerate the ConfigMap from the JSON (only needed when the JSON changed).
scripts/build_dashboard_configmap.sh

# Apply both manifests.
kubectl apply -f k8s/monitoring/dashboard-configmap.yaml
kubectl apply -f k8s/monitoring/alerts.yaml

# Confirm Prometheus picked the rule up.
kubectl get prometheusrule -n monitoring solar-forecasting-alerts
```

## Verify (end-to-end)

```bash
# 1. Make sure serving + replay are up (T10 / T11).
kubectl -n solar get pods

# 2. Start a baseline replay so the dashboard fills in.
kubectl -n solar apply -f k8s/replay/job-demo.yaml

# 3. Open Grafana and confirm the "Solar Forecasting — Overview" dashboard
#    loads with every row populated.
kubectl -n monitoring port-forward svc/kps-grafana 3000:80
# -> http://localhost:3000/d/solar-overview

# 4. Inject synthetic drift to trigger SolarDriftHigh.
#    The shift is applied to features in-flight; PSI on `k_t` should climb
#    above 0.2 within ~5 min, and the alert fires after sustained_minutes (10m).
kubectl -n solar delete job solar-replay-demo --ignore-not-found
kubectl -n solar create -f - <<'YAML'
apiVersion: batch/v1
kind: Job
metadata:
  name: solar-replay-drift
  labels: { app: solar-replay }
spec:
  template:
    metadata:
      labels: { app: solar-replay }
    spec:
      restartPolicy: Never
      containers:
        - name: replay
          image: solar-replay:latest
          imagePullPolicy: IfNotPresent
          env:
            - { name: SERVING_ENDPOINT, value: "http://solar-serve.solar.svc.cluster.local:80" }
          args:
            - "--drift-shift"
            - "k_t=0.4"
            - "--drift-shift"
            - "ghi_lag1=300"
YAML

# 5. Confirm the alert in Alertmanager.
kubectl -n monitoring port-forward svc/kps-alertmanager 9093:9093
# -> http://localhost:9093 — expect SolarDriftHigh `firing` after ~10 minutes.

# 6. Stop the drift run, restart a clean replay, confirm the alert recovers.
kubectl -n solar delete job solar-replay-drift
kubectl -n solar apply -f k8s/replay/job-demo.yaml
```

## Why a separate PSI gauge?

PromQL has no built-in PSI primitive. We could approximate drift in PromQL
using `histogram_quantile` and the existing `solar_input_feature_value`
histogram, but quantile drift and PSI are not interchangeable, and the
existing alert thresholds in `params.yaml` are calibrated against PSI. The
replay client therefore snapshots a reference distribution from
`data/features/train.parquet` at startup, maintains a rolling sample, and
emits `solar_replay_feature_psi{feature=...}` directly.

The implementation lives in [`src/replay/rolling.py`](../../src/replay/rolling.py)
and is unit-tested in
[`tests/unit/test_replay_rolling.py`](../../tests/unit/test_replay_rolling.py).

## Thresholds

| Alert | Source value (params.yaml) | Where it lives in `alerts.yaml` |
|---|---|---|
| Drift threshold (PSI) | `monitoring.drift.threshold = 3.0` | `expr` of `SolarDriftHigh` |
| Drift sustained window | `monitoring.drift.sustained_minutes = 10` | `for: 10m` |
| Skill threshold | `monitoring.skill_alert.threshold = 0.0` | `expr` of `SolarSkillScoreLow` |
| Skill sustained window | `monitoring.skill_alert.sustained_minutes = 10` | `for: 10m` |

`PrometheusRule` has no parameter substitution — if you change the values in
`params.yaml` you must also update the literals here. The replay client's
rolling window size still comes from `params.yaml` (`monitoring.drift.window_steps`).
